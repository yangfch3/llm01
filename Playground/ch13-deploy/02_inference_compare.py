"""ch13 练习 2：fp32 / fp16 / 模拟 int8 推理对比。

用 GPT-2 small 跑同一段输入，对比：
  - 输出 logits 的差异（量化是否破坏分布）
  - 单 token 推理耗时（fp16 在 GPU 上是否真省时间）
  - 模型权重显存/内存占用

精度档：
  - fp32：PyTorch 默认
  - fp16：half precision，CUDA / MPS 都支持；CPU 上 PyTorch 仅部分 op 支持，本脚本仅在非 CPU 设备启用
  - 模拟 int8：把所有 nn.Linear 权重用对称量化"反量化回 fp32 数值"（数值上与 int8 等价，
    只是为了能直接 forward），看精度退化。**不会**变快——这是"精度模拟"，不是真量化推理

为什么不用 PyTorch 的 torch.ao.quantization.quantize_dynamic？
  在 GPT-2 这类 attention 密集模型上：
    ① Conv1D（HF 的 GPT-2 Linear 实现）需特殊处理
    ② packed weight 存在 ScriptObject 里，state_dict 序列化反而比 fp32 大（含元数据）
    ③ 精度损失大到 top-1 都对不上，演示效果反而误导
  生产用 GGUF Q4_K_M（M6 阶段），那才是真正可用的量化方案。
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Echo.shared.device import get_device  # noqa: E402

PROMPT = "The future of artificial intelligence is"
N_WARMUP = 3
N_TIMED = 20


def model_size_mb(model: nn.Module, dtype_override: torch.dtype | None = None) -> float:
    """估算模型权重占用（MB）：sum(numel × bytes_per_element)。

    dtype_override：用于"假设全部参数按 X dtype 存"算理论大小（不真改模型）。
    """
    if dtype_override is not None:
        bytes_per = torch.zeros(1, dtype=dtype_override).element_size()
        return sum(p.numel() for p in model.parameters()) * bytes_per / (1024 ** 2)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / (1024 ** 2)


@torch.no_grad()
def time_forward(model: nn.Module, ids: torch.Tensor, n: int) -> float:
    """连续 forward n 次，返回平均单次毫秒。"""
    # 预热：避免首次 forward 包含编译/缓存开销
    for _ in range(N_WARMUP):
        _ = model(ids).logits
    # 同步设备（CUDA / MPS 是异步的，必须 sync 才能拿到准确耗时）
    if ids.is_cuda:
        torch.cuda.synchronize()
    elif ids.device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        _ = model(ids).logits
    if ids.is_cuda:
        torch.cuda.synchronize()
    elif ids.device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / n * 1000


@torch.no_grad()
def get_last_logits(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """取最后一个位置的 logits（即"下一个 token 分布"）。"""
    return model(ids).logits[0, -1].float().cpu()


def fake_quantize_to_intN(model: nn.Module, bits: int) -> nn.Module:
    """把模型里所有 Linear / Conv1D 的 weight 替换为"int{bits} 量化再反量化"的 fp32 值。

    数值上等价于"用 int{bits} 存权重 + 推理时反量化回 fp32"，但实际仍走 fp32 GEMM——
    所以只能演示精度退化，不能演示速度收益。
    """
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    quantized = copy.deepcopy(model)
    for module in quantized.modules():
        # GPT-2 内部用的是 transformers.pytorch_utils.Conv1D（不是 nn.Conv1d，是带转置的 Linear）
        # 这里偷懒：凡是 weight 形状 2D 且不是 Embedding 的，都量化
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            w = module.weight.data
            if w.dim() != 2 or isinstance(module, nn.Embedding):
                continue
            max_abs = w.abs().max().clamp(min=1e-8)
            scale = max_abs / qmax
            q = torch.clamp(torch.round(w / scale), qmin, qmax)
            module.weight.data = q * scale  # 反量化回 fp32 数值空间
    return quantized


def main() -> None:
    device = get_device()
    print(f"[device] {device}")

    print("加载 GPT-2 small...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    ids_cpu = tokenizer.encode(PROMPT, return_tensors="pt")

    # ---------- fp32 baseline（CPU）----------
    print("\n" + "=" * 64)
    print("[1/3] fp32 baseline（CPU）")
    print("=" * 64)
    model_fp32 = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    size_fp32 = model_size_mb(model_fp32)
    t_fp32 = time_forward(model_fp32, ids_cpu, N_TIMED)
    logits_fp32 = get_last_logits(model_fp32, ids_cpu)
    print(f"  权重大小   = {size_fp32:.1f} MB")
    print(f"  forward 平均 = {t_fp32:.2f} ms")

    # ---------- fp16（GPU/MPS only）----------
    print("\n" + "=" * 64)
    print("[2/3] fp16（half precision）")
    print("=" * 64)
    if device.type in ("cuda", "mps"):
        model_fp16 = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float16).to(device).eval()
        size_fp16 = model_size_mb(model_fp16)
        ids_dev = ids_cpu.to(device)
        t_fp16 = time_forward(model_fp16, ids_dev, N_TIMED)
        logits_fp16 = get_last_logits(model_fp16, ids_dev)
        max_diff = (logits_fp16 - logits_fp32).abs().max().item()
        same_top1 = logits_fp16.argmax().item() == logits_fp32.argmax().item()
        print(f"  权重大小   = {size_fp16:.1f} MB （≈ fp32 / 2）")
        print(f"  forward 平均 = {t_fp16:.2f} ms  (在 {device} 上)")
        print(f"  与 fp32 logits 的 max 差 = {max_diff:.4f}")
        print(f"  下一个 token top-1 与 fp32 一致：{same_top1}")
    else:
        size_fp16 = None
        t_fp16 = None
        print("  当前设备是 CPU，跳过 fp16 演示（PyTorch CPU fp16 op 覆盖不全）。")

    # ---------- 模拟 int8 / int4（仅看精度退化，不看速度）----------
    print("\n" + "=" * 64)
    print("[3/3] 模拟 int8 / int4：把 Linear 权重量化再反量化，观察精度退化")
    print("=" * 64)
    print("  （weight 数值已退化到 int{N} 网格上，但实际计算仍走 fp32 GEMM —— 不会变快）")
    for bits in [8, 4]:
        m_q = fake_quantize_to_intN(model_fp32, bits)
        m_q.eval()
        size_theory = model_size_mb(m_q, dtype_override=torch.float16) * (bits / 16)
        logits_q = get_last_logits(m_q, ids_cpu)
        max_diff = (logits_q - logits_fp32).abs().max().item()
        same_top1 = logits_q.argmax().item() == logits_fp32.argmax().item()
        # 跟 fp32 比 top-5 重合度，更能反映"分布是否大体对齐"
        top5_fp32 = set(logits_fp32.topk(5).indices.tolist())
        top5_q = set(logits_q.topk(5).indices.tolist())
        overlap = len(top5_fp32 & top5_q)
        print(f"\n  int{bits} 模拟：")
        print(f"    理论权重大小（若真存 int{bits}）≈ {size_theory:.1f} MB")
        print(f"    与 fp32 logits 的 max 差 = {max_diff:.4f}")
        print(f"    下一个 token top-1 与 fp32 一致：{same_top1}")
        print(f"    top-5 与 fp32 重合：{overlap}/5")

    # ---------- 总结 ----------
    print("\n" + "=" * 64)
    print("[总结]")
    print("=" * 64)
    print(f"  fp32         {size_fp32:>6.1f} MB   {t_fp32:>6.2f} ms (CPU)")
    if size_fp16 is not None:
        print(f"  fp16         {size_fp16:>6.1f} MB   {t_fp16:>6.2f} ms ({device})")
    print(f"  int8 (理论)  ≈ {size_fp32 / 4:>4.1f} MB   速度需走专用 kernel（GGUF / GPTQ / dyn-quant）")
    print(f"  int4 (理论)  ≈ {size_fp32 / 8:>4.1f} MB   同上")
    print("\n  - fp16 在 GPU 上既省一半显存又显著提速，是训练后推理的最佳起点")
    print("  - int8/int4 真要省时间，必须配套 int 矩阵乘 kernel —— PyTorch 原生不擅长，")
    print("    生产走 llama.cpp（GGUF）或 vLLM（GPTQ/AWQ）")
    print("  - 本脚本第 3 段是『精度沙盒』：用 fp32 算力模拟 int{N} 数值，看分布退化是否在可接受范围")


if __name__ == "__main__":
    main()
