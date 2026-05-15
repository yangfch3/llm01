"""ch13 练习 1：玩具版量化，看舍入误差与累积现象。

做三件事：
  1. 把一组 fp32 权重分别量化到 int8 / int4，看每种精度的"舍入误差分布"
  2. 用量化后的权重做矩阵乘法，对比与 fp32 baseline 的输出差异
  3. 模拟"多层叠加"：连续 N 层都用量化权重，误差是否累积放大

不依赖 bitsandbytes / llama.cpp，纯 PyTorch 玩具数据，理解量化"在做什么"。
"""

from __future__ import annotations

import torch


def quantize_symmetric(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    """对称量化：把 fp32 张量 w 映射到 [-(2^(b-1)), 2^(b-1)-1] 的整数范围。

    步骤：
      1. 找绝对值最大值 max_abs（量化范围由它定）
      2. scale = max_abs / qmax，把"实际值"映射到"整数级别"
      3. 四舍五入到整数 → clamp 到合法区间
      4. 反量化（× scale）回 fp32 数值，便于直接代入原计算流程

    返回 (反量化后的 fp32 权重, scale)。
    实际部署里只存"整数权重 + scale"，推理时再乘回 scale；这里直接给反量化值方便对比。
    """
    qmax = 2 ** (bits - 1) - 1   # int8: 127, int4: 7
    qmin = -(2 ** (bits - 1))    # int8: -128, int4: -8
    max_abs = w.abs().max().clamp(min=1e-8)
    scale = max_abs / qmax
    # 量化：除 scale → 取整 → clamp
    q = torch.clamp(torch.round(w / scale), qmin, qmax)
    # 反量化：× scale 回到 fp32 数值空间
    w_dq = q * scale
    return w_dq, scale.item()


def main() -> None:
    torch.manual_seed(0)

    # ===== 验证 1：单组权重的舍入误差分布 =====
    print("=" * 64)
    print("[验证 1] 同一组 fp32 权重，int8 / int4 量化后的误差分布")
    print("=" * 64)
    w = torch.randn(10000) * 0.1   # 模拟 LLM 权重的典型尺度（小、近正态）
    print(f"  原始 fp32：mean={w.mean():+.5f}, std={w.std():.5f}, |max|={w.abs().max():.4f}")

    for bits in [8, 4, 3, 2]:
        w_dq, scale = quantize_symmetric(w, bits)
        err = w - w_dq                        # 量化误差（fp32 - 反量化）
        rel = err.abs().mean() / w.abs().mean()
        print(
            f"  int{bits}: scale={scale:.6f}  "
            f"|err|.mean={err.abs().mean():.6f}  rel_err={rel.item() * 100:.2f}%"
        )
    print("\n  → 比特数越少，舍入步长（scale）越大，平均相对误差越大；")
    print("    int2 已基本失真，这就是社区量化下限通常停在 int3 的原因。")

    # ===== 验证 2：单层矩阵乘法的输出差异 =====
    print("\n" + "=" * 64)
    print("[验证 2] 用量化权重做一次 y = W·x，对比与 fp32 baseline 的输出 RMSE")
    print("=" * 64)
    in_dim, out_dim = 256, 256
    W = torch.randn(out_dim, in_dim) * 0.05    # LLM 风格权重
    x = torch.randn(in_dim)
    y_ref = W @ x                              # fp32 baseline 输出
    print(f"  baseline 输出范围：[{y_ref.min():+.3f}, {y_ref.max():+.3f}]")
    for bits in [8, 4, 3]:
        W_q, _ = quantize_symmetric(W, bits)
        y_q = W_q @ x
        # RMSE：均方根误差，量化输出与 baseline 的整体偏差
        rmse = (y_q - y_ref).pow(2).mean().sqrt().item()
        # 相对幅度：RMSE / baseline 的标准差，更能反映"误差占比"
        rel = rmse / y_ref.std().item()
        print(f"  int{bits}: RMSE={rmse:.5f}  RMSE/std(y_ref)={rel * 100:.2f}%")

    # ===== 验证 3：多层叠加，误差累积吗？=====
    print("\n" + "=" * 64)
    print("[验证 3] N 层连续量化矩阵乘，看误差是否被层数放大")
    print("=" * 64)
    n_layers = 12
    layers = [torch.randn(out_dim, out_dim) * 0.05 for _ in range(n_layers)]
    x0 = torch.randn(out_dim)

    for bits in [8, 4, 3]:
        # baseline：fp32 跑 N 层
        h_ref = x0
        for W_l in layers:
            h_ref = W_l @ h_ref
        # 量化版：每层都用反量化后的权重跑
        h_q = x0
        for W_l in layers:
            W_lq, _ = quantize_symmetric(W_l, bits)
            h_q = W_lq @ h_q
        rmse = (h_q - h_ref).pow(2).mean().sqrt().item()
        rel = rmse / h_ref.std().clamp(min=1e-6).item()
        print(f"  int{bits}, {n_layers} 层后: RMSE={rmse:.4e}  RMSE/std={rel * 100:.2f}%")

    print("\n  → 误差随层数累积（且因每层放大效应可能呈指数增长），")
    print("    这是为什么真实量化算法（GPTQ / AWQ / GGUF K-quant）要『非均匀』地保护重要权重，")
    print("    而非朴素地按本练习的『全张量统一 scale』做。")


if __name__ == "__main__":
    main()
