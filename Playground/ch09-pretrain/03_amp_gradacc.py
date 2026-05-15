"""ch09 练习 3：AMP + 梯度累积 + gradient checkpointing 三件套。

构造一个小 Transformer（3 层、d=128），跑 4 种配置：
  A. baseline (fp32, no accum, no ckpt)
  B. + AMP (bf16/fp16 autocast)
  C. + grad accumulation × 4
  D. + gradient checkpointing

每种配置打印：
  - 峰值显存（仅 CUDA 上有意义；MPS/CPU 显示"N/A"）
  - 单 step 平均耗时
  - 训完 final loss

注意：CPU/MPS 上 AMP 与 ckpt 的收益不显著，但 API 调用方式相同 → 验证代码能跑通即可。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_utils

from Echo.shared.device import get_device

# 超参（小模型，3060 / Mac 都能秒跑）
VOCAB = 256
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
SEQ = 64
BATCH = 16
N_STEPS = 30


class Block(nn.Module):
    """简版 Pre-LN block：MHA + FFN。"""

    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        # nn.MultiheadAttention(embed_dim, num_heads, batch_first)：
        #   batch_first=True → 输入 (B, L, D)，否则 (L, B, D)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        # MHA forward 返回 (out, attn_weights)；这里只要 out
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, use_ckpt: bool = False) -> None:
        super().__init__()
        self.use_ckpt = use_ckpt
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(SEQ, D_MODEL)
        self.blocks = nn.ModuleList([Block(D_MODEL, N_HEADS) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(L, device=ids.device))
        for blk in self.blocks:
            if self.use_ckpt and self.training:
                # ckpt_utils.checkpoint(fn, *args, use_reentrant=False)：
                #   前向只记 fn 的输入与函数引用，反向时重算 fn 内部激活；省激活显存换计算时间
                #   use_reentrant=False 是新版推荐写法，对自定义 module 更稳
                x = ckpt_utils.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.head(self.ln_f(x))


def make_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """生成假数据。labels 直接复用 ids 简化（本练习不关心 loss 绝对值）。"""
    ids = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)
    return ids, labels


def reset_mem(device: torch.device) -> None:
    """重置 CUDA 显存峰值统计；非 CUDA 时空操作。"""
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_mem_mb(device: torch.device) -> str:
    if device.type != "cuda":
        return "N/A (非 CUDA 无统一显存计)"
    torch.cuda.synchronize()
    return f"{torch.cuda.max_memory_allocated() / 1e6:.1f} MB"


def run(name: str, device: torch.device, *, amp: bool, accum: int, ckpt: bool) -> None:
    """跑一个配置。amp/accum/ckpt 三个开关组合。"""
    print(f"\n[{name}]  amp={amp}  accum={accum}  ckpt={ckpt}")
    torch.manual_seed(0)
    reset_mem(device)

    model = TinyGPT(use_ckpt=ckpt).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # AMP 在 CUDA + fp16 时需要 GradScaler 防梯度下溢；bf16 / 非 CUDA 不需要
    # 简化：只在 CUDA 上启用 AMP（其他设备 autocast 行为一致但收益小）
    use_amp = amp and device.type == "cuda"
    # bf16 在 Ampere+（30 系起）原生支持，不需要 GradScaler
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    model.train()
    t0 = time.time()
    optimizer.zero_grad()
    for step in range(1, N_STEPS + 1):
        ids, labels = make_batch(device)
        # torch.amp.autocast(device_type, dtype, enabled)：
        #   上下文内的算子尝试用 dtype 计算；enabled=False 退化为普通 fp32 forward
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(ids)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1))
            # 梯度累积时 loss 必须除以 accum，否则等效 lr 放大 accum 倍
            loss = loss / accum
        loss.backward()
        # 攒够 accum 步才真正 step
        if step % accum == 0:
            optimizer.step()
            optimizer.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()  # 等所有 CUDA kernel 跑完再计时，否则计时偏小
    elapsed = time.time() - t0

    print(f"  final loss     = {loss.item() * accum:.4f}  (×accum 还原)")
    print(f"  total time     = {elapsed:.2f}s  ({elapsed / N_STEPS * 1000:.1f} ms/step)")
    print(f"  peak GPU mem   = {peak_mem_mb(device)}")


def main() -> None:
    device = get_device()
    print(f"device: {device}")
    print(f"模型: TinyGPT  d={D_MODEL} layers={N_LAYERS} heads={N_HEADS}")
    print(f"数据: batch={BATCH} seq={SEQ} steps={N_STEPS}")

    run("A baseline           ", device, amp=False, accum=1, ckpt=False)
    run("B + AMP (bf16)       ", device, amp=True, accum=1, ckpt=False)
    run("C + grad accum × 4   ", device, amp=True, accum=4, ckpt=False)
    run("D + gradient ckpt    ", device, amp=True, accum=4, ckpt=True)

    print("\n[读图说明]")
    print("  CUDA 上：B→A 显存降 ~40%、速度升；D 激活显存再降一截但单 step 略慢")
    print("  CPU/MPS：autocast 与 ckpt 路径都跑通即可，显存与速度差异不显著")


if __name__ == "__main__":
    main()
