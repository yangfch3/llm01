"""ch09 练习 3 扩展：放大模型量级，柱状图对比三件套效果。

模型：d=512, layers=12, heads=8, seq=512, batch=32
配置：
  A. baseline (fp32, no accum, no ckpt)
  B. + AMP (bf16)
  C. + AMP + grad accumulation × 4
  D. + AMP + grad accum × 4 + gradient checkpointing

输出：两张柱状图（峰值显存 / 单 step 耗时），保存为 PNG。
要求：仅 CUDA 设备运行（需要真实显存统计）。
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

# ---------- 超参（放大量级，让差异可观） ----------
VOCAB = 8192
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 12
SEQ = 256
BATCH = 32
N_STEPS = 20  # 够统计耗时即可，不需要训收敛


# ---------- 模型定义（同 03，参数放大） ----------
class Block(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


class GPTBench(nn.Module):
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
                x = ckpt_utils.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.head(self.ln_f(x))


# ---------- 工具函数 ----------
def make_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)
    return ids, labels


def reset_mem() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_mem_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


# ---------- 单次运行 ----------
def run(name: str, device: torch.device, *, amp: bool, accum: int, ckpt: bool) -> dict:
    """跑一个配置，返回 {name, mem_mb, ms_per_step}。"""
    print(f"  运行: {name} ...", end="", flush=True)
    torch.manual_seed(0)
    reset_mem()

    model = GPTBench(use_ckpt=ckpt).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    amp_dtype = torch.bfloat16 if amp else torch.float32

    model.train()
    # warmup 2 steps（让 CUDA cache 稳定）
    for _ in range(2):
        ids, labels = make_batch(device)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            logits = model(ids)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1)) / accum
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 正式计时
    reset_mem()
    torch.cuda.synchronize()
    t0 = time.time()
    optimizer.zero_grad()
    for step in range(1, N_STEPS + 1):
        ids, labels = make_batch(device)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            logits = model(ids)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1)) / accum
        loss.backward()
        if step % accum == 0:
            optimizer.step()
            optimizer.zero_grad()

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    mem = peak_mem_mb()
    ms_step = elapsed / N_STEPS * 1000
    print(f"  显存={mem:.0f}MB  耗时={ms_step:.1f}ms/step")

    # 清理模型释放显存给下一个配置
    del model, optimizer
    torch.cuda.empty_cache()

    return {"name": name, "mem_mb": mem, "ms_per_step": ms_step}


# ---------- 作图 ----------
def plot(results: list[dict], save_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False

    names = [r["name"] for r in results]
    mems = [r["mem_mb"] for r in results]
    times = [r["ms_per_step"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 显存对比
    bars = axes[0].bar(names, mems, color=["#d62728", "#2ca02c", "#1f77b4", "#9467bd"])
    axes[0].set_ylabel("峰值显存 (MB)")
    axes[0].set_title("峰值显存对比")
    for bar, v in zip(bars, mems):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    # 耗时对比
    bars = axes[1].bar(names, times, color=["#d62728", "#2ca02c", "#1f77b4", "#9467bd"])
    axes[1].set_ylabel("单 step 耗时 (ms)")
    axes[1].set_title("单 step 耗时对比")
    for bar, v in zip(bars, times):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = REPO_ROOT / "Doc" / "Courseware" / "ch09-pretrain" / "03_bench_compare.png"
    plt.savefig(out, dpi=150)
    print(f"\n图表已保存: {out}")
    plt.close()


# ---------- main ----------
def main() -> None:
    device = get_device()
    if device.type != "cuda":
        print("此 benchmark 脚本仅支持 CUDA 设备（需要真实显存统计）。")
        print(f"当前设备: {device}")
        return

    print(f"device: {device}")
    print(f"模型: GPTBench  d={D_MODEL} layers={N_LAYERS} heads={N_HEADS}")
    print(f"数据: batch={BATCH} seq={SEQ} vocab={VOCAB} steps={N_STEPS}")
    print(f"参数量: ~{sum(p.numel() for p in GPTBench().parameters()) / 1e6:.1f}M")
    print()

    results = []
    results.append(run("A: baseline", device, amp=False, accum=1, ckpt=False))
    results.append(run("B: +AMP", device, amp=True, accum=1, ckpt=False))
    results.append(run("C: +AMP+Accum", device, amp=True, accum=4, ckpt=False))
    results.append(run("D: +AMP+Accum+Ckpt", device, amp=True, accum=4, ckpt=True))

    plot(results, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
