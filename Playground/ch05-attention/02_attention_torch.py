"""ch05 练习 2：PyTorch 单头注意力，对照官方 `F.scaled_dot_product_attention`。

确认你手写实现与 PyTorch 内置数值一致——这是后面写 MHA / Transformer 的信心基础。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from Echo.shared.device import get_device


def manual_attention(
    q: torch.Tensor,  # (B, n, d_k)
    k: torch.Tensor,  # (B, n, d_k)
    v: torch.Tensor,  # (B, n, d_v)
    is_causal: bool = False,
) -> torch.Tensor:
    d_k = q.size(-1)
    # (B, n, d_k) @ (B, d_k, n) = (B, n, n)，每个 batch 独立算
    scores = q @ k.transpose(-2, -1) / (d_k**0.5)  # 缩放点积：除 √d_k
    if is_causal:
        n = q.size(-2)
        # torch.triu(..., diagonal=1)：取严格上三角（不含主对角线）→ True 即"未来位置"
        causal_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=q.device), diagonal=1)
        # masked_fill：mask 为 True 的位置填指定值；注意要在 softmax 之前
        scores = scores.masked_fill(causal_mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)  # dim=-1：沿最后一维（key 维）归一化，每个 query 一行
    return attn @ v


def main() -> None:
    device = get_device()
    print(f"device: {device}")
    torch.manual_seed(0)

    B, n, d_k = 2, 6, 16
    q = torch.randn(B, n, d_k, device=device)
    k = torch.randn(B, n, d_k, device=device)
    v = torch.randn(B, n, d_k, device=device)

    # 无 mask 对照
    out_manual = manual_attention(q, k, v, is_causal=False)
    # F.scaled_dot_product_attention：PyTorch 2.0+ 内置融合实现，含数值优化（FlashAttention 等）
    out_torch = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    diff = (out_manual - out_torch).abs().max().item()
    print(f"无 mask  最大差异: {diff:.3e}")
    assert diff < 1e-5, f"手写与官方实现应数值一致，实际差异 {diff}"

    # causal mask 对照
    out_manual_c = manual_attention(q, k, v, is_causal=True)
    out_torch_c = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    diff_c = (out_manual_c - out_torch_c).abs().max().item()
    print(f"causal   最大差异: {diff_c:.3e}")
    assert diff_c < 1e-5, f"causal 模式也应数值一致，实际差异 {diff_c}"

    print(f"\n输出 shape: {tuple(out_manual.shape)}（应为 (B, n, d_k)）")
    print("PASS")


if __name__ == "__main__":
    main()
