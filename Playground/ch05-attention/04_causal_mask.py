"""ch05 练习 4：因果掩码可视化 + 不泄漏验证。

两个实验：
1. 打印 causal attention 权重矩阵，确认上三角全 0
2. 篡改"未来 token"的输入，观察当前 token 输出是否变化——不变即"未来未泄漏"
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F


def causal_attention_weights(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """只算 attention 权重（不用 V），方便可视化。"""
    n, d_k = q.shape
    scores = q @ k.T / (d_k**0.5)
    # torch.triu(..., diagonal=1)：严格上三角（不含主对角线）→ True 即"未来位置"，要屏蔽
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))  # 在 softmax 之前置 -inf，softmax 后权重变 0
    return F.softmax(scores, dim=-1)


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    attn = causal_attention_weights(q, k)
    return attn @ v


def main() -> None:
    torch.manual_seed(0)
    n, d = 6, 8

    # 1) 可视化 causal mask 后的 attention 权重
    q = torch.randn(n, d)
    k = torch.randn(n, d)
    attn = causal_attention_weights(q, k)
    print("Causal attention 权重（上三角应全 0）：")
    torch.set_printoptions(precision=3, sci_mode=False)  # 关掉科学计数法，0/1 直接显示
    print(attn)

    # 验证：严格上三角元素全为 0
    # torch.triu_indices(n, n, offset=1) 返回 (2, K) 张量，第 0 行是 row、第 1 行是 col；
    # .unbind() 拆成 (rows, cols) 两个 1D 张量，作为 advanced indexing 取出对应位置的值
    upper = attn[torch.triu_indices(n, n, offset=1).unbind()]
    assert torch.allclose(upper, torch.zeros_like(upper)), "未来位置权重必须为 0"

    # 验证：每行权重和为 1
    assert torch.allclose(attn.sum(dim=-1), torch.ones(n))

    # 2) 不泄漏验证：把"位置 t 之后"的输入换掉，前 t 个位置输出应该不变
    print("\n不泄漏验证：篡改未来 token，观察当前 token 输出是否变化\n")
    x_orig = torch.randn(n, d)
    # 共享 Q/K/V = X（self-attention 简化版）
    out_orig = causal_attention(x_orig, x_orig, x_orig)

    # 把第 t=2 之后的 token（位置 3,4,5）随便改
    t = 2
    x_tampered = x_orig.clone()
    x_tampered[t + 1 :] = torch.randn(n - t - 1, d) * 100  # 改得很离谱

    out_tampered = causal_attention(x_tampered, x_tampered, x_tampered)

    # 前 t+1 个位置的输出应完全不受未来扰动影响
    diff_past = (out_orig[: t + 1] - out_tampered[: t + 1]).abs().max().item()
    diff_future = (out_orig[t + 1 :] - out_tampered[t + 1 :]).abs().max().item()
    print(f"前 {t + 1} 个位置（过去）输出最大差异: {diff_past:.3e}  ← 应≈0")
    print(f"后 {n - t - 1} 个位置（被改）输出最大差异: {diff_future:.3e}  ← 应明显>0")
    assert diff_past < 1e-6, "causal mask 失效：未来 token 改动影响了过去位置的输出"
    assert diff_future > 1e-3, "未来位置本身被改了，输出应明显不同"

    print("\nPASS")


if __name__ == "__main__":
    main()
