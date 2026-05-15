"""ch06 练习 2：单个 Pre-LN Transformer Block。

验证项：
- 输入输出 shape 一致
- 残差路径存在（去掉 attention/FFN 子模块输出仍≈输入）
- 前向 + 反向能跑通，梯度健康
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from Echo.shared.device import get_device


class CausalSelfAttention(nn.Module):
    """多头因果自注意力。复用 ch05 §03 的实现思路，去掉 weight-copy 调试代码。"""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)  # 3 个投影合并成一次大乘法
        self.proj = nn.Linear(d_model, d_model, bias=True)     # 输出投影 W^O

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n, d = x.shape
        # qkv: (B, n, 3d) → 切 3 份 → 各 (B, n, d) → reshape 切头 → (B, H, n, d_k)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, n, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, n, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, n, self.n_heads, self.d_k).transpose(1, 2)
        # F.scaled_dot_product_attention：PyTorch 2.0+ 融合实现，is_causal=True 内部生成上三角 mask
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, n, d)  # 合头：transpose 后内存非连续，先 contiguous
        return self.proj(out)


class FFN(nn.Module):
    """两层 Linear + GELU。d_ff = 4 * d_model。"""

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))  # GELU = x·Φ(x)，比 ReLU 软一点，GPT 系标配


class Block(nn.Module):
    """Pre-LN Transformer Block：x → LN → MHA → +x → LN → FFN → +x。"""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # 残差 1：注意力（LN 在前 = Pre-LN）
        x = x + self.ffn(self.ln2(x))   # 残差 2：FFN
        return x


def main() -> None:
    device = get_device()
    print(f"device: {device}")
    torch.manual_seed(0)

    B, n, d, H = 2, 16, 64, 4
    block = Block(d, H).to(device)
    x = torch.randn(B, n, d, device=device, requires_grad=True)

    # 1) 形状检查
    y = block(x)
    print(f"输入  shape: {tuple(x.shape)}")
    print(f"输出  shape: {tuple(y.shape)}")
    assert y.shape == x.shape, "Block 输出形状必须与输入一致"

    # 2) 残差路径存在性：把 attn / ffn 的输出投影权重清零，输出应≈输入
    with torch.no_grad():  # 直接改权重不需建梯度图
        # tensor.zero_()：in-place 把所有元素置 0（与 `= torch.zeros(...)` 不同，保留 nn.Parameter 包装）
        block.attn.proj.weight.zero_()
        block.attn.proj.bias.zero_()
        block.ffn.fc2.weight.zero_()
        block.ffn.fc2.bias.zero_()
    y_residual_only = block(x)
    diff = (y_residual_only - x).abs().max().item()
    print(f"\n关掉子模块后输出 vs 输入 最大差异: {diff:.3e}（残差路径存在则≈0）")
    assert diff < 1e-5, "Pre-LN 残差路径应让 sublayer=0 时输出严格等于输入"

    # 3) 反向能跑通，梯度非 0 / 非 nan
    block_fresh = Block(d, H).to(device)
    x_fresh = torch.randn(B, n, d, device=device, requires_grad=True)
    loss = block_fresh(x_fresh).pow(2).mean()  # 假目标：输出平方均值
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in block_fresh.parameters() if p.grad is not None]
    print(f"\n反向后参数梯度范数范围: [{min(grad_norms):.3e}, {max(grad_norms):.3e}]")
    assert all(g > 0 and g < 1e3 for g in grad_norms), "梯度应全部健康（非 0 / 非爆炸）"
    assert not any(g != g for g in grad_norms), "梯度不应有 nan"

    # 4) 参数量速查
    n_params = sum(p.numel() for p in block_fresh.parameters())
    expected = 12 * d * d  # ≈ 12 d²（忽略 LN 和 bias 小项）
    print(f"\n参数量: {n_params}  ≈ 12·d² = {expected}（误差来自 bias/LN 小项）")

    print("\nPASS")


if __name__ == "__main__":
    main()
