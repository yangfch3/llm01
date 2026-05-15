"""ch05 练习 3：手写多头注意力，与 nn.MultiheadAttention 数值对齐。

关键点：把 d 维拆成 H 段，每段 d_k = d/H。Q/K/V 各用一个 (d, d) 大矩阵一次投影出来，
再 reshape 切头。最末加 W^O 让多头信息融合。
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


class MultiHeadAttention(nn.Module):
    """手写 MHA。对齐 nn.MultiheadAttention(batch_first=True, bias=True) 的行为。"""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # 三个大矩阵一次性投影出 Q/K/V，比 3H 个小矩阵高效
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)  # 输出投影：让多头融合

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        # x: (B, n, d_model)
        B, n, _ = x.shape
        # 投影后 reshape: (B, n, d) → (B, n, H, d_k) → (B, H, n, d_k)
        # transpose 把 head 维提到第 2 维，让 attention 在 (n, d_k) 维度上算
        q = self.w_q(x).view(B, n, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, n, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, n, self.num_heads, self.d_k).transpose(1, 2)

        # 多头并行：(B, H, n, d_k) @ (B, H, d_k, n) = (B, H, n, n)
        scores = q @ k.transpose(-2, -1) / (self.d_k**0.5)
        if is_causal:
            # 同 02：torch.triu(..., diagonal=1) 取严格上三角，masked_fill 把未来位置置 -inf
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)  # dim=-1 沿 key 维归一化
        out = attn @ v  # (B, H, n, d_k)

        # 合头：(B, H, n, d_k) → (B, n, H, d_k) → (B, n, d_model)
        # contiguous() 因为 transpose 后内存非连续，reshape 需要
        out = out.transpose(1, 2).contiguous().view(B, n, self.d_model)
        return self.w_o(out)


def copy_weights_from_torch(mine: MultiHeadAttention, torch_mha: nn.MultiheadAttention) -> None:
    """把 nn.MultiheadAttention 的权重拷到手写实现，便于数值对齐验证。

    nn.MultiheadAttention 内部把 Q/K/V 三个矩阵打包成一个 in_proj_weight (3d, d)：
    [W^Q; W^K; W^V] 纵向拼接。我们要拆开塞进 self.w_q/k/v。
    """
    d = torch_mha.embed_dim
    with torch.no_grad():  # 权重 copy 不需建梯度图
        in_w = torch_mha.in_proj_weight  # (3d, d)
        in_b = torch_mha.in_proj_bias    # (3d,)
        # tensor.copy_(src)：in-place 拷贝 src 数据到 self（与 `=` 赋值不同，不替换 tensor 对象本身，
        # 保留 nn.Parameter 包装与 requires_grad 状态）
        mine.w_q.weight.copy_(in_w[:d])
        mine.w_k.weight.copy_(in_w[d : 2 * d])
        mine.w_v.weight.copy_(in_w[2 * d :])
        mine.w_q.bias.copy_(in_b[:d])
        mine.w_k.bias.copy_(in_b[d : 2 * d])
        mine.w_v.bias.copy_(in_b[2 * d :])
        mine.w_o.weight.copy_(torch_mha.out_proj.weight)
        mine.w_o.bias.copy_(torch_mha.out_proj.bias)


def main() -> None:
    device = get_device()
    print(f"device: {device}")
    torch.manual_seed(0)

    B, n, d_model, H = 2, 8, 64, 4
    x = torch.randn(B, n, d_model, device=device)

    mine = MultiHeadAttention(d_model, H).to(device)
    # batch_first=True 让输入形状是 (B, n, d) 而不是 (n, B, d)，与手写对齐
    torch_mha = nn.MultiheadAttention(d_model, H, batch_first=True, bias=True).to(device)
    copy_weights_from_torch(mine, torch_mha)

    # 无 mask
    out_mine = mine(x, is_causal=False)
    # need_weights=False 让官方实现走融合 kernel，输出更快；attn_mask=None 表示无掩码
    out_torch, _ = torch_mha(x, x, x, need_weights=False)
    diff = (out_mine - out_torch).abs().max().item()
    print(f"无 mask  最大差异: {diff:.3e}")
    assert diff < 1e-5, f"手写 MHA 应与官方对齐，实际差异 {diff}"

    # causal mask
    out_mine_c = mine(x, is_causal=True)
    # nn.Transformer.generate_square_subsequent_mask(n)：返回 (n,n) 加性 mask（上三角 -inf、其余 0），
    # 与 nn.MultiheadAttention 的 attn_mask 接口对齐（attn_mask 是"加到 scores 上"而不是 bool 屏蔽）
    causal = nn.Transformer.generate_square_subsequent_mask(n).to(device)
    out_torch_c, _ = torch_mha(x, x, x, attn_mask=causal, need_weights=False)
    diff_c = (out_mine_c - out_torch_c).abs().max().item()
    print(f"causal   最大差异: {diff_c:.3e}")
    assert diff_c < 1e-5, f"causal 模式也应对齐，实际差异 {diff_c}"

    print(f"\n输出 shape: {tuple(out_mine.shape)}（应为 (B, n, d_model)）")
    print("PASS")


if __name__ == "__main__":
    main()
