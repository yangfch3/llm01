"""ch06 练习 1：正弦位置编码 vs RoPE。

两个目标：
1. 实现并可视化两种位置编码
2. 验证 RoPE 的核心性质：旋转后 Q·K 内积只依赖相对距离 (m-p)，与绝对位置无关
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch


def sinusoidal_pe(max_len: int, d: int) -> torch.Tensor:
    """正弦位置编码：PE[p, 2i] = sin(p / 10000^(2i/d))，PE[p, 2i+1] = cos(...)。"""
    pe = torch.zeros(max_len, d)
    position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (max_len,) → (max_len, 1)：升一维准备与 div_term 广播
    # div_term = 10000^(-2i/d)，i 从 0 到 d/2-1
    # 用 exp(... * -log(10000)/d) 等价计算 10000^(-2i/d)，避免直接幂运算的数值不稳
    div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * -(torch.log(torch.tensor(10000.0)) / d))
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维：sin；position·div_term 广播为 (max_len, d/2)
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维：cos
    return pe


def build_rope_cache(max_len: int, d: int) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 的 cos/sin 表。

    把 d 维按相邻两维分组成 d/2 个 2D 组，第 i 组用频率 θ_i = 10000^(-2i/d)。
    返回 cos/sin 形状均为 (max_len, d/2)。
    """
    assert d % 2 == 0, "RoPE 要求 d 为偶数"
    half = d // 2
    # θ_i = 10000^(-2i/d), i = 0..half-1
    theta = 10000.0 ** (-torch.arange(0, half, dtype=torch.float) * 2 / d)
    pos = torch.arange(max_len, dtype=torch.float)
    angles = torch.outer(pos, theta)  # (max_len, half)，第 (p, i) 项 = p * θ_i
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 x: (..., n, d) 应用 RoPE 旋转。

    把最后一维拆成两半 (x1, x2)，旋转：
        x1' = x1 * cos - x2 * sin
        x2' = x1 * sin + x2 * cos
    这是把 d/2 个 2D 向量分别旋转 p*θ_i 角度的等价向量化写法。
    """
    n = x.size(-2)
    cos_n = cos[:n]  # (n, d/2)
    sin_n = sin[:n]
    x1, x2 = x.chunk(2, dim=-1)  # 各 (..., n, d/2)
    rotated_1 = x1 * cos_n - x2 * sin_n
    rotated_2 = x1 * sin_n + x2 * cos_n
    return torch.cat([rotated_1, rotated_2], dim=-1)


def main() -> None:
    torch.manual_seed(0)

    # 1) 正弦位置编码 quick look
    pe = sinusoidal_pe(max_len=16, d=8)
    print(f"正弦位置编码 shape={tuple(pe.shape)}")
    torch.set_printoptions(precision=3, sci_mode=False)
    print("前 4 个位置：")
    print(pe[:4])
    # 验证：每个位置编码长度应在合理范围（每两维是单位圆上的点，d/2 组 → 范数 √(d/2)）
    norms = pe.norm(dim=-1)
    expected = (8 / 2) ** 0.5
    # torch.full_like(norms, v)：构造与 norms 同 shape/dtype/device，所有元素填充 v
    assert torch.allclose(norms, torch.full_like(norms, expected), atol=1e-5)
    print(f"每行范数 ≈ √(d/2) = {expected:.3f}\n")

    # 2) RoPE 相对性质验证
    # 构造同一对 q/k，分别放在位置 (p=2, m=5) 和 (p=10, m=13)，相对距离都是 3
    # 期望：旋转后两组的 q·k 内积应该相同（相对距离决定一切）
    d = 32
    cos, sin = build_rope_cache(max_len=64, d=d)

    q = torch.randn(d)
    k = torch.randn(d)

    def rotate_at(vec: torch.Tensor, p: int) -> torch.Tensor:
        # 把 vec 放在位置 p 上做 RoPE 旋转，扩成 (1, d) 走通用 apply_rope
        x = vec.unsqueeze(0)  # (1, d)
        cos_p = cos[p : p + 1]  # 只取位置 p 的 cos/sin
        sin_p = sin[p : p + 1]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos_p - x2 * sin_p, x1 * sin_p + x2 * cos_p], dim=-1).squeeze(0)

    # 配对 1：位置 (2, 5)
    q_rope_1 = rotate_at(q, 2)
    k_rope_1 = rotate_at(k, 5)
    inner_1 = torch.dot(q_rope_1, k_rope_1).item()

    # 配对 2：位置 (10, 13)，相对距离同样 3
    q_rope_2 = rotate_at(q, 10)
    k_rope_2 = rotate_at(k, 13)
    inner_2 = torch.dot(q_rope_2, k_rope_2).item()

    # 配对 3：位置 (2, 8)，相对距离 6 → 应不同
    q_rope_3 = rotate_at(q, 2)
    k_rope_3 = rotate_at(k, 8)
    inner_3 = torch.dot(q_rope_3, k_rope_3).item()

    print(f"RoPE 内积验证：")
    print(f"  (p=2, m=5)  距离=3 → q·k = {inner_1:.6f}")
    print(f"  (p=10, m=13) 距离=3 → q·k = {inner_2:.6f}")
    print(f"  (p=2, m=8)  距离=6 → q·k = {inner_3:.6f}")
    assert abs(inner_1 - inner_2) < 1e-4, "相对距离相同时内积必须相同（RoPE 核心性质）"
    assert abs(inner_1 - inner_3) > 1e-3, "相对距离不同时内积应该不同"

    # 对照：不加位置编码的内积
    inner_raw = torch.dot(q, k).item()
    print(f"  无位置编码         q·k = {inner_raw:.6f}（与位置完全无关）")

    print("\nPASS")


if __name__ == "__main__":
    main()
