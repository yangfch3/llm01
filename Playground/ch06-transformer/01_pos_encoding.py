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
    # 正确性检查：每两维是 (sinθ, cosθ)，sin²+cos²=1，d/2 组 → 范数应为 √(d/2)
    norms = pe.norm(dim=-1)
    expected = (8 / 2) ** 0.5
    # torch.full_like(norms, v)：构造与 norms 同 shape/dtype/device，所有元素填充 v
    assert torch.allclose(norms, torch.full_like(norms, expected), atol=1e-5)
    print(f"正确性检查：每行范数 = √(d/2) = {expected:.3f} ✓\n")

    # 1.5) embedding + 位置编码 前后对比
    # 模拟：3 个 token 的 embedding（随机初始化），加上正弦位置编码后观察变化
    print("=" * 60)
    print("embedding + 正弦位置编码 前后对比（d=8, 3 个位置）：")
    print("=" * 60)
    emb = torch.randn(2, 8) * 0.1  # 模拟小初始化的 embedding
    pe_3 = sinusoidal_pe(max_len=2, d=8)
    emb_with_pe = emb + pe_3
    print(f"原始 embedding（纯内容，不含位置）：\n{emb}")
    print(f"\n正弦位置编码（纯位置，不含内容）：\n{pe_3}")
    print(f"\nembedding + PE（内容 + 位置，喂给模型的实际输入）：\n{emb_with_pe}")
    print(f"\n观察：同一个 embedding 放在不同位置，加的 PE 不同 → 模型能区分位置\n")

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

    # 2.1) 展示同一个向量在不同位置旋转后的变化
    print("=" * 60)
    print("RoPE：同一个 q 向量旋转前后对比（前 8 维）")
    print("=" * 60)
    fmt = lambda t: "[" + ", ".join(f"{v:.3f}" for v in t[:8].tolist()) + "]"
    print(f"  原始 q（未旋转）: {fmt(q)}")
    for pos in [0, 1, 2, 5, 10]:
        rotated = rotate_at(q, pos)
        print(f"  位置 {pos:>2d} 旋转后:  {fmt(rotated)}")
    print("观察：位置 0 完全不变（旋转角 = 0），位置越大偏离原始值越多\n")

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
