"""ch05 练习 1：纯 NumPy 实现单头缩放点积注意力。

逐步打印 QK^T / softmax / 输出，看清"加权平均"在数值上长什么样。
不依赖 PyTorch，方便你不被框架抽象遮挡。
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # 减去 max 防止 exp 溢出（数值稳定 softmax 的标准做法）
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x_shift)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: np.ndarray,  # (n, d_k)
    k: np.ndarray,  # (n, d_k)
    v: np.ndarray,  # (n, d_v)
    mask: np.ndarray | None = None,  # (n, n)，True/1 表示能看，False/0 表示屏蔽
) -> tuple[np.ndarray, np.ndarray]:
    d_k = q.shape[-1]
    # QK^T: (n, d_k) @ (d_k, n) = (n, n)，每行是一个 query 对所有 key 的相似度
    scores = q @ k.T / np.sqrt(d_k)  # 缩放点积：除 √d_k 把方差拉回 1，softmax 不饱和
    if mask is not None:
        # 屏蔽位置置 -inf，softmax 后权重变 0；mask 必须在 softmax 之前应用
        scores = np.where(mask, scores, -np.inf)
    attn = softmax(scores, axis=-1)  # 行归一化：每个 query 对所有 key 的权重和=1
    out = attn @ v  # (n, n) @ (n, d_v) = (n, d_v)，加权求和 V
    return out, attn


def main() -> None:
    rng = np.random.default_rng(0)  # NumPy 现代随机数生成器，固定 seed → 可复现
    n, d_k, d_v = 4, 8, 8
    q = rng.standard_normal((n, d_k))
    k = rng.standard_normal((n, d_k))
    v = rng.standard_normal((n, d_v))

    print(f"Q shape={q.shape}  K shape={k.shape}  V shape={v.shape}\n")

    out, attn = scaled_dot_product_attention(q, k, v)

    print("attention 权重矩阵 (n × n)，每行 sum=1：")
    np.set_printoptions(precision=3, suppress=True)  # NumPy 全局打印格式：3 位精度 + 抑制科学计数
    print(attn)
    print(f"每行 sum: {attn.sum(axis=-1)}\n")

    print(f"输出 shape={out.shape}（应等于 (n, d_v)）")

    # 验证 1：每行权重和接近 1（softmax 性质）
    assert np.allclose(attn.sum(axis=-1), 1.0), "softmax 行和必须为 1"

    # 验证 2：不加 mask 时所有权重 > 0
    assert (attn > 0).all(), "无 mask 时所有 attention 权重应为正"

    # 验证 3：causal mask 下，i < j 的位置权重必须为 0
    causal = np.tril(np.ones((n, n), dtype=bool))  # np.tril 取下三角，主对角线及以下为 True，上三角为 False
    out_c, attn_c = scaled_dot_product_attention(q, k, v, mask=causal)
    upper_triangle = attn_c[np.triu_indices(n, k=1)]  # np.triu_indices(n, k=1) 返回严格上三角元素的 (row, col) 索引数组
    assert np.allclose(upper_triangle, 0.0), "causal mask 下未来位置权重必须为 0"
    print("\n加 causal mask 后的 attention：")
    print(attn_c)

    print("\nPASS")


if __name__ == "__main__":
    main()
