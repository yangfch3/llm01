"""ch02 练习 1：向量、矩阵、点积。

目标：
1. 手写点积，与 numpy 对照
2. 手写矩阵乘（三重循环），与 numpy 对照
3. 形状训练：常见 shape 错误的报错样子

跑法：
    uv run python Playground/ch02-math/vector_matrix.py
"""

from __future__ import annotations

import numpy as np


def dot_naive(a: np.ndarray, b: np.ndarray) -> float:
    """手写一维向量点积。"""
    assert a.shape == b.shape and a.ndim == 1
    total = 0.0
    for ai, bi in zip(a, b, strict=True):
        total += float(ai) * float(bi)
    return total


def matmul_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """手写矩阵乘 (m, k) @ (k, n) -> (m, n)。"""
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    m, k = a.shape
    _, n = b.shape
    out = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for kk in range(k):
                s += a[i, kk] * b[kk, j]
            out[i, j] = s
    return out


def main() -> None:
    rng = np.random.default_rng(0)

    # 1. 点积
    a = rng.standard_normal(5)
    b = rng.standard_normal(5)
    mine = dot_naive(a, b)
    ref = float(a @ b)
    print(f"[dot]    mine={mine:.6f}  numpy={ref:.6f}  diff={abs(mine - ref):.2e}")
    assert abs(mine - ref) < 1e-9

    # 2. 矩阵乘
    A = rng.standard_normal((3, 4))
    B = rng.standard_normal((4, 2))
    mine_mat = matmul_naive(A, B)
    ref_mat = A @ B
    print(f"[matmul] shape mine={mine_mat.shape}  numpy={ref_mat.shape}")
    print(f"[matmul] max abs diff = {np.max(np.abs(mine_mat - ref_mat)):.2e}")
    assert np.allclose(mine_mat, ref_mat)

    # 3. 形状错误演示
    try:
        bad = A @ A  # (3,4) @ (3,4) 内维不匹配
    except ValueError as e:
        print(f"[shape err] 预期失败：{e}")

    print("PASS")


if __name__ == "__main__":
    main()
