"""ch02 练习 2：softmax + 交叉熵。

目标：
1. 手写数值稳定版 softmax，验证溢出 trick
2. 手写交叉熵
3. 验证 d(CE)/d(z) = p - y_onehot

跑法：
    uv run python Playground/ch02-math/softmax_cross_entropy.py
"""

from __future__ import annotations

import numpy as np


def softmax_naive(z: np.ndarray) -> np.ndarray:
    """直接定义版，z 大会溢出。"""
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def softmax_stable(z: np.ndarray) -> np.ndarray:
    """减最大值版，等价但数值稳定。"""
    z_shift = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z_shift)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(p: np.ndarray, y_onehot: np.ndarray) -> float:
    """CE = -sum(y * log p)。p 是 softmax 输出，shape=(N, C)。"""
    eps = 1e-12  # log(0) 保护
    return float(-np.sum(y_onehot * np.log(p + eps)) / p.shape[0])


def numerical_grad_z(z: np.ndarray, y_onehot: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """对 logits z 做数值梯度。"""
    grad = np.zeros_like(z)
    it = np.nditer(z, flags=["multi_index"], op_flags=[["readwrite"]])
    while not it.finished:
        idx = it.multi_index
        orig = z[idx]
        z[idx] = orig + eps
        loss_plus = cross_entropy(softmax_stable(z), y_onehot)
        z[idx] = orig - eps
        loss_minus = cross_entropy(softmax_stable(z), y_onehot)
        z[idx] = orig
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


def main() -> None:
    # 1. 数值稳定 demo
    z_big = np.array([1000.0, 1001.0, 1002.0])
    print("[stable]")
    with np.errstate(over="ignore", invalid="ignore"):
        naive = softmax_naive(z_big)
    stable = softmax_stable(z_big)
    print(f"  naive  = {naive}    （含 nan/inf 即为溢出）")
    print(f"  stable = {stable}")
    print(f"  sum(stable) = {stable.sum():.6f}")

    # 2. CE + 解析梯度对照数值梯度
    rng = np.random.default_rng(0)
    N, C = 4, 5
    z = rng.standard_normal((N, C))
    labels = rng.integers(0, C, size=N)
    y_onehot = np.zeros((N, C))
    y_onehot[np.arange(N), labels] = 1.0

    p = softmax_stable(z)
    loss = cross_entropy(p, y_onehot)
    print(f"\n[ce] loss = {loss:.6f}")

    # 解析梯度：d(CE)/d(z) = (p - y) / N（因为 CE 取了 mean）
    grad_analytic = (p - y_onehot) / N
    grad_numeric = numerical_grad_z(z.copy(), y_onehot)

    diff = np.max(np.abs(grad_analytic - grad_numeric))
    print(f"[grad] analytic vs numeric max abs diff = {diff:.2e}")
    assert diff < 1e-6, "梯度对不上，重新检查公式"

    print("PASS")


if __name__ == "__main__":
    main()
