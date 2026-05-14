"""ch02 练习 3：手算两层网络反向传播，对比数值梯度。

网络：
    h = w1 @ x         (h2, in)  @ (in,)  -> (h2,)
    a = relu(h)
    y = w2 @ a         (out, h2) @ (h2,)  -> (out,)
    L = 0.5 * sum((y - t)^2)

跑法：
    uv run python Playground/ch02-math/gradient_chain_rule.py
"""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def forward(x: np.ndarray, t: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> tuple[float, dict]:
    h = w1 @ x
    a = relu(h)
    y = w2 @ a
    loss = 0.5 * np.sum((y - t) ** 2)
    cache = {"x": x, "h": h, "a": a, "y": y, "t": t, "w1": w1, "w2": w2}
    return float(loss), cache


def backward(cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """返回 (dL/dw1, dL/dw2)。"""
    x, h, a, y, t = cache["x"], cache["h"], cache["a"], cache["y"], cache["t"]
    w2 = cache["w2"]

    dy = y - t  # (out,)
    dw2 = np.outer(dy, a)  # (out, h2)
    da = w2.T @ dy  # (h2,)
    dh = da * (h > 0).astype(np.float64)  # ReLU 反向：>0 处为 1，否则 0
    dw1 = np.outer(dh, x)  # (h2, in)
    return dw1, dw2


def numerical_grad(
    fn,
    param: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """对 param 做数值梯度。fn() 内部应已闭包 param，返回标量 loss。"""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"], op_flags=[["readwrite"]])
    while not it.finished:
        idx = it.multi_index
        orig = param[idx]
        param[idx] = orig + eps
        lp = fn()
        param[idx] = orig - eps
        lm = fn()
        param[idx] = orig
        grad[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    return grad


def main() -> None:
    rng = np.random.default_rng(42)
    in_dim, h_dim, out_dim = 4, 5, 3

    x = rng.standard_normal(in_dim)
    t = rng.standard_normal(out_dim)
    w1 = rng.standard_normal((h_dim, in_dim)) * 0.5
    w2 = rng.standard_normal((out_dim, h_dim)) * 0.5

    loss, cache = forward(x, t, w1, w2)
    print(f"loss = {loss:.6f}")

    dw1, dw2 = backward(cache)

    # 数值梯度对照
    dw1_num = numerical_grad(lambda: forward(x, t, w1, w2)[0], w1)
    dw2_num = numerical_grad(lambda: forward(x, t, w1, w2)[0], w2)

    diff_w1 = np.max(np.abs(dw1 - dw1_num))
    diff_w2 = np.max(np.abs(dw2 - dw2_num))
    print(f"[w1] analytic vs numeric max abs diff = {diff_w1:.2e}")
    print(f"[w2] analytic vs numeric max abs diff = {diff_w2:.2e}")

    assert diff_w1 < 1e-6 and diff_w2 < 1e-6, "解析梯度公式可能有误"
    print("PASS")


if __name__ == "__main__":
    main()
