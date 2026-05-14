"""ch02 练习 4：纯 NumPy 两层 MLP，过拟合一个小合成分类任务。

任务：二维平面上的两个高斯团（二分类）。
网络：input(2) -> linear(16) -> relu -> linear(2) -> softmax + CE。
优化：SGD。

目的：作为 ch03 PyTorch 版的对照组，看清楚每一步在做什么。
跑法：
    uv run python Playground/ch02-math/mlp_numpy.py
"""

from __future__ import annotations

import numpy as np


def make_data(n_per_class: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    c0 = rng.standard_normal((n_per_class, 2)) + np.array([-2.0, 0.0])
    c1 = rng.standard_normal((n_per_class, 2)) + np.array([+2.0, 0.0])
    x = np.concatenate([c0, c1], axis=0)
    y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    # shuffle
    idx = rng.permutation(len(x))
    return x[idx], y[idx]


def softmax_stable(z: np.ndarray) -> np.ndarray:
    z_shift = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z_shift)
    return e / e.sum(axis=-1, keepdims=True)


def init_params(in_dim: int, hidden: int, out_dim: int, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    # He init for ReLU
    return {
        "W1": rng.standard_normal((in_dim, hidden)) * np.sqrt(2.0 / in_dim),
        "b1": np.zeros(hidden),
        "W2": rng.standard_normal((hidden, out_dim)) * np.sqrt(2.0 / hidden),
        "b2": np.zeros(out_dim),
    }


def forward(x: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """x: (N, in)  ->  logits: (N, out)。"""
    h_pre = x @ params["W1"] + params["b1"]  # (N, hidden)
    h = np.maximum(h_pre, 0.0)  # ReLU
    logits = h @ params["W2"] + params["b2"]  # (N, out)
    cache = {"x": x, "h_pre": h_pre, "h": h, "logits": logits}
    return logits, cache


def loss_and_grad(
    logits: np.ndarray,
    y: np.ndarray,
    cache: dict,
    params: dict,
) -> tuple[float, dict]:
    """CE loss + 梯度。y: (N,) int 类别。"""
    n = len(y)
    p = softmax_stable(logits)
    log_p = np.log(p[np.arange(n), y] + 1e-12)
    loss = float(-np.mean(log_p))

    # d(CE)/d(logits) = (p - onehot) / N
    dlogits = p.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n

    h = cache["h"]
    h_pre = cache["h_pre"]
    x = cache["x"]

    grads = {
        "W2": h.T @ dlogits,  # (hidden, out)
        "b2": dlogits.sum(axis=0),  # (out,)
    }
    dh = dlogits @ params["W2"].T  # (N, hidden)
    dh_pre = dh * (h_pre > 0).astype(np.float64)
    grads["W1"] = x.T @ dh_pre  # (in, hidden)
    grads["b1"] = dh_pre.sum(axis=0)  # (hidden,)
    return loss, grads


def sgd_step(params: dict, grads: dict, lr: float) -> None:
    for k in params:
        params[k] -= lr * grads[k]


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(logits, axis=-1)
    return float(np.mean(pred == y))


def main() -> None:
    x, y = make_data(n_per_class=100, seed=0)
    params = init_params(in_dim=2, hidden=16, out_dim=2, seed=1)

    epochs = 200
    lr = 0.1
    for epoch in range(epochs):
        logits, cache = forward(x, params)
        loss, grads = loss_and_grad(logits, y, cache, params)
        sgd_step(params, grads, lr)

        if epoch % 20 == 0 or epoch == epochs - 1:
            acc = accuracy(logits, y)
            print(f"epoch {epoch:>3d}  loss={loss:.4f}  acc={acc:.3f}")

    final_logits, _ = forward(x, params)
    final_acc = accuracy(final_logits, y)
    print(f"\n最终准确率 = {final_acc:.3f}")
    assert final_acc > 0.95, "两个明显可分的高斯团应能轻松超过 95%"
    print("PASS")


if __name__ == "__main__":
    main()
