"""ch02 练习 4：纯 NumPy 两层 MLP，过拟合一个小合成分类任务。

任务：二维平面上的两个高斯团（二分类）。
网络：input(2) -> linear(16) -> relu -> linear(2) -> softmax + CE。
优化：SGD。

目的：作为 ch03 PyTorch 版的对照组，看清楚每一步在做什么。

矩阵约定（重要）：
    本脚本用 `x @ W` 习惯，batch 在第一维：
        x: (N, in_dim)  W1: (in_dim, hidden)  ->  h: (N, hidden)
    这是 PyTorch / 业界主流写法。
    `03_gradient_chain_rule.py` 用的是 `W @ x` 习惯（W 在左、x 在右、单样本），
    两种写法**数学等价**，只是行列互为转置，别弄混。

跑法：
    uv run python Playground/ch02-math/04_mlp_numpy.py
"""

from __future__ import annotations

import numpy as np


def make_data(n_per_class: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # 在 (-2, 0) 和 (+2, 0) 周围各撒 n_per_class 个高斯点 → 两个可分的"团"
    # broadcasting：(N, 2) + (2,) → 每行加上同一个偏移
    c0 = rng.standard_normal((n_per_class, 2)) + np.array([-2.0, 0.0])
    c1 = rng.standard_normal((n_per_class, 2)) + np.array([+2.0, 0.0])
    # np.concatenate 沿 axis 拼接：(N,2) + (N,2) along axis=0 → (2N, 2)
    x = np.concatenate([c0, c1], axis=0)
    y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    # 打乱顺序，避免训练时前 N 个全是类 0、后 N 个全是类 1
    idx = rng.permutation(len(x))
    return x[idx], y[idx]


def softmax_stable(z: np.ndarray) -> np.ndarray:
    # 减最大值的数值稳定 softmax，原理见 02_softmax_cross_entropy.py
    z_shift = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z_shift)
    return e / e.sum(axis=-1, keepdims=True)


def init_params(in_dim: int, hidden: int, out_dim: int, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    # He 初始化（Kaiming 正态）：W ~ N(0, sqrt(2/fan_in))
    # 专为 ReLU 设计：保持每层输出方差稳定，避免深层网络梯度消失/爆炸
    # 对 sigmoid/tanh 该用 Xavier (sqrt(1/fan_in))
    return {
        "W1": rng.standard_normal((in_dim, hidden)) * np.sqrt(2.0 / in_dim),
        "b1": np.zeros(hidden),  # bias 初始为 0 是常规做法
        "W2": rng.standard_normal((hidden, out_dim)) * np.sqrt(2.0 / hidden),
        "b2": np.zeros(out_dim),
    }


def forward(x: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """x: (N, in)  ->  logits: (N, out)。"""
    # 仿射变换 + 广播：x @ W (N,hidden) + b1 (hidden,) → 自动广播到 (N,hidden)
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
    # 高级索引 p[行, 列]：取每个样本"正确类"那一项的概率 → shape (N,)
    # 等价于 sum(onehot * p, axis=1) 但避免显式构造 onehot
    log_p = np.log(p[np.arange(n), y] + 1e-12)  # +eps 防 log(0)
    loss = float(-np.mean(log_p))  # CE = -mean(log p_correct)

    # 解析梯度 d(CE)/d(logits) = (p - onehot) / N
    # 用 in-place 减法构造 (p - onehot)：先复制 p，再把正确类位置 -1
    dlogits = p.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n  # 对应 loss 的 mean

    h = cache["h"]
    h_pre = cache["h_pre"]
    x = cache["x"]

    # batched 反向传播：单样本是外积 outer(dy, a)，batch 版聚合成 a^T @ dy
    # h.T @ dlogits: (hidden, N) @ (N, out) → (hidden, out)，自动对 batch 求和
    grads = {
        "W2": h.T @ dlogits,  # (hidden, out)
        "b2": dlogits.sum(axis=0),  # bias 梯度 = batch 维求和 → (out,)
    }
    dh = dlogits @ params["W2"].T  # 反传到隐藏层 → (N, hidden)
    # ReLU 反向：>0 处过、≤0 处截断（同 03_gradient_chain_rule.py）
    dh_pre = dh * (h_pre > 0).astype(np.float64)
    grads["W1"] = x.T @ dh_pre  # (in, hidden)
    grads["b1"] = dh_pre.sum(axis=0)  # (hidden,)
    return loss, grads


def sgd_step(params: dict, grads: dict, lr: float) -> None:
    # SGD 更新规则：θ ← θ - lr · ∇L
    # 这里是"批梯度下降"（用全部样本算梯度），严格说不是"随机"SGD
    for k in params:
        params[k] -= lr * grads[k]


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    # argmax 沿类别维取最大 logit 的索引 = 预测类别
    # （softmax 单调，argmax(logits) == argmax(softmax(logits))，省一步算 softmax）
    pred = np.argmax(logits, axis=-1)
    return float(np.mean(pred == y))


def main() -> None:
    x, y = make_data(n_per_class=100, seed=0)
    params = init_params(in_dim=2, hidden=16, out_dim=2, seed=1)

    epochs = 200
    lr = 0.1
    for epoch in range(epochs):
        # 标准训练循环：forward → loss+grad → 更新参数
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
