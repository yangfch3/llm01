"""ch02 练习 2：softmax + 交叉熵。

目标：
1. 手写数值稳定版 softmax，验证溢出 trick
2. 手写交叉熵
3. 验证 d(CE)/d(z) = p - y_onehot

跑法：
    uv run python Playground/ch02-math/02_softmax_cross_entropy.py
"""

from __future__ import annotations

import numpy as np


def softmax_naive(z: np.ndarray) -> np.ndarray:
    """直接定义版，z 大会溢出。"""
    e = np.exp(z)  # 逐元素 e^z
    # axis=-1 沿最后一维求和，keepdims=True 保留维度便于广播除法
    return e / e.sum(axis=-1, keepdims=True)


def softmax_stable(z: np.ndarray) -> np.ndarray:
    """减最大值版，等价但数值稳定。"""
    # 平移 trick：softmax(z) = softmax(z - c)，取 c=max(z) 让 e^x 不溢出
    z_shift = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z_shift)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(p: np.ndarray, y_onehot: np.ndarray) -> float:
    """CE = -sum(y * log p)。p 是 softmax 输出，shape=(N, C)。"""
    eps = 1e-12  # log(0) 保护：避免 p=0 时 log(0) = -inf
    # 公式：mean over batch of CE = -(1/N) * Σ Σ y * log(p)
    return float(-np.sum(y_onehot * np.log(p + eps)) / p.shape[0])


def numerical_grad_z(z: np.ndarray, y_onehot: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """对 logits z 做数值梯度。"""
    grad = np.zeros_like(z)
    # np.nditer：N 维数组的"扁平化迭代器"，multi_index 给出每个元素的多维索引
    # readwrite 标志：允许通过 it 间接修改 z（这里我们直接用 z[idx] 读写，未真正用到该写权限）
    it = np.nditer(z, flags=["multi_index"], op_flags=[["readwrite"]])
    while not it.finished:
        idx = it.multi_index
        orig = z[idx]
        # 中心差分公式 ∂L/∂z_i ≈ (L(z+ε e_i) - L(z-ε e_i)) / (2ε)
        # 比前向差分 (L(z+ε) - L(z)) / ε 精度高一阶
        z[idx] = orig + eps
        loss_plus = cross_entropy(softmax_stable(z), y_onehot)
        z[idx] = orig - eps
        loss_minus = cross_entropy(softmax_stable(z), y_onehot)
        z[idx] = orig  # 还原，避免污染外部
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


def main() -> None:
    # 1. 数值稳定 demo
    z_big = np.array([1000.0, 1001.0, 1002.0])
    print("[stable]")
    # np.errstate 临时改浮点错误处理：这里允许 overflow/invalid 不报警，因为我们就是要观察溢出
    with np.errstate(over="ignore", invalid="ignore"):
        naive = softmax_naive(z_big)
    stable = softmax_stable(z_big)
    print(f"  naive  = {naive}    （含 nan/inf 即为溢出）")
    print(f"  stable = {stable}")
    print(f"  sum(stable) = {stable.sum():.6f}")

    # 2. CE + 解析梯度对照数值梯度
    rng = np.random.default_rng(0)
    N, C = 4, 5  # batch=4 样本，C=5 个类别
    z = rng.standard_normal((N, C))  # 模拟网络输出的 logits
    # rng.integers(low, high, size)：生成 [low, high) 区间的随机整数 → 模拟标签
    labels = rng.integers(0, C, size=N)
    y_onehot = np.zeros((N, C))
    # 高级索引：y_onehot[行索引, 列索引] = 1，把每行 labels[i] 对应的位置置 1
    # 例如 labels=[2,0,3,1] → y_onehot 第 0 行第 2 列、第 1 行第 0 列... 各置 1
    y_onehot[np.arange(N), labels] = 1.0

    p = softmax_stable(z)
    loss = cross_entropy(p, y_onehot)
    print(f"\n[ce] loss = {loss:.6f}")

    # 解析梯度：d(CE)/d(z) = (p - y) / N
    # 推导见 ch02 课件 §3.4 折叠块；除以 N 是因为 cross_entropy 对 batch 取了 mean
    grad_analytic = (p - y_onehot) / N
    grad_numeric = numerical_grad_z(z.copy(), y_onehot)  # copy 避免数值梯度污染原 z

    diff = np.max(np.abs(grad_analytic - grad_numeric))
    print(f"[grad] analytic vs numeric max abs diff = {diff:.2e}")
    assert diff < 1e-6, "梯度对不上，重新检查公式"

    print("PASS")


if __name__ == "__main__":
    main()
