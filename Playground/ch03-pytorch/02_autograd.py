"""ch03 练习 2：autograd 与 ch02 解析梯度对照。

ch02 `gradient_chain_rule.py` 用 NumPy 手算两层网络的解析梯度。
本脚本用 PyTorch autograd 算同样网络的梯度，与 ch02 数值对照。
目的是看清"框架替我们做的就是 ch02 那套链式法则"。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch


def demo_minimal() -> None:
    print("\n--- 1. 最小例子：手算 vs autograd ---")
    # 复刻 ch02 §2.3 标量例子：x=2, w1=0.5, w2=3, t=5
    x = torch.tensor(2.0)
    w1 = torch.tensor(0.5, requires_grad=True)  # 标记 leaf 才能拿 .grad
    w2 = torch.tensor(3.0, requires_grad=True)
    t = torch.tensor(5.0)

    h_pre = w1 * x  # forward 同时建图
    h = torch.relu(h_pre)
    y = w2 * h
    loss = 0.5 * (y - t) ** 2

    loss.backward()  # 沿图反向，自动填 .grad

    # ch02 手算结果：dw1 = -12, dw2 = -2
    print(f"w1.grad = {w1.grad.item():.4f}  (ch02 手算 = -12.0)")
    print(f"w2.grad = {w2.grad.item():.4f}  (ch02 手算 = -2.0)")
    assert abs(w1.grad.item() - (-12.0)) < 1e-5
    assert abs(w2.grad.item() - (-2.0)) < 1e-5


def demo_grad_accumulate() -> None:
    print("\n--- 2. 坑 1：梯度累加 ---")
    w = torch.tensor(1.0, requires_grad=True)
    loss1 = w * 2  # d/dw = 2
    loss1.backward()
    print(f"第一次 backward 后  w.grad = {w.grad.item()}")  # 2.0

    loss2 = w * 3  # d/dw = 3
    loss2.backward()
    print(f"第二次 backward 后  w.grad = {w.grad.item()}  ← 累加而非覆盖")  # 5.0
    assert w.grad.item() == 5.0

    # 训练循环里靠 optimizer.zero_grad() 解决：
    w.grad = None  # 等价于 optimizer.zero_grad(set_to_none=True)
    loss3 = w * 4
    loss3.backward()
    print(f"清零后再算          w.grad = {w.grad.item()}")  # 4.0


def demo_no_grad() -> None:
    print("\n--- 3. 推理：no_grad 关计算图 ---")
    w = torch.tensor(1.0, requires_grad=True)

    y_train = w * 2
    print(f"训练模式  y.requires_grad = {y_train.requires_grad}")  # True

    with torch.no_grad():  # 上下文内运算不建图，省显存提速
        y_infer = w * 2
    print(f"no_grad   y.requires_grad = {y_infer.requires_grad}")  # False

    # detach() 单点切断（更细粒度）
    y_det = (w * 2).detach()
    print(f"detach    y.requires_grad = {y_det.requires_grad}")  # False


def demo_vector() -> None:
    print("\n--- 4. 向量版：和 ch02 mlp_numpy 同形 ---")
    torch.manual_seed(0)
    # 一个 (1, 4) 输入过两层线性，与 ch02 §2.3 向量推广对照
    x = torch.randn(1, 4)
    W1 = torch.randn(4, 8, requires_grad=True)  # (in, hidden)
    W2 = torch.randn(8, 3, requires_grad=True)  # (hidden, out)
    target = torch.tensor([2])

    h = torch.relu(x @ W1)
    logits = h @ W2
    # 直接用 logits + CE，避免在 CE 前 softmax（ch02 §3.5 铁律）
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    print(f"loss = {loss.item():.4f}")
    print(f"W1.grad shape = {W1.grad.shape}  (应为 (4, 8))")
    print(f"W2.grad shape = {W2.grad.shape}  (应为 (8, 3))")
    # 验梯度形状必须和参数同形（ch02 §2.3 形状反推技巧）
    assert W1.grad.shape == W1.shape
    assert W2.grad.shape == W2.shape


def main() -> None:
    demo_minimal()
    demo_grad_accumulate()
    demo_no_grad()
    demo_vector()
    print("\nPASS")


if __name__ == "__main__":
    main()
