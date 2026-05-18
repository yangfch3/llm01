"""ch03 练习 3：用 nn.Module 重写 ch02 的 04_mlp_numpy。

对照组：`Playground/ch02-math/04_mlp_numpy.py` 全手写约 100 行。
本脚本同样任务用 PyTorch 写，看 framework 省了哪些代码（forward 反向自动、参数自动注册、优化器一行）。
任务沿用 ch02：二维平面两个高斯团二分类。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

from Echo.shared.device import get_device


def make_data(n_per_class: int = 100, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    # 与 ch02 04_mlp_numpy.make_data 同分布，方便对照
    rng = np.random.default_rng(seed)
    c0 = rng.standard_normal((n_per_class, 2)) + np.array([-2.0, 0.0])
    c1 = rng.standard_normal((n_per_class, 2)) + np.array([+2.0, 0.0])
    x = np.concatenate([c0, c1], axis=0).astype(np.float32)  # 转 float32 与权重 dtype 对齐
    y = np.concatenate([np.zeros(n_per_class, dtype=np.int64), np.ones(n_per_class, dtype=np.int64)])
    idx = rng.permutation(len(x))
    return torch.from_numpy(x[idx]), torch.from_numpy(y[idx])


class MLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 16, out_dim: int = 2) -> None:
        super().__init__()
        # nn.Linear 内部就是 ch02 的 W·x + b，权重默认 Kaiming 均匀
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        return self.fc2(h)  # 返回 logits，不在这 softmax（CE 内置 log_softmax）


def main() -> None:
    device = get_device()
    print(f"使用设备: {device}")

    x, y = make_data(n_per_class=100, seed=0)
    x, y = x.to(device), y.to(device)

    model = MLP().to(device)
    # SGD -> stochastic gradient descent, 随机梯度下降
    # SGD 与 ch02 04_mlp_numpy 的手写 sgd_step 等价；lr 也对齐
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()  # 内部含 log_softmax，所以 forward 不要 softmax

    # 看一眼参数（验证 nn.Module 自动注册）
    print("\n注册的参数：")
    for name, p in model.named_parameters():
        print(f"  {name:>10s}  shape={tuple(p.shape)}  requires_grad={p.requires_grad}")

    epochs = 200
    for epoch in range(epochs):
        # 标准训练循环五件套
        logits = model(x)  # 1. forward
        loss = loss_fn(logits, y)  # 2. loss
        optimizer.zero_grad()  # 3. 清零（必须，避免梯度累加）
        loss.backward()  # 4. autograd 反向
        optimizer.step()  # 5. SGD 更新

        if epoch % 20 == 0 or epoch == epochs - 1:
            # eval 阶段不需要梯度，no_grad 省显存
            with torch.no_grad():
                # argmax = "最大值的下标"。dim=-1 = 沿最后一维（这里就是类别那一维）。
                pred = logits.argmax(dim=-1)
                # 算准确率
                # | 子表达式        | 类型/值示例                                       | 作用                    |
                # | ----------- | -------------------------------------------- | --------------------- |
                # | `pred == y` | `tensor([True, False, True, ...])` 形状 `(N,)` | 逐元素比较                 |
                # | `.float()`  | `tensor([1., 0., 1., ...])`                  | bool → 0/1，才能求均值      |
                # | `.mean()`   | `tensor(0.95)` 0 维标量张量                       | 1 的比例 = 正确率           |
                # | `.item()`   | `0.95` Python float                          | 张量 → 普通数，方便 print/log |
                acc = (pred == y).float().mean().item()
            print(f"epoch {epoch:>3d}  loss={loss.item():.4f}  acc={acc:.3f}")

    # 最终评估
    model.eval()  # 本网络无 Dropout/BN，eval() 这里无效但养成习惯
    with torch.no_grad():
        final_acc = (model(x).argmax(dim=-1) == y).float().mean().item()
    print(f"\n最终准确率 = {final_acc:.3f}")
    assert final_acc > 0.95, "应能轻松超过 95%（与 ch02 04_mlp_numpy 同任务）"

    # —— 以下为 §3.3 保存与加载演示，不计入与 ch02 的代码量对照 ——
    ckpt_path = Path(__file__).parent / "mlp_ckpt.pt"
    torch.save(model.state_dict(), ckpt_path)  # 只存权重 dict（推荐）

    model2 = MLP().to(device)  # 重建同结构空壳
    model2.load_state_dict(torch.load(ckpt_path, weights_only=True))  # 灌权重
    with torch.no_grad():
        assert torch.equal(model(x), model2(x)), "加载后输出应与原模型一致"
    ckpt_path.unlink()  # 演示完清理，避免污染仓库
    print("save/load 验证通过")

    print("PASS")


if __name__ == "__main__":
    main()
