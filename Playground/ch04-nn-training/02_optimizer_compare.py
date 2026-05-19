"""ch04 练习 2：优化器对比。

四档：SGD / SGD+momentum / Adam / AdamW。
任务：合成回归（y = sin(x) + 噪声），用同一个小 MLP 拟合。
看 loss 下降曲线，理解"为什么 Adam 几乎不用调 lr 就能跑"。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from Echo.shared.device import get_device


def make_data(n: int = 1024, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    x = torch.linspace(-3.14, 3.14, n).unsqueeze(-1)  # (n, 1)
    # 加少量高斯噪声让任务非平凡
    y = torch.sin(x) + 0.1 * torch.randn(x.shape, generator=g)
    if device is not None:
        x, y = x.to(device), y.to(device)
    return x, y


def build_model() -> nn.Module:
    # 小 MLP：1 → 64 → 64 → 1
    return nn.Sequential(
        nn.Linear(1, 64),
        nn.Tanh(),  # 回归用 tanh 比 ReLU 平滑些
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )


def train_one(opt_name: str, x: torch.Tensor, y: torch.Tensor, device: torch.device, steps: int = 500) -> list[float]:
    torch.manual_seed(0)  # 同 init 才公平
    model = build_model().to(device)
    loss_fn = nn.MSELoss()

    # 注意 lr：SGD 系大、Adam 系小（ch04 §3.4 自检 1）
    if opt_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
    elif opt_name == "SGD+momentum":
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    elif opt_name == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    elif opt_name == "AdamW":
        # AdamW: weight decay 与梯度更新解耦（ch04 §3.4）
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    else:
        raise ValueError(opt_name)

    losses: list[float] = []
    for _ in range(steps):
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def main() -> None:
    device = get_device()
    print(f"使用设备: {device}")
    x, y = make_data(device=device)

    names = ["SGD", "SGD+momentum", "Adam", "AdamW"]
    history = {name: train_one(name, x, y, device) for name in names}

    # 抽几个时间点对照
    checkpoints = [0, 50, 100, 200, 500]
    print(f"\n{'优化器':<16}" + "".join(f"step{c:<6}" for c in (cp for cp in checkpoints)))
    print("-" * 60)
    for name in names:
        row = f"{name:<16}"
        for cp in checkpoints:
            idx = min(cp, len(history[name]) - 1)
            row += f"{history[name][idx]:<10.4f}"
        print(row)

    # 收敛性断言：四个最终 loss 都应低于 0.1
    for name in names:
        final = history[name][-1]
        assert final < 0.1, f"{name} 最终 loss {final:.4f} 超阈值，说明配置有问题"

    print("\n观察要点:")
    print("- SGD vs SGD+momentum：动量让早期下降更快")
    print("- Adam/AdamW vs SGD：自适应 lr，前 100 步基本收敛")
    print("- AdamW 与 Adam 在小任务上几乎无差，差异在大模型泛化上")

    # --plot：画 loss 走势对比图
    if "--plot" in sys.argv:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        for name in names:
            plt.plot(history[name], label=name)
        plt.xlabel("Step")
        plt.ylabel("MSE Loss")
        plt.title("Optimizer Comparison: Loss Curve")
        plt.legend()
        plt.yscale("log")  # log 尺度看差异更明显
        # 标注 Adam/AdamW 重合现象
        plt.annotate(
            "Adam ~ AdamW: weight decay has\nnear-zero effect on small tasks;\ndifference shows in large-model generalization",
            xy=(250, history["Adam"][250]),
            xytext=(300, history["SGD"][200]),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray"),
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
        )
        plt.tight_layout()
        out_path = REPO_ROOT / "Doc" / "Courseware" / "ch04-nn-training" / "optimizer_compare.png"
        plt.savefig(out_path, dpi=150)
        print(f"\n图已保存: {out_path.relative_to(REPO_ROOT)}")
        plt.show()

    print("\nPASS")


if __name__ == "__main__":
    main()
