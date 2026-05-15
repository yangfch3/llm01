"""ch03 练习 5：MNIST + MLP 综合实战。

把前 4 个练习的概念串起来：Tensor / autograd / nn.Module / DataLoader → 标准训练循环。
任务：MNIST 手写数字分类（10 类），用 2 层 MLP，目标测试集 ≥ 97%。

跑法：
    uv run python Playground/ch03-pytorch/05_mnist_mlp.py

数据：torchvision.datasets.MNIST 首次会下载到 ./data/mnist/（约 11MB，已 .gitignore）。
预期耗时：3060 ≈ 30s，Mac M 系列 ≈ 1min，纯 CPU ≈ 2min。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Echo.shared.device import get_device

DATA_DIR = REPO_ROOT / "data" / "mnist"


class MLP(nn.Module):
    """28x28 → flatten → 256 → 128 → 10 的两层 MLP。"""

    def __init__(self, hidden1: int = 256, hidden2: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # (N, 1, 28, 28) → (N, 784)
            nn.Linear(28 * 28, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),  # 输出 logits，CE 内含 log_softmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    # MNIST 像素 0~255 uint8 → ToTensor 把它转 (1,28,28) float32 ∈ [0,1]
    # Normalize 用官方均值/方差白化，加速收敛（这两个数是 MNIST 训练集的统计值）
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # download=True：首次运行会从镜像下载到 DATA_DIR；后续直接读本地
    train_set = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=transform)

    # num_workers=0：教学脚本最稳，避免 Win multiprocessing 坑（ch03 §4.3）
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)
    return train_loader, test_loader


@torch.no_grad()  # 整个函数关 autograd，等价于函数体外包 with torch.no_grad():
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()  # 切推理模式（影响 Dropout/BN，本网络无但养成习惯）
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def main() -> None:
    device = get_device()
    print(f"使用设备: {device}")

    torch.manual_seed(0)
    train_loader, test_loader = build_loaders(batch_size=128)
    model = MLP().to(device)
    # Adam 自适应学习率，比 SGD 在 MNIST 这种小任务上收敛更快、更省调参
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 3
    t0 = time.time()
    for epoch in range(epochs):
        model.train()  # 切训练模式
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()  # 必须，避免梯度累加
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(
                    f"  epoch {epoch} batch {batch_idx:>4d}/{len(train_loader)}  "
                    f"loss={loss.item():.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        test_acc = evaluate(model, test_loader, device)
        print(
            f"epoch {epoch} 完成  avg_train_loss={avg_loss:.4f}  test_acc={test_acc:.4f}  "
            f"用时={time.time() - t0:.1f}s"
        )

    final_acc = evaluate(model, test_loader, device)
    print(f"\n最终测试准确率 = {final_acc:.4f}")
    assert final_acc > 0.97, f"预期 ≥ 0.97，实际 {final_acc:.4f}"
    print("PASS")


if __name__ == "__main__":
    main()
