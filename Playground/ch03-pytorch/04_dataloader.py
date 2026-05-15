"""ch03 练习 4：自定义 Dataset + DataLoader。

演示三件事：
1. 自定义 Dataset 三件套（__init__ / __len__ / __getitem__）
2. DataLoader 怎么把单样本堆成 batch
3. shuffle / drop_last / batch_size 对一个 epoch 的影响
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset


class ToyDataset(Dataset):
    """10 条样本，每条 (x: shape (3,), y: scalar int)。"""

    def __init__(self, n: int = 10) -> None:
        # 确定性数据，便于观察 batch 拼装顺序
        self.x = torch.arange(n * 3, dtype=torch.float32).view(n, 3)
        self.y = torch.arange(n, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 返回单样本，DataLoader 会用 default_collate 自动堆成 batch
        return self.x[idx], self.y[idx]


def demo_basic_iter() -> None:
    print("\n--- 1. 基本迭代：batch_size=4 ---")
    loader = DataLoader(ToyDataset(10), batch_size=4, shuffle=False)
    for i, (xb, yb) in enumerate(loader):
        # 注意末尾 batch 只有 2 条（10 % 4 = 2）
        print(f"batch {i}: x.shape={tuple(xb.shape)}  y={yb.tolist()}")
    # 默认 drop_last=False，末尾不满的也保留


def demo_drop_last() -> None:
    print("\n--- 2. drop_last=True：丢弃末尾不足一个 batch 的样本 ---")
    loader = DataLoader(ToyDataset(10), batch_size=4, shuffle=False, drop_last=True)
    for i, (_, yb) in enumerate(loader):
        print(f"batch {i}: y={yb.tolist()}")  # 只剩 2 个完整 batch（共 8 样本）


def demo_shuffle() -> None:
    print("\n--- 3. shuffle=True：每个 epoch 顺序不同 ---")
    # 固定 seed 让两个 epoch 的打乱结果可复现，但彼此应不同
    g = torch.Generator().manual_seed(42)
    loader = DataLoader(ToyDataset(8), batch_size=4, shuffle=True, generator=g)
    for epoch in range(2):
        order = []
        for _, yb in loader:
            order.extend(yb.tolist())
        print(f"epoch {epoch} 实际顺序: {order}")
    # 训练才打乱；val/test 留 False 便于"按 batch 看错例"复现


def demo_num_workers_warning() -> None:
    print("\n--- 4. num_workers Win 注意事项 ---")
    print("  教学脚本一律 num_workers=0，最稳。")
    print("  num_workers > 0 时 Win 必须 if __name__ == '__main__': 守卫，")
    print("  否则 multiprocessing 会无限递归 fork（fork 在 Win 走 spawn）。")


def main() -> None:
    demo_basic_iter()
    demo_drop_last()
    demo_shuffle()
    demo_num_workers_warning()
    print("\nPASS")


if __name__ == "__main__":
    main()
