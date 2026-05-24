"""预训练数据加载。

数据格式：每个 .bin 文件为连续 uint16 token ids（由 prepare_data.py 生成）。
训练时按 seq_len+1 切块，前 seq_len 为 input，后 seq_len 为 target（shift-by-1）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PretrainDataset(Dataset):
    """Memory-mapped 预训练数据集。"""

    def __init__(self, data_path: Path, seq_len: int):
        """
        Args:
            data_path: .bin 文件路径（uint16 token ids）
            seq_len: 上下文长度
        """
        self.seq_len = seq_len
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        # 每个 sample 需要 seq_len + 1 个 token (input + 1 target)
        self.n_samples = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return {"input_ids": x, "targets": y}


def create_dataloader(
    data_path: Path,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """创建预训练 DataLoader。"""
    ds = PretrainDataset(data_path, seq_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
