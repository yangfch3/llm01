"""数据加载：预训练 + SFT。

PretrainDataset: .bin 文件连续 uint16 token ids，按 seq_len 切块。
SFTDataset: JSONL 对话数据，运行时 tokenize + 构造 labels (user 部分 mask 为 -100)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

# ============================================================
# Chat Template
# ============================================================

# 格式: <bos><|user|>{content}\n<|assistant|>{content}<eos>
# 角色标记使用专用特殊 token，单 token 即可标识角色切换
TURN_SEP = "\n"


def format_chat(messages: list[dict], bos_id: int, eos_id: int, tokenizer: Tokenizer) -> dict:
    """将 messages 转为 token ids + labels。

    user 部分的 label 设为 IGNORE_INDEX (-100)，只对 assistant 部分计算 loss。
    返回 {"input_ids": list[int], "labels": list[int]}。
    """
    ignore_index = -100
    user_token_id = tokenizer.token_to_id("<|user|>")
    assistant_token_id = tokenizer.token_to_id("<|assistant|>")

    input_ids: list[int] = [bos_id]
    labels: list[int] = [ignore_index]  # bos 不计 loss

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # 非最后一条消息，content 末尾加换行分隔
        if msg is not messages[-1]:
            content += TURN_SEP

        content_ids = tokenizer.encode(content).ids

        if role == "user":
            # <|user|> + content → 全部 mask
            input_ids.append(user_token_id)
            labels.append(ignore_index)
            input_ids.extend(content_ids)
            labels.extend([ignore_index] * len(content_ids))
        else:
            # <|assistant|> → mask, content → 计 loss
            input_ids.append(assistant_token_id)
            labels.append(ignore_index)
            input_ids.extend(content_ids)
            labels.extend(content_ids)

    # 结尾加 eos
    input_ids.append(eos_id)
    labels.append(eos_id)

    return {"input_ids": input_ids, "labels": labels}


# ============================================================
# Pretrain Dataset
# ============================================================


class PretrainDataset(Dataset):
    """Memory-mapped 预训练数据集。

    注意：bin 文件是多篇文档 token 拼接的连续流，按 seq_len 切块时一个 sample
    可能跨越文档边界。这是小规模预训练的常见简化做法；如需严格隔离可在
    prepare_data 阶段插入 <eos> 分隔（本项目已在 tokenize 阶段处理）。
    """

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


# ============================================================
# SFT Dataset
# ============================================================


class SFTDataset(Dataset):
    """JSONL 对话数据集。运行时 tokenize，支持 user mask。"""

    def __init__(self, jsonl_path: Path, tokenizer: Tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")
        self.pad_id = tokenizer.token_to_id("<pad>")

        # 加载全部数据到内存（数据量小，~10K-20K 条）
        self.samples: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        result = format_chat(sample["messages"], self.bos_id, self.eos_id, self.tokenizer)

        input_ids = result["input_ids"]
        labels = result["labels"]

        # 截断到 max_seq_len
        input_ids = input_ids[: self.max_seq_len]
        labels = labels[: self.max_seq_len]

        # 右填充到 max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        input_ids = input_ids + [self.pad_id] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ============================================================
# DataLoader 工厂
# ============================================================


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


def create_sft_dataloader(
    jsonl_path: Path,
    tokenizer: Tokenizer,
    max_seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """创建 SFT DataLoader。"""
    ds = SFTDataset(jsonl_path, tokenizer, max_seq_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

