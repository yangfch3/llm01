"""echo SFT 数据加载与处理。"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset


def load_sft_data(data_path: Path) -> Dataset:
    """从 JSONL 文件加载 SFT 数据为 HuggingFace Dataset。

    Args:
        data_path: JSONL 文件路径，每行一个 {"messages": [...]} 记录。

    Returns:
        HuggingFace Dataset，包含 "messages" 列。
    """
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def format_messages_for_trl(example: dict) -> dict:
    """保持 messages 格式不变，trl SFTTrainer 原生支持。

    trl >= 0.7 的 SFTTrainer 直接接受 messages 列（list of dicts），
    会自动调用 tokenizer 的 apply_chat_template。此函数为显式占位，
    方便后续加自定义处理逻辑。
    """
    return example
