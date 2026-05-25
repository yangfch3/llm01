"""echo SFT 数据下载与预处理。

从 HuggingFace 下载 sharegpt_gpt4 数据集，统一为 messages 格式后保存。

用法：
    uv run python scripts/prepare_data.py [--output DIR] [--max-samples N] [--force]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from rich.console import Console

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sft"

# HuggingFace 数据集 ID
DATASET_ID = "shibing624/sharegpt_gpt4"


def convert_to_messages(example: dict) -> dict | None:
    """将 sharegpt 格式转为 messages 格式。

    sharegpt 格式: {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}
    目标格式: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
    """
    conversations = example.get("conversations", [])
    if not conversations:
        return None

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    role_map = {"human": "user", "gpt": "assistant", "system": "system"}

    for turn in conversations:
        role = role_map.get(turn.get("from", ""), None)
        content = turn.get("value", "").strip()
        if role is None or not content:
            return None
        # 跳过数据里自带的 system（我们已加了统一的）
        if role == "system":
            messages[0]["content"] = content
            continue
        messages.append({"role": role, "content": content})

    # 至少有 system + user + assistant
    if len(messages) < 3:
        return None

    # 确保最后一条是 assistant
    if messages[-1]["role"] != "assistant":
        return None

    return {"messages": messages}


def quality_filter(record: dict, max_total_chars: int = 8192) -> bool:
    """基础质量过滤。"""
    messages = record["messages"]
    total_chars = sum(len(m["content"]) for m in messages)
    # 过短
    if total_chars < 20:
        return False
    # 过长
    if total_chars > max_total_chars:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="echo SFT data preparation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=20000, help="最多保留条数")
    parser.add_argument("--force", action="store_true", help="覆盖已存在数据")
    args = parser.parse_args()

    output_file = args.output / "train.jsonl"

    if output_file.exists() and not args.force:
        console.print(f"[yellow]已存在数据，跳过：{output_file}（用 --force 覆盖）[/yellow]")
        return

    args.output.mkdir(parents=True, exist_ok=True)

    # 下载数据集
    console.print(f"[bold]Downloading:[/bold] {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split="train")
    console.print(f"  原始条数: {len(ds)}")

    # 格式转换 + 过滤
    records = []
    skipped = 0
    for example in ds:
        converted = convert_to_messages(example)
        if converted is None:
            skipped += 1
            continue
        if not quality_filter(converted):
            skipped += 1
            continue
        records.append(converted)
        if len(records) >= args.max_samples:
            break

    console.print(f"  保留: {len(records)}, 跳过: {skipped}")

    # 写出 JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    console.print(f"[bold green]Done:[/bold green] {output_file} ({len(records)} samples)")


if __name__ == "__main__":
    main()
