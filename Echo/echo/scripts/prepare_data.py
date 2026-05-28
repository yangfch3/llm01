"""echo SFT 数据下载与预处理。

本项目锚点是 base 路线，故默认产出增广数据：
ShareGPT (sharegpt_gpt4) + 短问答 (alpaca-gpt4 英 + alpaca-gpt4-chinese 中)
混合后写入 data/sft/train_aug.jsonl + data/sft/val_aug.jsonl。

各数据源默认采样：
- ShareGPT: --max-samples 上限 20000（实际取 ~19K，受质量过滤影响）
- 英文短问答 (alpaca-gpt4): --short-qa-en 默认 5000
- 中文短问答 (alpaca-gpt4-chinese): --short-qa-zh 默认 5000

用法：
    # 默认：增广（base 路线用，约 29K 总量）
    uv run python scripts/prepare_data.py [--output DIR] [--force]

    # 仅 ShareGPT（instruct 路线对照用，约 19K 总量）
    uv run python scripts/prepare_data.py --mode sharegpt-only
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from rich.console import Console

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sft"

# HuggingFace 数据集 ID
DATASET_ID = "shibing624/sharegpt_gpt4"
SHORT_QA_EN_ID = "vicgalle/alpaca-gpt4"
SHORT_QA_ZH_ID = "silk-road/alpaca-data-gpt4-chinese"

# 增广默认采样量
DEFAULT_SHORT_QA_EN = 5000
DEFAULT_SHORT_QA_ZH = 5000


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


def convert_alpaca_to_messages(
    example: dict,
    instruction_key: str,
    input_key: str,
    output_key: str,
) -> dict | None:
    """alpaca 格式 → messages 格式。

    alpaca: {"instruction": ..., "input": ..., "output": ...}
    若 input 非空，user 内容为 "{instruction}\\n\\n{input}"，否则就是 instruction。
    """
    instruction = (example.get(instruction_key) or "").strip()
    input_text = (example.get(input_key) or "").strip()
    output = (example.get(output_key) or "").strip()

    if not instruction or not output:
        return None

    user_content = f"{instruction}\n\n{input_text}" if input_text else instruction

    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


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


def short_qa_filter(record: dict, max_assistant_chars: int = 800) -> bool:
    """短问答专用过滤：assistant 回复长度上限，去掉过长样本（降低与 ShareGPT 风格冲突）。"""
    if not quality_filter(record):
        return False
    assistant_chars = sum(len(m["content"]) for m in record["messages"] if m["role"] == "assistant")
    return assistant_chars <= max_assistant_chars


def write_jsonl(records: list[dict], path: Path) -> None:
    """写出 JSONL 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_sharegpt(max_samples: int) -> list[dict]:
    """下载并转换 ShareGPT，返回 messages 记录列表。"""
    console.print(f"[bold]Downloading:[/bold] {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split="train")
    console.print(f"  原始条数: {len(ds)}")

    records: list[dict] = []
    skipped = 0
    for example in ds:
        converted = convert_to_messages(example)
        if converted is None or not quality_filter(converted):
            skipped += 1
            continue
        records.append(converted)
        if len(records) >= max_samples:
            break
    console.print(f"  ShareGPT 保留: {len(records)}, 跳过: {skipped}")
    return records


def load_short_qa(
    n_en: int,
    n_zh: int,
    seed: int,
) -> list[dict]:
    """拉取英文 + 中文 alpaca-gpt4 短问答，返回混合 messages 记录列表。"""
    rng = random.Random(seed)
    out: list[dict] = []

    # 英文：vicgalle/alpaca-gpt4
    if n_en > 0:
        console.print(f"[bold]Downloading:[/bold] {SHORT_QA_EN_ID}")
        ds_en = load_dataset(SHORT_QA_EN_ID, split="train")
        console.print(f"  原始条数: {len(ds_en)}")
        records_en: list[dict] = []
        for example in ds_en:
            converted = convert_alpaca_to_messages(
                example, "instruction", "input", "output"
            )
            if converted is None or not short_qa_filter(converted):
                continue
            records_en.append(converted)
        rng.shuffle(records_en)
        records_en = records_en[:n_en]
        console.print(f"  英文短问答采样: {len(records_en)}")
        out.extend(records_en)

    # 中文：silk-road/alpaca-data-gpt4-chinese
    # 该数据集字段：instruction_zh / input_zh / output_zh（中文翻译版）
    if n_zh > 0:
        console.print(f"[bold]Downloading:[/bold] {SHORT_QA_ZH_ID}")
        ds_zh = load_dataset(SHORT_QA_ZH_ID, split="train")
        console.print(f"  原始条数: {len(ds_zh)}")
        # 自适应字段名：优先 _zh 后缀，回落到无后缀
        sample = ds_zh[0]
        zh_keys = ("instruction_zh", "input_zh", "output_zh")
        en_keys = ("instruction", "input", "output")
        if all(k in sample for k in zh_keys):
            keys = zh_keys
        elif all(k in sample for k in en_keys):
            keys = en_keys
        else:
            # 上游数据集结构变更时显式 raise，避免静默全过滤为 0
            raise RuntimeError(
                f"{SHORT_QA_ZH_ID} 字段不符合预期: 既无 {zh_keys} 也无 {en_keys}, "
                f"实际字段: {list(sample.keys())}"
            )
        console.print(f"  使用字段: {keys}")
        records_zh: list[dict] = []
        for example in ds_zh:
            converted = convert_alpaca_to_messages(example, *keys)
            if converted is None or not short_qa_filter(converted):
                continue
            records_zh.append(converted)
        rng.shuffle(records_zh)
        records_zh = records_zh[:n_zh]
        console.print(f"  中文短问答采样: {len(records_zh)}")
        out.extend(records_zh)

    return out


def split_and_write(
    records: list[dict],
    train_file: Path,
    val_file: Path,
    val_ratio: float,
    seed: int,
) -> None:
    """shuffle → train/val 切分 → 写出。"""
    rng = random.Random(seed)
    rng.shuffle(records)
    val_size = max(1, int(len(records) * val_ratio))
    val_records = records[:val_size]
    train_records = records[val_size:]
    write_jsonl(train_records, train_file)
    write_jsonl(val_records, val_file)
    console.print(
        f"[bold green]Done:[/bold green] "
        f"train={len(train_records)} → {train_file}, "
        f"val={len(val_records)} → {val_file}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="echo SFT data preparation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=20000, help="ShareGPT 最多保留条数")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="shuffle 随机种子")
    parser.add_argument("--force", action="store_true", help="覆盖已存在数据")
    parser.add_argument(
        "--mode",
        choices=("aug", "sharegpt-only"),
        default="aug",
        help="aug (默认): ShareGPT + 短问答 → train_aug.jsonl / val_aug.jsonl，base 路线用; "
             "sharegpt-only: 仅 ShareGPT → train.jsonl / val.jsonl，instruct 路线用",
    )
    parser.add_argument(
        "--short-qa-en",
        type=int,
        default=DEFAULT_SHORT_QA_EN,
        help="英文短问答采样数（仅 --mode aug 时生效）",
    )
    parser.add_argument(
        "--short-qa-zh",
        type=int,
        default=DEFAULT_SHORT_QA_ZH,
        help="中文短问答采样数（仅 --mode aug 时生效）",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "aug":
        # 增广路径：产出 train_aug.jsonl / val_aug.jsonl
        train_file = args.output / "train_aug.jsonl"
        val_file = args.output / "val_aug.jsonl"

        if train_file.exists() and not args.force:
            console.print(
                f"[yellow]已存在增广数据，跳过：{train_file}（用 --force 覆盖）[/yellow]"
            )
            return

        sharegpt_records = load_sharegpt(args.max_samples)
        short_qa_records = load_short_qa(args.short_qa_en, args.short_qa_zh, args.seed)
        all_records = sharegpt_records + short_qa_records
        console.print(
            f"[bold]混合总数:[/bold] {len(all_records)} "
            f"(ShareGPT {len(sharegpt_records)} + 短问答 {len(short_qa_records)})"
        )
        split_and_write(all_records, train_file, val_file, args.val_ratio, args.seed)
    else:
        # sharegpt-only：仅 ShareGPT，输出到 train.jsonl / val.jsonl
        train_file = args.output / "train.jsonl"
        val_file = args.output / "val.jsonl"

        if train_file.exists() and not args.force:
            console.print(
                f"[yellow]已存在数据，跳过：{train_file}（用 --force 覆盖）[/yellow]"
            )
            return

        records = load_sharegpt(args.max_samples)
        split_and_write(records, train_file, val_file, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
