"""echo DPO 偏好数据下载与预处理。

数据源：wenbopan/Chinese-dpo-pairs（中文多源混合偏好数据集，10735 条）。
- 字段：prompt / system / chosen / rejected / source / id
- 风格相关 source（sharegpt / ultrachat / evol_instruct / openorca）保留，
  flan 翻译题与过短闲聊默认过滤掉

输出：data/dpo/{train,val}.jsonl，trl DPOTrainer 标准 chat 格式
    {"prompt": [...], "chosen": [...], "rejected": [...]}

用法：
    # 默认（保留对话风格 source，约 8K 条）
    uv run python scripts/prepare_dpo_data.py

    # 自定义采样
    uv run python scripts/prepare_dpo_data.py --max-samples 5000 --force

    # 保留全部 source（含 flan 翻译题等）
    uv run python scripts/prepare_dpo_data.py --keep-all-sources
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
DEFAULT_OUTPUT = REPO_ROOT / "data" / "dpo"

DATASET_ID = "wenbopan/Chinese-dpo-pairs"

# 默认保留的 source —— 与 echo 日常对话目标契合度高
DEFAULT_KEEP_SOURCES = {
    "sharegpt",
    "ultrachat",
    "evol_instruct",
    "openorca",
    "truthy_dpo",
    "false_qa",
}

# 默认 system prompt，与 SFT 阶段保持一致（避免分布漂移）
DEFAULT_SYSTEM = "You are a helpful assistant."


def to_dpo_record(example: dict) -> dict | None:
    """单条原始记录 → trl DPO chat 格式。

    输出格式（trl >= 0.7）：
        {
          "prompt":   [{"role": "system", ...}, {"role": "user", ...}],
          "chosen":   [{"role": "assistant", "content": ...}],
          "rejected": [{"role": "assistant", "content": ...}],
        }
    """
    prompt = (example.get("prompt") or "").strip()
    chosen = (example.get("chosen") or "").strip()
    rejected = (example.get("rejected") or "").strip()
    system = (example.get("system") or "").strip() or DEFAULT_SYSTEM

    if not prompt or not chosen or not rejected:
        return None
    # chosen / rejected 完全相同的样本无信号
    if chosen == rejected:
        return None

    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def quality_filter(
    record: dict,
    *,
    min_chars: int = 10,
    max_prompt_chars: int = 2000,
    max_response_chars: int = 3000,
) -> bool:
    """基础长度过滤。

    - prompt user 内容上限 max_prompt_chars（DPO max_prompt_length=512 token 约对应 1500 中文字符）
    - chosen / rejected 上限 max_response_chars
    - chosen / rejected 都不能过短（< min_chars，避免单字回复）
    """
    user_content = next(
        (m["content"] for m in record["prompt"] if m["role"] == "user"), ""
    )
    chosen_content = record["chosen"][0]["content"]
    rejected_content = record["rejected"][0]["content"]

    if len(user_content) > max_prompt_chars:
        return False
    if len(chosen_content) < min_chars or len(rejected_content) < min_chars:
        return False
    if len(chosen_content) > max_response_chars or len(rejected_content) > max_response_chars:
        return False
    return True


def write_jsonl(records: list[dict], path: Path) -> None:
    """写出 JSONL 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    parser = argparse.ArgumentParser(description="echo DPO data preparation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=8000,
        help="过滤后最多保留条数（默认 8000，3060 12GB 单 epoch 可消化）",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="shuffle 随机种子")
    parser.add_argument("--force", action="store_true", help="覆盖已存在数据")
    parser.add_argument(
        "--keep-all-sources",
        action="store_true",
        help="保留全部 source（默认仅保留对话风格 source，过滤 flan 翻译题等）",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    train_file = args.output / "train.jsonl"
    val_file = args.output / "val.jsonl"

    if train_file.exists() and not args.force:
        console.print(
            f"[yellow]已存在 DPO 数据，跳过：{train_file}（用 --force 覆盖）[/yellow]"
        )
        return

    console.print(f"[bold]Downloading:[/bold] {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split="train")
    console.print(f"  原始条数: {len(ds)}")

    # source 过滤
    if not args.keep_all_sources:
        ds = ds.filter(lambda x: x["source"] in DEFAULT_KEEP_SOURCES)
        console.print(
            f"  source 过滤后（保留 {sorted(DEFAULT_KEEP_SOURCES)}）: {len(ds)}"
        )

    # 转换 + 质量过滤
    records: list[dict] = []
    skipped_convert = 0
    skipped_quality = 0
    for example in ds:
        rec = to_dpo_record(example)
        if rec is None:
            skipped_convert += 1
            continue
        if not quality_filter(rec):
            skipped_quality += 1
            continue
        records.append(rec)

    console.print(
        f"  保留: {len(records)}, 转换跳过: {skipped_convert}, 质量过滤跳过: {skipped_quality}"
    )

    # 采样上限
    if len(records) > args.max_samples:
        rng = random.Random(args.seed)
        rng.shuffle(records)
        records = records[: args.max_samples]
        console.print(f"  采样到上限: {len(records)}")

    split_and_write(records, train_file, val_file, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
