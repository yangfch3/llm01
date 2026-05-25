"""SFT 对话数据下载与处理。

流程：
1. 从 HuggingFace 下载 Alpaca-GPT4 (中英)
2. 统一格式为 {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
3. 过滤空/过长条目
4. 输出 data/sft/train.jsonl

用法：
    cd Echo/echo-mini
    uv run python scripts/prepare_sft_data.py --config configs/sft-full.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 将 src/ 加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import load_dataset
from tqdm import tqdm

from echo_mini.utils import load_config

ROOT = Path(__file__).resolve().parent.parent
SFT_DIR = ROOT / "data" / "sft"


# ============================================================
# 数据集处理函数
# ============================================================


def process_alpaca_sample(sample: dict) -> dict | None:
    """将 Alpaca 格式转为统一 messages 格式。

    Alpaca 字段: instruction, input, output
    如果 input 非空，拼到 instruction 后面。
    """
    instruction = (sample.get("instruction") or "").strip()
    inp = (sample.get("input") or "").strip()
    output = (sample.get("output") or "").strip()

    if not instruction or not output:
        return None

    user_content = f"{instruction}\n{inp}" if inp else instruction

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def download_alpaca_gpt4_en(max_samples: int) -> list[dict]:
    """下载 Alpaca-GPT4 英文数据。"""
    print(f"Downloading Alpaca-GPT4-EN (max {max_samples})...")
    ds = load_dataset("vicgalle/alpaca-gpt4", split="train", streaming=True)
    results = []
    for sample in tqdm(ds, total=max_samples, desc="alpaca-gpt4-en"):
        processed = process_alpaca_sample(sample)
        if processed is not None:
            results.append(processed)
        if len(results) >= max_samples:
            break
    print(f"  → {len(results)} samples")
    return results


def download_alpaca_gpt4_zh(max_samples: int) -> list[dict]:
    """下载 Alpaca-GPT4 中文数据。"""
    print(f"Downloading Alpaca-GPT4-ZH (max {max_samples})...")
    ds = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train", streaming=True)
    results = []
    for sample in tqdm(ds, total=max_samples, desc="alpaca-gpt4-zh"):
        processed = process_alpaca_sample(sample)
        if processed is not None:
            results.append(processed)
        if len(results) >= max_samples:
            break
    print(f"  → {len(results)} samples")
    return results


# ============================================================
# 过滤
# ============================================================


def filter_by_length(samples: list[dict], max_chars: int = 2000) -> list[dict]:
    """过滤 user+assistant 总字符数超过 max_chars 的条目。"""
    filtered = []
    for s in samples:
        total = sum(len(m["content"]) for m in s["messages"])
        if total <= max_chars:
            filtered.append(s)
    dropped = len(samples) - len(filtered)
    if dropped:
        print(f"  Filtered out {dropped} samples (>{max_chars} chars)")
    return filtered


# ============================================================
# 输出
# ============================================================


def save_jsonl(samples: list[dict], output_path: Path) -> None:
    """保存为 JSONL。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples → {output_path}")


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="echo-mini SFT data preparation")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sft_cfg = cfg.get("sft_data", {})
    max_en = sft_cfg.get("alpaca_en_samples", 10_000)
    max_zh = sft_cfg.get("alpaca_zh_samples", 10_000)
    max_chars = sft_cfg.get("max_chars_per_sample", 2000)

    # 下载
    en_data = download_alpaca_gpt4_en(max_en)
    zh_data = download_alpaca_gpt4_zh(max_zh)

    # 合并 + 过滤
    all_data = en_data + zh_data
    all_data = filter_by_length(all_data, max_chars)

    # 保存
    output_path = SFT_DIR / "train.jsonl"
    save_jsonl(all_data, output_path)

    print(f"\nDone. Total: {len(all_data)} samples (EN: {len(en_data)}, ZH: {len(zh_data)})")


if __name__ == "__main__":
    main()
