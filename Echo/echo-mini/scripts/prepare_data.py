"""数据下载与预处理。

流程：
1. 从 HuggingFace datasets 下载语料 (streaming 切片)
2. 清洗、过滤
3. 输出原始 .txt (供分词器训练)
4. 使用已训练好的分词器 tokenize → .bin (uint16 连续 token ids)

用法：
    # Step 1: 下载并存储原始文本（供分词器训练）
    python scripts/prepare_data.py download --config configs/pretrain-full.yaml

    # Step 2: 分词器训练完成后，tokenize 成 binary
    python scripts/prepare_data.py tokenize --config configs/pretrain-full.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将 src/ 加入搜索路径，使 echo_mini 包可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

from echo_mini.utils import load_config

# ============================================================
# 默认路径
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BIN_DIR = DATA_DIR / "bin"
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"


# ============================================================
# 下载
# ============================================================


def download_fineweb_edu(max_docs: int, output_path: Path) -> None:
    """下载 FineWeb-Edu 子集。"""
    print(f"Downloading FineWeb-Edu (max {max_docs} docs)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in tqdm(ds, total=max_docs, desc="fineweb-edu"):
            text = sample.get("text", "").strip()
            if len(text) < 100:
                continue
            f.write(text + "\n\n")
            count += 1
            if count >= max_docs:
                break
    print(f"  → {output_path} ({count} docs)")


def download_wiki_zh(max_docs: int, output_path: Path) -> None:
    """下载中文 Wikipedia 子集 (wikimedia/wikipedia parquet 格式)。"""
    print(f"Downloading Wikipedia ZH (max {max_docs} docs)...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.zh",
        split="train",
        streaming=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in tqdm(ds, total=max_docs, desc="wiki-zh"):
            text = sample.get("text", "").strip()
            if len(text) < 50:
                continue
            f.write(text + "\n\n")
            count += 1
            if count >= max_docs:
                break
    print(f"  → {output_path} ({count} docs)")


def download_skypile(max_docs: int, output_path: Path) -> None:
    """下载 SkyPile-150B 子集。"""
    print(f"Downloading SkyPile (max {max_docs} docs)...")
    ds = load_dataset(
        "Skywork/SkyPile-150B",
        split="train",
        streaming=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in tqdm(ds, total=max_docs, desc="skypile"):
            text = sample.get("text", "").strip()
            if len(text) < 50:
                continue
            f.write(text + "\n\n")
            count += 1
            if count >= max_docs:
                break
    print(f"  → {output_path} ({count} docs)")


def cmd_download(cfg: dict) -> None:
    """执行下载。"""
    data_cfg = cfg["data"]
    download_fineweb_edu(data_cfg.get("fineweb_docs", 50_000), RAW_DIR / "fineweb_edu.txt")
    download_wiki_zh(data_cfg.get("wiki_zh_docs", 30_000), RAW_DIR / "wiki_zh.txt")
    download_skypile(data_cfg.get("skypile_docs", 20_000), RAW_DIR / "skypile.txt")
    print("\nAll downloads complete. Now train tokenizer, then run `tokenize` command.")


# ============================================================
# Tokenize
# ============================================================


def tokenize_file(tokenizer: Tokenizer, input_path: Path, output_path: Path) -> int:
    """将 .txt 文件 tokenize 为 .bin (uint16)。返回 token 总数。

    每篇文档末尾追加 <eos> token，标记文档边界，
    使模型在预训练阶段学到序列终止信号。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eos_id = tokenizer.token_to_id("<eos>")
    all_ids: list[int] = []

    with open(input_path, encoding="utf-8") as f:
        # 分块读取避免内存爆炸
        chunk_size = 1024 * 1024  # 1MB
        buffer = ""
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            buffer += data
            # 按段落切分 tokenize
            paragraphs = buffer.split("\n\n")
            buffer = paragraphs[-1]  # 最后一段可能不完整
            for para in paragraphs[:-1]:
                para = para.strip()
                if not para:
                    continue
                encoded = tokenizer.encode(para)
                all_ids.extend(encoded.ids)
                all_ids.append(eos_id)
        # 处理剩余
        if buffer.strip():
            encoded = tokenizer.encode(buffer.strip())
            all_ids.extend(encoded.ids)
            all_ids.append(eos_id)

    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(output_path)
    return len(all_ids)


def cmd_tokenize(cfg: dict) -> None:
    """执行 tokenize。"""
    tokenizer_path = Path(cfg.get("tokenizer_path", str(TOKENIZER_PATH)))
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Train it first with train_tokenizer.py"
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Loaded tokenizer (vocab_size={tokenizer.get_vocab_size()})")

    total_tokens = 0
    for txt_file in sorted(RAW_DIR.glob("*.txt")):
        bin_file = BIN_DIR / (txt_file.stem + ".bin")
        n = tokenize_file(tokenizer, txt_file, bin_file)
        total_tokens += n
        print(f"  {txt_file.name} → {bin_file.name} ({n:,} tokens)")

    # 合并所有 bin 为 train.bin (方便 DataLoader)
    merge_bins(BIN_DIR, BIN_DIR / "train.bin")
    print(f"\nTotal tokens: {total_tokens:,}")


def merge_bins(bin_dir: Path, output_path: Path) -> None:
    """合并所有 .bin 文件为一个 train.bin。"""
    parts = sorted(f for f in bin_dir.glob("*.bin") if f.name != "train.bin")
    if not parts:
        print("  No .bin parts to merge.")
        return
    arrays = [np.fromfile(p, dtype=np.uint16) for p in parts]
    merged = np.concatenate(arrays)
    merged.tofile(output_path)
    print(f"  Merged → {output_path} ({len(merged):,} tokens)")


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="echo-mini data preparation")
    parser.add_argument("command", choices=["download", "tokenize"], help="Sub-command")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.command == "download":
        cmd_download(cfg)
    elif args.command == "tokenize":
        cmd_tokenize(cfg)


if __name__ == "__main__":
    main()
