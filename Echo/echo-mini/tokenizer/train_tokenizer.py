"""训练 BPE 分词器 (HuggingFace tokenizers 库)。

用法：
    python tokenizer/train_tokenizer.py --data_dir ../data/raw --vocab_size 16384

产物保存到 tokenizer/ 目录下 (tokenizer.json)。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders


def collect_text_files(data_dir: Path) -> list[str]:
    """收集 data_dir 下所有 .txt 文件路径。"""
    files = sorted(data_dir.glob("**/*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")
    print(f"Found {len(files)} text files for tokenizer training.")
    return [str(f) for f in files]


def train_tokenizer(data_dir: Path, vocab_size: int, output_dir: Path) -> None:
    """训练 BPE 分词器并保存。"""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,
    )

    files = collect_text_files(data_dir)
    tokenizer.train(files, trainer)

    # 设置特殊 token 的 id
    tokenizer.model.token_to_id("<pad>")  # 验证存在

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "tokenizer.json"
    tokenizer.save(str(save_path))
    print(f"Tokenizer saved to {save_path} (vocab_size={tokenizer.get_vocab_size()})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer for echo-mini")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="Directory containing .txt training files",
    )
    parser.add_argument("--vocab_size", type=int, default=16_384)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory for tokenizer.json",
    )
    args = parser.parse_args()
    train_tokenizer(args.data_dir, args.vocab_size, args.output_dir)


if __name__ == "__main__":
    main()
