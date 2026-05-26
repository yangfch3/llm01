"""训练 BPE 分词器 (HuggingFace tokenizers 库)。

用法：
    cd Echo/echo-mini
    uv run python tokenizer/train_tokenizer.py

产物保存到 tokenizer/ 目录下 (tokenizer.json)。
vocab_size = 16386，其中包含 6 个特殊 token（BpeTrainer 的 vocab_size 已含特殊 token）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders


# echo-mini 全部特殊 token，id 按顺序分配 (0-5)
SPECIAL_TOKENS = [
    "<pad>",        # 0 - padding
    "<bos>",        # 1 - beginning of sequence
    "<eos>",        # 2 - end of sequence
    "<unk>",        # 3 - unknown
    "<|user|>",     # 4 - user role marker
    "<|assistant|>",  # 5 - assistant role marker
]


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

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )

    files = collect_text_files(data_dir)
    tokenizer.train(files, trainer)

    # 验证特殊 token
    for tok in SPECIAL_TOKENS:
        tid = tokenizer.token_to_id(tok)
        assert tid is not None, f"Special token {tok} not in vocab!"
        print(f"  {tok}: {tid}")

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
    parser.add_argument("--vocab_size", type=int, default=16_386)
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

