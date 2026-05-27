"""对 adapter checkpoint 跑 eval loss / perplexity。

用法：
    uv run python scripts/eval_loss.py --adapter-dir checkpoints/sft/final
    uv run python scripts/eval_loss.py --adapter-dir checkpoints/sft/checkpoint-3000
    uv run python scripts/eval_loss.py --adapter-dir checkpoints/sft/final --val-file data/sft/val.jsonl
"""

from __future__ import annotations

# datasets 必须在 torch 之前（Windows pyarrow DLL 冲突）
import datasets  # noqa: F401, I001

import argparse
import math
import sys
from pathlib import Path

import torch
from peft import PeftModel
from rich.console import Console
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))

from device import get_device
from echo.data import load_sft_data
from echo.utils import load_config

console = Console()

DEFAULT_CONFIG = Path("configs/sft-8g.yaml")
DEFAULT_ADAPTER = Path("checkpoints/sft/final")
DEFAULT_VAL_FILE = Path("data/sft/val.jsonl")


def evaluate(args: argparse.Namespace) -> None:
    """加载 adapter + val set，逐 batch 跑 forward 计算 loss。"""
    cfg = load_config(args.config)
    model_id = cfg["model"]["model_id"]
    adapter_dir = Path(args.adapter_dir)
    val_file = Path(args.val_file)
    max_seq_length = cfg["training"].get("max_seq_length", 2048)

    if not adapter_dir.exists():
        console.print(f"[bold red]Error:[/bold red] adapter 目录不存在: {adapter_dir}")
        raise SystemExit(1)
    if not val_file.exists():
        console.print(f"[bold red]Error:[/bold red] val 文件不存在: {val_file}")
        raise SystemExit(1)

    console.print(f"[bold]Model:[/bold] {model_id}")
    console.print(f"[bold]Adapter:[/bold] {adapter_dir}")
    console.print(f"[bold]Val file:[/bold] {val_file}")

    # Tokenizer（从底座加载，保留完整 chat_template）
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型（QLoRA 4bit 以节省显存）
    quant_cfg = cfg.get("quantization")
    if quant_cfg and quant_cfg.get("enabled", True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        device = get_device()
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    # 数据
    dataset = load_sft_data(val_file)
    console.print(f"[bold]Val samples:[/bold] {len(dataset)}")

    # Tokenize
    def tokenize(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    tokenized.set_format("torch")

    # Collate with padding
    def collate_fn(batch: list[dict]) -> dict:
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        max_len = max(len(ids) for ids in input_ids)
        padded_ids, padded_labels, attention_mask = [], [], []
        for ids, lbl in zip(input_ids, labels):
            pad_len = max_len - len(ids)
            padded_ids.append(torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id)]))
            padded_labels.append(torch.cat([lbl, torch.full((pad_len,), -100)]))
            attention_mask.append(torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)]))
        return {
            "input_ids": torch.stack(padded_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(attention_mask),
        }

    batch_size = args.batch_size
    loader = DataLoader(tokenized, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Eval loop
    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device

    console.print("[bold]Running eval...[/bold]")
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # loss 是 per-token 均值，乘以有效 token 数还原总 loss
            valid_tokens = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    console.print(f"\n[bold green]Results:[/bold green]")
    console.print(f"  Adapter: {adapter_dir}")
    console.print(f"  Val samples: {len(dataset)}")
    console.print(f"  Valid tokens: {total_tokens:,}")
    console.print(f"  Avg loss: {avg_loss:.4f}")
    console.print(f"  Perplexity: {ppl:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate adapter checkpoint on val set")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="YAML config (for model_id)"
    )
    parser.add_argument(
        "--adapter-dir", type=Path, default=DEFAULT_ADAPTER, help="Adapter checkpoint dir"
    )
    parser.add_argument(
        "--val-file", type=Path, default=DEFAULT_VAL_FILE, help="Validation JSONL file"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Eval batch size")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
