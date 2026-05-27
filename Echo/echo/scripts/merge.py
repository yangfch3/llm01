"""合并 LoRA adapter 到底座，导出完整 bf16 权重。

用法：
    uv run python scripts/merge.py
    uv run python scripts/merge.py --adapter-dir checkpoints/sft/final --output-dir checkpoints/merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from echo.utils import load_config

console = Console()

DEFAULT_CONFIG = Path("configs/sft-full.yaml")
DEFAULT_ADAPTER = Path("checkpoints/sft/final")
DEFAULT_OUTPUT = Path("checkpoints/merged")


def merge(args: argparse.Namespace) -> None:
    """加载底座 + adapter，合并后保存。"""
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)

    # 从配置读取底座 model_id
    cfg = load_config(args.config)
    model_id = cfg["model"]["model_id"]

    if not adapter_dir.exists():
        console.print(f"[bold red]Error:[/bold red] adapter 目录不存在: {adapter_dir}")
        raise SystemExit(1)

    console.print(f"[bold]Base model:[/bold] {model_id}")
    console.print(f"[bold]Adapter:[/bold] {adapter_dir}")
    console.print(f"[bold]Output:[/bold] {output_dir}")

    # 加载底座（bf16，不量化）
    console.print("Loading base model (bf16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # 加载 adapter
    console.print("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    # 合并
    console.print("Merging adapter into base model...")
    model = model.merge_and_unload()

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Saving merged model to {output_dir}...")
    model.save_pretrained(str(output_dir), safe_serialization=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))

    console.print("[bold green]Done![/bold green] Merged model saved.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="YAML config (for model_id)"
    )
    parser.add_argument(
        "--adapter-dir", type=Path, default=DEFAULT_ADAPTER, help="Adapter checkpoint dir"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output dir for merged model"
    )
    args = parser.parse_args()
    merge(args)


if __name__ == "__main__":
    main()
