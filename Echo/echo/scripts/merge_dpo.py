"""合并 DPO LoRA adapter 到 merged-base，导出 Echo v2 完整 bf16 权重。

T6.3 产物：DPO 训练后的 adapter 与 SFT 已合并的 merged-base 再合并，
得到 Echo v2 完整模型。后续可继续走 export_gguf.py 量化部署链路。

用法：
    # 默认（dpo-base/final → merged-dpo/）
    uv run python scripts/merge_dpo.py

    # 指定 ckpt
    uv run python scripts/merge_dpo.py --adapter-dir checkpoints/dpo-base/checkpoint-N
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

console = Console()

DEFAULT_BASE = Path("checkpoints/merged-base")     # SFT 已合并的完整模型
DEFAULT_ADAPTER = Path("checkpoints/dpo-base/final")
DEFAULT_OUTPUT = Path("checkpoints/merged-dpo")    # Echo v2


def merge(args: argparse.Namespace) -> None:
    """加载 merged-base + DPO adapter，合并后保存为 Echo v2。"""
    base_dir = Path(args.base_dir)
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)

    if not base_dir.exists():
        console.print(f"[bold red]Error:[/bold red] 底座目录不存在: {base_dir}")
        raise SystemExit(1)
    if not adapter_dir.exists():
        console.print(f"[bold red]Error:[/bold red] adapter 目录不存在: {adapter_dir}")
        raise SystemExit(1)

    console.print(f"[bold]Base (SFT merged):[/bold] {base_dir}")
    console.print(f"[bold]DPO adapter:[/bold] {adapter_dir}")
    console.print(f"[bold]Output (Echo v2):[/bold] {output_dir}")

    # 加载 merged-base（bf16，CPU 合并避免占显存）
    console.print("Loading SFT merged base (bf16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # 加载 DPO adapter
    console.print("Loading DPO adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    # 合并
    console.print("Merging DPO adapter into base...")
    model = model.merge_and_unload()

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Saving Echo v2 to {output_dir}...")
    model.save_pretrained(str(output_dir), safe_serialization=True)

    # Tokenizer：优先从 adapter 取（含训练时写入的 chat_template），失败回落 base
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
        console.print(f"  tokenizer loaded from adapter: {adapter_dir}")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
        console.print(f"  tokenizer loaded from base: {base_dir}")
    tokenizer.save_pretrained(str(output_dir))

    console.print("[bold green]Done![/bold green] Echo v2 saved.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge DPO adapter into SFT merged base")
    parser.add_argument(
        "--base-dir", type=Path, default=DEFAULT_BASE, help="SFT merged base dir"
    )
    parser.add_argument(
        "--adapter-dir", type=Path, default=DEFAULT_ADAPTER, help="DPO adapter dir"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output dir for Echo v2"
    )
    args = parser.parse_args()
    merge(args)


if __name__ == "__main__":
    main()
