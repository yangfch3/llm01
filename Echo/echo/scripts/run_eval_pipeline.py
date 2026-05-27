"""串联评估流程：eval_loss(所有 ckpt) → generate → eval。

自动扫描 checkpoints/sft/ 下所有 checkpoint，跑 eval_loss 选出最优，
然后用最优 adapter 启动 generate 或 eval。

用法：
    # 全自动：eval_loss → eval（跳过交互式 generate）
    uv run python scripts/run_eval_pipeline.py

    # eval_loss → generate（交互对话）
    uv run python scripts/run_eval_pipeline.py --generate

    # 指定 checkpoint 目录
    uv run python scripts/run_eval_pipeline.py --ckpt-dir checkpoints/sft
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
from rich.table import Table
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))

from device import get_device
from echo.data import load_sft_data
from echo.utils import load_config

console = Console()

DEFAULT_CONFIG = Path("configs/sft-8g.yaml")
DEFAULT_CKPT_DIR = Path("checkpoints/sft")
DEFAULT_VAL_FILE = Path("data/sft/val.jsonl")


def find_checkpoints(ckpt_dir: Path) -> list[Path]:
    """扫描所有 adapter 目录（含 final 和 checkpoint-*）。"""
    candidates = []
    for p in sorted(ckpt_dir.iterdir()):
        if p.is_dir() and (p / "adapter_config.json").exists():
            candidates.append(p)
    return candidates


def eval_single(model_id: str, adapter_dir: Path, val_file: Path, cfg: dict) -> dict:
    """对单个 adapter 计算 val loss。返回 {name, avg_loss, ppl}。"""
    max_seq_length = cfg["training"].get("max_seq_length", 2048)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载底座 (QLoRA 4bit)
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
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    # 加载 val 数据
    dataset = load_sft_data(val_file)
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0

    for example in dataset:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        enc = tokenizer(text, truncation=True, max_length=max_seq_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        labels = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)

        n_tokens = input_ids.shape[1]
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    # 释放显存
    del model, base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {"name": adapter_dir.name, "path": adapter_dir, "avg_loss": avg_loss, "ppl": ppl}


def run_pipeline(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    model_id = cfg["model"]["model_id"]
    ckpt_dir = Path(args.ckpt_dir)
    val_file = Path(args.val_file)

    if not val_file.exists():
        console.print(f"[bold red]Error:[/bold red] {val_file} not found")
        raise SystemExit(1)

    # 1. 扫描 checkpoints
    checkpoints = find_checkpoints(ckpt_dir)
    if not checkpoints:
        console.print(f"[bold red]Error:[/bold red] No checkpoints found in {ckpt_dir}")
        raise SystemExit(1)

    console.print(f"[bold]Found {len(checkpoints)} checkpoints:[/bold]")
    for cp in checkpoints:
        console.print(f"  - {cp.name}")

    # 2. 逐个跑 eval_loss
    console.print(f"\n[bold]Running eval_loss on val set ({val_file})...[/bold]\n")
    results = []
    for cp in checkpoints:
        console.print(f"  Evaluating [cyan]{cp.name}[/cyan]...", end=" ")
        r = eval_single(model_id, cp, val_file, cfg)
        console.print(f"loss={r['avg_loss']:.4f}  ppl={r['ppl']:.2f}")
        results.append(r)

    # 3. 排序展示
    results.sort(key=lambda x: x["avg_loss"])
    best = results[0]

    console.print()
    table = Table(title="Eval Loss Results (sorted)")
    table.add_column("Checkpoint", style="cyan")
    table.add_column("Avg Loss", justify="right")
    table.add_column("PPL", justify="right")
    table.add_column("", justify="center")
    for r in results:
        marker = "★ best" if r["name"] == best["name"] else ""
        table.add_row(r["name"], f"{r['avg_loss']:.4f}", f"{r['ppl']:.2f}", marker)
    console.print(table)

    console.print(f"\n[bold green]Best checkpoint:[/bold green] {best['name']} (loss={best['avg_loss']:.4f})")

    # 4. 启动 generate 或 eval
    best_adapter = str(best["path"])

    if args.generate:
        console.print(f"\n[bold]Launching generate with {best['name']}...[/bold]\n")
        import subprocess
        subprocess.run(
            [sys.executable, "scripts/generate.py", "--adapter-dir", best_adapter, "--config", str(args.config)],
            check=False,
        )
    else:
        console.print(f"\n[bold]Launching eval with {best['name']}...[/bold]\n")
        import subprocess
        cmd = [sys.executable, "scripts/eval.py", "--adapter-dir", best_adapter, "--config", str(args.config)]
        if args.skip_questions:
            cmd.append("--skip-questions")
        subprocess.run(cmd, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval pipeline: eval_loss → generate/eval")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--val-file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--generate", action="store_true", help="Launch generate instead of eval")
    parser.add_argument("--skip-questions", action="store_true", help="Pass --skip-questions to eval.py")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
