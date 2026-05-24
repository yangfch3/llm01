"""echo-mini 预训练入口 (Accelerate 手写 training loop)。

用法：
    accelerate launch scripts/pretrain.py --config configs/pretrain-full.yaml
    accelerate launch scripts/pretrain.py --config configs/pretrain-tiny.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from rich.console import Console
from tqdm import tqdm

# 允许从 scripts/ 直接运行时找到 src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from echo_mini.config import EchoMiniConfig
from echo_mini.data import create_dataloader
from echo_mini.model import EchoMini
from echo_mini.utils import (
    Timer,
    find_latest_checkpoint,
    get_lr,
    load_checkpoint,
    load_config,
    save_checkpoint,
)

console = Console()


def build_model(cfg: dict) -> EchoMini:
    """从配置构建模型。"""
    model_cfg = EchoMiniConfig(**cfg.get("model", {}))
    model = EchoMini(model_cfg)
    param_count = sum(p.numel() for p in model.parameters())
    console.print(f"[bold]Model params:[/bold] {param_count:,} ({param_count/1e6:.1f}M)")
    return model


def train(args: argparse.Namespace) -> None:
    """主训练循环。"""
    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    # Accelerator
    accelerator = Accelerator(
        mixed_precision=train_cfg.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=train_cfg.get("grad_accum_steps", 1),
    )

    # Model
    model = build_model(cfg)

    # Data
    data_path = Path(cfg["data"]["train_bin"])
    seq_len = cfg["model"].get("max_seq_len", 1024)
    batch_size = train_cfg["batch_size"]
    dataloader = create_dataloader(
        data_path,
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["peak_lr"],
        betas=(train_cfg.get("beta1", 0.9), train_cfg.get("beta2", 0.95)),
        weight_decay=train_cfg.get("weight_decay", 0.1),
    )

    # Prepare with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Training params
    max_steps = train_cfg["max_steps"]
    warmup_steps = train_cfg.get("warmup_steps", int(max_steps * 0.02))
    peak_lr = train_cfg["peak_lr"]
    min_lr = train_cfg.get("min_lr", 0.0)
    log_interval = train_cfg.get("log_interval", 10)
    save_interval = train_cfg.get("save_interval", 1000)
    ckpt_dir = Path(train_cfg.get("ckpt_dir", "checkpoints/pretrain"))

    # Resume
    start_step = 0
    if args.resume:
        ckpt_path = find_latest_checkpoint(ckpt_dir)
        if ckpt_path is not None:
            unwrapped = accelerator.unwrap_model(model)
            start_step = load_checkpoint(ckpt_path, unwrapped, optimizer)

    # Training loop
    console.print(f"[bold green]Starting training:[/bold green] steps {start_step} → {max_steps}")
    console.print(f"  batch_size={batch_size}, grad_accum={train_cfg.get('grad_accum_steps', 1)}")
    console.print(f"  peak_lr={peak_lr}, warmup={warmup_steps}")

    model.train()
    timer = Timer()
    step = start_step
    total_loss = 0.0
    data_iter = iter(dataloader)

    pbar = tqdm(
        range(start_step, max_steps),
        desc="Pretrain",
        disable=not accelerator.is_main_process,
    )

    try:
        for step_idx in pbar:
            step = step_idx + 1

            # Get batch (循环 DataLoader)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # LR schedule
            lr = get_lr(step_idx, max_steps, peak_lr, warmup_steps, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward + backward
            with accelerator.accumulate(model):
                _, loss = model(batch["input_ids"], batch["targets"])
                accelerator.backward(loss)
                if train_cfg.get("max_grad_norm"):
                    accelerator.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            # Logging
            if step % log_interval == 0 and accelerator.is_main_process:
                avg_loss = total_loss / log_interval
                elapsed = timer.elapsed()
                tokens_per_sec = (log_interval * batch_size * seq_len) / elapsed
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{lr:.2e}",
                    tok_s=f"{tokens_per_sec:.0f}",
                )
                total_loss = 0.0
                timer.reset()

            # Save checkpoint
            if step % save_interval == 0 and accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint(unwrapped, optimizer, step, loss.item(), ckpt_dir)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow] Saving checkpoint...")

    # Final save
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_checkpoint(unwrapped, optimizer, step, loss.item(), ckpt_dir)
        console.print("[bold green]Training complete![/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="echo-mini pretrain")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
