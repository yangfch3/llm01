"""训练工具函数：lr scheduler, checkpoint 管理, 日志。"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import yaml
from rich.console import Console

console = Console()


# ============================================================
# 配置加载
# ============================================================


def load_config(path: Path) -> dict:
    """加载 YAML 配置文件。"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Learning Rate Scheduler
# ============================================================


def get_lr(step: int, total_steps: int, peak_lr: float, warmup_steps: int, min_lr: float = 0.0) -> float:
    """Cosine decay with linear warmup。"""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ============================================================
# Checkpoint
# ============================================================


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    ckpt_dir: Path,
) -> Path:
    """保存训练 checkpoint。返回保存路径。"""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    console.print(f"[green]Checkpoint saved:[/green] {path}")
    return path


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """加载 checkpoint，返回恢复的 step。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    console.print(f"[cyan]Resumed from:[/cyan] {path} (step {ckpt['step']})")
    return ckpt["step"]


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """找到最新的 checkpoint 文件。"""
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    return ckpts[-1] if ckpts else None


# ============================================================
# Timer
# ============================================================


class Timer:
    """简单计时器。"""

    def __init__(self):
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def reset(self) -> None:
        self._start = time.perf_counter()
