"""echo SFT 训练入口 (QLoRA + trl SFTTrainer)。

用法：
    uv run python scripts/sft.py --config configs/sft-full.yaml
    uv run python scripts/sft.py --config configs/sft-tiny.yaml
"""

from __future__ import annotations

# NOTE: datasets (pyarrow) 必须在 torch 之前 import，否则 Windows 上
# pyarrow DLL 与 CUDA DLL 加载顺序冲突会导致 segfault (0xC0000005)。
import datasets  # noqa: F401, I001

import argparse
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# 允许从 scripts/ 直接运行时找到 src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from echo.data import load_sft_data
from echo.utils import load_config

console = Console()


def build_bnb_config(cfg: dict) -> BitsAndBytesConfig | None:
    """构建 4bit 量化配置。tiny 配置不量化则返回 None。"""
    quant_cfg = cfg.get("quantization")
    if quant_cfg is None or not quant_cfg.get("enabled", True):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
    )


def build_lora_config(cfg: dict) -> LoraConfig:
    """构建 LoRA 配置。"""
    lora_cfg = cfg["lora"]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
    )


def train(args: argparse.Namespace) -> None:
    """主训练流程。"""
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    model_id = model_cfg["model_id"]
    max_seq_length = train_cfg.get("max_seq_length", 2048)

    # Tokenizer
    console.print(f"[bold]Loading tokenizer:[/bold] {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization
    bnb_config = build_bnb_config(cfg)

    # Model
    console.print(f"[bold]Loading model:[/bold] {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
        trust_remote_code=True,
    )

    # Prepare for kbit training (QLoRA)
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # LoRA
    lora_config = build_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    console.print(
        f"[bold]Trainable:[/bold] {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # Data
    data_path = Path(data_cfg["train_file"])
    console.print(f"[bold]Loading data:[/bold] {data_path}")
    dataset = load_sft_data(data_path)
    console.print(f"  samples: {len(dataset)}")

    # Training arguments (trl 1.4+ 使用 SFTConfig 替代 TrainingArguments)
    output_dir = train_cfg.get("output_dir", "checkpoints/sft")
    training_args = SFTConfig(
        output_dir=output_dir,
        max_length=max_seq_length,
        num_train_epochs=train_cfg.get("num_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("grad_accum_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03) if not train_cfg.get("warmup_steps") else 0,
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        max_steps=train_cfg.get("max_steps", -1),
        seed=train_cfg.get("seed", 42),
        report_to=train_cfg.get("report_to", "none"),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        dataloader_num_workers=train_cfg.get("num_workers", 0),
    )

    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    console.print("[bold green]Starting SFT training...[/bold green]")
    trainer.train(resume_from_checkpoint=args.resume)

    # Save final adapter
    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    console.print(f"[bold green]Done![/bold green] Adapter saved to {final_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="echo SFT (QLoRA)")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint dir (optional)"
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
