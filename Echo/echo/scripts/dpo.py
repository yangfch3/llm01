"""echo DPO 训练入口 (QLoRA + trl DPOTrainer)。

起点：checkpoints/merged-base/（SFT 后已合并的 bf16 完整权重）
产出：checkpoints/dpo-base/checkpoint-N + final，新一份 LoRA adapter

ref_model 处理：trl 1.4 在 PEFT + ref_model=None 时自动通过 disable_adapter
计算参考 logprob，无需显式加载第二份模型，省一份显存。

用法：
    # Win 3060 12GB QLoRA 生产配置
    uv run python scripts/dpo.py --config configs/dpo-8g-base.yaml

    # 大显存 QLoRA
    uv run python scripts/dpo.py --config configs/dpo-full-base.yaml

    # Mac/CPU 代码验证
    uv run python scripts/dpo.py --config configs/dpo-tiny-base.yaml
"""

from __future__ import annotations

# NOTE: datasets (pyarrow) 必须在 torch 之前 import，否则 Windows 上
# pyarrow DLL 与 CUDA DLL 加载顺序冲突会导致 segfault (0xC0000005)。
import datasets  # noqa: F401, I001

import argparse
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer

# 允许从 scripts/ 直接运行时找到 src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))

from device import get_device
from echo.data import load_dpo_data
from echo.utils import CHATML_INFER_TEMPLATE, load_config

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
    """构建 LoRA 配置。

    DPO 阶段不再需要训练 embed_tokens（SFT 已学好 ChatML），
    用常规 LoRA 配置即可，rank 默认 64。
    """
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

    # 起点是已合并的 SFT 模型目录（不是 HF base id）
    model_path = Path(model_cfg["model_path"])
    if not model_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] 模型目录不存在: {model_path}\n"
            "  请先运行: uv run python scripts/merge.py"
        )
        raise SystemExit(1)

    max_length = train_cfg.get("max_length", 1024)

    # Tokenizer：从 merged-base 加载（SFT 阶段已写入推理版 chat_template）
    console.print(f"[bold]Loading tokenizer:[/bold] {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = CHATML_INFER_TEMPLATE

    # base 路线：eos 必须是 <|im_end|>，与 SFT 训练时一致
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id >= 0:
        tokenizer.eos_token = "<|im_end|>"
        console.print(
            f"[yellow]eos_token → <|im_end|> (id={im_end_id}); pad={tokenizer.pad_token}[/yellow]"
        )

    # Quantization
    bnb_config = build_bnb_config(cfg)

    # Model（SFT 后已 merge 的完整权重作为起点；trl 自动加 LoRA + 用 disable_adapter 当 ref）
    console.print(f"[bold]Loading model:[/bold] {model_path}")
    if bnb_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        if train_cfg.get("bf16"):
            load_dtype = torch.bfloat16
        elif train_cfg.get("fp16"):
            load_dtype = torch.float16
        else:
            load_dtype = torch.float32
        device = get_device()
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=load_dtype,
            trust_remote_code=True,
        ).to(device)

    # PEFT 配置：交给 DPOTrainer，trl 内部 get_peft_model + ref_model 走 disable_adapter
    lora_config = build_lora_config(cfg)

    # Data
    train_path = Path(data_cfg["train_file"])
    val_path = Path(data_cfg["val_file"]) if data_cfg.get("val_file") else None
    if not train_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] 数据文件不存在: {train_path}\n"
            "  请先运行: uv run python scripts/prepare_dpo_data.py"
        )
        raise SystemExit(1)
    console.print(f"[bold]Loading data:[/bold] {train_path}")
    train_dataset = load_dpo_data(train_path)
    console.print(f"  train samples: {len(train_dataset)}")

    eval_dataset = None
    if val_path and val_path.exists():
        eval_dataset = load_dpo_data(val_path)
        console.print(f"  val samples: {len(eval_dataset)}")

    eval_enabled = train_cfg.get("eval_strategy", "epoch" if eval_dataset else "no") != "no"
    if not eval_enabled:
        eval_dataset = None

    # DPOConfig（trl 1.4：max_length 单参数覆盖 prompt+completion，不再分 max_prompt_length）
    output_dir = train_cfg.get("output_dir", "checkpoints/dpo-base")
    dpo_args = DPOConfig(
        output_dir=output_dir,
        beta=train_cfg.get("beta", 0.1),
        loss_type=train_cfg.get("loss_type", "sigmoid"),
        max_length=max_length,
        num_train_epochs=train_cfg.get("num_epochs", 1),
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("grad_accum_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy="no" if not eval_enabled else train_cfg.get("eval_strategy", "epoch"),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 5),
        load_best_model_at_end=eval_enabled and eval_dataset is not None,
        metric_for_best_model="eval_loss" if (eval_enabled and eval_dataset) else None,
        max_steps=train_cfg.get("max_steps", -1),
        seed=train_cfg.get("seed", 42),
        report_to=train_cfg.get("report_to", "none"),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        dataloader_num_workers=train_cfg.get("num_workers", 0),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
    )

    # DPOTrainer（ref_model=None：PEFT 场景下 trl 用 disable_adapter 当 ref）
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Train
    console.print("[bold green]Starting DPO training...[/bold green]")
    trainer.train(resume_from_checkpoint=args.resume)

    # Save final adapter
    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    # 保存推理版 chat_template，下游 generate / eval 直接复用
    tokenizer.chat_template = CHATML_INFER_TEMPLATE
    tokenizer.save_pretrained(str(final_dir))
    console.print(f"[bold green]Done![/bold green] DPO adapter saved to {final_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="echo DPO (QLoRA)")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint dir (optional)"
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
