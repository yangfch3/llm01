"""echo SFT 训练入口 (QLoRA + trl SFTTrainer)。

用法：
    # base 主线（默认配置，项目锚点）
    uv run python scripts/sft.py --config configs/sft-8g-base.yaml

    # instruct 对照
    uv run python scripts/sft.py --config configs/sft-8g-instruct.yaml

    # 代码验证（任选 base / instruct 的 tiny 配置）
    uv run python scripts/sft.py --config configs/sft-tiny-base.yaml
    uv run python scripts/sft.py --config configs/sft-tiny-instruct.yaml
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))

from device import get_device
from echo.data import load_sft_data
from echo.utils import CHATML_INFER_TEMPLATE, CHATML_TRAIN_TEMPLATE, load_config

console = Console()

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
    kwargs = dict(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        # modules_to_save 让指定模块全量参与训练（不走 LoRA 低秩近似）。
        # base 路线下用于训练 embed_tokens / lm_head，让 <|im_end|> 等
        # special token 的 embedding 真正被学到。
        modules_to_save=lora_cfg.get("modules_to_save"),
        bias="none",
    )
    # ensure_weight_tying：Qwen2.5-1.5B tie_word_embeddings=True，PEFT 默认会
    # 拆出独立的 embed 和 lm_head 副本破坏 tie。开启此参数后 PEFT 会让
    # lm_head 跟随 embed_tokens 同步更新，保持 tie 关系不破坏。
    # 仅 PEFT >= 0.12 支持；config 里没显式关闭就默认开。
    if lora_cfg.get("ensure_weight_tying", True):
        kwargs["ensure_weight_tying"] = True
    return LoraConfig(**kwargs)


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

    # Base 模型特殊处理：base 的 eos_token 是 <|endoftext|>，但 ChatML 对话
    # 终止符是 <|im_end|>。若不修正，trl 内部 collator/packing 会用错 EOS，
    # 导致训练信号稀释，模型学不会输出 <|im_end|>。
    #
    # 注意：上方 pad_token 已经在 eos 修改前设为 <|endoftext|>，这是有意为之——
    # base 模式下 pad 与 eos 故意解耦：pad=<|endoftext|>、eos=<|im_end|>，
    # 避免 padding 位置被误识别为对话终止符。
    if model_cfg.get("is_base_model", False):
        im_end = "<|im_end|>"
        im_end_id = tokenizer.convert_tokens_to_ids(im_end)
        if im_end_id is None or im_end_id < 0:
            raise RuntimeError(
                f"is_base_model=true 但 tokenizer 找不到 {im_end}，请检查底座是否为 Qwen2.5 系列"
            )
        tokenizer.eos_token = im_end
        console.print(
            f"[yellow]Base 模式：tokenizer.eos_token → {im_end} (id={im_end_id}); "
            f"pad_token 保留 {tokenizer.pad_token} 与 eos 解耦[/yellow]"
        )

    # Patch chat_template: 在 assistant 内容处加 {% generation %} 标记，
    # 让 trl 的 assistant_only_loss 能正确识别哪些 token 算 loss。
    tokenizer.chat_template = CHATML_TRAIN_TEMPLATE

    # Quantization
    bnb_config = build_bnb_config(cfg)

    # Model
    console.print(f"[bold]Loading model:[/bold] {model_id}")
    if bnb_config is not None:
        # 量化加载：必须走 device_map="auto" 让 accelerate 分配层
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # 无量化：根据配置决定 dtype，手动放到目标设备
        if train_cfg.get("bf16"):
            load_dtype = torch.bfloat16
        elif train_cfg.get("fp16"):
            load_dtype = torch.float16
        else:
            load_dtype = torch.float32
        device = get_device()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=load_dtype,
            trust_remote_code=True,
        ).to(device)

    # LoRA
    lora_config = build_lora_config(cfg)
    model = get_peft_model(model, lora_config)

    # gradient_checkpointing + modules_to_save (或 PEFT) 需要显式打开输入梯度，
    # 否则反向传播时报 "element 0 of tensors does not require grad"。
    # 4bit 量化路径下 prepare_model_for_kbit_training 已经处理过；非量化或安全起见统一调用。
    if train_cfg.get("gradient_checkpointing", False) and hasattr(
        model, "enable_input_require_grads"
    ):
        model.enable_input_require_grads()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    console.print(
        f"[bold]Trainable:[/bold] {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # Tie weights 检查：Qwen2.5-1.5B 默认 tie_word_embeddings=True，embed_tokens
    # 与 lm_head 共享同一权重张量。若 modules_to_save 同时列出这两个模块，PEFT
    # 会包出独立副本破坏 tie，导致 adapter 体积翻倍且训练时两个权重异步漂移。
    # 此处主动检测并打印实际 tie 状态，便于发现配置错误。
    try:
        from peft.utils.other import ModulesToSaveWrapper

        peft_base = model.base_model.model  # PEFT 包装后实际模型
        embed_w = peft_base.get_input_embeddings().weight
        head_w = peft_base.get_output_embeddings().weight
        is_tied = embed_w.data_ptr() == head_w.data_ptr()
        configured = (cfg.get("lora") or {}).get("modules_to_save") or []

        # PEFT 实际包装的 ModulesToSaveWrapper 列表（含 ensure_weight_tying 自动追加的 tied 模块）
        wrapped: list[str] = [
            name for name, mod in model.named_modules()
            if isinstance(mod, ModulesToSaveWrapper)
        ]
        # 简化显示：只保留尾部模块名而非全路径
        wrapped_short = [name.rsplit(".", 1)[-1] for name in wrapped]

        console.print(
            f"[bold]Tie weights:[/bold] embed/lm_head tied={is_tied}; "
            f"yaml.modules_to_save={configured}; PEFT 实际包装={wrapped_short}"
        )
        if (
            not is_tied
            and "embed_tokens" in configured
            and "lm_head" not in configured
        ):
            console.print(
                "[yellow]警告：embed_tokens 在 modules_to_save 但 tie 已断开，"
                "lm_head 不会跟随训练[/yellow]"
            )
    except (AttributeError, RuntimeError, ImportError) as e:
        console.print(f"[dim]Tie 检查跳过: {e}[/dim]")

    # Data
    data_path = Path(data_cfg["train_file"])
    if not data_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] 数据文件不存在: {data_path}\n"
            "  请先运行: uv run python scripts/prepare_data.py"
        )
        raise SystemExit(1)
    console.print(f"[bold]Loading data:[/bold] {data_path}")
    dataset = load_sft_data(data_path)
    console.print(f"  train samples: {len(dataset)}")

    eval_dataset = None
    if data_cfg.get("val_file"):
        val_path = Path(data_cfg["val_file"])
        eval_dataset = load_sft_data(val_path)
        console.print(f"  val samples: {len(eval_dataset)}")

    # 判断是否实际启用 eval
    eval_enabled = train_cfg.get("eval_strategy", "epoch" if eval_dataset else "no") != "no"
    if not eval_enabled:
        eval_dataset = None

    # Training arguments (trl 1.4+ 使用 SFTConfig 替代 TrainingArguments)
    output_dir = train_cfg.get("output_dir", "checkpoints/sft")
    training_args = SFTConfig(
        output_dir=output_dir,
        max_length=max_seq_length,
        assistant_only_loss=True,
        num_train_epochs=train_cfg.get("num_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("grad_accum_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy="no" if not eval_enabled else train_cfg.get("eval_strategy", "epoch"),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        # save_only_model: True 只存 adapter，不存 optimizer/scheduler/rng（无法 resume，但
        # 单 ckpt 体积从 ~3GB 降到 ~600MB，磁盘紧张时建议开启；
        # False 保留完整 ckpt，可 resume_from_checkpoint。
        save_only_model=train_cfg.get("save_only_model", False),
        load_best_model_at_end=eval_enabled and eval_dataset is not None,
        metric_for_best_model="eval_loss" if (eval_enabled and eval_dataset) else None,
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
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    console.print("[bold green]Starting SFT training...[/bold green]")
    trainer.train(resume_from_checkpoint=args.resume)

    # Save final adapter
    # 训练模板带 trl 专用的 {% generation %} 标记，不适合推理。
    # 收尾时切换为推理版 ChatML 模板，下游 generate / eval 直接 from_pretrained
    # 加载 adapter tokenizer 即可拿到正确 chat_template。
    tokenizer.chat_template = CHATML_INFER_TEMPLATE
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
