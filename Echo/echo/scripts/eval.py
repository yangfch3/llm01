"""echo 评测脚本 — 自动化对话质量评估。

评测项（对应 SPEC §8）：
  1. Val PPL（复用 eval_loss.py 逻辑）
  2. 常识/算术题（eval/questions.jsonl）
  3. 对话样例生成（人工查阅用）

默认评 base 路线 final adapter；instruct 路线需显式传 --config / --adapter-dir / --val-file。

用法：
    # base 路线（默认）
    uv run python scripts/eval.py

    # base 路线指定 ckpt
    uv run python scripts/eval.py --adapter-dir checkpoints/sft-base/checkpoint-N

    # instruct 路线对照
    uv run python scripts/eval.py --config configs/sft-8g-instruct.yaml \\
        --adapter-dir checkpoints/sft/final --val-file data/sft/val.jsonl
"""

from __future__ import annotations

# datasets 必须在 torch 之前（Windows pyarrow DLL 冲突）
import datasets  # noqa: F401, I001

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))

from device import get_device
from echo.data import load_sft_data
from echo.utils import load_config, load_inference_tokenizer

console = Console()

DEFAULT_CONFIG = Path("configs/sft-8g-base.yaml")
DEFAULT_ADAPTER = Path("checkpoints/sft-base/final")
DEFAULT_VAL_FILE = Path("data/sft/val_aug.jsonl")
DEFAULT_QUESTIONS = Path("eval/questions.jsonl")
EVAL_OUTPUT = Path("eval/results.json")


def load_model_and_tokenizer(args: argparse.Namespace):
    """加载模型 + tokenizer。"""
    cfg = load_config(args.config)
    model_id = cfg["model"]["model_id"]

    tokenizer = load_inference_tokenizer(
        adapter_dir=Path(args.adapter_dir) if not args.merged_dir else None,
        merged_dir=Path(args.merged_dir) if args.merged_dir else None,
        base_model_id=model_id,
    )

    # 显式声明 sdpa attention：O(seq) 显存，与 eager 数学等价，不影响 PPL/生成
    attn_impl = "sdpa"

    if args.merged_dir:
        model = AutoModelForCausalLM.from_pretrained(
            str(Path(args.merged_dir)),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
    else:
        adapter_dir = Path(args.adapter_dir)
        quant_cfg = cfg.get("quantization")
        if quant_cfg and quant_cfg.get("enabled", True):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
                bnb_4bit_compute_dtype=getattr(
                    torch, quant_cfg.get("compute_dtype", "bfloat16")
                ),
                bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )
        else:
            device = get_device()
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            ).to(device)
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    model.eval()
    return model, tokenizer, cfg


def eval_ppl(model, tokenizer, val_file: Path, max_seq_length: int) -> dict:
    """计算 val set perplexity。"""
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

        # 防止 reserved 内存随长样本累积
        del outputs, input_ids, labels, enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return {"avg_loss": round(avg_loss, 4), "perplexity": round(ppl, 2), "valid_tokens": total_tokens}


def _get_stop_token_ids(tokenizer) -> list[int]:
    """Qwen2.5 停止 token: <|im_end|> + <|endoftext|>。

    convert_tokens_to_ids 对未注册 token 可能返回 None / unk_id，过滤掉；
    全空时退回 tokenizer.eos_token_id。
    """
    ids = [
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    ]
    ids = [tid for tid in ids if tid is not None and tid >= 0]
    return ids or [tokenizer.eos_token_id]


def eval_questions(model, tokenizer, questions_file: Path, max_new_tokens: int = 256) -> dict:
    """对 eval/questions.jsonl 生成回答并判断正确率。

    questions.jsonl 格式：
    {"question": "...", "expected": "...", "type": "choice|short_answer"}
    """
    if not questions_file.exists():
        console.print(f"[yellow]Skip:[/yellow] {questions_file} not found")
        return {"skipped": True, "reason": "file not found"}

    device = next(model.parameters()).device
    questions = []
    with open(questions_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if not questions:
        return {"skipped": True, "reason": "empty file"}

    correct = 0
    results = []

    for q in questions:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q["question"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                eos_token_id=_get_stop_token_ids(tokenizer),
            )

        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        expected = q.get("expected", "")
        is_correct = expected.lower().strip() in generated.lower() if expected else None

        if is_correct:
            correct += 1

        results.append({
            "question": q["question"],
            "expected": expected,
            "generated": generated.strip()[:200],
            "correct": is_correct,
        })

    total = len(questions)
    accuracy = correct / total if total > 0 else 0
    return {"total": total, "correct": correct, "accuracy": round(accuracy, 4), "details": results}


def eval_dialogue_samples(model, tokenizer, max_new_tokens: int = 256) -> list[dict]:
    """生成固定对话样例供人工检查。"""
    device = next(model.parameters()).device
    prompts = [
        "你好，请介绍一下你自己。",
        "What is the capital of France?",
        "请用简单的语言解释什么是机器学习。",
        "1+1等于几？",
        "写一首关于春天的短诗。",
    ]

    samples = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=_get_stop_token_ids(tokenizer),
            )

        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        samples.append({"user": prompt, "echo": generated.strip()})

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Echo evaluation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--merged-dir", type=Path, default=None)
    parser.add_argument("--val-file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=EVAL_OUTPUT)
    parser.add_argument("--skip-ppl", action="store_true", help="Skip PPL evaluation")
    parser.add_argument("--skip-questions", action="store_true", help="Skip question eval")
    args = parser.parse_args()

    model, tokenizer, cfg = load_model_and_tokenizer(args)
    max_seq_length = cfg["training"].get("max_seq_length", 2048)

    # 复位峰值计数，便于观察各阶段显存
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    all_results = {}

    # 1. PPL
    if not args.skip_ppl and Path(args.val_file).exists():
        console.print("[bold]Evaluating PPL...[/bold]")
        ppl_result = eval_ppl(model, tokenizer, Path(args.val_file), max_seq_length)
        all_results["ppl"] = ppl_result
        console.print(f"  Loss: {ppl_result['avg_loss']}  PPL: {ppl_result['perplexity']}")
    else:
        console.print("[yellow]PPL evaluation skipped.[/yellow]")

    # 2. Questions
    if not args.skip_questions:
        console.print("[bold]Evaluating questions...[/bold]")
        q_result = eval_questions(model, tokenizer, Path(args.questions))
        all_results["questions"] = q_result
        if not q_result.get("skipped"):
            console.print(f"  Accuracy: {q_result['correct']}/{q_result['total']} ({q_result['accuracy']:.1%})")
    else:
        console.print("[yellow]Question evaluation skipped.[/yellow]")

    # 3. Dialogue samples
    console.print("[bold]Generating dialogue samples...[/bold]")
    samples = eval_dialogue_samples(model, tokenizer)
    all_results["dialogue_samples"] = samples

    table = Table(title="Dialogue Samples")
    table.add_column("User", style="cyan", max_width=30)
    table.add_column("Echo", style="green", max_width=60)
    for s in samples:
        table.add_row(s["user"], s["echo"][:100] + ("..." if len(s["echo"]) > 100 else ""))
    console.print(table)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    console.print(f"\n[bold green]Results saved to {output_path}[/bold green]")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        console.print(
            f"[dim]GPU peak allocated: {peak_gb:.2f} GB | reserved: {reserved_gb:.2f} GB[/dim]"
        )


if __name__ == "__main__":
    main()
