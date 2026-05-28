"""echo 推理 CLI — 加载 adapter 或 merged 模型进行交互式对话。

默认加载 base 路线 final adapter；instruct 路线需显式传参。

用法：
    # base 路线（默认，加载 checkpoints/sft-base/final）
    uv run python scripts/generate.py

    # base 路线指定 ckpt
    uv run python scripts/generate.py --adapter-dir checkpoints/sft-base/checkpoint-N

    # instruct 路线对照
    uv run python scripts/generate.py --config configs/sft-8g-instruct.yaml \\
        --adapter-dir checkpoints/sft/final

    # 加载已合并模型
    uv run python scripts/generate.py --merged-dir checkpoints/merged-base

    # 调整生成参数
    uv run python scripts/generate.py --temperature 0.7 --top-k 50 --top-p 0.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))

from device import get_device
from echo.utils import load_config, load_inference_tokenizer

console = Console()

DEFAULT_CONFIG = Path("configs/sft-8g-base.yaml")
DEFAULT_ADAPTER = Path("checkpoints/sft-base/final")


def load_model_and_tokenizer(args: argparse.Namespace):
    """根据参数加载模型 + tokenizer。"""
    cfg = load_config(args.config)
    model_id = cfg["model"]["model_id"]

    # Tokenizer：merged_dir > adapter_dir > base_model_id 兜底，
    # chat_template 缺失自动注入推理版 ChatML（见 echo.utils）。
    tokenizer = load_inference_tokenizer(
        adapter_dir=Path(args.adapter_dir) if not args.merged_dir else None,
        merged_dir=Path(args.merged_dir) if args.merged_dir else None,
        base_model_id=model_id,
    )

    if args.merged_dir:
        # 加载已合并的完整模型
        merged_dir = Path(args.merged_dir)
        console.print(f"[bold]Loading merged model:[/bold] {merged_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            str(merged_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # 加载底座 + adapter（QLoRA 4bit 节省显存）
        adapter_dir = Path(args.adapter_dir)
        console.print(f"[bold]Base model:[/bold] {model_id}")
        console.print(f"[bold]Adapter:[/bold] {adapter_dir}")

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
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            device = get_device()
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(device)

        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()
    return model, tokenizer


def chat_loop(model, tokenizer, args: argparse.Namespace) -> None:
    """交互式多轮对话循环。"""
    device = next(model.parameters()).device
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    console.print("\n[bold green]Echo Chat[/bold green] (输入 /quit 退出, /clear 清空历史)\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "/quit":
            break
        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            console.print("[dim]History cleared.[/dim]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        # 构造输入
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # 生成
        # Qwen2.5 停止 token: <|im_end|> + <|endoftext|>
        # convert_tokens_to_ids 对未注册 token 返回 unk_id 或 None，过滤掉
        stop_token_ids = [
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        stop_token_ids = [
            tid for tid in stop_token_ids if tid is not None and tid >= 0
        ]
        if not stop_token_ids:
            stop_token_ids = [tokenizer.eos_token_id]
        print("Echo: ", end="", flush=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.temperature > 0,
                eos_token_id=stop_token_ids,
                pad_token_id=tokenizer.pad_token_id,  # 显式传，避免 transformers fallback 警告
                streamer=streamer,
            )

        # 提取 assistant 回复
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        assistant_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": assistant_text})
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Echo inference CLI")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config")
    parser.add_argument(
        "--adapter-dir", type=Path, default=DEFAULT_ADAPTER, help="Adapter dir"
    )
    parser.add_argument("--merged-dir", type=Path, default=None, help="Merged model dir")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args)
    chat_loop(model, tokenizer, args)


if __name__ == "__main__":
    main()
