"""echo base 模型推理 CLI — 加载裸 base 模型，作为 SFT 后效果对照。

与 generate.py 的关键区别：
  - 仅加载 base 模型，不挂 adapter / merged
  - 不预设 system prompt
  - 默认 EOS 用 base 自带的 <|endoftext|>，不是 SFT 学到的 <|im_end|>
  - 提供两种交互模式：
      * raw    ：纯续写，prompt 进、续写出，看 base 原始倾向
      * chatml ：手拼 ChatML 包装 prompt，测 base 能否被提示出对话能力
                 （作为"如果不做 SFT，光靠 prompt 工程能走多远"的对照）

适用范围：
  - **base 路线主要场景**：底座是真 base 模型（如 Qwen2.5-1.5B），
    raw / chatml 两种模式都有对照价值
  - instruct 路线：底座本身已是 instruct（如 Qwen2.5-1.5B-Instruct），
    自带 chat_template + 已学停止，raw 模式无意义，chatml 模式接近 SFT 后
    → instruct 路线请直接拿 instruct 底座 vs SFT 后 instruct 比，不必走本脚本

用法：
    # 纯续写模式（默认，对应 base 路线对照）
    uv run python scripts/generate_base.py

    # 伪 ChatML 模式：手拼 system + user 包装，看 base 对 ChatML 的反应
    uv run python scripts/generate_base.py --mode chatml

    # 自定义 model_id（默认从 sft-8g-base.yaml 读）
    uv run python scripts/generate_base.py --model-id Qwen/Qwen2.5-1.5B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))

from device import get_device
from echo.utils import load_config

console = Console()

DEFAULT_CONFIG = Path("configs/sft-8g-base.yaml")

# 伪 ChatML 包装模板（与 SFT 训练时拼接形态一致，用于探测 base 的提示工程上限）
CHATML_WRAP = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def load_base_model(args: argparse.Namespace):
    """加载裸 base 模型 + tokenizer，复用 SFT 配置里的量化参数。"""
    cfg = load_config(args.config)
    model_id = args.model_id or cfg["model"]["model_id"]

    console.print(f"[bold]Base model:[/bold] {model_id}")
    console.print(f"[bold]Mode:[/bold] {args.mode}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    model.eval()
    return model, tokenizer


def build_prompt(user_input: str, mode: str) -> str:
    """根据 mode 构造送入 base 的文本。

    raw   ：原样送入，模型走预训练续写逻辑
    chatml：手拼 ChatML 包装，看 base 是否能被提示出对话格式
    """
    if mode == "raw":
        return user_input
    if mode == "chatml":
        return CHATML_WRAP.format(user=user_input)
    raise ValueError(f"unknown mode: {mode}")


def collect_stop_token_ids(tokenizer, mode: str) -> list[int]:
    """根据 mode 选择停止 token。

    raw   ：仅 base 自带 EOS（<|endoftext|>）
    chatml：额外加 <|im_end|>，让"如果 base 真按 ChatML 输出"也能停
    """
    stop_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)
    if mode == "chatml":
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and im_end >= 0 and im_end != tokenizer.eos_token_id:
            stop_ids.append(im_end)
    return stop_ids


def chat_loop(model, tokenizer, args: argparse.Namespace) -> None:
    """单轮交互循环 — base 不维护多轮历史，每次独立生成。"""
    device = next(model.parameters()).device
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    stop_token_ids = collect_stop_token_ids(tokenizer, args.mode)

    hint = (
        "续写模式：输入会被 base 当文章接着写"
        if args.mode == "raw"
        else "伪 ChatML 模式：输入会被包装成 user 消息送入"
    )
    console.print(
        f"\n[bold green]Base Chat[/bold green] ({hint}, /quit 退出)\n"
    )

    while True:
        try:
            user_input = input("Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "/quit":
            break

        prompt = build_prompt(user_input, args.mode)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        print("Base: ", end="", flush=True)
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.temperature > 0,
                eos_token_id=stop_token_ids if stop_token_ids else None,
                pad_token_id=tokenizer.pad_token_id,
                streamer=streamer,
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Echo base model inference CLI")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config")
    parser.add_argument(
        "--model-id", type=str, default=None, help="覆盖 config 里的 model_id"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["raw", "chatml"],
        default="raw",
        help="raw=纯续写, chatml=手拼 ChatML 包装",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()

    model, tokenizer = load_base_model(args)
    chat_loop(model, tokenizer, args)


if __name__ == "__main__":
    main()
