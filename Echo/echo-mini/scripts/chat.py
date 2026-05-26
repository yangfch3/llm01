"""echo-mini 推理 CLI — 交互式对话 / 续写。

支持两种模式：
- chat: 对话模式（使用 SFT chat template）
- complete: 续写模式（纯文本续写）

用法：
    cd Echo/echo-mini
    uv run python scripts/chat.py --ckpt checkpoints/sft/step_003000.pt --mode chat
    uv run python scripts/chat.py --ckpt checkpoints/pretrain/step_010000.pt --mode complete
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将 src/ 和 shared/ 加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

from tokenizers import Tokenizer  # noqa: I001

from device import get_device
from echo_mini.inference import generate, load_model


def format_chat_prompt(user_input: str, tokenizer: Tokenizer) -> list[int]:
    """构造 chat 模式的 prompt token ids。

    格式: <bos><|user|>{input}\n<|assistant|>
    模型从 <|assistant|> 之后开始生成。
    """
    bos_id = tokenizer.token_to_id("<bos>")
    user_token_id = tokenizer.token_to_id("<|user|>")
    assistant_token_id = tokenizer.token_to_id("<|assistant|>")
    content_ids = tokenizer.encode(user_input + "\n").ids
    return [bos_id, user_token_id] + content_ids + [assistant_token_id]


def format_complete_prompt(text: str, tokenizer: Tokenizer) -> list[int]:
    """构造续写模式的 prompt token ids。"""
    bos_id = tokenizer.token_to_id("<bos>")
    encoded = tokenizer.encode(text)
    return [bos_id] + encoded.ids


def main() -> None:
    parser = argparse.ArgumentParser(description="echo-mini inference CLI")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path")
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"), help="Tokenizer path"
    )
    parser.add_argument("--mode", choices=["chat", "complete"], default="chat", help="Inference mode")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model = load_model(args.ckpt, device)

    if args.mode == "chat":
        print("\n=== echo-mini Chat ===")
        print("Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not user_input or user_input.lower() in ("quit", "exit"):
                print("Bye!")
                break

            prompt_ids = format_chat_prompt(user_input, tokenizer)
            generated_ids = generate(
                model, tokenizer, prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
            response = tokenizer.decode(generated_ids)
            print(f"Echo: {response.strip()}\n")

    else:  # complete
        print("\n=== echo-mini Complete ===")
        print("Enter text to continue. Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                text = input("Prompt: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not text or text.lower() in ("quit", "exit"):
                print("Bye!")
                break

            prompt_ids = format_complete_prompt(text, tokenizer)
            generated_ids = generate(
                model, tokenizer, prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
            continuation = tokenizer.decode(generated_ids)
            print(f"→ {text}{continuation}\n")


if __name__ == "__main__":
    main()
