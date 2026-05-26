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

import torch

# 将 src/ 加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokenizers import Tokenizer

from echo_mini.config import EchoMiniConfig
from echo_mini.model import EchoMini


def load_model(ckpt_path: Path, device: torch.device) -> EchoMini:
    """加载模型权重。兼容 vocab_size 不匹配的旧 checkpoint。

    模型使用 weight tying，只需 resize tok_emb.weight 并跳过 lm_head.weight。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = EchoMiniConfig()
    model = EchoMini(cfg)

    state_dict = ckpt["model"]
    pretrain_vocab = state_dict["tok_emb.weight"].shape[0]

    if pretrain_vocab < cfg.vocab_size:
        old_emb = state_dict["tok_emb.weight"]
        new_emb = torch.randn(cfg.vocab_size - pretrain_vocab, old_emb.shape[1]) * 0.02
        state_dict["tok_emb.weight"] = torch.cat([old_emb, new_emb], dim=0)

    # Weight tying: 删除 lm_head.weight，让 tying 自动生效
    state_dict.pop("lm_head.weight", None)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path} (step {ckpt['step']})")
    return model


@torch.inference_mode()
def generate(
    model: EchoMini,
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
) -> list[int]:
    """带 KV cache 的自回归生成。"""
    eos_id = tokenizer.token_to_id("<eos>")
    model.reset_cache()

    # Prefill: 处理整个 prompt
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, _ = model(input_ids, use_cache=True, start_pos=0)
    # 取最后一个 token 的 logits
    next_logits = logits[:, -1, :]

    generated: list[int] = []
    pos = len(prompt_ids)

    for _ in range(max_new_tokens):
        # Sampling
        if temperature <= 0:
            next_token = next_logits.argmax(dim=-1).item()
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            # Top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_indices[0, torch.multinomial(sorted_probs[0], 1)].item()

        if next_token == eos_id:
            break

        generated.append(next_token)

        # Decode step: 单 token 输入
        input_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits, _ = model(input_ids, use_cache=True, start_pos=pos)
        next_logits = logits[:, -1, :]
        pos += 1

        if pos >= model.cfg.max_seq_len:
            break

    return generated


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


def get_device() -> torch.device:
    """统一设备选择。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
