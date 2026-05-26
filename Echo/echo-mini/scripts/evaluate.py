"""echo-mini 评测脚本 — PPL 计算 + 对话样例生成。

用法：
    cd Echo/echo-mini
    uv run python scripts/evaluate.py --ckpt checkpoints/sft/step_003000.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# 将 src/ 和 shared/ 加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

from tokenizers import Tokenizer  # noqa: I001

from device import get_device
from echo_mini.data import SFTDataset
from echo_mini.inference import generate, load_model
from echo_mini.model import EchoMini

# 评测用对话样例（中英混合）
EVAL_PROMPTS = [
    "你好，请介绍一下你自己。",
    "What is machine learning?",
    "请解释什么是Transformer。",
    "How does attention mechanism work?",
    "写一首关于春天的短诗。",
    "What is the capital of France?",
    "1+1等于几？",
    "Tell me a joke.",
]


@torch.inference_mode()
def compute_ppl(
    model: EchoMini,
    tokenizer: Tokenizer,
    jsonl_path: Path,
    max_samples: int = 200,
    max_seq_len: int = 1024,
    device: torch.device = torch.device("cpu"),
) -> float:
    """在 SFT 验证集上计算 perplexity。"""
    ds = SFTDataset(jsonl_path, tokenizer, max_seq_len)
    total_loss = 0.0
    total_tokens = 0
    n = min(max_samples, len(ds))

    for i in range(n):
        sample = ds[i]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)

        logits, _ = model(input_ids[:, :-1])
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_tokens = (shift_labels != -100).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def format_chat_prompt(user_input: str, tokenizer: Tokenizer) -> list[int]:
    """构造 chat prompt token ids。"""
    bos_id = tokenizer.token_to_id("<bos>")
    user_token_id = tokenizer.token_to_id("<|user|>")
    assistant_token_id = tokenizer.token_to_id("<|assistant|>")
    content_ids = tokenizer.encode(user_input + "\n").ids
    return [bos_id, user_token_id] + content_ids + [assistant_token_id]


def main() -> None:
    parser = argparse.ArgumentParser(description="echo-mini evaluation")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"))
    parser.add_argument("--sft_data", type=Path, default=Path("data/sft/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("logs/eval_results.json"))
    parser.add_argument("--ppl_samples", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model = load_model(args.ckpt, device)
    print(f"Loaded: {args.ckpt}")

    # 1. PPL
    print("\n--- Perplexity ---")
    ppl = compute_ppl(model, tokenizer, args.sft_data, args.ppl_samples, device=device)
    print(f"PPL ({args.ppl_samples} samples): {ppl:.2f}")

    # 2. 对话样例
    print("\n--- Dialogue Samples ---")
    samples = []
    for prompt in EVAL_PROMPTS:
        prompt_ids = format_chat_prompt(prompt, tokenizer)
        generated_ids = generate(
            model, tokenizer, prompt_ids,
            temperature=args.temperature, device=device,
        )
        response = tokenizer.decode(generated_ids)
        samples.append({"user": prompt, "assistant": response})
        print(f"User: {prompt}")
        print(f"Echo: {response}")
        print()

    # 保存结果
    results = {
        "checkpoint": str(args.ckpt),
        "ppl": round(ppl, 2),
        "dialogue_samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
