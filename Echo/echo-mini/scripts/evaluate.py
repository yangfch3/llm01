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

# 将 src/ 加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokenizers import Tokenizer

from echo_mini.config import EchoMiniConfig
from echo_mini.data import SFTDataset
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


def load_model(ckpt_path: Path, device: torch.device) -> EchoMini:
    """加载模型。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = EchoMiniConfig()
    model = EchoMini(cfg)

    state_dict = ckpt["model"]
    pretrain_vocab = state_dict["tok_emb.weight"].shape[0]
    if pretrain_vocab < cfg.vocab_size:
        old_emb = state_dict["tok_emb.weight"]
        new_emb = torch.randn(cfg.vocab_size - pretrain_vocab, old_emb.shape[1]) * 0.02
        state_dict["tok_emb.weight"] = torch.cat([old_emb, new_emb], dim=0)

    state_dict.pop("lm_head.weight", None)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


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


@torch.inference_mode()
def generate_response(
    model: EchoMini,
    tokenizer: Tokenizer,
    user_input: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
) -> str:
    """生成单条回答。"""
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    user_token_id = tokenizer.token_to_id("<|user|>")
    assistant_token_id = tokenizer.token_to_id("<|assistant|>")

    content_ids = tokenizer.encode(user_input + "\n").ids
    prompt_ids = [bos_id, user_token_id] + content_ids + [assistant_token_id]

    model.reset_cache()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, _ = model(input_ids, use_cache=True, start_pos=0)
    next_logits = logits[:, -1, :]

    generated: list[int] = []
    pos = len(prompt_ids)

    for _ in range(max_new_tokens):
        if temperature <= 0:
            next_token = next_logits.argmax(dim=-1).item()
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_indices[0, torch.multinomial(sorted_probs[0], 1)].item()

        if next_token == eos_id:
            break
        generated.append(next_token)

        input_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits, _ = model(input_ids, use_cache=True, start_pos=pos)
        next_logits = logits[:, -1, :]
        pos += 1
        if pos >= model.cfg.max_seq_len:
            break

    return tokenizer.decode(generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="echo-mini evaluation")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"))
    parser.add_argument("--sft_data", type=Path, default=Path("data/sft/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("logs/eval_results.json"))
    parser.add_argument("--ppl_samples", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        response = generate_response(
            model, tokenizer, prompt,
            temperature=args.temperature, device=device,
        )
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
