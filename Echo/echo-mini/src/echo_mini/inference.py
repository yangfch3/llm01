"""推理工具：模型加载、自回归生成。

供 chat.py / evaluate.py 等推理脚本复用。
注意：调用 load_model 时如不传 device 参数，需确保 Echo/shared/ 已在 sys.path 中。
"""

from __future__ import annotations

from pathlib import Path

import torch
from rich.console import Console
from tokenizers import Tokenizer

from .config import EchoMiniConfig
from .model import EchoMini

console = Console()


def load_model(ckpt_path: Path, device: torch.device | None = None) -> EchoMini:
    """加载模型权重。兼容 vocab_size 不匹配的旧 checkpoint。

    模型使用 weight tying，只需 resize tok_emb.weight 并跳过 lm_head.weight。

    Args:
        ckpt_path: checkpoint 文件路径
        device: 目标设备。None 时调用 shared/device.py::get_device()（需确保已在 sys.path）
    """
    if device is None:
        from device import get_device
        device = get_device()

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
    console.print(f"[cyan]Loaded checkpoint:[/cyan] {ckpt_path} (step {ckpt['step']})")
    return model


@torch.inference_mode()
def generate(
    model: EchoMini,
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: torch.device | None = None,
) -> list[int]:
    """带 KV cache 的自回归生成。

    Args:
        model: 已加载到目标 device 的模型（eval 模式）
        tokenizer: 分词器（用于获取 eos_id）
        prompt_ids: prompt 的 token id 列表
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度，<=0 时 greedy
        top_p: nucleus sampling 阈值
        device: 推理设备，None 时从 model 参数推断

    Returns:
        生成的 token id 列表（不含 prompt，不含 eos）
    """
    if device is None:
        device = next(model.parameters()).device

    eos_id = tokenizer.token_to_id("<eos>")
    model.reset_cache()

    # Prefill: 处理整个 prompt
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, _ = model(input_ids, use_cache=True, start_pos=0)
    next_logits = logits[:, -1, :]

    generated: list[int] = []
    pos = len(prompt_ids)

    for _ in range(max_new_tokens):
        # Sampling
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

        # Decode step: 单 token 输入
        input_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits, _ = model(input_ids, use_cache=True, start_pos=pos)
        next_logits = logits[:, -1, :]
        pos += 1

        if pos >= model.cfg.max_seq_len:
            break

    return generated
