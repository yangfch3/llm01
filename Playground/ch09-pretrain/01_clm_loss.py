"""ch09 练习 1：CLM loss 与 input/labels 对齐。

演示三件事：
  1. 手动 shift 算 loss
  2. 等价的 "labels=tokens 让模型内部 shift" 写法（HF 风格）
  3. ignore_index=-100 让 padding 位不计入 loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLM(nn.Module):
    """极简 LM：embedding + 单个线性头。够用来验证 loss 行为。"""

    def __init__(self, vocab_size: int, d_model: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # 不做因果 mask 也无所谓——本练习只关心 loss 形态，不关心生成质量
        return self.head(self.emb(ids))  # (B, L, V)


def loss_manual_shift(model: TinyLM, tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """手动 shift：input 是除最后一位、labels 是除第一位。"""
    input_ids = tokens[:, :-1]  # (B, L)
    labels = tokens[:, 1:]      # (B, L)
    logits = model(input_ids)   # (B, L, V)
    # F.cross_entropy 期望 (N, V) + (N,)，把 batch 和 seq 维度拍平
    return F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1))


def loss_hf_style(model: TinyLM, tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """HF 风格：模型吃完整 tokens，内部 shift。
    模型 forward 的是 tokens[:, :-1]，labels 是 tokens[:, 1:]，本质等价。
    这里手动模拟"内部 shift"以证等价。
    """
    logits = model(tokens)               # (B, L+1, V)
    shift_logits = logits[:, :-1, :]     # (B, L, V) 丢掉最后一个位置（没有"下一个"可预测）
    shift_labels = tokens[:, 1:]         # (B, L)    丢掉第一个位置（没有"上一个"作为输入）
    return F.cross_entropy(shift_logits.reshape(-1, vocab_size), shift_labels.reshape(-1))


def main() -> None:
    torch.manual_seed(0)
    vocab_size = 50
    batch, seq = 2, 8
    # randint(low, high, size)：[low, high) 范围整数；这里 0 到 vocab-1 做假 token
    tokens = torch.randint(0, vocab_size, (batch, seq + 1))  # (B, L+1)
    model = TinyLM(vocab_size, d_model=16)

    # 验证 1：两种写法 loss 数值完全相同
    l1 = loss_manual_shift(model, tokens, vocab_size)
    l2 = loss_hf_style(model, tokens, vocab_size)
    print(f"[等价验证] manual_shift = {l1.item():.6f}")
    print(f"[等价验证] hf_style     = {l2.item():.6f}")
    assert torch.allclose(l1, l2), "两种写法应数值完全一致"
    print("PASS: 手动 shift 与 HF 风格内部 shift 等价\n")

    # 验证 2：ignore_index=-100 让 pad 位不参与 loss
    # 构造：第一个样本后 4 位是 pad，labels 设 -100
    labels = tokens[:, 1:].clone()  # (B, L)
    labels[0, -4:] = -100  # 第一个样本的最后 4 个位置不计 loss
    logits = model(tokens[:, :-1])  # (B, L, V)
    # ignore_index=-100：cross_entropy 跳过 label==-100 的位置（不计入分子也不计入分母）
    loss_with_ignore = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )
    # 对照：手动只在非 -100 位置算
    mask = labels != -100  # (B, L) bool
    # .clamp(min=0)：把 -100 替成合法 class id（0），否则 cross_entropy 越界报错；
    # 这些位置的 loss 反正会被下面的 mask 过滤掉，填什么数值不影响结果
    loss_per_pos = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1).clamp(min=0),
        reduction="none",
    ).reshape(batch, seq)
    loss_manual_mask = loss_per_pos[mask].mean()
    print(f"[ignore 验证] cross_entropy(ignore_index=-100) = {loss_with_ignore.item():.6f}")
    print(f"[ignore 验证] 手动 mask 平均                    = {loss_manual_mask.item():.6f}")
    assert torch.allclose(loss_with_ignore, loss_manual_mask, atol=1e-6)
    print("PASS: ignore_index=-100 等价于手动 mask 后取平均\n")

    # 验证 3：input==labels 时 loss 应该比正确 shift 时低（模型可作弊抄当前 token）
    # 训几步让 emb 与 head 学会 identity
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(50):
        logits = model(tokens[:, :-1])
        loss_cheat = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            tokens[:, :-1].reshape(-1),  # labels 与 input 完全相同：作弊
        )
        optimizer.zero_grad()
        loss_cheat.backward()
        optimizer.step()
    print(f"[对照] input==labels 训 50 步后 loss = {loss_cheat.item():.4f}")
    print("  能轻松降到极低 → 任务退化为 identity，没学到'预测下一个'")


if __name__ == "__main__":
    main()
