"""ch10 练习 2：SFT loss mask 对照实验。

构造一个简化的 SFT 样本（prompt + response），分别用：
  A. 全序列算 loss（错误做法，模拟"忘了 mask"）
  B. 只对 response 算 loss（正确做法，prompt 部分 -100）

然后训同一个 TinyLM 若干步，对比：
  - response 部分的 loss 下降情况
  - prompt 部分的 loss 变化（B 不应主动学，但 emb 共享会有间接影响）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLM(nn.Module):
    """单层 GRU + 线性头的极简 LM，能体现序列依赖即可。"""

    def __init__(self, vocab_size: int, d_model: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        # batch_first=True：输入输出形状 (B, L, d) 而非 (L, B, d)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.emb(ids))  # (B, L, d)
        return self.head(h)             # (B, L, V)


def make_sample(vocab_size: int, prompt_len: int = 12, response_len: int = 8) -> tuple[torch.Tensor, int]:
    """造一个假 SFT 样本：随机 prompt + 一个固定模式的 response（让 loss 信号清晰）。

    返回 (tokens, prompt_len)。tokens.shape == (1, prompt_len + response_len)
    """
    torch.manual_seed(0)
    prompt = torch.randint(1, vocab_size, (1, prompt_len))
    # response 用一个简单循环模式（如 1,2,3,1,2,3,...），便于模型学且观察明显
    pattern = torch.tensor([1, 2, 3])
    # repeat(n) 沿 dim 0 复制 n 次 → 切片裁到 response_len → unsqueeze(0) 加 batch 维：(R,) → (1, R)
    response = pattern.repeat((response_len + 2) // 3)[:response_len].unsqueeze(0)
    tokens = torch.cat([prompt, response], dim=1)  # (1, P+R)
    return tokens, prompt_len


def make_labels(tokens: torch.Tensor, prompt_len: int, mask_prompt: bool) -> torch.Tensor:
    """构造 labels（与 input_ids 同 shape，cross_entropy 用 ignore_index=-100 跳过）。

    注意：CLM 的 loss 是"用 t 位置预测 t+1 位置"。这里为简化，直接把 labels 设为 input_ids 同 shape，
    在内部做 shift。详细对齐写法见 ch09。

    mask_prompt=True：对应正确 SFT 做法，prompt 部分 label 设 -100
    mask_prompt=False：错误做法，全序列都算 loss
    """
    labels = tokens.clone()
    if mask_prompt:
        # 注意 shift 后的语义：position t 的 label 实际是 tokens[t+1]
        # 想要"不学 prompt 内的下一个 token 预测"，应该 mask labels[:prompt_len-1]（含）
        # 即让模型从"prompt 最后一个 token 预测 response 第一个 token"开始算 loss
        labels[:, : prompt_len - 1] = -100
    return labels


def compute_loss(model: TinyLM, tokens: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """标准 CLM shift + cross_entropy(ignore_index=-100)。"""
    input_ids = tokens[:, :-1]      # (B, L-1)
    shift_labels = labels[:, 1:]    # (B, L-1) 对齐：input 的 t 位置去预测 t+1 的 label
    logits = model(input_ids)       # (B, L-1, V)
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def loss_per_segment(
    model: TinyLM, tokens: torch.Tensor, prompt_len: int, vocab_size: int
) -> tuple[float, float]:
    """诊断用：单独算 prompt 段与 response 段的平均 loss（无 mask）。"""
    with torch.no_grad():
        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]
        logits = model(input_ids)
        # reduction='none' → 每个位置一个 loss 值，便于按段切片
        per_pos = F.cross_entropy(
            logits.reshape(-1, vocab_size), labels.reshape(-1), reduction="none"
        ).reshape(tokens.shape[0], -1)
        # shift 后 prompt 段长度 = prompt_len - 1（最后一个 prompt token 用来预测 response 首位）
        prompt_loss = per_pos[:, : prompt_len - 1].mean().item()
        response_loss = per_pos[:, prompt_len - 1 :].mean().item()
    return prompt_loss, response_loss


def train_one_setting(mask_prompt: bool, n_steps: int = 200) -> tuple[list[float], list[float]]:
    """跑一种 mask 设置的训练，返回 (prompt_loss 历史, response_loss 历史)。"""
    vocab_size = 50
    tokens, prompt_len = make_sample(vocab_size)
    labels = make_labels(tokens, prompt_len, mask_prompt)

    torch.manual_seed(42)  # 模型初始化固定，两次实验可比
    model = TinyLM(vocab_size, d_model=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    p_hist, r_hist = [], []
    for step in range(n_steps):
        opt.zero_grad()
        loss = compute_loss(model, tokens, labels, vocab_size)
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == n_steps - 1:
            p, r = loss_per_segment(model, tokens, prompt_len, vocab_size)
            p_hist.append(p)
            r_hist.append(r)
    return p_hist, r_hist


def main() -> None:
    print("=" * 70)
    print("对照：mask_prompt=False（错误，全序列算 loss）")
    print("=" * 70)
    p_no, r_no = train_one_setting(mask_prompt=False, n_steps=200)
    print(f"  prompt 段最终 loss   = {p_no[-1]:.4f}（被强行学，但 prompt 是随机噪声 → 难降）")
    print(f"  response 段最终 loss = {r_no[-1]:.4f}\n")

    print("=" * 70)
    print("对照：mask_prompt=True（正确，prompt 部分 -100 不算 loss）")
    print("=" * 70)
    p_yes, r_yes = train_one_setting(mask_prompt=True, n_steps=200)
    print(f"  prompt 段最终 loss   = {p_yes[-1]:.4f}（不参与训练，自然不降）")
    print(f"  response 段最终 loss = {r_yes[-1]:.4f}（容量集中在 response，应更低）\n")

    print("=" * 70)
    print("[结论]")
    print("=" * 70)
    print(f"  response loss 对比：mask={r_yes[-1]:.4f}  vs  no-mask={r_no[-1]:.4f}")
    if r_yes[-1] < r_no[-1]:
        print("  → mask=True 时 response 学得更好（容量没被随机 prompt 浪费）")
    else:
        print("  → 两者差不多（玩具样本/小模型差异不显著，但真实大模型差异明显）")
    print(f"  prompt   loss 对比：mask={p_yes[-1]:.4f}  vs  no-mask={p_no[-1]:.4f}")
    print("  → no-mask 强行让模型记忆随机 prompt → 浪费参数容量、推理时容易复读 prompt")


if __name__ == "__main__":
    main()
