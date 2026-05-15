"""ch11 练习 2：DPO 中 β 的影响实验。

固定一对 (prompt, chosen, rejected)，分别用不同 β 训若干步，对比：
  - loss 下降速度
  - chosen / rejected 的 logp 分化程度
  - 训完后"模型相对 ref 的偏移"幅度（隐式 KL 强度）

直觉（注意：单步梯度幅值与收敛终点的 drift 方向相反）：
  β 大 → sigmoid σ(β·m) 饱和更快 → margin 略大就让 loss 趋零、梯度归零
         → 收敛时 logp 分化"小"、param drift "小"（更靠近 ref）
  β 小 → sigmoid 平缓 → 需要把 margin 推得很大才能压低 loss
         → 收敛时 logp 分化"大"、param drift "大"（远离 ref）
  这就是为什么 DPO 论文把 β 直接称作"KL 强度"：β 越大约束越强。
  （01 验证 3 测的是固定 margin 下单步梯度的线性放大，与本实验的"收敛终点"不冲突）
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.emb(ids))


def sequence_logp(model: nn.Module, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    # 与 01_dpo_loss.py 一致；gather / shift / response 切片的语义注释见该文件
    input_ids = full_ids[:, :-1]
    target_ids = full_ids[:, 1:]
    logits = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    response_logps = token_logps[:, prompt_len - 1 :]
    return response_logps.sum(dim=-1)


def dpo_loss(lp_pi_c, lp_pi_r, lp_ref_c, lp_ref_r, beta):
    # DPO loss：margin = β·((logp_π(c) - logp_ref(c)) - (logp_π(r) - logp_ref(r)))
    # 详细公式与 logsigmoid 数值稳定性说明见 01_dpo_loss.py
    margin = beta * ((lp_pi_c - lp_ref_c) - (lp_pi_r - lp_ref_r))
    return -F.logsigmoid(margin).mean()


def param_drift(model: nn.Module, ref: nn.Module) -> float:
    """衡量 model 相对 ref 的参数空间偏移（L2 范数）。"""
    diff_sq = 0.0
    for p, q in zip(model.parameters(), ref.parameters(), strict=True):
        diff_sq += (p - q).pow(2).sum().item()
    return diff_sq**0.5


def run_one_beta(beta: float, n_steps: int = 200, vocab_size: int = 50):
    """跑一个 β 的训练，返回 (最终 loss, Δlogp_c, Δlogp_r, drift)。"""
    torch.manual_seed(0)
    prompt = torch.randint(1, vocab_size, (1, 6))
    chosen = torch.tensor([[5, 6, 7, 8, 9, 10]])
    rejected = torch.tensor([[20, 21, 22, 23, 24, 25]])
    chosen_ids = torch.cat([prompt, chosen], dim=1)
    rejected_ids = torch.cat([prompt, rejected], dim=1)
    prompt_len = 6

    torch.manual_seed(42)
    policy = TinyLM(vocab_size, d_model=32)
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad = False

    # 训前 ref logp 缓存（ref 不变，只算一次）
    with torch.no_grad():
        lp_ref_c = sequence_logp(ref, chosen_ids, prompt_len)
        lp_ref_r = sequence_logp(ref, rejected_ids, prompt_len)

    opt = torch.optim.Adam(policy.parameters(), lr=5e-3)
    final_loss = None
    for _ in range(n_steps):
        opt.zero_grad()
        lp_pi_c = sequence_logp(policy, chosen_ids, prompt_len)
        lp_pi_r = sequence_logp(policy, rejected_ids, prompt_len)
        loss = dpo_loss(lp_pi_c, lp_pi_r, lp_ref_c, lp_ref_r, beta)
        loss.backward()
        opt.step()
        final_loss = loss.item()

    with torch.no_grad():
        delta_c = (sequence_logp(policy, chosen_ids, prompt_len) - lp_ref_c).item()
        delta_r = (sequence_logp(policy, rejected_ids, prompt_len) - lp_ref_r).item()
    drift = param_drift(policy, ref)
    return final_loss, delta_c, delta_r, drift


def main() -> None:
    print("=" * 78)
    print("DPO β 扫描：固定数据 + 固定初始化 + 固定 200 步训练，只变 β")
    print("=" * 78)
    print(f"  {'β':>6}  {'final loss':>10}  {'Δlogp_c':>9}  {'Δlogp_r':>9}  {'分化':>8}  {'param drift':>11}")
    print(f"  {'-' * 6}  {'-' * 10}  {'-' * 9}  {'-' * 9}  {'-' * 8}  {'-' * 11}")
    rows = []
    for beta in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
        loss, dc, dr, drift = run_one_beta(beta)
        rows.append((beta, loss, dc, dr, drift))
        print(
            f"  {beta:>6.2f}  {loss:>10.4f}  {dc:>+9.3f}  {dr:>+9.3f}  "
            f"{(dc - dr):>8.3f}  {drift:>11.4f}"
        )

    print("\n" + "=" * 78)
    print("[读图]")
    print("=" * 78)
    print("  - β 大：sigmoid 饱和快 → 收敛 margin 小、drift 小（贴 ref）")
    print("  - β 小：sigmoid 平缓 → 需更大 margin 才能压低 loss → drift 大、分化更猛")
    print("  - 工业实践 β 通常 0.1–0.5：约束适中，分化够大但不至崩坏 SFT 学到的能力")
    print("\n  注意 'final loss' 在所有 β 下都接近 0，这是玩具数据的特征（chosen 和 rejected")
    print("  完全独立 token，模型容易把它们分开）。真实数据 chosen/rejected 共享大量 token，")
    print("  loss 不会这么快趋零，β 的影响主要体现在收敛时的 drift 与生成质量。")


if __name__ == "__main__":
    main()
