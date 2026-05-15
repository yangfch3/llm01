"""ch11 练习 1：手写 DPO loss，验证公式行为。

DPO loss:
  L = -log σ( β · ( logp_π(c) - logp_ref(c) - logp_π(r) + logp_ref(r) ) )

验证三件事：
  1. 训前 π == π_ref，loss = -log σ(0) = log 2 ≈ 0.693（理论值）
  2. 训若干步后：chosen 的 logp 上升、rejected 的 logp 下降（相对 π_ref）
  3. β 越大，对同样的 logp 差距，梯度越大（公式直觉）

用玩具 LM（embedding + linear），不依赖任何对齐库。
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.emb(ids))  # (B, L, V)


def sequence_logp(model: nn.Module, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """对一条 (prompt + response) 序列，只计算 response 部分的对数似然之和。

    返回 shape (B,)。等价于 SFT 里 mask 掉 prompt 后的总 logp。
    """
    input_ids = full_ids[:, :-1]   # (B, L-1) 用 t 预测 t+1
    target_ids = full_ids[:, 1:]   # (B, L-1) 真实下一个 token
    logits = model(input_ids)      # (B, L-1, V)
    # log_softmax 沿 V 维 → 每个位置每个 vocab 的对数概率
    log_probs = F.log_softmax(logits, dim=-1)
    # gather(dim, index)：沿 dim 按 index 取值。这里 dim=-1，index 形状 (B, L-1, 1)
    # → 输出 (B, L-1, 1) → squeeze 掉最后维 → (B, L-1) 即每个位置真实 token 的 logp
    token_logps = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    # response 部分对应的位置：原序列里 prompt 占 [0, prompt_len)，对应 target 位置是 [prompt_len-1, L-1)
    # 即"用 prompt 最后一个 token 预测 response 第一个 token"开始
    response_logps = token_logps[:, prompt_len - 1 :]  # (B, response_len)
    return response_logps.sum(dim=-1)  # (B,)


def dpo_loss(
    logp_pi_c: torch.Tensor, logp_pi_r: torch.Tensor,
    logp_ref_c: torch.Tensor, logp_ref_r: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """DPO loss 公式（按 batch 求平均）。"""
    # 各自的 "policy 相对 ref 的 logp 差"
    pi_logratios_c = logp_pi_c - logp_ref_c
    pi_logratios_r = logp_pi_r - logp_ref_r
    # 偏好"差距"：希望它越大（chosen 比 rejected 更被 π 偏好）
    margin = beta * (pi_logratios_c - pi_logratios_r)
    # -log σ(margin) = softplus(-margin)，等价于二元 logistic loss
    # 用 F.logsigmoid 而非 log(sigmoid(·))：数值稳定，避免大负 margin 时 sigmoid 下溢到 0 → log(0)=-inf
    return -F.logsigmoid(margin).mean()


def make_data(vocab_size: int = 50) -> tuple[torch.Tensor, torch.Tensor, int]:
    """造一对 (prompt, chosen, rejected)，prompt 共享，response 不同。"""
    torch.manual_seed(0)
    prompt_len = 6
    response_len = 6
    prompt = torch.randint(1, vocab_size, (1, prompt_len))
    # chosen / rejected 设成两段截然不同的 response，便于看 logp 分化
    chosen = torch.tensor([[5, 6, 7, 8, 9, 10]])
    rejected = torch.tensor([[20, 21, 22, 23, 24, 25]])
    chosen_full = torch.cat([prompt, chosen], dim=1)
    rejected_full = torch.cat([prompt, rejected], dim=1)
    return chosen_full, rejected_full, prompt_len


def main() -> None:
    vocab_size = 50
    chosen_ids, rejected_ids, prompt_len = make_data(vocab_size)

    torch.manual_seed(42)
    policy = TinyLM(vocab_size, d_model=32)
    # ref 必须是 policy 的"训前快照"且冻结。deepcopy 后关掉所有 require_grad
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()

    beta = 0.1

    # ---------- 验证 1：训前 loss ≈ log 2 ----------
    print("=" * 60)
    print("[验证 1] 训练前 π == π_ref，DPO loss 应等于 log 2")
    print("=" * 60)
    with torch.no_grad():
        logp_pi_c = sequence_logp(policy, chosen_ids, prompt_len)
        logp_pi_r = sequence_logp(policy, rejected_ids, prompt_len)
        logp_ref_c = sequence_logp(ref, chosen_ids, prompt_len)
        logp_ref_r = sequence_logp(ref, rejected_ids, prompt_len)
    init_loss = dpo_loss(logp_pi_c, logp_pi_r, logp_ref_c, logp_ref_r, beta)
    print(f"  实际 loss  = {init_loss.item():.6f}")
    print(f"  log 2 理论 = {math.log(2):.6f}")
    assert abs(init_loss.item() - math.log(2)) < 1e-5
    print("  PASS\n")

    # ---------- 验证 2：训若干步，看 logp 分化 ----------
    print("=" * 60)
    print(f"[验证 2] 训 200 步 (β={beta})，观察 chosen / rejected logp 演化")
    print("=" * 60)
    opt = torch.optim.Adam(policy.parameters(), lr=5e-3)
    print(f"  {'step':>5}  {'loss':>8}  {'logp_pi_c':>10}  {'logp_pi_r':>10}  {'margin/β':>10}")
    for step in range(201):
        opt.zero_grad()
        logp_pi_c = sequence_logp(policy, chosen_ids, prompt_len)
        logp_pi_r = sequence_logp(policy, rejected_ids, prompt_len)
        # ref 的 logp 不需要梯度，但每步都要算（因为它是固定参考点）
        with torch.no_grad():
            logp_ref_c_v = sequence_logp(ref, chosen_ids, prompt_len)
            logp_ref_r_v = sequence_logp(ref, rejected_ids, prompt_len)
        loss = dpo_loss(logp_pi_c, logp_pi_r, logp_ref_c_v, logp_ref_r_v, beta)
        loss.backward()
        opt.step()
        if step % 40 == 0 or step == 200:
            margin_div_beta = (logp_pi_c - logp_ref_c_v - logp_pi_r + logp_ref_r_v).item()
            print(
                f"  {step:>5}  {loss.item():>8.4f}  "
                f"{logp_pi_c.item():>10.3f}  {logp_pi_r.item():>10.3f}  "
                f"{margin_div_beta:>10.3f}"
            )

    # 最终验证：相对 ref，chosen 的 logp 升、rejected 的 logp 降
    with torch.no_grad():
        delta_c = (sequence_logp(policy, chosen_ids, prompt_len) - logp_ref_c).item()
        delta_r = (sequence_logp(policy, rejected_ids, prompt_len) - logp_ref_r).item()
    print(f"\n  Δlogp(chosen)   = π - ref = {delta_c:+.4f}（应 > 0）")
    print(f"  Δlogp(rejected) = π - ref = {delta_r:+.4f}（应 < 0）")
    assert delta_c > 0 and delta_r < 0, "DPO 应该拉开 chosen 与 rejected 的 logp"
    print("  PASS: chosen 被推高，rejected 被压低\n")

    # ---------- 验证 3：β 改变梯度幅值 ----------
    print("=" * 60)
    print("[验证 3] 同样的 logp 差距，β 越大梯度越大")
    print("=" * 60)
    # 重置一个干净的 policy，看初始 step 的梯度范数随 β 变化
    for trial_beta in [0.05, 0.1, 0.5, 1.0]:
        torch.manual_seed(42)
        m = TinyLM(vocab_size, d_model=32)
        ref2 = copy.deepcopy(m)
        for p in ref2.parameters():
            p.requires_grad = False
        # 给 m 一点扰动让它略偏离 ref（否则梯度全为 0）
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01 * torch.randn_like(p))
        logp_pi_c = sequence_logp(m, chosen_ids, prompt_len)
        logp_pi_r = sequence_logp(m, rejected_ids, prompt_len)
        with torch.no_grad():
            lref_c = sequence_logp(ref2, chosen_ids, prompt_len)
            lref_r = sequence_logp(ref2, rejected_ids, prompt_len)
        loss = dpo_loss(logp_pi_c, logp_pi_r, lref_c, lref_r, trial_beta)
        loss.backward()
        # 梯度范数：所有 trainable param 的梯度拼起来求 2-范数
        grad_norm = math.sqrt(sum(p.grad.pow(2).sum().item() for p in m.parameters() if p.grad is not None))
        print(f"  β = {trial_beta:>4}  loss = {loss.item():.4f}   grad_norm = {grad_norm:.4f}")
    print("\n  → β 越大，梯度幅值越大（公式中 β 直接放大 margin → 梯度按比例放大）")


if __name__ == "__main__":
    main()
