"""ch10 练习 3：手撕 LoRA 包装一层 nn.Linear。

无 peft 依赖，验证三件事：
  1. LoRALinear 在 B=0 初始化下与原 Linear 数值完全一致（"空旁路"）
  2. 训练时只有 LoRA 参数（A、B）有梯度，base.weight 永远 None
  3. 参数量对比：原 Linear vs LoRA 增量
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """对一个冻结的 nn.Linear 加一条低秩旁路。

    y = base(x) + (alpha/r) * (B @ A @ x)

    base: 冻结的原线性层（require_grad=False）
    A: (r, in_features)   Kaiming 类小方差初始化
    B: (out_features, r)  零初始化 → 训练起点等价原模型
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16) -> None:
        super().__init__()
        self.base = base
        # 冻结 base 的所有参数：requires_grad=False → 反向不计算梯度，optimizer 也忽略
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r  # 推理时把旁路缩放，避免与 r 大小耦合

        in_f, out_f = base.in_features, base.out_features
        # nn.Parameter 包裹的 Tensor 会被 .parameters() 遍历到，自动加入梯度计算
        # A: kaiming_uniform，与 nn.Linear 默认初始化方差量级对齐
        self.A = nn.Parameter(torch.empty(r, in_f))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)  # PyTorch 默认 nn.Linear 用的就是这个 a 值
        # B: 全零 → 训练起点 BAx = 0 → forward 输出与原 base 完全一致
        self.B = nn.Parameter(torch.zeros(out_f, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主路：原模型，权重冻结
        base_out = self.base(x)
        # 旁路：x @ A.T @ B.T（与 base.weight @ x 形状对齐）
        # F.linear(x, W) 等价 x @ W.T，输入 (..., in)，权重 (out, in)，输出 (..., out)
        lora_out = F.linear(F.linear(x, self.A), self.B)  # (..., r) → (..., out)
        return base_out + self.scaling * lora_out


def count_params(module: nn.Module) -> tuple[int, int]:
    """返回 (总参数, 可训参数)。"""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    torch.manual_seed(0)
    in_f, out_f = 512, 512
    base = nn.Linear(in_f, out_f, bias=False)

    # ---------- 验证 1：B=0 时 LoRALinear 与 base 数值完全一致 ----------
    lora = LoRALinear(base, r=8, alpha=16)
    x = torch.randn(2, 4, in_f)  # (B, L, in)
    y_base = base(x)
    y_lora = lora(x)
    print("=" * 60)
    print("[验证 1] B=0 初始化下 LoRA 旁路应输出 0")
    print("=" * 60)
    diff = (y_lora - y_base).abs().max().item()
    print(f"  max |y_lora - y_base| = {diff:.2e}")
    assert diff < 1e-6, "B=0 时旁路必须为 0"
    print("  PASS: 训练起点完全等价原模型\n")

    # ---------- 验证 2：参数量对比 ----------
    base_total, base_trainable = count_params(nn.Linear(in_f, out_f, bias=False))  # 重建一个干净的算
    lora_total, lora_trainable = count_params(lora)
    print("=" * 60)
    print("[验证 2] 参数量对比（in=out=512, r=8, alpha=16）")
    print("=" * 60)
    print(f"  原 Linear           总参数 = {base_total:>10,}    可训 = {base_trainable:>10,}")
    print(f"  LoRALinear (r=8)    总参数 = {lora_total:>10,}    可训 = {lora_trainable:>10,}")
    lora_only = lora.A.numel() + lora.B.numel()
    print(f"  LoRA 增量参数       = A({lora.A.shape}) + B({lora.B.shape}) = {lora_only:,}")
    print(f"  可训参数减少倍数    = {base_trainable / lora_trainable:.1f}×")
    print(f"  公式：r×(in+out) = 8×({in_f}+{out_f}) = {8 * (in_f + out_f):,}\n")

    # ---------- 验证 3：一次前反向，只有 A、B 有梯度 ----------
    opt = torch.optim.Adam(lora.parameters(), lr=1e-3)
    x = torch.randn(2, 4, in_f)
    target = torch.randn(2, 4, out_f)
    opt.zero_grad()
    loss = F.mse_loss(lora(x), target)
    loss.backward()

    print("=" * 60)
    print("[验证 3] 反向后梯度归属")
    print("=" * 60)
    print(f"  base.weight.grad is None: {lora.base.weight.grad is None}  (应为 True，base 冻结)")
    print(f"  A.grad shape:    {tuple(lora.A.grad.shape)}  norm = {lora.A.grad.norm().item():.4f}")
    print(f"  B.grad shape:    {tuple(lora.B.grad.shape)}  norm = {lora.B.grad.norm().item():.4f}")
    assert lora.base.weight.grad is None, "base 必须无梯度"
    assert lora.A.grad is not None and lora.B.grad is not None
    print("  PASS: 梯度只流向 LoRA 参数\n")

    # ---------- 验证 4：训几步，旁路实际激活 ----------
    print("=" * 60)
    print("[验证 4] 训 30 步后旁路有实际贡献")
    print("=" * 60)
    for _step in range(30):
        opt.zero_grad()
        loss = F.mse_loss(lora(x), target)
        loss.backward()
        opt.step()
    with torch.no_grad():
        y_lora = lora(x)
        y_base = lora.base(x)
        side_norm = (y_lora - y_base).norm().item()
        full_norm = y_lora.norm().item()
    print(f"  旁路输出范数 = {side_norm:.4f}")
    print(f"  总输出范数   = {full_norm:.4f}")
    print(f"  旁路占比     = {side_norm / full_norm:.1%}")
    print(f"  最终 loss    = {loss.item():.4f}")
    print("  → 旁路从 0 学出了非零增量，base 始终未动\n")

    # ---------- 显存账小算（只是数字，没真分配） ----------
    print("=" * 60)
    print("[显存账] 假设 7B 模型，attention q/k/v/o 全加 LoRA r=8（直觉数字）")
    print("=" * 60)
    print("  base weights (bf16):     7B × 2B = 14 GB     ← 冻结但仍要存")
    print("  base gradients:          0                    ← LoRA 不算 base 梯度")
    print("  LoRA params (bf16):      ~8.4M × 2B = 17 MB")
    print("  LoRA gradients (bf16):   ~17 MB")
    print("  Adam states (fp32 m+v):  8.4M × 8B = 67 MB    ← 只对 LoRA 算")
    print("  ————————————————————————————————————————")
    print("  总计 ~14.5 GB，对比全参 ~85 GB，砍掉 ~70 GB")


if __name__ == "__main__":
    main()
