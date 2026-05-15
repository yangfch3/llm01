"""ch06 练习 3：完整 MiniGPT 模型 + 朴素贪心 generate。

把 ch05 的 attention 和 02 的 Block 拼成一个能 forward / 能生成的 Decoder-only 模型。
不接训练，仅验证：
- 前向 logits shape 正确
- weight tying 生效（embedding 与 lm_head 是同一个 tensor）
- 朴素贪心 generate 能跑、能停在 max_new_tokens
- 参数量与公式估算一致
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# 复用 02 的 Block 实现
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from Echo.shared.device import get_device

# 局部 import，复用 02 的 Block（避免重复实现）
from importlib import import_module
_block_mod = import_module("02_block")
Block = _block_mod.Block


class MiniGPT(nn.Module):
    """最小 Decoder-only GPT。

    设计选型（与 ch06 §8 一致）：
    - 学习式位置 embedding（最简，4 节末尾的伪代码版；RoPE 留给 echo-mini）
    - Pre-LN block × n_layers
    - final LN + lm_head（与 token_emb 共享权重）
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int) -> None:
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)  # 学习式位置编码
        # nn.ModuleList：容器版的 list，区别在于会把内部 module 注册为子模块，
        # 让 .parameters() / .to(device) / state_dict 都能找到它们；普通 list 则不会
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying：lm_head 与 token_emb 共享同一个权重矩阵，省 V·d 参数
        self.lm_head.weight = self.token_emb.weight

        # GPT-2 风格初始化：std=0.02 的小正态。M2 提过，M3 ch09 详讲
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # nn.init.normal_(tensor, mean, std)：in-place 用正态分布填充 tensor（带下划线表示原地）
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # in-place 置 0
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (B, n)，n ≤ max_len
        B, n = ids.shape
        assert n <= self.max_len, f"序列长度 {n} 超过 max_len {self.max_len}"
        pos = torch.arange(n, device=ids.device)  # (n,)
        # 广播：token_emb (B, n, d) + pos_emb (n, d) → (B, n, d)
        x = self.token_emb(ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_final(x)  # Pre-LN 模型必须的最末归一化
        return self.lm_head(x)  # (B, n, V)

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """朴素贪心：每步取 argmax。ch07 会加 top-k / top-p / KV cache。"""
        self.eval()
        for _ in range(max_new_tokens):
            # 上下文超长就截断左侧（推理时无 KV cache 的简单做法）
            ids_cond = ids[:, -self.max_len :]
            logits = self(ids_cond)              # (B, n, V)
            next_logits = logits[:, -1, :]       # 只看最后一步：(B, V)
            next_id = next_logits.argmax(dim=-1, keepdim=True)  # 贪心：argmax
            ids = torch.cat([ids, next_id], dim=1)
        return ids


def count_params(model: nn.Module) -> int:
    # 注意：weight tying 后 token_emb.weight 与 lm_head.weight 是同一 tensor，不重复计数
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def main() -> None:
    device = get_device()
    print(f"device: {device}")
    torch.manual_seed(0)

    V, d, H, L, max_len = 256, 64, 4, 4, 32  # 玩具配置
    model = MiniGPT(vocab_size=V, d_model=d, n_heads=H, n_layers=L, max_len=max_len).to(device)

    # 1) 权重共享检查
    # tensor.data_ptr() 返回底层存储的内存地址；两个 tensor 地址相同 = 共享同一份数据
    assert model.lm_head.weight.data_ptr() == model.token_emb.weight.data_ptr(), "weight tying 必须生效"
    print("weight tying: 共享同一权重 tensor ✓")

    # 2) 前向 shape 检查
    B, n = 3, 16
    ids = torch.randint(0, V, (B, n), device=device)
    logits = model(ids)
    print(f"\n输入 ids shape: {tuple(ids.shape)}")
    print(f"输出 logits shape: {tuple(logits.shape)}（应为 (B, n, V)）")
    assert logits.shape == (B, n, V)

    # 3) 训练目标 = 下一个 token cross-entropy（CLM，ch09 详讲）
    targets = torch.randint(0, V, (B, n), device=device)
    # F.cross_entropy(logits, targets)：内部先做 log_softmax 再 NLL，所以传入的是原始 logits 而非概率
    # logits[:, :-1] 与 targets[:, 1:] 对齐 = "用前 t 个 token 预测第 t+1 个"
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, V),  # (B*(n-1), V)
        targets[:, 1:].reshape(-1),     # (B*(n-1),)
    )
    print(f"\n随机权重 loss: {loss.item():.3f}  ≈ ln(V) = {torch.log(torch.tensor(float(V))).item():.3f}")
    # 随机初始化时 loss ≈ ln(V)（均匀分布的交叉熵）；偏差 ±20% 内都正常
    assert 0.8 * torch.log(torch.tensor(float(V))) < loss < 1.2 * torch.log(torch.tensor(float(V)))

    # 4) 参数量 vs 公式
    n_params = count_params(model)
    block_params = 12 * d * d * L  # ≈ 12 L d²
    emb_params = V * d             # tying 后只算一次
    expected = emb_params + block_params
    print(f"\n实际参数量: {n_params}")
    print(f"公式估算  : V·d + 12·L·d² = {V}·{d} + 12·{L}·{d}² = {expected}")
    print(f"误差来自 LN/bias/final_ln 小项 = {n_params - expected}")

    # 5) 朴素贪心生成
    prompt = torch.randint(0, V, (1, 5), device=device)
    out = model.generate(prompt, max_new_tokens=10)
    print(f"\nprompt 长度: {prompt.size(1)}, 生成后长度: {out.size(1)}（应 = 15）")
    assert out.size(1) == prompt.size(1) + 10

    print("\nPASS")


if __name__ == "__main__":
    main()
