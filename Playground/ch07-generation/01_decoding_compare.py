"""ch07 练习 1：解码策略对比。

复用 ch06 的 MiniGPT，即时在 Tiny Shakespeare 上训 200 步（足以让 loss 下降但不过度），
对比五种解码方式的续写质量：
  - greedy
  - temperature=0.8
  - top-k=20
  - top-p=0.9
  - 三件套 (T=0.8 + top-k=50 + top-p=0.95)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
# 复用 ch06 的 MiniGPT 与 04 的数据加载
CH06_DIR = REPO_ROOT / "Playground" / "ch06-transformer"
sys.path.insert(0, str(CH06_DIR))

import torch
import torch.nn.functional as F

from Echo.shared.device import get_device
from importlib import import_module
MiniGPT = import_module("03_model").MiniGPT
load_dataset = import_module("04_train_shakespeare").load_dataset
get_batch = import_module("04_train_shakespeare").get_batch


@torch.no_grad()
def generate(
    model: MiniGPT,
    ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """统一采样 generate。temperature=0 走贪心，其它情况按需叠加 top-k / top-p。"""
    model.eval()
    for _ in range(max_new_tokens):
        ids_cond = ids[:, -model.max_len :]
        logits = model(ids_cond)[:, -1, :]  # 只看最后一步：(B, V)

        if temperature == 0:
            # 贪心：等价 T → 0⁺ 的极限，避免除零直接走 argmax
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature  # 温度缩放：T 越大分布越平
            if top_k is not None:
                # torch.topk 返回 (values, indices)：values 是降序排好的前 k 大值，indices 是它们在原 tensor 的位置
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = v[:, -1:]  # 第 k 大值（topk 降序排，最后一个就是阈值）
                # torch.where(cond, a, b)：按 cond 逐元素选 a 或 b；这里把 < 阈值的位置替换成 -inf
                logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
            if top_p is not None:
                # nucleus: 累计概率 ≥ top_p 的最小 token 集合，其余置 -inf
                # torch.sort(descending=True, dim=-1) 返回 (sorted_values, original_indices)
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                # cumsum(dim=-1)：沿最后一维累加，得到前缀和；softmax 后的累加 = 累计概率
                cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                # 标记累计概率超过阈值的位置（要剔除的）；把"恰好达到"的那一位保留
                remove = cum_probs > top_p
                # 右移 1 位：让"第一个使累计 ≥ p 的 token"留下，从它后面开始才剔除
                # 用 .clone() 避免 inplace 写入与右侧切片视图冲突
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                # 把排序后要剔除的位置映射回原索引：
                # tensor.scatter_(dim, index, src)：沿 dim 把 src 的值按 index 写入 self（gather 的反操作）
                logits_filtered = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.empty_like(logits).scatter_(dim=-1, index=sorted_idx, src=logits_filtered)
            probs = F.softmax(logits, dim=-1)
            # torch.multinomial(probs, num_samples=1)：按 probs 概率分布抽样 1 个索引（采样而非 argmax）
            next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


def main() -> None:
    device = get_device()
    print(f"device: {device}")
    torch.manual_seed(42)

    text = load_dataset()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    # dtype=torch.long 因为后续要喂给 nn.Embedding，必须是 LongTensor 整数索引
    data = torch.tensor(encode(text), dtype=torch.long)

    block_size = 64
    model = MiniGPT(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=4, max_len=block_size).to(device)

    # 训 500 步：贪心退化现象更明显（200 步不够，模型还没足够确定地循环）
    print("训练 500 步...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    model.train()
    for step in range(1, 501):
        xb, yb = get_batch(data, batch_size=32, block_size=block_size, device=device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"  step {step}  loss {loss.item():.3f}")

    # 对比五种解码
    prompt = "ROMEO:"
    # .unsqueeze(0)：在最前加 batch 维 → (n,) → (1, n)
    prompt_ids = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    configs = [
        ("greedy            ", dict(temperature=0)),
        ("T=0.8             ", dict(temperature=0.8)),
        ("top-k=20          ", dict(temperature=1.0, top_k=20)),
        ("top-p=0.9         ", dict(temperature=1.0, top_p=0.9)),
        ("三件套 T=0.8+k50+p=0.95", dict(temperature=0.8, top_k=50, top_p=0.95)),
    ]

    print(f"\nprompt: {prompt!r}\n" + "=" * 70)
    for name, cfg in configs:
        torch.manual_seed(0)  # 固定种子让采样可复现
        out_ids = generate(model, prompt_ids, max_new_tokens=120, **cfg)[0].tolist()
        text_out = decode(out_ids).replace("\n", "\\n")  # 单行打印更清晰
        print(f"\n[{name}]")
        print(text_out)

    # 验证 1：greedy 输出含明显重复（连续 5 个相同字符出现）
    torch.manual_seed(0)
    greedy_out = decode(generate(model, prompt_ids, max_new_tokens=200, temperature=0)[0].tolist())
    has_repeat = any(greedy_out[i] == greedy_out[i + 1] == greedy_out[i + 2] == greedy_out[i + 3] for i in range(len(greedy_out) - 4))
    print(f"\ngreedy 是否出现连续 4 个相同字符（贪心退化迹象）: {has_repeat}")

    # 验证 2：采样模式输出多样性更高（去重字符数更多）
    torch.manual_seed(0)
    sample_out = decode(generate(model, prompt_ids, max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.95)[0].tolist())
    greedy_unique = len(set(greedy_out))
    sample_unique = len(set(sample_out))
    print(f"greedy 输出 unique 字符数: {greedy_unique}")
    print(f"三件套输出 unique 字符数: {sample_unique}")
    assert sample_unique > greedy_unique, "采样模式应比贪心多样性更高"

    print("\nPASS")


if __name__ == "__main__":
    main()
