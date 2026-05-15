"""ch09 练习 2：数据 packing 对比。

构造一批长度差异巨大的"假文档"，对比两种喂数据方式：
  A. batch padding：每 batch 内 pad 到最长
  B. packing：拼成长流后切固定窗口

观察：有效 token 比例（非 pad / 总）。
"""

from __future__ import annotations

import random

import torch
from torch.nn.utils.rnn import pad_sequence

EOS = 0
PAD = 0  # demo 里用同一个 id 简化；真实场景 pad 与 eos 通常分离

random.seed(0)
torch.manual_seed(0)


def make_fake_docs(n_docs: int = 200) -> list[list[int]]:
    """构造 n_docs 个长度服从长尾分布的假文档（少量长文档 + 大量短文档）。"""
    docs = []
    for _ in range(n_docs):
        # random.lognormvariate(mu, sigma)：对数正态分布，模拟真实文档长度长尾
        # 中位数约 e^3.5 ≈ 33，但少数样本会拖到几百（mean 因长尾被拉到 40+）
        length = max(5, int(random.lognormvariate(3.5, 0.8)))
        # 随机 token id（避开 0，0 留给 pad/eos）
        doc = [random.randint(1, 1000) for _ in range(length)]
        docs.append(doc)
    return docs


def stats_padding(docs: list[list[int]], batch_size: int = 16) -> tuple[int, int]:
    """方案 A：按 batch padding。返回 (有效 token, 总 token)。"""
    valid, total = 0, 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        max_len = max(len(d) for d in batch)
        for d in batch:
            valid += len(d)
            total += max_len  # 每条都 pad 到 max_len
    return valid, total


def stats_packing(docs: list[list[int]], block_size: int = 256) -> tuple[int, int]:
    """方案 B：拼接 + 切块。返回 (有效 token, 总 token)。"""
    # 拼成长流，每个 doc 后塞 EOS 当边界
    flat: list[int] = []
    for d in docs:
        flat.extend(d)
        flat.append(EOS)
    n_chunks = len(flat) // block_size  # 不足 block_size 的尾部丢弃
    total = n_chunks * block_size
    # packing 几乎无 pad；EOS 算"有效 token"（它是真实信号，不是无意义填充）
    valid = total
    return valid, total


def main() -> None:
    docs = make_fake_docs(200)
    lengths = [len(d) for d in docs]
    print(f"假文档数: {len(docs)}")
    print(f"长度分布: min={min(lengths)}  max={max(lengths)}  mean={sum(lengths) / len(lengths):.1f}")
    print(f"总 token: {sum(lengths)}\n")

    # 方案 A
    a_valid, a_total = stats_padding(docs, batch_size=16)
    a_ratio = a_valid / a_total
    print(f"[A. batch padding, batch_size=16]")
    print(f"  有效 token = {a_valid}")
    print(f"  总 token   = {a_total}")
    print(f"  有效比例   = {a_ratio:.1%}")
    print(f"  浪费       = {a_total - a_valid} 个 pad\n")

    # 方案 B
    b_valid, b_total = stats_packing(docs, block_size=256)
    b_ratio = b_valid / b_total
    print(f"[B. packing, block_size=256]")
    print(f"  有效 token = {b_valid}")
    print(f"  总 token   = {b_total}")
    print(f"  有效比例   = {b_ratio:.1%}")
    print(f"  浪费       = {b_total - b_valid} 个 pad（理论上仅尾部丢弃部分）\n")

    print(f"[结论] packing 比 padding 高 {(b_ratio - a_ratio) * 100:.1f} 个百分点的算力效率")
    assert b_ratio > a_ratio, "packing 应显著优于 padding"

    # 演示：用 pad_sequence 实际造一个 batch，看 shape
    batch = docs[:4]
    # pad_sequence(seqs, batch_first, padding_value)：把变长 list[Tensor] padding 成等长 tensor
    # batch_first=True → 输出 (B, L)；False → (L, B)
    padded = pad_sequence(
        [torch.tensor(d) for d in batch],
        batch_first=True,
        padding_value=PAD,
    )
    print(f"\n[示例 padding batch]")
    print(f"  shape = {tuple(padded.shape)}")
    print(f"  各文档原长 = {[len(d) for d in batch]}")
    print(f"  非 pad 比例 = {(padded != PAD).float().mean().item():.1%}")


if __name__ == "__main__":
    main()
