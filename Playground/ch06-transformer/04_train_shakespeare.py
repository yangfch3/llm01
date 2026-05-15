"""ch06 练习 4：char-level Tiny Shakespeare 过拟合训练。

目标：让 MiniGPT 在 Tiny Shakespeare（~1MB）上跑几百步，loss 显著下降，能续写出"看起来像莎士比亚"的字符序列。

跨平台：3060 ~30 秒；CPU/Mac ~3-5 分钟。
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from Echo.shared.device import get_device
from importlib import import_module
MiniGPT = import_module("03_model").MiniGPT


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILE = DATA_DIR / "tinyshakespeare.txt"


def load_dataset() -> str:
    DATA_DIR.mkdir(exist_ok=True)
    if not DATA_FILE.exists():
        print(f"下载 Tiny Shakespeare → {DATA_FILE} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)  # 一次性下载，~1MB
    return DATA_FILE.read_text(encoding="utf-8")


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """从 data 里随机采样 batch_size 个长度 block_size 的窗口。"""
    # torch.randint(low, high, size)：在 [low, high) 上均匀采样整数；留 1 给 target shift
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    # torch.stack：沿"新建的"第 0 维堆叠 list of tensors，得到 (B, block_size)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])  # 右移 1 位作 target
    return x.to(device), y.to(device)


def main() -> None:
    device = get_device()
    print(f"device: {device}")
    torch.manual_seed(0)

    # ---- 数据：char-level，~65 个唯一字符 ----
    text = load_dataset()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    data = torch.tensor(encode(text), dtype=torch.long)
    print(f"语料长度: {len(text):,} 字符, 词表大小: {vocab_size}")

    # ---- 模型：~1M 参数（小到 CPU 也能玩，大到能学出 shakespeare 韵味）----
    block_size = 64
    model = MiniGPT(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=4, max_len=block_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 注意：weight tying 让两个 .parameters() 共享，重复计了 token_emb 一次
    n_params_unique = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    print(f"模型参数: {n_params_unique:,}（去重后）")

    # ---- 训练 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    n_steps = 500
    batch_size = 32
    eval_every = 100

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, n_steps + 1):
        xb, yb = get_batch(data, batch_size, block_size, device)
        logits = model(xb)                                             # (B, T, V)
        # cross_entropy 要求 (N, V) 与 (N,)，把 batch 和时间维 flatten
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ch04 §1.3 的事后救火
        optimizer.step()
        losses.append(loss.item())
        if step % eval_every == 0 or step == 1:
            recent = sum(losses[-eval_every:]) / min(eval_every, len(losses))
            print(f"step {step:4d}  loss {loss.item():.4f}  avg {recent:.4f}")

    elapsed = time.time() - t0
    print(f"\n训练耗时: {elapsed:.1f}s, 平均 {elapsed * 1000 / n_steps:.1f} ms/step")

    # ---- 验证：loss 必须显著下降 ----
    avg_first = sum(losses[:50]) / 50
    avg_last = sum(losses[-50:]) / 50
    print(f"\nloss 前 50 步均值: {avg_first:.4f}")
    print(f"loss 后 50 步均值: {avg_last:.4f}")
    assert avg_last < avg_first - 0.5, f"训 500 步 loss 应显著下降，实际 {avg_first:.3f} → {avg_last:.3f}"

    # ---- 续写示例 ----
    prompt = "ROMEO:"
    # encode → list[int] → tensor，dtype=long 因为后续要走 nn.Embedding 必须是 LongTensor 索引
    # .unsqueeze(0)：在最前加一维 → (n,) → (1, n)，凑出 batch 维
    ids = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    out_ids = model.generate(ids, max_new_tokens=200)[0].tolist()
    print(f"\n=== 续写示例（贪心，前 200 字符）===")
    print(decode(out_ids))
    print("=" * 50)
    print("\n注：贪心解码 + 500 步欠训练 → 输出极易陷入 'the the the' 重复循环。")
    print("    这不是 bug，是预期现象。ch07 会引入 top-k / top-p / temperature 解决。")
    print("    想看更像 shakespeare 的输出可以把 n_steps 调到 3000+（3060 ~30s）。")

    print("\nPASS")


if __name__ == "__main__":
    main()
