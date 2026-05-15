"""ch12 练习 1：算 PPL，验证它是个"内部监控指标"而非"模型能力指标"。

做三件事：
  1. 用 GPT-2 small 在两段不同分布的文本上算 PPL，观察分布漂移对 PPL 的影响
  2. 验证 PPL = exp(平均 token NLL) 与 CLM loss 的等价
  3. 演示"跨分词器不可比"——同一段文本用 GPT-2 vs 字符级朴素 LM 算"PPL"差几个数量级

依赖 transformers + GPT-2 small（首次运行约 500MB 下载）。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# transformers 用于加载 GPT-2 small；模型仅用于推理（不训练）
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# 项目统一设备选择：cuda → mps → cpu
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Echo.shared.device import get_device  # noqa: E402

# 两段分布迥异的文本：英文文学 vs 中文新闻
ENGLISH_LIT = (
    "It was the best of times, it was the worst of times, "
    "it was the age of wisdom, it was the age of foolishness, "
    "it was the epoch of belief, it was the epoch of incredulity."
)

CHINESE_NEWS = (
    "新华社北京电  国务院常务会议今日召开，会议研究部署了"
    "进一步推动经济高质量发展的若干举措，强调要坚持稳中求进工作总基调。"
)


@torch.no_grad()
def compute_ppl(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast, text: str, device: torch.device) -> tuple[float, int]:
    """对一段文本算 PPL：返回 (ppl, n_tokens)。

    实现要点：
      - reduction='sum' 后手动除以总有效 token 数，避免 batch 间长度差导致的平均偏差
      - labels 与 input_ids 一致；HF 模型内部自动 shift（input[:-1] 对 labels[1:]）
    """
    # 编码 → tensor，加 batch 维 (1, L)
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    # 模型 forward：传 labels 时 HF 会自动算 CLM loss（已 shift + 平均），
    # 但为了"按 token 求和再除"，这里自己拿 logits 算
    logits = model(ids).logits  # (1, L, V)
    # CLM 对齐：用前 L-1 个位置的 logits 预测后 L-1 个位置的 token
    shift_logits = logits[:, :-1, :].contiguous()  # (1, L-1, V)
    shift_labels = ids[:, 1:].contiguous()         # (1, L-1)
    # cross_entropy 输入要 (N, V) + (N,)；reduction='sum' 拿到总 NLL
    nll_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="sum",
    ).item()
    n_tokens = shift_labels.numel()
    ppl = math.exp(nll_sum / n_tokens)
    return ppl, n_tokens


def char_level_uniform_ppl(text: str) -> float:
    """玩具"字符级朴素 LM"：假设每个字符在词表上均匀分布（最差模型）。

    PPL = 词表大小（即字符 unique 数）。用来对比"PPL 数值跨分词器没意义"。
    """
    vocab = set(text)
    return float(len(vocab))


def main() -> None:
    device = get_device()
    print(f"[device] {device}\n")

    # ------- 加载 GPT-2 small -------
    print("加载 GPT-2 small（首次运行会从 HF Hub 下载约 500MB）...")
    # GPT2TokenizerFast：Rust 实现的 BPE 分词器，与 HF 模型权重配对
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # GPT2LMHeadModel：GPT-2 base + LM head（输出 vocab 维 logits）
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()  # 关 dropout，eval 模式

    # ------- 验证 1：两段分布不同的文本 -------
    print("\n" + "=" * 60)
    print("[验证 1] 同模型，不同语料的 PPL")
    print("=" * 60)
    for name, text in [("英文文学（GPT-2 训练分布内）", ENGLISH_LIT),
                       ("中文新闻（训练分布外）", CHINESE_NEWS)]:
        ppl, n = compute_ppl(model, tokenizer, text, device)
        print(f"  {name}")
        print(f"    token 数 = {n}, PPL = {ppl:.2f}")
    print("\n  → 中文 PPL 远高于英文：GPT-2 训练语料几乎全英文，对中文是分布外。")
    print("    但这并不意味着'GPT-2 在中文上比英文差 N 倍'，因为：")
    print("    ① 中文被 GPT-2 BPE 切成 byte 级碎片，每个 token 信息量小")
    print("    ② PPL 受分词粒度影响极大，跨语料/跨分词器都不可直接比")

    # ------- 验证 2：PPL = exp(loss) -------
    print("\n" + "=" * 60)
    print("[验证 2] PPL 与 HF 模型自带 CLM loss 的等价性")
    print("=" * 60)
    ids = tokenizer.encode(ENGLISH_LIT, return_tensors="pt").to(device)
    with torch.no_grad():
        # 传 labels 让 HF 自己算 loss（内部已 shift，按 token 求平均）
        out = model(ids, labels=ids)
    hf_loss = out.loss.item()
    hf_ppl = math.exp(hf_loss)
    my_ppl, _ = compute_ppl(model, tokenizer, ENGLISH_LIT, device)
    print(f"  HF 内部 loss      = {hf_loss:.4f}  → exp(loss) = {hf_ppl:.4f}")
    print(f"  我们手算 PPL      = {my_ppl:.4f}")
    print(f"  差异（应趋零）     = {abs(hf_ppl - my_ppl):.6f}")
    assert abs(hf_ppl - my_ppl) < 1e-3, "PPL 与 exp(CLM loss) 应等价"
    print("  PASS：PPL 就是 exp(token 平均 CE loss)，不是新指标")

    # ------- 验证 3：跨分词器"PPL"对比是无意义的 -------
    print("\n" + "=" * 60)
    print("[验证 3] 跨分词器比 PPL 没有意义")
    print("=" * 60)
    gpt2_ppl, gpt2_n = compute_ppl(model, tokenizer, ENGLISH_LIT, device)
    naive_ppl = char_level_uniform_ppl(ENGLISH_LIT)
    naive_n = len(ENGLISH_LIT)
    print(f"  GPT-2 BPE 分词           token 数 = {gpt2_n:>3},  PPL = {gpt2_ppl:.2f}")
    print(f"  字符级均匀 LM (玩具)     token 数 = {naive_n:>3},  PPL = {naive_ppl:.2f}")
    print("\n  → 字符级均匀 LM 的 PPL 数字看着可能比 GPT-2 还小（取决于语料），")
    print("    但这只能说明'分子分母都换了'，不能说它比 GPT-2 强。")
    print("    PPL 跨分词器不可直接比，这是 ch12 §1.4 的核心警示。")


if __name__ == "__main__":
    main()
