"""ch12 练习 2：手写 loglikelihood scoring，模拟 MMLU/C-Eval 多选题打分。

LLM 不是分类器，benchmark 用 loglikelihood scoring 把"选哪个"变成
"4 个候选答案谁的条件概率最高"。

本脚本做三件事：
  1. 构造一道 4 选 1 题，对每个候选算 log P(候选 | prompt)，看模型选哪个
  2. 对比 zero-shot 与 1-shot：示范一条带答案的样例后，模型能否更稳地"按格式"答
  3. 改 prompt 模板（"答案：" vs "Answer:" vs "正确答案是 "），看分数怎么漂

依赖 GPT-2 small（与 01_ppl.py 共用，已下载则不重复）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Echo.shared.device import get_device  # noqa: E402


# 玩具题：选项里只有 B 是正确的（哺乳动物）；用英文，迁就 GPT-2 的训练分布
QUESTION = (
    "Question: Which of the following is a mammal?\n"
    "A. Shark\n"
    "B. Dolphin\n"
    "C. Octopus\n"
    "D. Starfish\n"
)

# few-shot 示范：另一道完整带答案的题，喂给模型"看格式"
FEWSHOT_EXAMPLE = (
    "Question: Which of the following is a fruit?\n"
    "A. Carrot\n"
    "B. Potato\n"
    "C. Apple\n"
    "D. Onion\n"
    "Answer: C\n\n"
)

CHOICES = ["A", "B", "C", "D"]
CORRECT = "B"


@torch.no_grad()
def score_choice(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt: str,
    choice: str,
    device: torch.device,
) -> tuple[float, int]:
    """算 log P(choice | prompt)：把 prompt+choice 拼起来，只把 choice 部分的 logp 加起来。

    返回 (choice 部分总 logp, choice 占的 token 数)。

    注意候选前的空格——分词结果会因此完全不同（GPT-2 BPE 把 " B" 和 "B" 视作不同 token）。
    标准做法是给候选加前导空格再拼。如果不加，BPE 可能把 prompt 末尾的字符与 choice
    合并成同一个 token，导致 choice 占 0 个 token、scoring 失效。
    """
    # 先单独编码 prompt，记录长度，用于切出 choice 对应的位置
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    full_ids = tokenizer.encode(prompt + choice, return_tensors="pt").to(device)
    prompt_len = prompt_ids.size(1)
    n_choice_tokens = full_ids.size(1) - prompt_len
    if n_choice_tokens <= 0:
        # BPE 把 choice 合进了 prompt 末尾 token —— 典型"分词坍塌"
        return float("nan"), 0

    # forward 一次拿全程 logits
    logits = model(full_ids).logits  # (1, L, V)
    # CLM shift：用前 L-1 位预测后 L-1 位
    shift_logits = logits[:, :-1, :]      # (1, L-1, V)
    shift_labels = full_ids[:, 1:]        # (1, L-1)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # gather：取每个位置真实 token 的 logp → (1, L-1)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # choice 部分对应的位置：原序列里 prompt 占 [0, prompt_len)，对应 target 索引 [prompt_len-1, L-1)
    # （即"用 prompt 最后一个 token 预测 choice 第一个 token"开始）
    choice_logps = token_logps[0, prompt_len - 1 :]
    return choice_logps.sum().item(), n_choice_tokens


def evaluate(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt: str,
    suffix_template: str,
    device: torch.device,
    label: str,
) -> None:
    """对 4 个候选打分，打印分数、模型选哪个、是否对。

    suffix_template 形如 " {}"（候选前加空格）或 ".  {}." 等，体现"prompt 模板"对结果的影响。
    """
    print(f"\n  [{label}] prompt 末尾模板: {suffix_template!r}")
    scores = {}
    n_toks = {}
    for c in CHOICES:
        suffix = suffix_template.format(c)
        scores[c], n_toks[c] = score_choice(model, tokenizer, prompt, suffix, device)
    # 若所有候选都坍塌（n_token=0），说明 prompt 末尾与候选拼接出问题
    if all(n == 0 for n in n_toks.values()):
        print("    [警告] 4 个候选都被 BPE 合进了 prompt 末尾 token（分词坍塌）")
        print("    这本身就是一个值得记的坑：prompt 末尾不留空格 + 候选无前导空格 → 无法 scoring")
        return
    # 选 logp 最大的候选（坍塌的设为 -inf 排除）
    valid = {c: s for c, s in scores.items() if n_toks[c] > 0}
    pred = max(valid, key=valid.get)
    for c in CHOICES:
        mark = " ←" if c == pred else ""
        truth = " ★" if c == CORRECT else ""
        if n_toks[c] == 0:
            print(f"    {c}: [坍塌，0 token]{truth}")
        else:
            print(f"    {c}: logp = {scores[c]:>8.3f}  ({n_toks[c]} tok){mark}{truth}")
    ok = "✓" if pred == CORRECT else "✗"
    print(f"    预测 = {pred}  正确 = {CORRECT}  {ok}")


def main() -> None:
    device = get_device()
    print(f"[device] {device}")
    print("加载 GPT-2 small（若 01_ppl.py 跑过则走本地缓存）...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    # ------- 验证 1：zero-shot 多选 -------
    print("\n" + "=" * 64)
    print("[验证 1] zero-shot 多选打分")
    print("=" * 64)
    zero_shot_prompt = QUESTION + "Answer:"
    evaluate(model, tokenizer, zero_shot_prompt, " {}", device, "zero-shot")

    # ------- 验证 2：1-shot 多选 -------
    print("\n" + "=" * 64)
    print("[验证 2] 1-shot 多选（先给一道完整带答案的样例）")
    print("=" * 64)
    one_shot_prompt = FEWSHOT_EXAMPLE + QUESTION + "Answer:"
    evaluate(model, tokenizer, one_shot_prompt, " {}", device, "1-shot")

    # ------- 验证 3：prompt 模板敏感性 -------
    print("\n" + "=" * 64)
    print("[验证 3] 同样的题，换 3 种 prompt 模板看打分变化")
    print("=" * 64)
    for tail, suffix_tmpl, label in [
        ("Answer:",          " {}",  "tail='Answer:' + ' X'"),
        ("Answer: ",         "{}",   "tail='Answer: ' + 'X'（无前导空格）"),
        ("The answer is",    " {}",  "tail='The answer is' + ' X'"),
    ]:
        p = QUESTION + tail
        evaluate(model, tokenizer, p, suffix_tmpl, device, label)

    print("\n" + "=" * 64)
    print("[读图]")
    print("=" * 64)
    print("  - GPT-2 small 仅 124M，且未对齐，常常选错——这是预期")
    print("  - 重点不是分数高低，而是观察：")
    print("    ① 同一道题，prompt 模板小变动 → 候选 logp 显著变化（ch12 §3.2）")
    print("    ② 候选前是否带空格 → 分词结果不同 → 打分不同")
    print("    ③ few-shot 通常会让模型分布更『按格式』，但 124M 太小，效果不稳定")
    print("  - 这就是为什么 lm-evaluation-harness 把 prompt 模板作为『契约』严格固定")


if __name__ == "__main__":
    main()
