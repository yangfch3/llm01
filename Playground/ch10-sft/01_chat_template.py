"""ch10 练习 1：ChatML 对话模板渲染与反解。

演示三件事：
  1. 把多轮 (role, content) 渲染成 ChatML 字符串
  2. add_generation_prompt：训练样本与推理输入的差异
  3. 把渲染后的字符串反解回 turns（演示 special token 是边界这件事）

不依赖 HF transformers，纯字符串操作，体感为主。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


@dataclass
class Turn:
    role: str  # system / user / assistant
    content: str


def render_chatml(turns: list[Turn], add_generation_prompt: bool = False) -> str:
    """把一系列 turn 渲染成 ChatML 文本。

    add_generation_prompt=True 时末尾追加 `<|im_start|>assistant\\n`，
    告诉模型"该你说话了"——推理时必开，训练时不开。
    """
    parts: list[str] = []
    for t in turns:
        # 每个 turn 形如：<|im_start|>role\n{content}<|im_end|>\n
        parts.append(f"{IM_START}{t.role}\n{t.content}{IM_END}\n")
    if add_generation_prompt:
        # 末尾不闭合，停在 assistant 前缀，让模型从下一 token 开始生成
        parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def parse_chatml(text: str) -> list[Turn]:
    """把 ChatML 文本反解回 turns。仅处理已闭合的 turn，未闭合的 generation prompt 忽略。"""
    # re.findall 用非贪婪 .*? + DOTALL 匹配多行 content
    # 模式：<|im_start|>{role}\n{content}<|im_end|>
    pattern = re.escape(IM_START) + r"(\w+)\n(.*?)" + re.escape(IM_END)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return [Turn(role=role, content=content) for role, content in matches]


def main() -> None:
    turns = [
        Turn("system", "你是一个简洁的助手。"),
        Turn("user", "帮我写一首关于秋天的诗"),
        Turn("assistant", "秋风扫落叶，孤雁向南飞。"),
        Turn("user", "再来一首"),
        Turn("assistant", "层林尽染霜，归雁过寒江。"),
    ]

    # 1. 训练样本：完整对话，所有 turn 都闭合
    train_text = render_chatml(turns, add_generation_prompt=False)
    print("=" * 60)
    print("[训练样本] 完整对话，所有 <|im_end|> 都到位（含末尾）")
    print("=" * 60)
    print(train_text)
    print(f"末尾字符 repr: {train_text[-20:]!r}\n")

    # 2. 推理输入：只到第二个 user 问完，追加 generation prompt
    infer_turns = turns[:-1]  # 砍掉最后一个 assistant，让模型自己生成
    infer_text = render_chatml(infer_turns, add_generation_prompt=True)
    print("=" * 60)
    print("[推理输入] 末尾用 generation prompt 收口，等模型续写")
    print("=" * 60)
    print(infer_text)
    print(f"末尾字符 repr: {infer_text[-30:]!r}\n")

    # 3. 训练 vs 推理的关键差异
    diff = train_text[len(infer_text):]
    print("=" * 60)
    print("[差异] 推理输入 → 训练样本，模型需要在续写中生成的 token：")
    print("=" * 60)
    print(repr(diff))
    print("→ 模型必须学会输出最后一个 assistant 的 content + <|im_end|> + \\n\n")

    # 4. 反解验证 round-trip
    parsed = parse_chatml(train_text)
    print("=" * 60)
    print(f"[反解] 拆出 {len(parsed)} 个 turn：")
    print("=" * 60)
    for i, t in enumerate(parsed):
        print(f"  turn {i}  role={t.role:9s}  content={t.content!r}")
    assert len(parsed) == len(turns)
    for orig, got in zip(turns, parsed, strict=True):
        assert orig.role == got.role and orig.content == got.content
    print("\nPASS: 渲染 → 反解 round-trip 内容一致\n")

    # 5. 反解推理输入：未闭合的 generation prompt 不会被解析为完整 turn
    parsed_infer = parse_chatml(infer_text)
    print(f"[反解推理输入] 拆出 {len(parsed_infer)} 个 turn（最后的未闭合 assistant 不计）")
    assert len(parsed_infer) == len(infer_turns)
    print("→ 这正是 generation prompt 的设计：用'未闭合'状态告诉模型该说话了\n")


if __name__ == "__main__":
    main()
