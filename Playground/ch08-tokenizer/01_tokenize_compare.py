"""ch08 练习 1：三种分词粒度对比。

同一段中英混合文本，分别用：
  - 字符级（直接 list(text)）
  - 词级（按空格切，中文整段当一"词"）
  - GPT-2 byte-level BPE（HF tokenizers）

观察：token 数、token 形态、对中文的处理差异。
"""

from __future__ import annotations

from tokenizers import Tokenizer

SAMPLE = (
    "Hello world! 这是一段中英混合文本。"
    "Transformers are awesome. 注意力机制 attention is all you need."
)


def tokenize_char(text: str) -> list[str]:
    """字符级：每个字符一个 token，中文每字独立、空格也算一个。"""
    return list(text)


def tokenize_word(text: str) -> list[str]:
    """词级（朴素）：按空白切。中文无空格 → 整段算一个 token，必然 OOV。"""
    return text.split()


def tokenize_bpe(text: str) -> tuple[list[str], list[int]]:
    """GPT-2 byte-level BPE。返回 (token 字符串, id 列表)。"""
    # Tokenizer.from_pretrained 走 HF Hub 拉 tokenizer.json（含 vocab + merges）
    # gpt2 是 byte-level BPE，词表 50257，无 OOV
    tok = Tokenizer.from_pretrained("gpt2")
    enc = tok.encode(text)
    return enc.tokens, enc.ids


def main() -> None:
    print(f"原文 ({len(SAMPLE)} 字符):")
    print(f"  {SAMPLE}\n")

    # 字符级
    chars = tokenize_char(SAMPLE)
    print(f"[字符级]  token 数 = {len(chars)}")
    print(f"  前 20 个: {chars[:20]}")
    print(f"  unique 字符数 = {len(set(chars))}（即所需词表下限）\n")

    # 词级
    words = tokenize_word(SAMPLE)
    print(f"[词级]    token 数 = {len(words)}")
    print(f"  前 10 个: {words[:10]}")
    print("  注意：中文段整体被当成一个 token，真实词表必然爆炸\n")

    # BPE
    bpe_tokens, bpe_ids = tokenize_bpe(SAMPLE)
    print(f"[BPE]     token 数 = {len(bpe_tokens)}")
    print(f"  前 20 个 token: {bpe_tokens[:20]}")
    print(f"  前 20 个 id:    {bpe_ids[:20]}")
    # GPT-2 BPE 把不可显示 byte 映射成可见 unicode（如 'Ġ' 表示空格前缀），
    # 中文字符通常被拆成 2-3 个 byte token（如 '是' → 'æ','ĺ','¯'）
    print("  观察：英文常见词整词成 token；中文字按 utf-8 拆成多 byte token\n")

    # 压缩率对比：BPE / 字符级
    ratio = len(bpe_tokens) / len(chars)
    print(f"[压缩率]  BPE token 数 / 字符数 = {ratio:.2f}")
    print("  英文越多比值越小（BPE 收益越大），纯中文场景比值接近 1（甚至 >1）")

    # 验证 1：BPE 编解码 round-trip 必须可逆
    tok = Tokenizer.from_pretrained("gpt2")
    decoded = tok.decode(bpe_ids)
    assert decoded == SAMPLE, f"round-trip 失败：\n  原文: {SAMPLE!r}\n  解码: {decoded!r}"
    print("\nPASS: BPE round-trip 编解码一致")


if __name__ == "__main__":
    main()
