"""ch08 练习 1：三种分词粒度对比。

同一段中英混合文本，分别用：
  - 字符级（直接 list(text)）
  - 词级（按空格切，中文整段当一"词"）
  - GPT-2 byte-level BPE（HF tokenizers）
  - Qwen2.5 BPE（中英均衡语料训练，词表 ~150k，中文压缩率更优）

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


def tokenize_bpe_gpt2(text: str) -> tuple[list[str], list[int]]:
    """GPT-2 byte-level BPE。返回 (token 字符串, id 列表)。"""
    # gpt2 是 byte-level BPE，词表 50257，英文语料为主
    tok = Tokenizer.from_pretrained("gpt2")
    enc = tok.encode(text)
    return enc.tokens, enc.ids


def tokenize_bpe_qwen(text: str) -> tuple[list[str], list[int]]:
    """Qwen2.5 BPE（中英均衡语料训练，词表 ~150k）。返回 (token 字符串, id 列表)。"""
    # Qwen2.5 词表对中文覆盖好，常见汉字/词组被合并为单 token
    tok = Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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

    # GPT-2 BPE
    bpe_tokens, bpe_ids = tokenize_bpe_gpt2(SAMPLE)
    print(f"[GPT-2 BPE]  token 数 = {len(bpe_tokens)}")
    print(f"  前 20 个 token: {bpe_tokens[:20]}")
    print(f"  前 20 个 id:    {bpe_ids[:20]}")
    # GPT-2 BPE 把不可显示 byte 映射成可见 unicode（如 'Ġ' 表示空格前缀），
    # 中文字符通常被拆成 2-3 个 byte token（如 '是' → 'æ','ĺ','¯'）
    print("  观察：英文常见词整词成 token；中文字按 utf-8 拆成多 byte token\n")

    # Qwen2.5 BPE
    qwen_tokens, qwen_ids = tokenize_bpe_qwen(SAMPLE)
    print(f"[Qwen2.5 BPE]  token 数 = {len(qwen_tokens)}")
    print(f"  前 20 个 token: {qwen_tokens[:20]}")
    print(f"  前 20 个 id:    {qwen_ids[:20]}")
    print("  观察：中文常见字/词被合并为单 token，不再拆成 byte 碎片\n")

    # 压缩率对比
    ratio_gpt2 = len(bpe_tokens) / len(chars)
    ratio_qwen = len(qwen_tokens) / len(chars)
    print(f"[压缩率]  GPT-2 BPE token 数 / 字符数 = {ratio_gpt2:.2f}")
    print(f"          Qwen2.5 BPE token 数 / 字符数 = {ratio_qwen:.2f}")
    print("  中文优化后的分词器压缩率明显更好（比值更小）\n")

    # 纯中文片段单独对比
    zh_only = "注意力机制是深度学习的核心创新之一"
    tok_gpt2 = Tokenizer.from_pretrained("gpt2")
    tok_qwen = Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    enc_gpt2 = tok_gpt2.encode(zh_only)
    enc_qwen = tok_qwen.encode(zh_only)
    print(f"[纯中文对比] \"{zh_only}\"")
    print(f"  字符数 = {len(zh_only)}")
    print(f"  GPT-2 BPE  token 数 = {len(enc_gpt2.tokens)}  (ratio = {len(enc_gpt2.tokens)/len(zh_only):.2f})")
    print(f"  Qwen2.5    token 数 = {len(enc_qwen.tokens)}  (ratio = {len(enc_qwen.tokens)/len(zh_only):.2f})")
    print(f"  GPT-2 tokens:  {enc_gpt2.tokens[:15]}...")
    print(f"  Qwen2.5 tokens: {enc_qwen.tokens[:15]}...")

    # 验证：BPE 编解码 round-trip 必须可逆
    decoded_gpt2 = tok_gpt2.decode(bpe_ids)
    decoded_qwen = tok_qwen.decode(qwen_ids)
    assert decoded_gpt2 == SAMPLE, f"GPT-2 round-trip 失败：\n  原文: {SAMPLE!r}\n  解码: {decoded_gpt2!r}"
    assert decoded_qwen == SAMPLE, f"Qwen round-trip 失败：\n  原文: {SAMPLE!r}\n  解码: {decoded_qwen!r}"
    print("\nPASS: GPT-2 & Qwen2.5 round-trip 编解码一致")


if __name__ == "__main__":
    main()
