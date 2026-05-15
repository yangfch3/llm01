"""ch08 练习 2：用 HF tokenizers 训一个小 byte-level BPE。

语料用 ch06 已下载的 tinyshakespeare（~1MB 英文）。词表 ~2000。
跑完观察：
  - 学到了哪些 token（高频词被合成单 token，罕见词拆成子串）
  - 编解码 round-trip 正确
  - 与 gpt2 BPE 对比同一句的 token 数差异
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "Playground" / "ch06-transformer" / "data" / "tinyshakespeare.txt"
OUT_PATH = Path(__file__).parent / "shakespeare_bpe.json"

VOCAB_SIZE = 2000


def train_bpe(corpus_path: Path, out_path: Path, vocab_size: int) -> Tokenizer:
    """训一个 byte-level BPE 并保存到 out_path。"""
    # models.BPE：BPE 模型本体，初始可空，靠 trainer 喂数据填充
    tok = Tokenizer(models.BPE(unk_token=None))
    # pre-tokenizer：进 BPE 前的预切分
    # ByteLevel(add_prefix_space=False)：把字符流转成 byte 流，每个 byte 映射成可见 unicode
    # 这一步保证「最小单元是 byte」，所以无 OOV
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # decoder 必须与 pre_tokenizer 配套，否则解码出来是 byte 字符乱码
    tok.decoder = decoders.ByteLevel()

    # BpeTrainer：BPE 训练器
    # vocab_size：目标词表大小（含 special token + 256 个 byte）
    # initial_alphabet：强制把所有 256 个 byte 加进起始词表，否则未在语料出现的 byte 永远学不到
    # special_tokens：保留 token，不会被合并；放在词表最前
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<pad>", "<bos>", "<eos>"],
        show_progress=False,
    )
    # train(files, trainer)：按文件流式训练，files 接受路径列表
    tok.train([str(corpus_path)], trainer)

    tok.save(str(out_path))
    return tok


def main() -> None:
    print(f"语料: {CORPUS_PATH} ({CORPUS_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"目标词表: {VOCAB_SIZE}\n")

    tok = train_bpe(CORPUS_PATH, OUT_PATH, VOCAB_SIZE)
    actual = tok.get_vocab_size()
    print(f"训练完成。实际词表 = {actual}（含 256 byte + 3 special + 合并产物）")
    print(f"保存到: {OUT_PATH}\n")

    # 看几个学到的高频 token：从词表里挑 id 较大（= 较晚合并 = 较长子串）的几个
    vocab = tok.get_vocab()  # dict: token_str → id
    # 按 id 倒序，取前 20 个（id 越大表示训练越后期才合成 → 越长越完整的 n-gram）
    top = sorted(vocab.items(), key=lambda kv: -kv[1])[:20]
    print("[学到的较晚合并 token]（id 越大 = 合并越晚 = 越长）")
    for t, i in top:
        print(f"  id={i:4d}  token={t!r}")

    # 编解码示例
    sample = "ROMEO: But, soft! what light through yonder window breaks?"
    enc = tok.encode(sample)
    print(f"\n[示例] {sample!r}")
    print(f"  token 数 = {len(enc.tokens)}")
    print(f"  tokens = {enc.tokens}")
    print(f"  ids    = {enc.ids}")

    # 验证 1：round-trip
    decoded = tok.decode(enc.ids)
    assert decoded == sample, f"round-trip 失败:\n  原文 {sample!r}\n  解码 {decoded!r}"
    print(f"\nPASS: round-trip 一致 → {decoded!r}")

    # 验证 2：与 gpt2 50k 词表对比同一句的 token 数
    try:
        gpt2 = Tokenizer.from_pretrained("gpt2")
        gpt2_tokens = gpt2.encode(sample).tokens
        print(f"\n[对比] 同一句:")
        print(f"  本章 2k BPE  → {len(enc.tokens)} tokens")
        print(f"  gpt2 50k BPE → {len(gpt2_tokens)} tokens")
        print("  词表越大，平均 token 数越少（更多 n-gram 合并成单 token）")
    except Exception as e:  # noqa: BLE001
        print(f"\n（跳过 gpt2 对比：{e}）")


if __name__ == "__main__":
    main()
