# ch06 · Transformer 架构（Decoder-only）

> ch05 把"心脏"（attention）拆开看了。本章把整副"骨架"装起来：
> 位置编码 + 残差 + LayerNorm + FFN + lm_head，凑出一个能跑能训的 Decoder-only Transformer。
> **这是 echo-mini 的原型**——本章 `MiniGPT` 类放大 10 倍换上 BPE 分词器，就是 M4 要做的事。

## 学习目标

1. 能默写 Decoder-only 的完整数据流（token id → loss）
2. 理解位置编码的必要性，能解释 RoPE 与正弦位置编码的本质区别
3. 区分 Pre-LN 和 Post-LN，知道为什么现代 LLM 全走 Pre-LN
4. 能估算给定超参的 Transformer 参数量（不用查表）

## 前置依赖

- ch04 §2 初始化、§6 LayerNorm、§6.3 Pre/Post-LN 的悬念
- ch05 全章（注意力是核心子模块）

---

## 1. 全景图

```
input_ids: (B, n)
   ↓ token embedding (V × d)
   ↓ + 位置编码（绝对正弦 / RoPE 等）
x: (B, n, d)
   ↓
┌─────────────── Block × N ───────────────┐
│  x → LN → MHA(causal) → +x  (残差 1)    │
│  x → LN → FFN          → +x  (残差 2)   │
└─────────────────────────────────────────┘
   ↓
final LN
   ↓ lm_head: Linear(d → V)
logits: (B, n, V)
   ↓ shift + cross_entropy
loss
```

数据流就这几步。这一章每一节解释**其中一步为什么这样设计**。

---

## 2. 位置编码

### 2.1 为什么需要

Attention 是**集合操作**——`softmax(QK^T/√d)V` 把 token 当成无序集合处理，"我爱你"和"你爱我"对它一样。语言显然不行。必须把"位置"信息塞进去。

三大流派：

| 方案 | 怎么塞 | 代表模型 |
|---|---|---|
| 绝对正弦位置编码 | 加在 embedding 上 | 原版 Transformer、BERT 早期 |
| 学习式位置 embedding | 加在 embedding 上，可训练 | GPT-2、BERT |
| **RoPE（旋转位置编码）** | 在 Q、K 上做旋转 | LLaMA、Qwen、GLM、几乎所有现代 LLM |

### 2.2 正弦位置编码

\[
\mathrm{PE}_{(p, 2i)} = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \quad
\mathrm{PE}_{(p, 2i+1)} = \cos\!\left(\frac{p}{10000^{2i/d}}\right)
\]

`p` 是位置，`i` 是维度对的索引。**核心性质**：不同维度对用不同频率的正弦/余弦——位置 `p` 的编码是一个独特的"频率指纹"。

> 10000 是人为选的基数。指数 `2i/d` 让不同维度对的频率从高频（i=0，周期 ≈ 2π）到低频（i=d/2，周期 ≈ 2π·10000）均匀分布在对数尺度上，类似二进制计数器的不同位——低位变化快、高位变化慢，组合起来给每个位置一个独特编码。

为什么这么设计？因为 `sin(a+b)`、`cos(a+b)` 可以用 `sin(a), cos(a), sin(b), cos(b)` 线性组合出来——理论上模型能从两个位置编码"算出"它们的相对距离。但**模型实际是否学到、学到多少**全看运气，效果不如直接告诉它。

### 2.3 RoPE 的核心思想

> **数学不熟可直接跳到结论**：RoPE 让 attention 看到的是"相对距离"，不是"绝对位置"。下面的推导只需理解大意。

> 不要把位置"加"到 embedding 上，而是让位置在 attention 计算时**直接出现在 Q·K 的相对距离里**。

把 d 维 Q（或 K）按相邻两维分组成 d/2 个 2D 向量。对位置 `p` 的第 `i` 个 2D 组，乘一个旋转矩阵：

\[
R_{p, i} = \begin{pmatrix} \cos(p\theta_i) & -\sin(p\theta_i) \\ \sin(p\theta_i) & \cos(p\theta_i) \end{pmatrix},
\quad \theta_i = 10000^{-2i/d}
\]

旋转后 Q·K 算点积时，神奇的事发生：

\[
(R_p q) \cdot (R_m k) = q^\top R_{m-p} k
\]

——结果**只依赖相对距离 `m-p`**（推导用到旋转矩阵正交性：`R_p^\top = R_{-p}`，所以 `R_p^\top R_m = R_{m-p}`）。这就是 RoPE 比绝对位置编码强的根本原因：注意力天然具备相对位置感知。

工程实现就是对 Q、K 做几次三角函数 + 复数乘法，没有新增可学习参数。echo-mini 就用 RoPE。完整推导见 RoFormer 论文。

### 自检

1. 为什么不能把 embedding 直接当成 "位置 0,1,2..." 一并加进去？
2. RoPE 相比正弦绝对位置编码，最大的优势是什么？

<details markdown="1">
<summary>答案速查</summary>

1. 那等于把 token 含义和位置混在一个维度里，数值范围从 [-1, 1] 直接漂移到 [0, n]，破坏 embedding 的方差结构，attention 也分不清"加了 5 是因为这个词重要还是因为它在第 5 位"

2. attention 计算时直接看到相对距离 `m-p`，**不是绝对位置 p 和 m 各自**。这让模型的"位置感"具备平移不变性：训练时见过 `(1,3)` 距离 2 的对，推理时看到 `(100,102)` 距离 2 的对也能复用知识；也是它能轻松外推到长上下文（NTK / YaRN 等技巧）的基础

</details>

---

## 3. Pre-LN vs Post-LN

ch04 §6.3 留过悬念，这里兑现。

### 3.1 两种放法

```
Post-LN（原版 Transformer，Vaswani 2017）：
  x → MHA  → +x → LN → FFN → +x → LN

Pre-LN（GPT-2 起的现代主流）：
  x → LN → MHA → +x → LN → FFN → +x
```

差别就一个：**LN 在残差加法之前还是之后**。

### 3.2 为什么 Pre-LN 训得稳

直觉：Post-LN 把残差路径上的输出也归一化了，**残差信号被 LN 的可学习 γ/β 反复重塑**——梯度反传时必须穿过每层 LN 的 Jacobian，深堆几十层后不稳，训练初期需要精细 warmup 才能收敛。

Pre-LN 的残差是**纯加法路径**，从输入到输出有一条不被任何归一化打扰的"高速公路"——梯度可以绕过 LN 沿残差直通，深层网络也能训稳。

代价：模型最终输出方差会随层数累积，所以 Pre-LN 模型最末必须**额外加一个 final LN** 把输出拉回正常尺度。

### 3.3 现状

GPT-2、GPT-3、LLaMA 1/2/3、Qwen、几乎一切 ≥2020 年的 LLM 全是 **Pre-LN**。echo-mini 跟随。

---

## 4. Transformer Block

```
def block(x):                    # x: (B, n, d)
    h = x + mha(layernorm(x))    # 残差 1：注意力
    h = h + ffn(layernorm(h))    # 残差 2：FFN
    return h
```

每个 Block 两条残差通路。MHA 用 ch05 写的（带 causal mask）。FFN 接下来讲。

---

## 5. FFN（前馈网络）

### 5.1 结构

\[
\mathrm{FFN}(x) = W_2 \cdot \mathrm{GELU}(W_1 x + b_1) + b_2
\]

形状：`d → d_ff → d`，其中 **d_ff 通常等于 4d**（GPT-2/3 经验）。LLaMA 系用 `SwiGLU` 变种（gate/up/down 三个 `d × d_ff` 矩阵），取 `d_ff ≈ 8d/3` 使总参数 `3 · d · d_ff ≈ 8d²`，**与标准 d_ff=4d 双层 FFN 的 `2 · d · 4d = 8d²` 等价**（实际实现会把 d_ff 对齐到 256 等硬件友好倍数，如 LLaMA-7B d=4096 对应 d_ff=11008 而非精确的 10923）。本章先用最朴素的 GELU 双层版。

### 5.2 为什么需要 FFN

attention 是"token 之间混合"，FFN 是"每个位置独立做特征变换"。两者互补：

- 没有 FFN：模型只是反复做线性混合，本质是个深的 attention 平均
- 没有 attention：每个 token 闷头变换自己，没有上下文

经验：FFN 占整个 Transformer **2/3 参数量**——它才是模型存知识的主要地方。

### 5.3 GELU vs ReLU

GELU = `x · Φ(x)`，其中 Φ 是标准正态 CDF。直觉：ReLU 的"软化版"，负半轴不是硬砍而是逐渐衰减。GPT 系一律 GELU。

```python
import torch.nn.functional as F
F.gelu(x)              # 直接用，PyTorch 原生
```

---

## 6. Embedding 与 lm_head 共享（weight tying）

token embedding 形状 `(V, d)`，lm_head 形状 `(d, V)`——是同一个矩阵的转置。共享：

```python
self.lm_head.weight = self.token_emb.weight
```

省 `V × d` 参数（V=32k、d=768 → ~2500 万参数，能省一大笔），还轻微提升性能（输入空间和输出空间共享几何）。GPT-2 原版就这么做。

---

## 7. 参数量估算

记号：层数 L、模型维 d、词表 V、d_ff = 4d。

```
embedding (含 tying):  V × d
每个 Block：
  4 个 d×d (Q/K/V/O):      4 d²
  2 个 d×4d (FFN):          8 d²
  小项 (LN/bias):          忽略
  合计：                  ~12 d²
final LN + 其它：          忽略
位置编码：                RoPE 无参数；学习式 PE 多 max_len × d，通常是次要项
总参数 ≈ V·d + L · 12 d²
```

举例：echo-mini 计划 L=8、d=384、V=16k：
`16000·384 + 8·12·384² ≈ 6.1M + 14.2M ≈ 20M`——量级正确。

> **快速估算口诀**：参数量 `≈ 12 · L · d²`（embedding 在词表小时是次要项）。

### 自检

1. 为什么 FFN 占 2/3 参数？
2. Pre-LN 模型为什么必须在最后再加一个 LN？

<details markdown="1">
<summary>答案速查</summary>

1. 每个 Block 注意力部分 4 个 d×d 矩阵共 4d²；FFN 两个 d×4d 矩阵共 8d²。8/(4+8) = 2/3

2. Pre-LN 残差路径上 LN 不再压缩输出方差，每层加完都"放大"一点，深堆几十层后输出尺度可能漂得很远。final LN 把它强制拉回均值 0、方差 1，让 lm_head 的 softmax 不至于饱和

</details>

---

## 8. 完整模型骨架（伪代码）

```python
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len, d_ff=None):
        ...
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)   # 教学简化：学习式位置；echo-mini 实际用 RoPE，见练习 01
        self.blocks    = nn.ModuleList([Block(...) for _ in range(n_layers)])
        self.ln_final  = nn.LayerNorm(d_model)
        self.lm_head   = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight       # weight tying

    def forward(self, ids):                                # ids: (B, n)
        B, n = ids.shape
        pos = torch.arange(n, device=ids.device)
        x = self.token_emb(ids) + self.pos_emb(pos)        # (B, n, d)
        for blk in self.blocks: x = blk(x)
        x = self.ln_final(x)
        return self.lm_head(x)                             # (B, n, V)
```

训练时把 `logits[:, :-1]` 与 `ids[:, 1:]` 算 cross_entropy——这就是 ch09 要正经讲的 **CLM 目标**。

---

## 9. 练习

落到 `Playground/ch06-transformer/`：

| 脚本 | 内容 |
|---|---|
| `01_pos_encoding.py` | 正弦位置编码 vs RoPE 对比；验证 RoPE 让 Q·K 只依赖相对距离 |
| `02_block.py` | 单个 Pre-LN Block 实现，shape / 梯度健康度检查 |
| `03_model.py` | 完整 `MiniGPT` 类（~1M 参数），forward + 朴素贪心 generate |
| `04_train_shakespeare.py` | char-level tiny shakespeare 过拟合训练，loss 显著下降，能续写 |

数据：tiny shakespeare（~1MB）首次运行自动下载到 `Playground/ch06-transformer/data/`（已被 `.gitignore` 排除）。3060 上 ~30 秒训完，CPU/Mac 上 3-5 分钟。

## 思考题

1. 为什么 LLaMA 用 RMSNorm 替换 LN？省的那点计算真有意义吗？
2. weight tying 一定是正收益吗？什么场景下可能不共享更好？
3. 把 `d_ff` 从 4d 提到 8d，参数量会怎么变？性能呢？

## 参考资料

- **Vaswani et al., "Attention is All You Need"**：原版 Transformer
- **Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2)**：Decoder-only + Pre-LN 的现代范式
- **Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"**：RoPE 原论文
- **Karpathy, "nanoGPT" / "Let's build GPT"**：本章练习的精神向导
- **Xiong et al., "On Layer Normalization in the Transformer Architecture"**：Pre-LN vs Post-LN 的稳定性分析
