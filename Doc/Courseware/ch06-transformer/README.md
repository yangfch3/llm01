# ch06 · Transformer 架构

> ch05 把 "心脏"（attention）拆开看了。本章把整副 "骨架" 装起来：位置编码 + 残差 + LayerNorm + FFN + lm_head，凑出一个能跑能训的 Decoder-only Transformer。
> 
> **这是 echo-mini 的原型**——本章 `MiniGPT` 类放大 10 倍换上 BPE（Byte-Pair Encoding，字节对编码）分词器，就是 M4 要做的事。
>
> Reading 部分有《图解 Transformer》可扩展阅读。

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

原始 Transformer 有 Encoder 和 Decoder 两部分。名字来自信息论隐喻：**Encoder 把输入"编码"成紧凑的内部表示（理解），Decoder 从表示"解码"出目标序列（生成）**。

具体来说：Encoder 双向（每个 token 能看左右所有 token）看完整个输入生成上下文向量，Decoder 拿着这些向量单向（每个 token 只能看它左边已有的 token）逐 token 生成输出。

现代 LLM 几乎全是 **Decoder-only**，只保留 Decoder（自回归生成）堆叠，把 "理解" 和 "生成" 合并到同一个续写过程中完成。原因：

1. **范式统一**：把输入和输出都当作 token 序列的续写，问答/翻译/写代码用同一种 next-token-prediction（给定前文预测下一个 token）解决，不需要区分"编码端"和"解码端"。
2. **架构简单**：砍掉 Encoder 和 cross-attention（Decoder 用来读取 Encoder 输出的注意力），只保留 causal self-attention + FFN，训练目标一个 loss 打天下。
3. **参数效率高**：同等参数预算全集中在同一组重复堆叠的 Block 里而非分散到两套子网络，模型做大后表现优于同规模 Encoder-Decoder（GPT-3 / LLaMA / Qwen 等实证）。

因此本章以及后续的 echo-mini / echo 都只涉及 Decoder-only。下面看它训练的前向传播的完整数据流：

```
input_ids: (B, n)      ← B 条长度为 n 的 token id 序列（B = batch size）
   ↓ token embedding (V × d)      ← 查表：每个整数 id → d 维向量
   ↓ + 位置编码（绝对正弦 / RoPE 等）
x: (B, n, d)
   ↓
┌─────────────── Block × N ───────────────┐
│  x → LN → MHA(causal) → +x  (残差 1)   │
│  x → LN → FFN         → +x  (残差 2)   │
└─────────────────────────────────────────┘
   ↓
final LN
   ↓ lm_head: Linear(d → V)  ← lm_head（Language Model Head）：把隐状态映射为词表上的预测分布
logits: (B, n, V)
   ↓ shift + cross_entropy   ← shift：logits 和真实 token 错开一位，让每个位置的输出预测下一个 token
loss
```

数据流就这几步。这一章每一节解释**其中一步为什么这样设计**。

---

## 2. 位置编码

### 2.1 为什么需要

Attention 是**集合操作**——`softmax(QK^T/√d)V` 把 token 当成无序集合处理，打乱输入顺序结果不变，它只看内容相似度，不知道谁在前谁在后。"我爱你"和"你爱我"对它一样。语言显然不行。必须把"位置"信息塞进去。

三大流派：

| 方案 | 怎么塞 | 代表模型 |
|---|---|---|
| 绝对正弦位置编码 | 加在 embedding 上 | 原版 Transformer、BERT 早期 |
| 学习式位置 embedding | 加在 embedding 上，可训练 | GPT-2、BERT |
| **RoPE（Rotary Position Embedding，旋转位置编码）** | 在 Q、K 上做旋转 | LLaMA、Qwen、GLM、几乎所有现代 LLM |

> 想直观看到 embedding 加上位置编码前后的数值变化，可运行练习 `Playground/ch06-transformer/01_pos_encoding.py`。

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

做法：把 d 维 Q（或 K）按相邻两维分组成 d/2 个 2D 向量，对每个 2D 向量做平面旋转——旋转角度由位置决定。对位置 `p` 的第 `i` 个 2D 组，乘一个旋转矩阵（作用：把该 2D 向量绕原点旋转 `pθ_i` 角度）：

\[
R_{p, i} = \begin{pmatrix} \cos(p\theta_i) & -\sin(p\theta_i) \\ \sin(p\theta_i) & \cos(p\theta_i) \end{pmatrix},
\quad \theta_i = 10000^{-2i/d} \text{（和正弦编码同一组频率）}
\]

旋转后 Q·K 算点积时，神奇的事发生：

\[
(R_p q) \cdot (R_m k) = q^\top R_{m-p} k
\]

——结果**只依赖相对距离 `m-p`**。这就是 RoPE 比绝对位置编码强的根本原因：注意力天然具备相对位置感知。

工程实现就是对 Q、K 做几次三角函数运算（实际代码常用复数乘法作为等价高效写法，理解上不需要复数知识），没有新增可学习参数。echo-mini 和 echo 都用 RoPE。完整推导见 RoFormer 论文。

### 自检

1. 为什么不能把 embedding 直接当成 "位置 0,1,2..." 一并加进去？
2. RoPE 相比正弦绝对位置编码，最大的优势是什么？

<details markdown="1">
<summary>答案速查</summary>

1. 那等于把 token 含义和位置混在一个维度里，数值范围从 [-1, 1] 直接漂移到 [0, n]，破坏 embedding 的方差结构，attention 也分不清"加了 5 是因为这个词重要还是因为它在第 5 位"

2. attention 计算时直接看到相对距离 `m-p`，**不是绝对位置 p 和 m 各自**。这让模型的"位置感"具备平移不变性：训练时见过 `(1,3)` 距离 2 的对，推理时看到 `(100,102)` 距离 2 的对也能复用知识；也是它能轻松外推到长上下文（NTK（Neural Tangent Kernel，神经切线核）/ YaRN（Yet another RoPE extensioN，RoPE 长上下文扩展方法）等技巧）的基础

</details>

---

## 3. Pre-LN vs Post-LN

ch04 §6.3 留过悬念，这里兑现。

### 3.1 两种放法

```
Post-LN（原版 Transformer，Vaswani 2017）：
  x → MHA → +x → LN → FFN → +x → LN

Pre-LN（GPT-2 起的现代主流）：
  x → LN → MHA → +x → LN → FFN → +x
```

差别就一个：**LN 在残差加法之前还是之后**。

用主干/分支视角对比一个 Block：

```
Post-LN（LN 在主干上，主干被 LN 打断）：
主干：x ─────────── +x → LN ─────────── +x → LN
                  ↑                   ↑
分支：    x → MHA ─┘          x → FFN ─┘

Pre-LN（LN 在分支里，主干是纯加法）：
主干：x ─────────── +x ─────────── +x
                  ↑               ↑
分支： x → LN → MHA ┘  x → LN → FFN ┘
```

主干 = x 原封不动传过去做加法的路径；分支 = 从 x 岔出去经过处理再加回来。

### 3.2 为什么 Pre-LN 训得稳

直觉：Post-LN 中，残差加完之后才做 LN——残差路径上的信号每过一层都被 LN 归一化一次。LN 不只是缩放：它有可学习参数 γ/β（归一化后的缩放/偏移，ch04 §6 讲过），相当于每层都对残差信号做一次非线性变换。深堆几十层后，梯度反传时必须逐层穿过这些 LN 的导数（Jacobian，即多元函数的导数矩阵），容易累积衰减或爆炸，训练初期需要精细 warmup 才能收敛。

Pre-LN 把 LN 放在分支里（MHA/FFN 之前），而不是主干上——残差路径变成纯加法，从第一层到最后一层不经过任何归一化。梯度反传时可以沿残差路径直通回去，不会被 LN 的 Jacobian 逐层衰减，所以深层网络也能训稳。

代价：残差路径上每层都是纯加法（分支输出叠加上去），没有 LN 压制方差，输出方差随层数累积。因此 Pre-LN 模型最末必须**额外加一个 final LN** 把输出拉回正常尺度。

### 3.3 现状

GPT-2、GPT-3、LLaMA 1/2/3、Qwen、几乎一切 ≥2020 年的 LLM 全是 **Pre-LN**。echo-mini 与 echo 跟随。

---

## 4. Transformer Block

Transformer 的主体是 N 个相同结构的 Block 重复堆叠。一个 Block 是模型的最小重复单元，内含一组 MHA + FFN + 残差 + LN。工业界说"32-layer Transformer"里的 "32-layer" 就是指 32 个 Block（如 LLaMA-2-7B），不是指 32 个 Linear。

```
def block(x):                    # x: (B, n, d)
    h = x + mha(layernorm(x))    # 残差 1：注意力
    h = h + ffn(layernorm(h))    # 残差 2：FFN
    return h
```

每个 Block 两条残差通路。MHA 用 ch05 写的（带 causal mask）。FFN 接下来讲。

---

## 5. FFN（前馈网络）

### 5.1 为什么需要 FFN

attention 解决的是"看哪些 token、看多少"——它把不同位置的信息加权汇聚起来。但汇聚完之后，每个位置拿到的只是其他 token 的加权平均，缺少对信息的深层加工。

FFN 解决的是"拿到信息后怎么处理"——它对每个位置独立做非线性变换，把汇聚来的信息进一步提炼、组合成更抽象的特征。

两者互补：

- 只有 attention 没有 FFN：模型只是反复在 token 之间混合信息，无法深层加工
- 只有 FFN 没有 attention：每个 token 闷头变换自己，看不到上下文

类比：attention 像"每个人听取其他人的发言并汇总"，FFN 像"每个人独立消化加工自己拿到的信息"。

经验：FFN 占整个 Transformer **2/3 参数量**（§7 会算出来），它是模型存储事实知识的主要载体——可以简单印象化理解为 attention 负责"查找"，FFN 负责"记忆"。

### 5.2 结构

\[
\mathrm{FFN}(x) = W_2 \cdot \mathrm{GELU}(W_1 x + b_1) + b_2
\]

即：本质为两 Linear 层，输入 x (d 维) → W1 升维到 d_ff 维 → GELU 激活 → W2 降回 d 维。先展开再压缩，中间的非线性（GELU）是加工的关键。

形状：`d → d_ff → d`，其中 **d_ff 通常等于 4d**（GPT-2/3 经验），两个 Linear（W1: d×4d, W2: 4d×d）共 8d² 参数。本章先用这种最朴素的 GELU 双层版。

> 补充：LLaMA 系改用 SwiGLU（Swish-Gated Linear Unit，Swish 门控线性单元）变种，三个矩阵（gate/up/down），取 `d_ff ≈ 8d/3` 使总参数量与标准版持平（≈ 8d²）。

### 5.3 GELU vs ReLU

GELU = `x · Φ(x)`，其中 Φ 是标准正态（μ=0, σ=1）的 CDF（Cumulative Distribution Function，累积分布函数），Φ 的值域为 (0,1)。直觉：ReLU 的"软化版"，负半轴不是硬砍而是逐渐衰减。GPT-1/2/3 使用 GELU。

```python
import torch.nn.functional as F
F.gelu(x)              # 直接用，PyTorch 原生
```

> 补充：2022 年后主流开源 LLM（PaLM、LLaMA、Mistral、Qwen 等）几乎全面转向 SwiGLU 等 GLU 变体，GELU 双层版已退居"经典教学结构"。

---

## 6. Emb 与 lm_head 共享 weight

lm_head（Language Model Head）是模型最后一层 `Linear(d → V)`，负责把每个位置的 d 维隐状态映射到词表大小 V 维的 logits，再经 softmax 得到下一个 token 的概率分布。

token embedding 形状 `(V, d)`，lm_head 形状 `(d, V)`——是同一个矩阵的转置。共享：

```python
self.lm_head.weight = self.token_emb.weight
```

省 `V × d` 参数（V=32k、d=768 → ~2500 万参数，能省一大笔），还轻微提升性能（输入空间和输出空间共享几何）。GPT-2 原版就这么做。

现状：大模型（≥7B）主流已**不共享**（LLaMA、Mistral、Qwen 等）。原因是词表变大后，输入端和输出端对 embedding 的需求分化——强制共享反而互相制约；且模型够大时多出的 V×d 参数占比很小。小模型（≤1B）参数预算紧张，共享仍是合理选择。echo-mini 用共享。

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
        self.max_len   = max_len
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

    # 逐 token 自回归生成（详见 ch07）
    def generate(self, ids, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self.forward(ids[:, -self.max_len:])  # 截断到最大上下文长度
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)  # 贪心：取概率最大的 token
            ids = torch.cat([ids, next_id], dim=1)
        return ids
```

以上 `forward` 对应 §1 全景图中 input_ids → logits 的部分。`generate` 是推理环节的逻辑（什么是自回归生成、为什么贪心不够、更好的解码策略，均在 ch07 展开），此处仅给出最朴素的贪心版供练习 03/04 验证模型能跑。

对于全景图中的从 logits 到 loss，在练习代码中可见：

```python
# 03_model.py
loss = F.cross_entropy(
   logits[:, :-1].reshape(-1, V),  # (B*(n-1), V)
   targets[:, 1:].reshape(-1),     # (B*(n-1),)
)
```

训练时把 `logits[:, :-1]` 与 `ids[:, 1:]` 做 shift 错位对齐后算 cross_entropy——这就是 ch09 要正经讲的 **CLM（Causal Language Modeling，因果语言建模）目标**。

---

## 9. 练习

落到 `Playground/ch06-transformer/`：

| 脚本 | 内容 |
|---|---|
| `01_pos_encoding.py` | 正弦位置编码 vs RoPE 对比；验证 RoPE 让 Q·K 只依赖相对距离 |
| `02_block.py` | 单个 Pre-LN Block 实现，shape / 梯度健康度检查 |
| `03_model.py` | 完整 `MiniGPT` 类（~1M 参数），forward + 朴素贪心 generate |
| `04_train_shakespeare.py` | char-level tiny shakespeare 过拟合训练，loss 显著下降，能续写 |

数据：tiny shakespeare（~1MB）首次运行自动下载到 `Playground/ch06-transformer/data/`（已被 `.gitignore` 排除）。3060 12GB 上 ~30 秒训完，CPU/Mac 上 3-5 分钟。

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
