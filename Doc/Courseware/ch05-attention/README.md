# ch05 · 注意力（Attention）

> M2 起步章。注意力是 Transformer 的心脏，也是 LLM 全部能力的源头。
> 
> 本章只讲"注意力本身"，不碰位置编码、不碰残差、不碰 FFN（Feed-Forward Network，前馈网络）——那些留给 ch06。
> 
> 目标：你看到 `softmax(QK^T/√d_k)V` 这行公式时，每个符号都能讲清楚为什么。

## 学习目标

1. 能用"加权平均"语言解释注意力，知道 Q/K/V 各自代表什么
2. 能徒手写出缩放点积公式，解释 `√d_k` 不是装饰
3. 能区分 padding mask 与 causal mask，知道掩码在 softmax 哪一步生效
4. 理解多头注意力是"拆维度 → 并行算 → concat → 投影"，不是"复制 H 份算 H 遍"

## 前置依赖

- ch02 §3 矩阵乘法 + §4 softmax、ch03 §3 `nn.Module` 写法

---

## 1. 为什么需要注意力

### 1.1 RNN/seq2seq 的痛点

前面几章的 MLP 处理的是固定维度输入。当任务变成"一段话翻译成另一段话"——输入输出都是变长序列——历史上的解决方案是 RNN（Recurrent Neural Network，循环神经网络）组成的 seq2seq 架构。

经典 seq2seq（sequence-to-sequence，序列到序列）用 Encoder-Decoder + RNN 做翻译。RNN 按时间步逐个处理 token，每步把当前输入和上一步的隐状态合并产出新隐状态——本课程不展开 RNN 细节，只需知道它的瓶颈：**整句信息要被压进一个固定向量 h_n**（想象把一本 100 页的书只用一句话转述给别人）。

```
输入: I love NLP
       ↓ Encoder RNN 一步步吃
       ↓
       最后一个 hidden state h_n  ← 整句压成一个固定向量
       ↓
       Decoder RNN 据此生成: 我 爱 自然 语言 处理
```

问题暴露：

- **信息瓶颈**：长句压成一个向量，前面的词必然被稀释
- **顺序依赖**：必须 t-1 算完才能算 t，没法并行
- **长程衰减**：第 50 步看第 1 步要传 49 次梯度，消失/爆炸两难

Bahdanau 2014 的 attention 是修补 RNN 用的（绕开了"只靠 h_n 一个向量传递信息"的瓶颈）："Decoder 每一步可以**回看**所有 Encoder 输出，按需取用"。后来 Vaswani 2017《Attention is All You Need》直接把 RNN 砍了，**只留 attention**——Transformer 诞生。

### 1.2 一句话直觉

> 注意力 = **可微分的字典查找**。

普通字典：给 key `"apple"` → 返回 value。

注意力：给一个 **query**，对所有 keys 算"相似度权重"，把对应 values 加权求和返回。"相似度"是连续的（点积+softmax），所以可微，能反传。

> 注意：这里的 query/key/value 不是 token 本身，而是 token 嵌入经过投影矩阵后的产物（§2.1 详述）。类比里 "apple" 是 token，"你拿着 apple 想查什么"才是 query。

---

## 2. 缩放点积注意力

> 从这里开始，忘掉 RNN。下面讲的是 Transformer 的 self-attention——输入一个序列，输出同长序列，没有时间步、没有 Encoder-Decoder 的区分。

### 2.1 Q/K/V 三元组

设输入序列 `X ∈ R^{n × d}`（n 个 token，每个 d 维）。token 是文本被分词器切出的最小单元（可以是一个字、一个词或子词片段，ch08 详讲）。三个**独立**的可学习投影矩阵：

\[
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
\]

形状：Q、K 是 `(n, d_k)`，V 是 `(n, d_v)`。对应投影矩阵 W^Q、W^K 为 `(d, d_k)`，W^V 为 `(d, d_v)`。工程上常令 `d_v = d_k` 简化。

| 角色 | 类比 | 说明 |
|---|---|---|
| Query (Q) | "我想找什么" | 每个 token 各有一个 query，代表它想关注谁 |
| Key (K) | "我是什么" | 每个 token 各有一个 key，供所有 query 算相似度 |
| Value (V) | "我能贡献什么" | 每个 token 各有一个 value，按权重被取用 |

在 self-attention 里 Q/K/V 都从**同一个 X** 投影出来——同一份输入，三种身份。

### 2.2 公式

\[
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]

语义：每个 token 用自己的 query 去和所有 token 的 key 算相似度，得到权重，再用权重加权混合所有 token 的 value——结果是一个融合了上下文信息的新向量，供后续层进一步处理（ch06 详述完整架构）。

逐步拆：

1. `QK^T`：形状 `(n, n)`（n 为上方提到的输入 X 的 token 长度），第 `(i, j)` 项 = query_i 与 key_j 的点积，即"i 想看 j 的程度"
2. `/√d_k`：缩放（d_k 为每个 key 向量的维度），下面 §2.3 详讲
3. `softmax(行)`：每行归一化成概率，每个 query 对所有 key 的注意力权重和为 1
4. `× V`：用权重加权求和 V，得到形状 `(n, d_v)` 的输出

> 回顾 ch02 §3：点积衡量两个向量的方向接近程度。`QK^T` 的每一项正是这个意思——query_i 和 key_j 方向越近，"i 想看 j"的原始分数越高。

### 2.3 为什么要 √d_k

直觉：d_k 越大，`QK^T` 数值越大 → softmax 越尖锐 → 梯度越接近 0（softmax 饱和区）。

数学：假设 q、k 各分量 i.i.d.（independent and identically distributed，独立同分布：各分量互相独立且服从同一分布）均值 0 方差 1，**且 q 与 k 相互独立**（投影矩阵随机初始化时近似成立），那么 `q·k = Σ q_i k_i` 的方差 = `d_k`。除以 `√d_k` 把方差拉回 1，softmax 输出分布不会过于尖锐，梯度健康。

> 工程结论：**忘了除 √d_k 会训不动**。这是注意力的 Kaiming 初始化级别的"事前防火"。

### 2.4 一个 4 token 玩具示例

```
n=4, d_k=2, d_v=d_k

Q <- XW^Q      K <- XW^K      V <- XW^V
[1.0  1.0]     [1.0  1.0]     [1.0  0.0]
[0.0  1.0]     [0.0  1.0]     [0.0  1.0]
[1.0  0.0]     [1.0  0.0]     [1.0  1.0]
[0.5  0.5]     [0.5  0.5]     [0.5  0.5]

Step 1: QK^T (4×4)，每项 = q_i · k_j
        j=0   j=1   j=2   j=3
i=0    [2.0   1.0   1.0   1.0]    ← q_0·k_0 = 1*1+1*1=2
i=1    [1.0   1.0   0.0   0.5]
i=2    [1.0   0.0   1.0   0.5]
i=3    [1.0   0.5   0.5   0.5]

Step 2: /√d_k = /√2 ≈ /1.41
i=0    [1.41  0.71  0.71  0.71]
i=1    [0.71  0.71  0.00  0.35]
i=2    [0.71  0.00  0.71  0.35]
i=3    [0.71  0.35  0.35  0.35]

Step 3: softmax（逐行归一化，以 token 0 为例，其余行同理）
i=0    [0.40  0.20  0.20  0.20]    ← token 0 对自己关注最多
...

Step 4: output_0 = 加权求和 V （其余行 output_i 同理）
        = 0.40*[1,0] + 0.20*[0,1] + 0.20*[1,1] + 0.20*[0.5,0.5]
        = [0.40+0+0.20+0.10, 0+0.20+0.20+0.10]
        = [0.70, 0.50]            ← token 0 的最终输出向量
        ...
```

### 自检

1. 把 `√d_k` 拿掉，d_k=64 时 softmax 输出大概会变成什么样？
2. self-attention 里 W^Q、W^K、W^V 三个矩阵能合并成一个吗？

<details markdown="1">
<summary>答案速查</summary>

1. logit 方差变成 64 而不是 1，softmax 接近 one-hot（一个值 ≈1 其余 ≈0），梯度几乎全是 0，训不动

2. 不能。三者的"语义角色"不同：Q 表达"想找谁"，K 表达"自己是谁"，V 表达"自己能贡献什么"。共享会强行让"是什么"和"能贡献什么"必须相同，表达力大幅下降。极少数 ALBERT 类工作做过 K=V 共享，效果有损

</details>

---

## 3. 掩码

注意力公式没说"哪些 token 不能看"。但实际场景有两种 token 不该被关注：一是 padding（补位用的占位符，不含信息）；二是未来 token（自回归训练时，看到未来等于作弊）。

掩码就是干这个的：在 softmax **之前**把不该看的位置加上 `-∞`，softmax 后这些位置权重变 0。

### 3.1 Padding mask

batch 里每个序列长度不一样，短的右边补 `<pad>`。pad token 不该被 attend。这里的 logits 即 §2.2 中 `QK^T / √d_k` 的结果（softmax 之前的原始分数）：

```
原始 logits (n=4):  [3.0  0.5  -1.0   2.0]
mask:               [ 1    1    0     0 ]   ← 1=真实token可看，0=pad不可看
应用 mask:          [3.0  0.5  -inf  -inf]
softmax:            [0.92 0.08  0     0 ]
```

实现技巧：用 `logits.masked_fill(mask == 0, float("-inf"))`。

### 3.2 Causal mask（因果掩码）

LLM 训练时**当前 token 只能看自己和之前**，不能偷看未来——否则就是开卷考试，学不到东西。做法：mask 矩阵的上三角为 0（不可看），应用后 logits 矩阵的上三角被填成 `-∞`，softmax 后这些位置权重归零：

```
n=4 的 causal mask（1 = 能看，0 = 不能）：
        j=0 j=1 j=2 j=3
i=0      1   0   0   0       ← 第 0 个 token 只能看自己
i=1      1   1   0   0       ← 第 1 个 token 看 0 和 1
i=2      1   1   1   0
i=3      1   1   1   1
```

PyTorch 一行：

```python
mask = torch.tril(torch.ones(n, n))               # 下三角全 1
logits = logits.masked_fill(mask == 0, float("-inf"))
```

> **关键点**：mask 在 `softmax` **之前**加，不是之后。之后再 mask 已经把概率分给未来了，破坏归一化。

### 3.3 两种 mask 同时存在

实战训练里两者通常**一起**应用：causal mask 防偷看 + padding mask 排除补位。实现上只需合并成一个 mask——某位置只要任一条件说 "不可看"，最终就不可看（实现方式：两个 Mask 矩阵相 AND / 加性 MASK 相加）。

### 自检

1. 为什么 mask 一定要在 softmax 之前加，不能之后？
2. 推理时（自回归生成）每生成一个新 token，causal mask 长什么样？

<details markdown="1">
<summary>答案速查</summary>

1. softmax 后概率已经分配给未来位置了，再置零会让该行概率和不为 1，破坏归一化语义。在 softmax 前加 `-∞`，`exp(-∞)=0`，未来位置自然不参与归一化

2. 推理一步步走，第 t 步只算第 t 行 attention，query 是当前 token、keys/values 是前 t 个 token——形状本来就是 `(1, t)`，**不需要显式 mask**。这就是 KV cache 能省一大笔算力的根本原因（ch07 详讲）

</details>

---

## 4. 多头注意力 MHA（Multi-Head Attention，多头注意力）

### 4.1 动机

单头注意力只能学**一种**注意力模式。但语言里"注意力"是多维的：

- 句法关系（动词 ↔ 主语）
- 共指关系（代词 ↔ 名词）
- 修饰关系（形容词 ↔ 名词）

强行用单头去拟合多种模式 → 互相打架。**让网络并行学好几种**就是多头思想。

### 4.2 不是"复制 H 份算 H 遍"

常见误区：以为多头是用 H 套独立的 d×d 参数各算一遍全维度 attention。错。

**真相**：把 d 维**拆**成 H 段，每段 `d_k = d / H` 维：

```
输入 X: (n, d=512) -> n 行 × 512 列 的矩阵
  ↓ W^Q (512 × 512)
Q: (n, 512)
  ↓ reshape: (n, H=8, d_k=64) → transpose: (H=8, n, d_k=64)
对 H 个头并行算 attention，每个头算出 (n, d_v=64)
  ↓ concat 头维度: (n, H × d_v = 512)
  ↓ W^O (512 × 512) 输出投影
output: (n, 512)
```

**总参数量与单头同维相同**——只是把同一份 d 维表示分给多头分工。参数量没多花一分，**表达力却更强**：每个头在自己的 d_k 维子空间独立学一种注意力模式，互不打架。

> 为什么拆开不会破坏信息？因为 W^Q 本身就是学出来的——网络会自动学会把相关信息投影到同一个头的 64 维子空间里。拆头只是让每组子空间**独立做 softmax**，互不干扰。

### 4.3 公式

\[
\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_H) W^O
\]

\[
\mathrm{head}_i = \mathrm{Attention}(X W^Q_i, X W^K_i, X W^V_i)
\]

工程上不会真的搞 3H 个独立小矩阵，而是用 3 个大矩阵 `W^Q, W^K, W^V`（各 `d × d`）一次投影出来，再 reshape 切头。代码效率与可读性都高。

```python
# 理论写法：3×H=24 个小矩阵，循环 H 次（慢，不实用）
q_heads = [x @ W_Q_list[i] for i in range(H)]   # 每个 (n, d_k)
k_heads = [x @ W_K_list[i] for i in range(H)]
v_heads = [x @ W_V_list[i] for i in range(H)]

# 工程写法：1 个大矩阵一次算完 + reshape 切头（快，实际都这么写）
Q = x @ W_Q                             # (n, d) → (n, d)
Q = Q.reshape(n, H, d_k).transpose(0, 1)  # → (H, n, d_k)
# K、V 同理
```

### 4.4 形状记忆口诀

batch=B、序列长 n、模型维 d、头数 H、每头维 d_k = d/H：

```
X            (B, n, d)
Q,K,V        (B, n, d)             ← 三次 Linear(d → d)
切头后       (B, H, n, d_k)         ← reshape + transpose
QK^T         (B, H, n, n)
softmax      (B, H, n, n)
× V          (B, H, n, d_k)
合头         (B, n, d)              ← transpose + reshape
× W^O        (B, n, d)
```

记住这串形状变换，写 MHA 就是肌肉记忆。

### 自检

1. d=512、H=8 和 d=512、H=64 哪个参数量大？
2. 为什么 MHA 末尾还要一个 `W^O`？

<details markdown="1">
<summary>答案速查</summary>

1. 一样大。Q/K/V/O 四个矩阵各 d×d=512×512，与 H 无关。H 只决定"拆几段"，不影响参数总量

2. concat 出来的 (n, d) 各头之间是"硬拼"的，没有任何信息交互。`W^O` 让各头输出再做一次线性混合，给网络一个"决定怎么融合多头信息"的自由度

</details>

---

## 5. 复杂度速记

| 项 | 复杂度 | 说明 |
|---|---|---|
| 时间 | O(n² · d) | `QK^T` 是 n×n |
| 空间 | O(n²) | attention 矩阵存下来 |
| 参数 | O(d²) | 4 个 d×d 矩阵，与 n 无关 |

n² 是 Transformer 长上下文的核心瓶颈。后来的 FlashAttention / Linear Attention / 滑窗 等都在攻这个问题，M3 之后会零星提到，本课程主线不深入。

---

## 6. 练习

落到 `Playground/ch05-attention/`：

| 脚本 | 内容 |
|---|---|
| `01_attention_numpy.py` | 纯 NumPy 实现单头缩放点积，逐步打印 QK^T / softmax / 输出 |
| `02_attention_torch.py` | PyTorch 单头版，与 `F.scaled_dot_product_attention` 输出对齐 |
| `03_multihead.py` | 手写 `MultiHeadAttention` 模块，与 `nn.MultiheadAttention` 数值对齐 |
| `04_causal_mask.py` | 因果掩码可视化 + 用"未来 token 替换"实验验证不泄漏 |

跑法同 ch04，CPU 秒级。

## 思考题

1. 如果让 Q 与 K 共享同一个投影矩阵（即 `W^Q = W^K`），attention 的对称性会变成什么样？训练上会出什么问题？
2. 多头注意力的 H 选 8、16、32 各有什么权衡？d=768 时为什么 H=12 是 GPT-2 的选择？
3. n²·d 的复杂度里，n=2k 和 n=8k 时 attention 矩阵显存分别多少（fp16（16-bit floating point，半精度浮点），单头单 batch）？

## 参考资料

- **Vaswani et al., "Attention is All You Need"**：原论文，Transformer + MHA 的源头
- **Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate"**：attention 的早期形态（RNN + 加性注意力）
- **The Annotated Transformer (Harvard NLP)**：逐行注释的 PyTorch 实现，最佳辅助读物
- **Jay Alammar, "The Illustrated Transformer"**：图示派经典
