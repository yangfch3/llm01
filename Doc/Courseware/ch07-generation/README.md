# ch07 · 生成策略与 KV cache

> ch06 训出来的 MiniGPT，贪心解码下输出 "the the the the..."。本章解决两件事：
> 1. **怎么解码** — 让生成的文本既不傻（贪心退化）也不乱（纯随机崩坏）
> 2. **怎么提速** — KV cache 让生成 n 个 token 的复杂度从 O(n³) 降到 O(n²)
>
> 本章是 M2 收官，也是 echo-mini 推理 CLI 的全部理论基础。

## 学习目标

1. 能解释 temperature / top-k / top-p 的数学定义与各自直觉
2. 能说明 KV cache 为什么有效、能省多少、形状如何变化
3. 知道训练阶段为什么用不上 KV cache，推理阶段为什么必须用

## 前置依赖

- ch06 全章（要给 ch06 的 MiniGPT 加策略和 cache）

---

## 1. 为什么贪心不够

ch06 §04 训完的 MiniGPT 续写 ROMEO 的结果：

```
ROMEO:
The the the the the the the the the the the the the the the the...
```

诊断：**贪心解码每步取 argmax**。一旦模型学到"逗号后大概率接 the"、"the 后大概率接 the"（小模型欠训练时常见），就会陷入**确定性循环**。

数学上：贪心 = 温度 0 的概率分布退化为 one-hot，永远走概率最高的那条路 → 没有任何探索 → 一旦进入循环态出不来。

解药：**给概率分布加一点不确定性**，让模型有机会跳出 local 最优。三种主流做法：温度、top-k、top-p。

> Beam search 是另一类（保留多条路径取总概率最高的），在机器翻译时代曾是主流，**LLM 时代基本不用** — beam 倾向短而保守的输出，对开放式生成（写故事、对话）反而不利。本章不展开。

---

## 2. Temperature：拉伸或压缩 softmax

```
softmax_T(z)_i = exp(z_i / T) / Σ exp(z_j / T)
```

| T | 效果 | 直觉 |
|---|---|---|
| T → 0⁺ | softmax 趋于 one-hot | 等价贪心，最确定 |
| T = 1 | 原始 softmax | 模型学到的原始分布 |
| T → ∞ | softmax 趋于均匀 | 完全随机 |

工程经验：

- T ∈ [0.7, 1.0]：对话/续写默认范围
- T = 0：需要确定性输出（代码补全、抽取式 QA）
- T > 1.5：极少用，输出失控

实现一行：`logits = logits / T` 然后接 softmax 与采样。

### 自检

1. T=0.5 与 T=2.0 哪个分布更"尖"？
2. 为什么 T 不能等于 0？

<details markdown="1">
<summary>答案速查</summary>

1. T=0.5。`exp(z/0.5)` 比 `exp(z)` 把高 logit 的优势进一步放大 → 分布更尖

2. 数值上除零会爆炸。等价行为应该走 argmax 路径，不要走 softmax。代码里通常 `if T == 0: return argmax` 早退

</details>

---

## 3. Top-k 采样

> 只在概率最高的 k 个 token 里采样，其余直接丢弃。

```
1. 对 logits 取 topk → 得到 k 个 logit 与索引
2. 把其余位置 logit 置为 -inf
3. softmax + multinomial 采样 1 个
```

直觉：**截掉长尾**。模型分布尾部往往是大量"明显不该出现"的 token，加起来概率不小但都是噪声。砍掉后采样更安全。

经验值：k ∈ [20, 100]。k=1 等价贪心，k=V（词表大小）等价无截断。

```python
def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    # logits: (B, V)；返回同形状，非 top-k 位置为 -inf
    v, _ = torch.topk(logits, k)
    threshold = v[:, -1:]                                # 第 k 大的值
    return torch.where(logits < threshold, float("-inf"), logits)
```

---

## 4. Top-p (nucleus) 采样

> 选择**累计概率 ≥ p 的最小 token 集合**，在集合内重新归一化采样。

```
1. logits → softmax → 概率
2. 按概率降序排
3. 从最高概率开始累加，一旦累计概率 ≥ p，**保留已累加的所有 token**（构成 nucleus），截掉剩余
4. 重新归一化
5. multinomial 采样 1 个
```

直觉：**自适应 k**。

- 模型很确定时（如填"今天天气真"后接什么），分布尖锐，p=0.9 可能只保留 3-5 个 token
- 模型不确定时（如开放性创意），分布扁平，p=0.9 可能保留几十上百个

top-k 是"硬截断"，top-p 是"软截断"——后者更主流。经验值：p ∈ [0.8, 0.95]。

> 这两者不互斥。GPT-2/LLaMA 的常见配方是 **top-k=50 + top-p=0.95 + T=0.8** 三件套，先 top-k 砍掉离谱长尾、再 top-p 自适应、最后温度调锐度。

### 自检

1. top-p=1.0 等价于什么？top-k=词表大小呢？
2. 为什么 top-p 比 top-k 主流？

<details markdown="1">
<summary>答案速查</summary>

1. 都等价于"无截断的纯采样"，整个词表都参与归一化采样

2. top-k 的 k 是固定的，无法适配分布形状。模型很确定时 k=50 可能让尾部噪声混入；模型不确定时 k=50 又可能截太狠。top-p 按概率累计自适应，分布尖时自动 k 小、分布平时自动 k 大

</details>

---

## 5. 解码配方与 ch06 退化的解药

回到 ch06 的 "the the the" 问题。三种解药：

| 配方 | 效果 | 适用 |
|---|---|---|
| `greedy` | 死循环依旧 | 仅诊断 |
| `T=0.8` | 偶尔跳出循环 | 最小修补 |
| `T=0.8 + top-k=20 + top-p=0.9` | 几乎不死循环 | 推荐默认 |

ch07 §06 练习 1 会跑出对照打印。

> **另一类正交手段：repetition / frequency penalty**。直接对"已生成过的 token"的 logit 减一个惩罚值（OpenAI API 的 `presence_penalty` / `frequency_penalty`），从源头打压重复，可与上面三件套叠加。本章不展开实现。

---

## 6. KV cache

### 6.1 朴素自回归生成的浪费

ch06 §03 的 `generate` 每生成一个新 token，**整段历史重新跑一遍**：

```
生成第 t 个 token：
  forward([id_0, id_1, ..., id_{t-1}])  → logits  → 采样 id_t
生成第 t+1 个 token：
  forward([id_0, id_1, ..., id_t])      → logits  → 采样 id_{t+1}
                                                ↑↑↑
                                    前 t 个 token 的 K/V 又算了一遍
```

attention 的 K/V 只依赖各自位置的输入 token——**之前算过的 K/V 永远不会变**。每步重算就是纯浪费。

### 6.2 KV cache 怎么省

```
第 1 步：forward([id_0])      → 算出 K_0, V_0，存入 cache
第 2 步：forward([id_1])      → 只算 K_1, V_1，cache 拼接 → [K_0,K_1], [V_0,V_1]
                              → query 只是 q_1（单个 token）
                              → attention(q_1, [K_0,K_1], [V_0,V_1]) → logits
第 t 步：forward([id_{t-1}])  → 只算 K_{t-1}, V_{t-1}，append 到 cache
                              → attention(q_{t-1}, K_cache, V_cache)
```

每步：

- 输入只是**一个 token**（不是整段历史）
- Q 只算 1 个，K/V 只新增 1 个，与历史 K/V cache 拼起来算 attention
- attention 矩阵形状从 `(t, t)` 退化成 `(1, t)`，对应推理时**没有 mask 也是因果的**（只有一个 query 在最末位）

### 6.3 复杂度对比

设序列长 n，模型维 d，层数 L。

| 阶段 | 无 cache | 有 cache |
|---|---|---|
| 第 t 步 forward | O(t² · d · L) | O(t · d · L) |
| 生成 n 个 token 总计 | O(n³ · d · L) | O(n² · d · L) |
| 显存 | 仅 weights | weights + KV cache (O(L · n · d)) |

**结论**：KV cache 用显存换计算，n 越大省得越多。生成 1k token，无 cache 算 ~10⁹ 次，有 cache ~10⁶ 次，**1000 倍**差距。

### 6.4 形状变化（必背）

|  | 无 cache（训练 / 首次 prefill） | 有 cache（生成第 t 步） |
|---|---|---|
| 输入 ids | (B, n) | (B, 1) |
| Q | (B, H, n, d_k) | (B, H, 1, d_k) |
| K cache | — | (B, H, t, d_k) |
| V cache | — | (B, H, t, d_k) |
| attention 矩阵 | (B, H, n, n) | (B, H, 1, t) |
| 输出 | (B, n, d) | (B, 1, d) |

### 6.5 边界

- **训练用不上**：训练时一次性给全长序列，并行算所有位置的 loss，没有"上一步"概念
- **首次 prefill 走"无 cache"路径**：把 prompt 一次性 forward 进去同时填充 cache，之后才进入"每步一 token"
- **batch 内不同序列长度问题**：若 batch 内序列等长，KV cache 形状对齐；不等长需要 padding + 合理的 attention mask。生成时通常用 **left-padding**——因为右边是要新增 token 的"生成区"，pad 必须靠左才能让所有序列的"末位"对齐到同一列（M3 工程细节）

### 自检

1. 为什么 KV cache 不需要存 Q？
2. 推理时第 t 步的 attention 矩阵是 (1, t) 而不是 (t, t)，那 causal mask 哪去了？

<details markdown="1">
<summary>答案速查</summary>

1. Q 是当前要查询的 token（query），每步只用一次就丢，不会复用。K 和 V 是被查询的对象，所有未来步都会反复看到 → 缓存它们才有意义

2. 当 query 只有 1 个（最末位的 token），它本来就只能看自己和之前——形状 (1, t) 自动满足因果性，无需显式 mask。这也是 KV cache 实现起来不麻烦的根本原因

</details>

---

## 7. 练习

落到 `Playground/ch07-generation/`：

| 脚本 | 内容 |
|---|---|
| `01_decoding_compare.py` | 即时训练 200 步小 MiniGPT，对比 greedy / T=0.8 / top-k / top-p / 三件套 五种续写 |
| `02_kv_cache.py` | 给 ch06 的 `CausalSelfAttention` 加 KV cache，①验证数值一致 ②实测加速 |

跑法同 ch06。01 ~10s，02 几秒。

## 思考题

1. 为什么 LLM 时代 beam search 失宠？
2. KV cache 的显存占用 = `2 · L · B · n · d · sizeof(dtype)`。LLaMA-7B（L=32, d=4096）batch=1、n=4096、fp16，cache 多大？这是为什么长上下文 LLM 推理特别吃显存的根本原因
3. 假设你要做 **batch 推理**，batch 内 prompt 长度不一，KV cache 怎么管理？（提示：padding / left-padding / continuous batching）

## 参考资料

- **Holtzman et al., "The Curious Case of Neural Text Degeneration"**：top-p (nucleus) 采样原论文，附"为什么贪心/beam 在开放式生成会退化"的精彩分析
- **Pope et al., "Efficiently Scaling Transformer Inference"**：KV cache 与 batch 推理优化的工程参考
- **vLLM PagedAttention 论文**：把 KV cache 当虚拟内存管理，是当前推理引擎的主流方案
- **Karpathy, "nanoGPT" generate.py**：极简 KV cache 参考实现
