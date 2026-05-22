# ch10 · 监督微调（SFT）

> Pretrain 教模型 "接龙"，但接出来的东西不一定是你想要的——它可能续写一段维基、丢一串代码、或者陷入复读。
>
> **SFT（Supervised Fine-Tuning）** 把模型从 "会接龙" 调教成 "会按对话格式回答问题"。
>
> 本章是 M5 echo 微调的理论基础——数据怎么拼、loss 怎么算、显存不够怎么办，全在这里。

## 学习目标

1. 能写出 SFT 训练样本的 `input_ids` / `labels` 形态，正确标出 -100 的位置
2. 能解释 ChatML（Chat Markup Language，对话标记语言）类对话模板的三件事：role 标记、turn 边界、generation prompt
3. 能讲清 LoRA 的低秩分解为什么省显存、QLoRA 在它之上又叠了什么 trick

## 前置依赖

- ch08（分词器，特别是 special token）、ch09（CLM loss、ignore_index、AMP）
- 对 `nn.Linear` 的 weight shape `(out, in)` 要熟

---

## 1. SFT vs Pretrain

| 维度 | Pretrain | SFT |
|---|---|---|
| 数据 | 海量原始文本（爬虫/书/代码），无结构 | 几千–几十万条 (prompt, response) 对话样本，**人工或合成** |
| 目标 | 学语言分布 P(token \| 前文) | 学 "在 prompt 后输出符合期望的 response" |
| Loss | 全序列 CLM | **只对 response 部分算 CLM**，prompt 部分 mask 掉 |
| 模型变动 | 从随机初始化训起 | **基于 pretrain 权重继续训**，lr 小一两个量级 |
| 数据量级 | TB 级 token | MB–GB 级 token |
| 训练时长 | 数天–数月 | 数小时–几天 |

一句话区分：**Pretrain 让模型 "知晓规律"，SFT 让模型 "按格式表达"。**

**SFT 训练流程一图流**：
```
原始对话 
→ chat template 渲染成文本 
→ tokenize 成 `input_ids`
→ 构造 `labels`（prompt 部分置 -100） 
→ 喂进模型算 CLM loss
```

### 1.1 为什么必须基于 pretrain

直接用 SFT 数据从零训会怎样？几万条对话样本撑不起一个语言模型 —— 连基础语法都学不全。SFT 是**站在 pretrain 巨人肩上做轻量微调**，调的是**输出风格、格式、指令跟随**，不是语言能力本身。

### 1.2 通用能力会丢吗

不一定。SFT 数据若分布太窄（如全是问答），模型会**遗忘** pretrain 学到的部分能力（如续写、代码、长文）。这叫 **catastrophic forgetting / alignment tax**。
缓解办法：

- 数据混合（SFT 数据 + 少量 pretrain 数据混训）
- LoRA（不动 base，效果限制在 adapter 上）
- 多阶段 SFT，逐步收窄

### 自检

1. SFT 训练时如果 prompt 部分也算 loss，会发生什么？
2. 为什么 SFT 的 lr 普遍比 Pretrain 小一两个量级？

<details markdown="1">
<summary>答案速查</summary>

1. 模型把 "复述用户提问" 也当成训练目标 → 推理时容易把 prompt 又输出一遍，且把宝贵的容量浪费在学习 user 输入的分布上，response 学不专心

2. SFT 在 pretrain 权重附近做小幅修正，lr 大了会把 pretrain 学到的语言能力冲掉（catastrophic forgetting）；典型 SFT lr 在 `1e-5` ~ `5e-5`，pretrain 常 `1e-4` ~ `6e-4`

</details>

---

## 2. 对话模板（chat template）

### 2.1 模型怎么知道该它说了

Pretrain 模型只见过大量原始文本，不知道 user / assistant 的概念。SFT 阶段必须**用文本格式编码这个结构信息** —— 这就是 chat template。

最朴素的写法：

```
User: 帮我写一首关于秋天的诗
Assistant: 秋风扫落叶，... 

User: 再来一首
Assistant: ...
```

够用，但有歧义：模型不知道 "User:" 这五个字是格式还是用户真的说了 "User"。所以现代模型都用 **special token** 显式标记边界 —— 用户基本不会主动说的保留 token。

### 2.2 ChatML 类模板

ChatML 是 GPT-3.5 起的事实标准。

```
<|im_start|>system
你是一个简洁的助手。<|im_end|>
<|im_start|>user
帮我写一首关于秋天的诗<|im_end|>
<|im_start|>assistant
秋风扫落叶，孤雁向南飞。<|im_end|>
<|im_start|>user
再来一首<|im_end|>
<|im_start|>assistant
```

要点：

- `<|im_start|>` / `<|im_end|>` 是分词器里**预留的 special token**（不是 BPE 出来的子串），单 token id
- 每个 turn 形如 `<|im_start|>{role}\n{content}<|im_end|>\n`
- **末尾有意不闭合**：以 `<|im_start|>assistant\n` 结尾告诉模型 "该你说了"。这叫 **generation prompt**
- system 是可选的全局指令，约定放最前

> Qwen / Yi / 国内大多数对话模型都用 ChatML 或其变体。LLaMA-2/3 用自家 `[INST] ... [/INST]`，本质等价。

代码示意（HF tokenizer 的 `apply_chat_template`）：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

messages = [
    {"role": "system", "content": "你是一个简洁的助手。"},
    {"role": "user", "content": "帮我写一首关于秋天的诗"},
]

# 推理时：add_generation_prompt=True → 末尾拼上 <|im_start|>assistant\n
prompt_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# 训练时：补上 response + <|im_end|>，不加 generation prompt（完整对话已含）
messages_with_response = messages + [
    {"role": "assistant", "content": "秋风扫落叶，孤雁向南飞。"}
]
train_text = tokenizer.apply_chat_template(
    messages_with_response, tokenize=False, add_generation_prompt=False
)
```

### 2.3 generation prompt

generation prompt 是 SFT 的核心机关。训练时给模型看**完整对话**（含最后一个 `<|im_end|>`）；推理时给模型看**到 generation prompt 为止**，让它从 `\n` 后开始续写。

```
训练样本:
<|im_start|>user\n问<|im_end|>\n<|im_start|>assistant\n答<|im_end|>\n
                                                      |           └─模型学着在这里收尾
                                                      └─模型学着在这里续写
                                                          

推理输入:
<|im_start|>user\n问<|im_end|>\n<|im_start|>assistant\n
                                                      └─模型从这里开始生成
```

如果训练时没把 `<|im_end|>` 放进 response 末尾，推理时模型不知道何时停 → 一直生成下去。**EOS (End of Sequence) 训练数据里必须出现**。

**安全延伸：特殊 token 注入。** 用户输入里如果夹带 `<|im_end|>\n<|im_start|>assistant\n` 等字符串，且 tokenizer 将其识别为真正的 special token，模型看到的 turn 结构就被篡改（经典 prompt injection 攻击面）。防御：tokenizer encode 用户输入时设 `allowed_special=set()` 或在入口处过滤/转义控制标记，确保用户文本中的控制串只被当作普通字符拆分、不触发边界语义。

### 自检

1. 为什么 `<|im_start|>` 要作为 special token 而不是普通子串？
2. 训练时不把对话末尾的 `<|im_end|>` 当作 response 的一部分会怎样？

<details markdown="1">
<summary>答案速查</summary>

1. 普通子串会被 BPE 拆成多个 token（`<`, `|`, `im`, ...），模型必须凑齐这串才能识别边界，鲁棒性差；special token 是单 id，分词器永远把它整体输出，模型一眼就识别

2. 模型学不到 "该停了" 的信号，推理时会续写无止境，直到撞上 max_new_tokens。`<|im_end|>` 既是边界也是 EOS，必须算进 response 部分让模型学

</details>

---

## 3. Loss mask（SFT 的灵魂）

### 3.1 哪些位置不算 loss

回顾 ch09 的 CLM 公式 `L = -Σ log P(x_t | x_{<t})`。SFT 沿用这个公式，但 **t 只跑 response 的位置**，prompt 与模板 token 全跳过。

```
Full sequence (one training sample):

<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\nAI Answer<|im_end|>\n
|--------- user turn ---------||----- gen prompt ----||---- response ---|
          (mask, -100)               (mask, -100)        (compute loss)

input_ids: [t_user_0, ..., t_user_k,  t_gp_0, ..., t_gp_m,  t_resp_0, ..., t_resp_n, t_eos]
labels:    [   -100,  ...,    -100,     -100,  ...,   -100,   t_resp_0, ..., t_resp_n, t_eos]
            |--- prompt + template: all -100 ---||--- response text + <|im_end|>: loss ---|
```

- user turn: `<|im_start|>user\nQ<|im_end|>\n` — mask（格式标记 + 提问）
- gen prompt: `<|im_start|>assistant\n` — mask（格式标记，非模型输出）
- response: `A<|im_end|>` — compute loss（`<|im_end|>` = EOS, 模型必须学会触发）
- `<|im_end|>` 后的 `\n`：属于模板格式，非模型输出

实现就一行：把 prompt 长度（user turn + gen prompt）内的 label 都设 `-100`，cross_entropy 自动跳过（ch09 §1.3 已铺垫）。

```python
# prompt_len = len(user_turn_ids) + len(gen_prompt_ids)
labels = input_ids.clone()
labels[:prompt_len] = -100  # prompt + gen prompt 全部 mask

loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels.view(-1),
    ignore_index=-100,
)
```

### 3.2 为什么 prompt 不算 loss

SFT 时不算 prompt 的 loss == 不贡献梯度 ≠ 不需要看到 prompt 部分。

prompt 部分虽然不贡献梯度（label=-100），但它参与前向计算 —— 模型通过 attention 看到完整 prompt 才能在 response 位置产出正确的条件概率 **P (response_t | prompt, response_{<t})**。

如果 prompt 算了 loss: 

- **目标错位**：算 prompt loss -> 会让模型学着拟合问题的表述，你要的是模型学回答，不是学习 user 提问口吻以及复述用户输入
- **容量浪费**：模型有限容量被迫同时拟合 user 输入分布（高噪声、每条不同）和 response 分布，后者空间被挤压、浪费
- **推理偏移**：训练时如果学过 "看到 user 输入就概率输出 user 风格的话、概率输出问题中的字眼" → 推理时容易把 user 的话再重复一遍
- **生成污染**：turn 含义的 special token 也被拟合进去了，到时模型就有概率生成伪 turn 标记

### 3.3 多轮对话的 mask

为了让模型能学习到合理的多轮对话回复，SFT 的训练语料中会有多轮对话样本。多轮样本里 assistant 会出现多次。**所有 assistant 的 response 部分都算 loss**，所有 user / system / 模板 token 都 mask：

```
<|im_start|>system\n你是助手<|im_end|>\n    ← 全 mask
<|im_start|>user\n问1<|im_end|>\n           ← 全 mask
<|im_start|>assistant\n  答1<|im_end|>\n    ← generation prompt mask，仅 "答1<|im_end|>" 算 loss
└---------------------┘---generation prompt
<|im_start|>user\n问2<|im_end|>\n           ← 全 mask
<|im_start|>assistant\n  答2<|im_end|>\n    ← 同上
```

> `<|im_start|>assistant\n` 这段 generation prompt 是格式而非内容，**通常也 mask 掉**。HF 的 `apply_chat_template` 配 `return_assistant_tokens_mask=True` 能直接拿到 mask。

### 3.4 SFT 还要 packing 吗

ch09 讲过 pretrain 必做 packing。SFT 数据通常较短（几百–几千 token），且不希望跨样本污染（一个对话的尾巴影响下一个对话的开头），**默认不做 packing**，每个样本独立 padding 到 batch 最大长度即可。

少量框架（如 Axolotl）支持 "sample packing"，把多个样本拼一条但用 attention mask 切开（document-level mask）——工程复杂度上去了，吞吐能涨 2–3×，看是否瓶颈在 GPU。echo 系列默认不上。

### 自检

1. 一个 SFT 样本里 prompt 占 200 token，response 占 50 token。labels 里有多少个 -100？
2. 多轮对话样本，第一轮 assistant 已经回答了，为什么还要让模型继续学第二轮？

<details markdown="1">
<summary>答案速查</summary>

1. 200 个（prompt 部分），加上前面的 system / 模板 token；response 50 个 token 对应 50 个真实 label

2. 多轮样本前几轮的 assistant 答提供了**上下文**，让模型学会 "在已经聊过 N 轮的状态下继续合理回复"。每一轮 response 都是独立的训练信号，多轮样本相当于一次训了 N 个有递进上下文的小样本

</details>

---

## 4. LoRA / QLoRA

### 4.1 全参微调贵在哪

我们先来算一下一个 7B 模型全参 SFT 的资源开销：

```
weights:    7B × 2B (bf16)   = 14GB
gradients:  7B × 2B (bf16)   = 14GB
optimizer:  7B × 8B (Adam fp32 m+v) = 56GB
activations: 与 batch×seq 成正比，2-15GB (seq 512~4096)
─────────────────────────────────────
总计：85GB + activations，A100 80GB 都嫌挤
```

这对于资源不充足的团队或个人，是巨量的资源开销，业内有何针对性解决方案呢？

下面会先介绍 LoRA，**LoRA 把可训参数砍掉 99%。**

### 4.2 LoRA 核心公式

思路来自论文：Hu et al. 2021。一句话描述便是：**冻结原矩阵 W，旁路加一个低秩分解 BA 学增量。**"低秩分解" = 用两个瘦矩阵（B、A）的乘积代替一个大矩阵（W 的规模），牺牲表达能力（秩被限制）换取参数量大幅缩减。

对一个线性层 `y = Wx`，LoRA 改成：

```
y = Wx + (α/r) · BAx       # W 冻结，B、A 可训
W: (out, in)               # 原权重，冻结
A: (r, in)                 # 低秩矩阵，r 通常 8/16/32/64
B: (out, r)                # 低秩矩阵，初始化为 0
α: 缩放系数（超参）
```

最简 PyTorch 实现：

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad_(False)  # 冻结 W
        out_dim, in_dim = original.weight.shape
        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        self.scale = alpha / r

    def forward(self, x):
        base_out = self.original(x)             # Wx
        lora_out = (x @ self.A.T) @ self.B.T    # xA^T B^T = (BA)x 的等价写法
        return base_out + self.scale * lora_out
```

**关键直觉/假设**：

- 微调过程中权重的 "变化量" `ΔW` 在低秩子空间（8~64）里就够表达 —— 大模型参数高度冗余
- B 初始化为 0 → 训练开始时旁路输出 = 0 → 等价于完全用原模型
- 训练只更新 B、A，待训参数量从 `out×in` 降到 `r×(in+out)`

旁路中的 **`α/r` 是干嘛的**：经验上调大 r 会推大旁路输出尺度，`α/r` 缩放让你 **调 r 时不必重调 lr**。约定 α 与初始 r 一起定（如 α=16, r=8 → scale=2），后续只动 r、α 不动。这是 LoRA 论文的工程经验，不是数学必然。

> 为什么能抵消：r 增大 → A 行数增加 → `Ax` 方差与 r 成正比 → 旁路输出尺度线性增长；同时 `α/r` 线性缩小，两者相消 → 输出尺度（从而梯度尺度）对 r 不敏感 → 同一 lr 基本可沿用。
>
> 简版：r↑ → Ax 方差 ∝ r → α/r 线性抵消 → lr 不敏感

参数量对比（7B 模型 attention 的 q/k/v/o 全加 LoRA，r=8）：

```
原 attention：每层 4 × d² ≈ 4 × 4096² = 67M（×32 层 = 2.1B）
LoRA r=8： 4 × r × 2d ≈ 4 × 8 × 8192 = 0.26M（×32 层 = 8.4M）
```

可训参数从 2.1B 降到 8.4M，**减 250 倍**，这对于资源不充裕的团队或个人来说是巨大的节约。当然，资源充足时全参微调追求效果上限 —— LoRA 的低秩约束会牺牲少量表达能力，大厂在追求最后几个点性能时仍选择全参。

### 4.3 显存账重新算

继续用上面 7B 模型的例子，算算 LoRA 下的开销：

```
weights:        7B × 2B = 14GB      # 冻结但仍要存
gradients:      8.4M × 2B = 17MB    # 只对 LoRA 算梯度
optimizer:      8.4M × 8B = 67MB    # Adam 状态只对 LoRA
activations:    与全参差不多          # 前向还是要走全网络
─────────────────────────────
总计 14.5GB + activations
```

注意 weights 那行还是 14GB —— LoRA 不省 base model 的存储，只省**梯度 + 优化器状态**。activations 也不省，因为前向必须走完整 W。**LoRA 不是 "压缩模型"，是 "压缩 SFT 的可训参数"。**

想再砍 activations？可叠 ch09 §3.3 的 gradient checkpointing，LoRA + ckpt 是 7B 微调的标配组合。

### 4.4 QLoRA 多走一步

来自 Dettmers et al. 2023，思路是 LoRA（旁路仍用 bf16 正常训练） + 把 base model 量化到 4bit。

```
weights:    7B × 0.5B (NF4 4bit) = 3.5GB    ← 砍 4 倍
gradients:  8.4M × 2B = 17MB
optimizer:  8.4M × 8B = 67MB
activations: 与 LoRA 同
─────────────────────────────
总计：4GB + activations
```

QLoRA 做了三件事：

1. **NF4（NormalFloat 4-bit）**：针对正态分布权重设计的非均匀 4bit 量化，比对称 int4 损失更小
2. **double quantization**：量化时每组权重需要存一个 fp32 scale（缩放因子，用于把 int4 还原回浮点），这些 scale 本身也占显存；double quant 对 scale 再做一次量化，再省 ~0.5 GB
3. **paged optimizer**：将 optimizer 状态存放在主机 RAM 而非 GPU 显存，`step()` 时按需 swap 到 GPU 完成更新。以 PCIe 带宽换显存空间，适合显存紧张的场景开启

> 注意区分：ch09 的混合精度（AMP）是用 bf16 浮点加速训练，不叫量化；量化特指把权重离散化到 int8/int4 等整数格式，QLoRA 是主流方案中**唯一把量化用在训练阶段**的做法。

代价：

- 前向反向都要把 4bit 解量化回 bf16 算，吞吐略降（~10–30%）
- base model 量化是有损的，最终模型效果比全精度 LoRA 训练略差（多数 benchmark 差 1–2 个点）

**训练产出物**：QLoRA 的 4bit 只是训练过程中的运行时状态（加载时即时量化到显存），**不改变磁盘上的 base model 权重**。训练结束后你拿到的是原始全精度 base + 几十 MB 的 LoRA adapter（bf16）。

> 部署时先 merge adapter 回 base 得到全精度微调模型，再按需做独立的 PTQ 量化（GGUF/GPTQ 等）。

### 4.5 选型速查

消费卡 SFT 规模承载速查表：

| 场景 | 选 |
|---|---|
| <1B 模型，单卡 12GB | 全参 SFT 没问题 |
| 7B 模型，单卡 12GB | **QLoRA**（必须 4bit） |
| 7B 模型，单卡 24GB | LoRA bf16 |
| 70B 模型，单卡 24GB | QLoRA + offload，能跑但慢 |
| Mac MPS | LoRA（bf16/fp16）；bitsandbytes 对 MPS 支持不成熟，QLoRA 不建议在 Mac 上跑 |

> echo（M5）走 Qwen2.5-0.5B/1.5B 底座，3060 上**纯 LoRA 即可**，QLoRA 在这个量级反而吞吐亏；Mac 也能直接 LoRA。

### 自检

1. LoRA 的 B 矩阵为什么必须初始化为 0？A 初始化呢？
2. QLoRA 的 base model 量化了，反向传播怎么算梯度？

<details markdown="1">
<summary>答案速查</summary>

1. B=0 → BAx=0 → 训练起点等价原模型，避免一开始就破坏 pretrain。A 必须非零（Kaiming 类小方差初始化），否则 B 的梯度 `∂L/∂B = (∂L/∂y)·(Ax)^T` 含 Ax=0 因子 → B 永远学不动。**两个都为 0 时双向卡死**：B=0 让 A 的梯度 `B^T·(∂L/∂y)·x^T` 也为 0。所以 LoRA 选 "B=0 + A 随机" 这一组合：起点等价 base，且 B 第一步就有非零梯度

2. 前向时把 4bit base 解量化回 bf16 与输入相乘；反向只对 LoRA 的 B、A 算梯度，base 是冻结的不需要梯度。**4bit 权重本身不参与梯度计算**，只参与前向矩阵乘的解量化数值

</details>

---

## 5. 数据质量 > 数量

LIMA 论文（Zhou et al. 2023）：用 1000 条**精挑细选**的样本 SFT，效果接近 5 万条平均质量数据。后续 Tülu / OpenHermes 都验证了 "数据质量 > 数量" 这个倾向。

### 5.1 高质量 SFT 数据的特征

- **回答完整、格式一致**：每条 response 都按预期 markdown / 段落组织
- **指令多样性**：覆盖问答、改写、总结、推理、代码、拒绝等不同任务
- **拒答样本**：教模型说 "我不知道" 或拒绝越界请求
- **难度梯度**：既有简单事实也有需要多步推理的复杂问题

### 5.2 常见数据来源

| 来源 | 量级 | 质量 | 备注 |
|---|---|---|---|
| 人工标注 | 千–万 | 高 | 贵，慢 |
| 从 SOTA 模型蒸馏 | 万–几十万 | 中–高 | Alpaca/WizardLM 路线，注意 license |
| 公开数据集组合 | 万–百万 | 参差 | OpenHermes/Tülu 系，过滤后用 |
| 真实日志 | 海量 | 低 | 需重度清洗 + 反馈打分 |

> echo 系列：M4 echo-mini 用公开 + 少量 GPT 生成补足；M5 echo 用 Tülu/Alpaca 子集 + 自构 Echo 人设样本。

### 自检

1. SFT 数据 5000 条但全是闲聊，跟 500 条覆盖 10 类任务，哪个效果好？
2. "拒答样本" 在 SFT 里起什么作用？

<details markdown="1">
<summary>答案速查</summary>

1. 后者大概率好。指令跟随能力的核心是**任务多样性**而非样本绝对数；闲聊单一分布会让模型在编程、数学等任务上不会回答，泛化差

2. 教模型识别越界 / 不会 / 有害请求并给出合规回复。没拒答样本的模型遇到不会的问题会乱编（hallucinate），或对有害请求照做

</details>

---

## 6. 练习

落到 `Playground/ch10-sft/`：

| 脚本 | 内容 |
|---|---|
| `01_chat_template.py` | 手写 ChatML 渲染器 + 反解（拆出每个 turn 的 role/content），并演示 generation prompt 的差异 |
| `02_loss_mask.py` | 构造一个 SFT 样本的 input_ids/labels，对照 "全序列算 loss" 与 "只对 response 算 loss" 两种训练动力学 |
| `03_lora_demo.py` | 手撕 LoRALinear（无 peft 依赖），打印参数量对比，验证一次前反向只更新 LoRA 参数 |

跑法：

```bash
uv run python Playground/ch10-sft/01_chat_template.py
uv run python Playground/ch10-sft/02_loss_mask.py
uv run python Playground/ch10-sft/03_lora_demo.py
```

## 思考题

1. 为什么 LoRA 通常只挂在 attention 的 q/k/v/o 上，不挂 FFN？挂 FFN 会怎样？
2. SFT 之后再做对齐（ch11 DPO（Direct Preference Optimization，直接偏好优化））能进一步提升，反过来 "先 DPO 再 SFT" 行不行？为什么？
3. 数据混合时 pretrain 数据该占多大比例？小了没用，大了 SFT 信号被淹没

## 参考资料

- **Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"** (2021)：LoRA 原论文
- **Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs"** (2023)：QLoRA 三件套
- **Zhou et al., "LIMA: Less Is More for Alignment"** (2023)：数据质量 > 数量的实证
- **HuggingFace TRL 文档**：<https://huggingface.co/docs/trl> — `SFTTrainer` 与 chat template 实践
- **OpenAI ChatML spec**：<https://github.com/openai/openai-python/blob/main/chatml.md>（已归档但仍是事实标准）
