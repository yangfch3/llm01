# ch00 · AI 全景与概念速查

> **不用现在全懂。** 本文是地图，不是考试。先建立整体印象，遇到陌生词翻到末尾的[术语速查表](#附录术语速查表)即可。

## 0. 引言

2022 年底，ChatGPT 一夜之间让全世界知道了"大语言模型"。但在它背后，是从感知机到 Transformer 长达六十年的积累，是 Scaling Law 的量化预言，是 RLHF 对齐技术的临门一脚。

这篇文档帮你快速建立全局认知：AI 怎么走到今天、核心概念之间什么关系、一个模型从训练到上线经历了哪些阶段、当下前沿在做什么。不求深，但求"知道自己不知道什么"。

---

## 1. 前史速览

| 年代 | 事件 | 意义 |
|------|------|------|
| 1958 | 感知机 (Perceptron) | 第一个可学习的神经网络单元 |
| 1969 | Minsky《Perceptrons》出版 | 数学证明感知机局限性，AI 第一次寒冬开始——此后神经网络沉寂近 20 年 |
| 1986 | 反向传播算法普及 | 多层网络可训练，寒冬回暖 |
| 2012 | AlexNet 夺冠 ImageNet | 深度学习实用化信号，GPU 训练起飞 |

> 以上仅供定位"深度学习从哪里来"。本文重点从下一节开始。

---

## 2. 当代大模型时间线（2013–2025）

| 年份 | 里程碑 | 一句话意义 |
|------|--------|-----------|
| 2013 | Word2Vec | 词可以变成向量，向量可以做算术（king - man + woman ≈ queen） |
| 2014 | Seq2Seq / GAN | 序列到序列翻译框架；对抗生成范式诞生 |
| 2015 | Attention 机制 | 解码时动态聚焦源端相关位置，性能跃升 |
| 2017 | **Transformer** | "Attention Is All You Need"，抛弃循环，全靠注意力+并行 |
| 2018 | GPT-1 / BERT | 预训练+微调范式确立；自回归 vs 双向掩码两条路线分野 |
| 2020 | **Scaling Law / GPT-3** | Scaling Law 量化了"规模→性能"规律；GPT-3 (1750亿参数) 是其实践产物 |
| 2020 | Vision Transformer (ViT) | Transformer 入侵视觉，证明注意力不止做 NLP |
| 2021 | DALL·E 1 / CLIP | 文本↔图像对齐，多模态时代序幕（注：DALL·E 1 用 dVAE+Transformer，非扩散） |
| 2022 | **ChatGPT / InstructGPT** | RLHF 对齐落地，LLM 进入大众视野 |
| 2022 | Stable Diffusion | 潜空间扩散，开源图像生成爆发 |
| 2023 | **GPT-4 / LLaMA** | 多模态闭源标杆；开源权重潮涌 |
| 2023 | MoE 规模化 (Mixtral) | 稀疏激活降本，同等效果少用算力 |
| 2024 | Sora / 视频生成 | DiT 架构驱动长视频生成 |
| 2024 | 开源追平 (Qwen2/LLaMA3/DeepSeek) | 开源模型能力逼近闭源前沿 |
| 2025 | **DeepSeek-R1 / 推理模型 / Agent** | 开源推理模型比肩闭源；模型学会"思考"与"使用工具" |

---

## 3. 概念分类树

```
AI（人工智能）
└─ ML（机器学习）
   ├─ 学习范式（数据怎么用）
   │   ├─ 监督学习
   │   ├─ 无监督学习
   │   ├─ 自监督学习
   │   └─ 强化学习
   └─ 模型架构（网络怎么搭）
       ├─ 传统 ML
       └─ 神经网络
           ├─ MLP
           ├─ CNN
           ├─ RNN → LSTM / GRU
           └─ Transformer
               ├─ Encoder-only（BERT 系）
               ├─ Decoder-only（GPT 系）
               └─ Encoder-Decoder（T5 / BART）
```

### 学习范式

| 范式 | 核心思路 | 典型应用 |
|------|---------|---------|
| 监督学习 | 有标签，学 输入→标签 映射 | 图像分类、情感分析、序列标注 |
| 无监督学习 | 无标签，学数据内部结构 | 聚类、降维、异常检测 |
| 自监督学习 | 从数据自身构造标签（遮挡/预测下一个） | MLM (BERT)、CLM (GPT)、对比学习 (CLIP) |
| 强化学习 | 智能体与环境交互，最大化累积奖励 | 游戏 AI、机器人控制、RLHF |

> **生动类比**
>
> - **监督学习** — 像做带答案的习题集：老师给你题目和标准答案，你反复练到能举一反三。
> - **无监督学习** — 像整理一堆没标签的照片：没人告诉你分类标准，你自己发现"这些是风景、那些是人像"。
> - **自监督学习** — 像完形填空：把文章挖几个空让你猜，答案就藏在原文里，不需要额外标注。GPT 的"预测下一个词"本质就是这个。
> - **强化学习** — 像训练小狗：做对了给零食（奖励），做错了没有。小狗通过反复试错学会"坐下"和"握手"。

注：还有一种**半监督学习**（少量标签 + 大量无标签数据混合训练），工业界常见但在当代大模型链路中不是主角，此处不展开。

### 模型架构

| 架构 | 核心思想 | 擅长 | 主要瓶颈 |
|------|---------|------|---------|
| MLP | 全连接层堆叠 | 通用基础积木 | 无法捕捉空间/序列结构 |
| CNN | 局部感受野 + 权重共享 + 池化 | 图像、空间特征 | 全局依赖需要很深 |
| RNN / LSTM / GRU | 隐状态逐步传递 | 序列、时序 | 长距离遗忘，无法并行 |
| Transformer | 自注意力 + 位置编码 | 全局依赖，天然并行 | 计算量随序列长度二次增长 |

### 两维度正交

> 学习范式和模型架构是独立的两个选择。任何架构都能搭配任何范式。

| | MLP | CNN | RNN/LSTM | Transformer |
|---|:---:|:---:|:---:|:---:|
| 监督 | ✓ | ✓ | ✓ | ✓ |
| 无监督 | ✓ (AE) | ✓ (VAE) | ✓ | ✓ |
| 自监督 | △ (罕见) | ✓ (SimCLR) | △ (罕见) | ✓ (GPT/BERT/CLIP) |
| 强化学习 | ✓ (DQN) | ✓ | ✓ | ✓ (Decision Transformer) |

---

## 4. 架构演进路线

```
MLP（全连接）
 │
 ├──→ CNN ──────────────────────────┐
 │    "用卷积核扫描局部特征"          │
 │                                   │
 └──→ RNN → LSTM / GRU             │ 2017: 全部汇聚
      "隐状态传递记忆"               │
       │                             │
       └→ Seq2Seq + Attention ──────→ Transformer
           "解码器动态看源端"         "全靠注意力，并行训练"
```

### 每一站的故事

**MLP** — 网络基础形态。每层全连接，能拟合任意函数（万能近似定理），但对空间/序列结构完全无感，参数量爆炸。类比：一张全是连线的蜘蛛网，什么都连但什么结构都不懂。

**CNN** — 观察：图像有局部相关性。方案：小卷积核滑动扫描 → 权重共享大幅降参数；池化下采样压缩空间维度并提供平移不变性。成就：ImageNet 革命。局限：全局信息需要堆很多层或很大核。类比：拿放大镜逐块扫描一张图片，先看局部细节再拼出整体。

**RNN → LSTM/GRU** — 观察：语言/时序有前后依赖。方案：隐状态逐步传递，理论上"记住"所有历史。现实：梯度消失/爆炸 → LSTM 引入门控缓解。局限：串行无法并行，长序列仍会遗忘。类比：逐字朗读一本书，同时在脑中默记上文——读到第 500 页时，第 1 页的细节早忘了。

**Seq2Seq + Attention** — 观察：翻译时不是每个输出都依赖所有输入。方案：解码每步动态加权聚焦源端。效果：翻译质量飞跃。但仍依赖 RNN 做编码/解码。类比：翻译一句话时，翻到某个词会回头重点看原文对应的几个词，而不是把整句话平均地记在脑子里。

**Transformer** — 激进一步：把 RNN 全扔掉，编码和解码都用自注意力。位置信息靠位置编码注入。收益：完全并行训练 + 全局依赖一步到位。代价：注意力计算 O(n²)。自此之后，几乎所有 SOTA 模型都基于 Transformer 变体。类比：一间教室里所有学生同时互相交流，每个人瞬间就能获取任何人的信息——沟通效率极高，但人数多了开销爆炸。

---

## 5. 当下模型方向

| 模态 | 主流架构 | 训练方法 | 代表 |
|------|---------|---------|------|
| 文本生成 | Decoder-only Transformer | CLM 自回归 | GPT-4、LLaMA、Qwen |
| 文本理解 | Encoder-only Transformer | MLM 掩码 | BERT、RoBERTa |
| 图像生成 | DiT / U-Net | 扩散 (Diffusion) | Stable Diffusion、DALL·E 3 |
| 视频生成 | DiT 变体 | 扩散 + 时序建模 | Sora、Kling |
| 多模态理解 | VLM（视觉编码器 + LLM） | 对比学习 + CLM | GPT-4V、Qwen-VL |
| 代码 | Decoder-only Transformer | CLM + Fill-in-Middle | Codex、DeepSeek-Coder |

> 注：以上训练方法均属于**自监督学习**（从数据自身构造学习信号，无需人工标注）。监督学习（SFT）和强化学习（RLHF/DPO）在后续对齐阶段介入。

### 为什么文本走自回归、图像走扩散？

- **文本**天然是离散序列，左到右逐 token 生成符合语言本质，CLM 自回归简洁高效。
- **图像**是高维连续信号，像素之间强空间相关性；扩散模型通过逐步去噪，能在潜空间稳定生成高质量图像，比 GAN 更易训练、比自回归像素生成更高效。

### 扩散模型一句话原理

训练时对图像逐步加噪直到变成纯噪声，然后训练网络学会逆过程（去噪）。推理时从纯随机噪声出发，逐步去噪，最终生成清晰图像。

---

## 6. LLM 全链路地图

### 什么是预训练？

预训练 = 拿海量无标注文本，让模型反复做一件事：**根据前文预测下一个 token**（即 CLM）。不需要人工标注，模型通过数万亿次预测，自己学会语言规律、事实知识和推理模式。这一步成本最高（数千张 GPU 跑数周），产出称为 base 模型。

### 为什么预训练不够？

- **Pretrain** 产出 base 模型 → 会续写文本，但不会对话。你说"北京天气怎样"，它可能接"预报显示明天晴转多云……后天……"无限续写，而不是回答你的问题。
- **SFT** 教会对话格式 → 能正常对话了，但可能输出有害内容或质量参差不齐。它学了"怎么说话"，没学"什么该说什么不该说"。
- **Alignment** 对齐人类偏好 → 好用且安全了。但数百亿参数的模型普通人跑不起来。
- **量化** 精度换显存 → FP16 压到 INT4，显存省 4 倍，终于能在消费级硬件跑起来。

```
数据 → Pretrain → SFT → Alignment → 量化 → 部署推理 → 上层应用
       (base模型)  (能对话)  (好用安全)  (跑得动)  (跑起来)    (用起来)
```

### 训练阶段

| 阶段 | 做什么 | 产出 |
|------|--------|------|
| Pretrain | 自监督 next token prediction，海量文本，烧大算力 | Base 模型（会续写，不会对话） |
| SFT | 有监督微调，学对话格式与指令服从 | Chat 模型（能对话） |
| Alignment | 对齐人类偏好（RLHF / DPO），变好用+无害 | 对齐后模型 |
| 量化 | FP16 → INT8/INT4，精度换显存 | 可本地部署的模型 |

**SFT 具体怎么做？**

核心：用"指令 + 标准回答"的配对数据，教 base 模型学会对话格式。

```
训练样本:
[User] 用一句话解释什么是黑洞
[Assistant] 黑洞是时空中引力极强、连光都无法逃逸的区域。
```

Base 模型只会无限续写，SFT 让它学到：看到 `[User]` 就该在 `[Assistant]` 后给出简洁回答然后停下来。本质是用监督学习把"对话行为模式"注入模型。数据质量 > 数量——几万条高质量样本就能显著改变模型行为。

**Alignment 具体怎么做？**

核心：让人类当裁判，告诉模型"哪个回答更好"，模型据此调整行为。

```
Prompt: "如何减肥？"
回答 A (chosen ✓): "建议控制饮食+适量运动，具体可以……"
回答 B (rejected ✗): "直接绝食三天，效果立竿见影"
→ 模型学到：往 A 靠，远离 B
```

- **RLHF**：先训一个"裁判"（奖励模型），再用 PPO 让 LLM 最大化裁判打分。两阶段，贵且不稳定。
- **DPO**：跳过裁判，直接拿偏好对数据优化 LLM（数学上等价于隐式奖励模型）。一阶段，简单稳定，开源主流。

### 推理与部署（概述）

模型训完后需要高效跑起来。类比：训练像"培养一个专家"，推理像"让专家上岗接客"——上岗时要考虑响应速度和接待能力。

核心优化思路：缓存已算结果（KV Cache）、压缩精度（量化）、并行调度（Continuous Batching）。

### 上层应用

| 概念 | 一句话 |
|------|--------|
| Prompt Engineering | 不改模型只改输入——CoT（思维链）/ few-shot / role play |
| RAG | 检索外部知识注入 prompt，缓解幻觉，让模型"有据可查" |
| Function Calling | 模型按格式输出工具调用指令，外部执行后回传结果 |
| MCP | Model Context Protocol，标准化模型与外部工具/数据源的连接协议 |
| Skill | Agent 的可复用能力单元，封装特定任务的 prompt + 工具组合 |
| Agent | 模型 + 工具调用 + 多步规划，从"回答问题"到"完成任务" |

> 类比：如果 LLM 是大脑，那么 Prompt Engineering 是"问对问题的技巧"、RAG 是"查资料"、Function Calling 是"动手操作"、MCP 是"统一的工具接口标准"、Skill 是"学会的一项具体技能"、Agent 是"自主规划+执行一整套流程"。

想看工业界真实案例如何落地以上全链路？见下方[附录](#附录前沿模型完整链路deepseek-v3--r1--qwen25)。

---

## 附录：前沿模型完整链路（DeepSeek V3 / R1 & Qwen2.5）

> 以下展示 2024–2025 年前沿开源模型的真实全链路。信息密度较高，建议对注意力机制有基本了解后再细读。

### 架构层

```
┌─────────────────────────────────────────────────────────┐
│  DeepSeek V3 架构 (671B total / ~37B active per token)  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input Embedding + RoPE                                 │
│       ↓                                                 │
│  ┌─ Transformer Block × 61 ──────────────────────┐     │
│  │                                                │     │
│  │  ┌─ Multi-head Latent Attention (MLA) ──────┐ │     │
│  │  │  Q/K/V 压缩到低秩潜向量                   │ │     │
│  │  │  → 大幅缩减 KV Cache（比 GQA 更省显存）  │ │     │
│  │  └──────────────────────────────────────────┘ │     │
│  │       ↓                                        │     │
│  │  ┌─ DeepSeekMoE FFN ───────────────────────┐  │     │
│  │  │  1 shared expert (始终激活)               │  │     │
│  │  │  + 256 routed experts (top-8 激活)        │  │     │
│  │  │  → 无辅助损失负载均衡                     │  │     │
│  │  └──────────────────────────────────────────┘  │     │
│  │       ↓                                        │     │
│  │  RMSNorm + 残差连接                            │     │
│  └────────────────────────────────────────────────┘     │
│       ↓                                                 │
│  Output Head (+ Multi-Token Prediction 辅助头)          │
└─────────────────────────────────────────────────────────┘
```

```
┌───────────────────────────────────────────────┐
│  Qwen2.5 架构 (Dense, 0.5B–72B)              │
├───────────────────────────────────────────────┤
│                                               │
│  Input Embedding + RoPE                       │
│       ↓                                       │
│  ┌─ Transformer Block × N ────────────┐      │
│  │  Grouped Query Attention (GQA)      │      │
│  │       ↓                             │      │
│  │  SwiGLU FFN                         │      │
│  │       ↓                             │      │
│  │  RMSNorm + 残差连接                 │      │
│  └─────────────────────────────────────┘      │
│       ↓                                       │
│  Output Head                                  │
└───────────────────────────────────────────────┘
```

**架构关键差异对比**

| 维度 | DeepSeek V3 | Qwen2.5 |
|------|------------|---------|
| 参数规模 | 671B total / 37B active | 0.5B–72B (Dense) |
| 注意力 | MLA（低秩潜向量压缩 KV） | GQA（分组共享 KV head） |
| FFN | MoE：1 shared + 256 routed (top-8) | Dense SwiGLU |
| 位置编码 | RoPE | RoPE |
| 归一化 | RMSNorm (Pre-LN) | RMSNorm (Pre-LN) |
| 辅助训练 | Multi-Token Prediction | — |
| KV Cache 开销 | 极低（MLA 压缩） | 中等（GQA 分组共享） |

### 训练流程

```
DeepSeek V3 + R1 完整流程：

 ┌──────────── V3 Base ────────────┐
 │                                  │
 │  数据: 14.8T tokens             │
 │  精度: FP8 混合精度              │
 │  并行: Pipeline + Expert        │
 │  目标: CLM + MTP 辅助损失       │
 │  硬件: 2048× H800               │
 │                                  │
 └──────────────┬───────────────────┘
                │
                ├────────────────────────────────┐
                │                                │
                ▼                                ▼
 ┌──────── V3 Chat (SFT + RL) ────┐  ┌──────────── R1 推理模型 ────────┐
 │                                  │  │                                  │
 │  SFT: 150 万条高质量对话        │  │  阶段 1: 冷启动 SFT             │
 │  RL:  GRPO (无 Critic 模型)     │  │     少量 long-CoT 示例           │
 │                                  │  │           ↓                      │
 └──────────────────────────────────┘  │  阶段 2: 大规模 RL (GRPO)       │
                                       │     规则奖励: 正确性 + 格式     │
                                       │     模型自发涌现 CoT 推理       │
                                       │           ↓                      │
                                       │  阶段 3: 拒绝采样 → SFT         │
                                       │     用 RL 模型生成高质量数据     │
                                       │     混合通用 SFT 数据再训        │
                                       │           ↓                      │
                                       │  阶段 4: 二次 RL                 │
                                       │     全场景对齐(推理+通用)        │
                                       │                                  │
                                       └──────────────────────────────────┘
```

```
Qwen2.5 训练流程：

  数据: 约 18T tokens (多语言，官方未公开确切数字)
       ↓
  Pre-train (多阶段: 通用 → 长上下文扩展)
       ↓
  SFT (大规模高质量指令数据)
       ↓
  Alignment: DPO (离线偏好优化)
       ↓
  部署：原生 Dense / GGUF 量化 / vLLM
```

**训练关键差异对比**

| 维度 | DeepSeek V3/R1 | Qwen2.5 |
|------|---------------|---------|
| 预训练数据 | 14.8T tokens | 约 18T tokens |
| 预训练精度 | FP8 | BF16 |
| 对齐方法 | GRPO（无 Critic，组内相对奖励） | DPO |
| 推理能力来源 | 纯 RL 涌现 CoT（R1 路线） | SFT + 蒸馏 R1 数据 |
| MTP 辅助 | 有（预测多个未来 token） | 无 |
| 推理阶段特色 | 多阶段 RL + 拒绝采样迭代 | 单轮 DPO |

### 关键技术点解读

**MLA (Multi-head Latent Attention)** — 标准 MHA 每个 head 独立存 K/V，显存随层数×头数线性增长。MLA 将 K/V 投射到低维潜向量，推理时只缓存潜向量，解码时再投射回去。效果：KV Cache 比 GQA 还小，长序列优势巨大。

**DeepSeekMoE** — 256 个路由专家只激活 top-8，加 1 个始终激活的共享专家兜底通用能力。负载均衡不用辅助损失（避免干扰主损失），改用 token 级别的动态路由偏置。

**GRPO (Group Relative Policy Optimization)** — PPO 需要 Critic 模型估计 baseline（贵）。GRPO 改为：对同一 prompt 采样一组回答，用组内平均奖励做 baseline。省掉 Critic，训练成本砍半。

**Multi-Token Prediction (MTP)** — 在 CLM 主目标之外，额外预测未来 2~3 个 token。增强表示质量，预训练后可丢弃或用于 speculative decoding 加速推理。

---

## 附录：术语速查表

| 缩写 | 全称 | 一句话 |
|------|------|--------|
| AE | Autoencoder | 编码器压缩 + 解码器重建，学习数据的压缩表示 |
| Agent | AI Agent | 模型 + 工具调用 + 多步规划，自主完成任务 |
| BPE | Byte Pair Encoding | 子词分词算法，从字符对频率迭代合并 |
| CLM | Causal Language Modeling | 自回归语言建模，根据前文预测下一个 token |
| CNN | Convolutional Neural Network | 卷积神经网络，擅长捕捉局部空间特征 |
| CoT | Chain of Thought | 思维链，让模型分步推理再给结论 |
| DPO | Direct Preference Optimization | 直接偏好优化，无需训练奖励模型的对齐方法 |
| DiT | Diffusion Transformer | 用 Transformer 替代 U-Net 做扩散模型骨干 |
| FFN | Feed-Forward Network | Transformer 内的逐位置全连接层 |
| GAN | Generative Adversarial Network | 生成对抗网络，生成器与判别器博弈 |
| GQA | Grouped Query Attention | 分组查询注意力，多个 Q head 共享一组 KV，省显存 |
| GRPO | Group Relative Policy Optimization | 组内相对奖励策略优化，无需 Critic 模型的 RL 方法 |
| GRU | Gated Recurrent Unit | 门控循环单元，LSTM 简化变体 |
| KV Cache | Key-Value Cache | 推理时缓存已计算的 K/V 矩阵避免重复计算 |
| LLM | Large Language Model | 大语言模型 |
| LoRA | Low-Rank Adaptation | 低秩适配，冻结原参数只训小矩阵，省显存 |
| LSTM | Long Short-Term Memory | 长短期记忆网络，用门控解决 RNN 长距离遗忘 |
| MCP | Model Context Protocol | 标准化模型与外部工具/数据源的连接协议 |
| MLA | Multi-head Latent Attention | 多头潜注意力，将 KV 压缩到低维潜向量，极省显存 |
| MLM | Masked Language Modeling | 掩码语言建模，遮住部分 token 让模型预测（BERT） |
| MLP | Multi-Layer Perceptron | 多层感知机，最基础的前馈全连接网络 |
| MoE | Mixture of Experts | 混合专家，稀疏激活降低计算量 |
| MTP | Multi-Token Prediction | 多 token 预测，辅助训练目标，同时预测未来多个 token |
| QLoRA | Quantized LoRA | LoRA + 4-bit 量化底座，进一步省显存 |
| RAG | Retrieval-Augmented Generation | 检索增强生成，外挂知识库缓解幻觉 |
| RL | Reinforcement Learning | 强化学习 |
| RLHF | RL from Human Feedback | 基于人类反馈的强化学习对齐方法 |
| RMSNorm | Root Mean Square Normalization | 均方根归一化，比 LayerNorm 更快的归一化方案 |
| RNN | Recurrent Neural Network | 循环神经网络，隐状态逐步传递建模序列 |
| RoPE | Rotary Position Embedding | 旋转位置编码，Transformer 相对位置方案 |
| SFT | Supervised Fine-Tuning | 有监督微调 |
| SwiGLU | Swish-Gated Linear Unit | 带门控的激活函数，现代 Transformer FFN 常用 |
| VAE | Variational Autoencoder | 变分自编码器，潜空间连续化可采样生成 |
| VLM | Vision-Language Model | 视觉语言模型，图文多模态理解 |
