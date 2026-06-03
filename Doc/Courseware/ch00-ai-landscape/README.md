# ch00 · AI 全景与概念速查

> **不用现在全懂。** 本文是地图，不是考试。先建立整体印象，遇到陌生词翻到末尾的[术语速查表](#附录术语速查表)即可。

## 0. 引言

2022 年底，ChatGPT 一夜之间让全世界知道了 "大语言模型"。但在它背后，是从感知机到 Transformer 长达六十年的积累，是 Scaling Law 的量化预言，是 RLHF 对齐技术的临门一脚。

这篇文档帮你快速建立全局认知：AI 怎么走到今天、核心概念之间什么关系、一个模型从训练到上线经历了哪些阶段、当下前沿在做什么。不求深，但求 "知道自己不知道什么"。

---

## 1. 前史速览

| 年代 | 事件 | 意义 |
|------|------|------|
| 1958 | 感知机 (Perceptron) | 第一个可学习的神经网络单元 |
| 1969 | Minsky《Perceptrons》出版 | 数学证明感知机局限性，成为 AI 第一次寒冬的导火索——此后神经网络沉寂近 20 年 |
| 1986 | 反向传播算法普及 | 多层网络可训练，寒冬回暖 |
| 2012 | AlexNet 夺冠 ImageNet | 深度学习实用化信号，GPU 训练起飞 |

> 以上仅供定位"深度学习从哪里来"。本文重点从下一节开始。

---

## 2. 当代大模型时间线（2013–2025）

| 年份 | 里程碑 | 一句话意义 |
|------|--------|-----------|
| 2013 | Word2Vec | 词可以变成向量，向量可以做算术（king - man + woman ≈ queen） |
| 2014 | RNN/Seq2Seq / GAN | 序列到序列翻译框架；对抗生成范式诞生 |
| 2015 | Attention 机制 | 解码时动态聚焦源端相关位置，性能跃升 |
| 2017 | **Transformer** | "Attention Is All You Need"，抛弃 RNN 的循环，全靠注意力+并行 |
| 2018 | GPT-1 / BERT | 预训练+微调范式确立；自回归 (Autoregressive) vs 双向掩码 (Bidirectional Masked) 两条路线分野 |
| 2020 | **Scaling Law / GPT-3** | Scaling Law（2020 初）量化了"规模→性能"规律；GPT-3 (1750亿参数) 是其验证产物 |
| 2020 | Vision Transformer (ViT) | Transformer 入侵视觉领域，证明注意力不止能做 NLP |
| 2021 | DALL·E 1 / CLIP | 文本↔图像对齐，多模态时代序幕 |
| 2022 | **ChatGPT / InstructGPT** | RLHF 对齐落地，LLM 进入大众视野 |
| 2022 | Stable Diffusion | 潜空间扩散，开源图像生成爆发 |
| 2023 | **GPT-4 / LLaMA** | 多模态闭源标杆；开源权重潮涌 |
| 2023 | MoE 规模化 (Mixtral) | 稀疏激活降本，同等效果少用算力 |
| 2024 | Sora / 视频生成 | DiT 架构驱动长视频生成 |
| 2024 | 开源追平 (Qwen2/LLaMA3/DeepSeek) | 开源模型能力逼近闭源前沿 |
| 2025 | **DeepSeek-R1 / 推理模型 / Agent** | 开源推理模型比肩闭源；模型学会 "思考" 与 "使用工具" |

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
       └─ 神经网络（多层时称"深度学习"）
           ├─ MLP (多层感知机)
           ├─ CNN (卷积神经网络)
           ├─ RNN → LSTM / GRU (循环神经网络)
           └─ Transformer (Attention 机制)
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

> 注：还有一种**半监督学习**（少量标签 + 大量无标签数据混合训练），工业界常见但在当代大模型链路中不是主角，此处不展开。

**生动类比**

- **监督学习** — 像做带答案的习题集：老师给你题目和标准答案，你反复练到能举一反三。
- **无监督学习** — 像整理一堆没标签的照片：没人告诉你分类标准，你自己发现"这些是风景、那些是人像"。
- **自监督学习** — 像完形填空：把文章挖几个空让你猜，答案就藏在原文里，不需要额外标注。GPT 等当代 LLM 被训练出来的 "预测下一个词" 本质就是这个。
- **强化学习** — 像训练小狗：做对了给零食（奖励），做错了没有。小狗通过反复试错学会"坐下"和"握手"。

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
 │    "用卷积核扫描局部特征"           │
 │                                  │
 └──→ RNN → LSTM / GRU              │ 2017: 全部汇聚
      "隐状态传递记忆"                │
       │                            │
       └→ Seq2Seq + Attention ──────→ Transformer
           "解码器动态看源端"         "全靠注意力，并行训练"
```

### 每一站的故事

| 架构 | 核心思想 | 擅长 | 主要瓶颈 |
|------|---------|------|---------|
| MLP | 全连接层堆叠 | 通用基础积木 | 无法捕捉空间/序列结构 |
| CNN | 局部感受野 + 权重共享 + 池化 | 图像、空间特征 | 全局依赖需要很深 |
| RNN / LSTM / GRU | 隐状态逐步传递 | 序列、时序 | 长距离遗忘，无法并行 |
| Transformer | 自注意力 + 位置编码 | 全局依赖，天然并行 | 计算量随序列长度二次增长 |

---

**MLP（多层感知机）**

> 类比：一堆人站成几排传纸条——每个人能看到上一排所有人递来的内容，但完全不知道"顺序"和"位置"有什么含义。

```
输入 [x₁, x₂, ..., xₙ]
      ↓ 全连接（每个输入连到每个神经元）
隐藏层 [h₁, h₂, ..., hₘ]  ← 激活函数（ReLU 等）
      ↓ 全连接
输出 [y₁, y₂, ..., yₖ]
```

能拟合任意函数（万能近似定理），但对空间/序列结构完全无感——把图片像素打乱顺序喂进去，它毫无察觉。参数量随输入维度爆炸（1000×1000 图片 → 百万维全连接）。

---

**CNN（卷积神经网络）**

> 类比：拿一个小放大镜在图片上逐块扫描——每次只看局部，但同一个放大镜（同一组权重）扫遍整张图，找出所有"边缘""纹理"。

```
输入图像 [H × W × C]
      ↓ 卷积核滑动扫描（提取局部特征）
特征图 [H' × W' × F]   ← 多个卷积核 = 多种特征
      ↓ 池化（下采样，压缩空间尺寸）
更小特征图
      ↓ 重复 N 层（浅层→边缘 / 中层→纹理 / 深层→物体部件）
      ↓ 展平 → MLP
分类输出（经典应用 = 图像识别/分类）
```

关键设计：**权重共享**（一个卷积核全图复用，参数极少）+ **池化**（压缩空间 + 平移不变性）。成就：2012 AlexNet 引爆深度学习。局限：感受野有限，全局依赖需堆很多层。

---

**RNN → LSTM/GRU**

> 类比：逐字朗读一篇文章，同时在脑中维护一份"到目前为止的摘要"。每读一个字就更新摘要，用摘要指导后续理解。读到第 500 页时，第 1 页的细节早忘了。

```
输入序列: x₁, x₂, x₃, ..., xₜ
           ↓    ↓    ↓         ↓
RNN:    [h₀]→[h₁]→[h₂]→[h₃]→...→[hₜ] → 输出
         初始  每步: hₜ = f(hₜ₋₁, xₜ)
         状态  "用上一步的记忆 + 当前输入 → 新记忆"
```

问题：信息在反复覆盖中稀释殆尽（梯度消失）。LSTM 加三个"门"（遗忘门/输入门/输出门）控制信息存取，缓解遗忘但未根治。致命局限：必须**串行**（h₂ 依赖 h₁），无法并行，训练慢。

---

**Seq2Seq + Attention**

> 类比：同声传译——翻译每个词时不是死记整句话，而是**回头扫一眼**原文中最相关的部分，动态聚焦。

```
编码器 (RNN):  [法语 x₁...xₙ] → 隐状态序列 [h₁, h₂, ..., hₙ]

解码器 (RNN):  生成每个英语词时:
               ① 算注意力权重 αᵢ = 对齐(当前解码状态, hᵢ)
               ② 加权求和 context = Σ αᵢ·hᵢ  ← "聚焦源端相关位置"
               ③ context + 上一个输出 → 生成下一个词
```

效果：翻译质量飞跃（终于不用把整句话压成一个固定向量了）。但编码/解码仍是 RNN，串行瓶颈未消。

---

**Transformer**

> 类比：一间教室里所有学生同时互相交流，每个人瞬间获取任何人的信息——沟通效率极高，但人数多了开销爆炸。

```
输入 tokens + 位置编码
      ↓
┌─ Transformer Block × N ─────────────────┐
│                                          │
│  自注意力: 每个 token 同时看所有其他 token │
│     Q·Kᵀ → 权重 → 加权求和 V            │
│           ↓                              │
│  前馈网络 (FFN): 逐位置独立变换           │
│           ↓                              │
│  残差连接 + 归一化                        │
└──────────────────────────────────────────┘
      ↓
输出
```

激进一步：RNN 全扔掉，编码和解码都只用注意力。每个块两大组件：自注意力（捕捉 token 间关系）+ FFN（逐位置独立变换，提供非线性拟合能力）。位置信息靠位置编码注入。收益：**完全并行**（所有 token 同时互看）+ **全局依赖一步到位**（第 1 个和第 1000 个 token 直接交互）。代价：注意力计算 O(n²)。自此之后，几乎所有 SOTA 模型都基于 Transformer 变体。

---

## 5. 当下模型方向

> 本节概览各模态主流技术路线，后续内容聚焦 LLM（文本生成）。

| 模态 | 主流架构 | 训练方法 | 代表 |
|------|---------|---------|------|
| 文本生成 | Decoder-only Transformer | CLM 自回归 | GPT-4、LLaMA、Qwen |
| 文本理解 | Encoder-only Transformer | MLM 掩码 | BERT、RoBERTa |
| 图像生成 | DiT / U-Net | 扩散 (Diffusion) | Stable Diffusion、DALL·E 3 |
| 视频生成 | DiT 变体 | 扩散 + 时序建模 | Sora、Kling |
| 多模态理解 | VLM（视觉编码器 + LLM） | 对比学习 + CLM | GPT-4V、Qwen-VL |
| 代码 | Decoder-only Transformer | CLM + Fill-in-Middle | Codex、DeepSeek-Coder |

> 注：以上训练方法均属于**自监督学习**（从数据自身构造学习信号，无需人工标注）。**监督学习**（SFT）和**强化学习**（RLHF/DPO）在后续微调、对齐阶段介入。

### 为什么文本走自回归、图像走扩散？

扩散模型一句话原理：训练时对图像**逐步**加噪直到变成纯噪声，然后训练网络学会逆过程（去噪）。推理时从纯随机噪声出发，**逐步**去噪，最终生成清晰图像。

- **文本**天然是离散序列，左到右逐 token 生成符合语言本质，CLM 自回归简洁高效。
- **图像**是高维连续信号，像素之间强空间相关性；扩散模型通过逐步去噪，能在潜空间稳定生成高质量图像，比 GAN 更易训练、生成质量和训练稳定性优于自回归像素生成。

> 注：图像与视频生成不再展开，本系列课件聚焦 LLM。

---

## 6. LLM 全链路地图

### 什么是预训练？

预训练 = 拿海量无标注文本，让模型反复做一件事：**根据前文预测下一个 token**（即 CLM）。不需要人工标注，模型通过数万亿次预测，自己学会语言规律、事实知识和推理模式。这一步成本最高（数千张 GPU 跑数周），产出称为 base 模型。

### 为什么预训练不够？

- **Pretrain** 产出 base 模型 → 会续写文本，但不会对话。你说 "北京天气怎样"，它可能接 "预报显示明天晴转多云……后天……" 无限续写，而不是回答你的问题。
- **SFT** 教会对话格式 → 能正常对话了，但可能输出有害内容或质量参差不齐。它学了 "怎么说话"，没学 "什么该说什么不该说"。
- **Alignment** 对齐人类偏好 → 好用且安全了。但数百亿参数的模型普通人跑不起来。
- **量化** 精度换显存 → FP16 压到 INT4，显存省 4 倍，终于能在消费级硬件跑起来。

```
数据  → Pretrain  → SFT  → Alignment  → 量化  → 部署推理  → 上层应用
       (base模型)  (能对话)  (好用安全)  (跑得动)  (跑起来)    (用起来)
```

### 训练阶段

| 阶段 | 做什么 | 产出 |
|------|--------|------|
| Pretrain | 自监督 next token prediction，海量文本，烧大算力 | Base 模型（会续写，不会对话） |
| SFT | 有监督微调，学对话格式与指令服从 | Chat 模型（能对话） |
| Alignment | 对齐人类偏好（RLHF / DPO），变好用+无害 | 对齐后模型 |
| 量化 | FP16 → INT8/INT4，精度换显存 | 可本地部署的模型 |

#### SFT 具体怎么做？

核心：用 "指令 + 标准回答" 的配对数据，教 base 模型学会对话格式。

```
训练样本:
[User] 用一句话解释什么是黑洞
[Assistant] 黑洞是时空中引力极强、连光都无法逃逸的区域。
```

Base 模型只会无限续写，SFT 让它学到：看到 `[User]` 就该在 `[Assistant]` 后给出简洁回答然后停下来。本质是用监督学习把 "对话行为模式" 注入模型。数据质量 > 数量 —— 早期 7B 级模型几万条高质量样本就能显著改变行为，前沿大模型（如 DeepSeek V3）则用到百万级。

#### Alignment 具体怎么做？

核心：让人类当裁判，告诉模型 "哪个回答更好"，模型据此调整行为。

```
Prompt: "如何减肥？"
回答 A (chosen ✓): "建议控制饮食+适量运动，具体可以……"
回答 B (rejected ✗): "直接绝食三天，效果立竿见影"
→ 模型学到：往 A 靠，远离 B
```

- **RLHF**：先训一个"裁判"（奖励模型），再用 PPO 让 LLM 最大化裁判打分。两阶段，贵且不稳定。
- **DPO**：跳过裁判，直接拿偏好对数据优化 LLM（数学上等价于隐式奖励模型）。一阶段，简单稳定，开源主流。

### 推理与部署（概述）

模型训完后需要高效跑起来。类比：训练像 "培养一个专家"，推理像 "让专家上岗接客" —— 上岗时要考虑响应速度和接待能力。

核心优化思路：

- **KV Cache** — 类比：写长文章时边写边在旁边记 "要点清单"，每写完一句就追加一条，下一句直接看清单，不用从头重读全文重新提炼。模型生成每个新 token 时，把已生成 token 的 K、V 向量缓存下来，下一步直接复用，避免重算。
- **量化** — 类比：把精装全彩教材换成黑白口袋本——内容几乎一样，但体积小得多。将模型参数从高精度（FP16）压缩到低精度（INT8/INT4），显存开销大幅下降，精度损失很小。
- **Continuous Batching** — 多个用户请求动态拼批处理（多 slot 独立 KV Cache、采样规则），提升 GPU 利用率。

> 本课程后续章节课件会展开。

### 上层应用

| 概念 | 一句话 |
|------|--------|
| Prompt Engineering | 不改模型只改输入——CoT（思维链）/ few-shot / role play |
| RAG | 检索外部知识注入 prompt，缓解幻觉，让模型 "有据可查" |
| Function Calling | 模型按格式输出工具调用指令，外部执行后回传结果 |
| MCP | Model Context Protocol，标准化模型与外部工具/数据源的连接协议 |
| Skill | Agent 的可复用能力单元，封装特定任务的 prompt + 工具组合 |
| Agent | 模型 + 工具调用 + 多步规划，从 "回答问题" 到 "完成任务" |

> 类比：如果 LLM 是大脑，那么
> 
> - Prompt Engineering 是 "问对问题的技巧"
> - RAG 是 "查资料"
> - Function Calling 是 "动手操作"
> - MCP 是 "统一的工具接口标准"
> - Skill 是 "学会的一项具体技能"
> - Agent 是 "自主规划+执行一整套流程"。

想看工业界真实案例如何落地以上全链路？见下方[附录](#附录前沿模型完整链路deepseek-v3--r1--qwen25)。

---

## 附录：前沿模型完整链路（DeepSeek V3 / R1 & Qwen2.5）

> 以下展示 2024–2025 年前沿开源模型的真实全链路。信息密度较高，初次阅读完全可以跳过——等学完注意力机制再回来细读效果更好。

### Dense vs MoE

- **Dense（稠密）**：每个 token 经过网络的全部参数。简单直接，但参数量 ≈ 计算量，大了就贵。
- **MoE（混合专家）**：把 FFN 拆成多个"专家"子网络，每个 token 只路由到少数几个专家。总参数量很大（知识容量大），但单次推理只激活一小部分（计算成本可控）。

下方 DeepSeek V3 采用 MoE（671B 总参数 / 37B 激活），Qwen2.5 采用 Dense。

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
│  Qwen2.5 架构 (Dense, 0.5B–72B)               │
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

> 下方 ASCII 流程图较宽，建议在宽屏下查看。

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
 ┌──────── V3 Chat (SFT + RL) ───┐  ┌──────────── R1 推理模型 ────────┐
 │                               │  │                                │
 │  SFT: 150 万条高质量对话        │  │  阶段 1: 冷启动 SFT             │
 │  RL:  GRPO (无 Critic 模型)    │  │     少量 long-CoT 示例          │
 │                               │  │           ↓                    │
 └─────────────┬─────────────────┘  │  阶段 2: 大规模 RL (GRPO)       │
               │                    │     规则奖励: 正确性 + 格式      │
               ▼                    │     模型自发涌现 CoT 推理        │
 部署：FP8 推理 / GGUF 量化 /         │           ↓                    │
       SGLang / vLLM                │  阶段 3: 拒绝采样 → SFT         │
                                    │     用 RL 模型生成高质量数据     │
                                    │     混合通用 SFT 数据再训        │
                                    │           ↓                    │
                                    │  阶段 4: 二次 RL                │
                                    │     全场景对齐(推理+通用)        │
                                    │           ↓                    │
                                    │  部署：动态量化 / GGUF /         │
                                    │        SGLang / vLLM           │
                                    │                                │
                                    └────────────────────────────────┘
```

```
Qwen2.5 训练流程：

  数据: 约 18T tokens (多语言，官方未公开确切数字)
       ↓
  Pre-train (多阶段: 通用 → 长上下文扩展)
       ↓
  SFT (大规模高质量指令数据)
       ↓
  Alignment: DPO 为主 (离线偏好优化)
       ↓
  部署：原生 Dense / GGUF 量化 / vLLM
```

**训练关键差异对比**

| 维度 | DeepSeek V3/R1 | Qwen2.5 |
|------|---------------|---------|
| 预训练数据 | 14.8T tokens | 约 18T tokens |
| 预训练精度 | FP8 | BF16 |
| 对齐方法 | GRPO（无 Critic，组内相对奖励） | DPO 为主 |
| 推理能力来源 | 纯 RL 涌现 CoT（R1 路线） | SFT + DPO（数学/代码专项 RL） |
| MTP 辅助 | 有（预测多个未来 token） | 无 |
| 推理阶段特色 | 多阶段 RL + 拒绝采样迭代 | 单轮 DPO |

### 关键技术点解读

**MLA (Multi-head Latent Attention)** — 标准 MHA 每个 head 独立存 K/V，显存随层数×头数线性增长。MLA 将 K/V 投射到低维潜向量，推理时只缓存潜向量，解码时再投射回去。效果：KV Cache 比 GQA 还小，长序列优势巨大。

**DeepSeek MoE** — 256 个路由专家只激活 top-8，加 1 个始终激活的共享专家兜底通用能力。负载均衡不用辅助损失（避免干扰主损失），改用 token 级别的动态路由偏置。

**GRPO (Group Relative Policy Optimization)** — PPO 需要 Critic 模型估计 baseline（贵）。GRPO 改为：对同一 prompt 采样一组回答，用组内平均奖励做 baseline。省掉 Critic，训练成本砍半。

**Multi-Token Prediction (MTP)** — 在 CLM 主目标之外，额外预测未来 2~3 个 token。增强表示质量，预训练后可丢弃或用于 speculative decoding 加速推理。

---

## 附录：主流 LLM 架构差异速查

横向对比代表性模型的关键架构选型。部分闭源模型架构未公开，标注为推测。

| 维度 | GPT-2 | LLaMA-3-8B | Qwen-2.5-72B | DeepSeek-V3-671B | Qwen-3-235B | DeepSeek-4.0 | GPT-4o |
|------|-------|------------|--------------|-------------|-------------|--------------|--------|
| **参数量** | 1.5B | 8B | 72B | 671B (37B active) | 235B (22B active) | ~1T (37-50B active) | 未公开 |
| **n_layers** | 48 | 32 | 80 | 61 | 94 | 64-80 | 未公开 |
| **d_model** | 1600 | 4096 | 8192 | 7168 | ~8192 | 8192-10240 | 未公开 |
| **n_heads** | 25 | 32 | 64 | 128 | 64 (Q) / 4 (KV) | 128 | 未公开 |
| **归一化类型** | LayerNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm | 未公开 |
| **归一化位置** | Pre-LN | Pre-LN | Pre-LN | Pre-LN | Pre-LN | Pre-LN | 推测 Pre-LN |
| **位置编码** | 学习式 | RoPE | RoPE | RoPE | RoPE + ABF + YARN + DCA | RoPE（高基频，原生 1M） | 推测 RoPE |
| **注意力类型** | MHA | GQA | GQA | MLA | GQA | MLA v2 | 未公开 |
| **KV heads** | 25 (全 MHA) | 8 | 8 | — (潜向量) | 4 | — (潜向量) | 未公开 |
| **FFN 激活** | GELU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | 未公开 |
| **FFN 结构** | Dense | Dense | Dense | MoE (1+256, top-8) | MoE (128, top-8) | MoE (1+512, top-8) | 推测 MoE |
| **词表大小** | 50,257 | 128,256 | 152,064 | 129,280 | 151,669 | 150,000-200,000 | 未公开 |
| **原生上下文长度** | 1,024 | 8,192 | 131,072 | 128,000 | 128,000 | 256K-1M | 128,000 |

> 数据截止至 2026-05-20

**趋势观察**：

- 归一化：LayerNorm → RMSNorm（省计算，效果持平）
- 位置编码：学习式 → RoPE（支持长上下文外推）
- 注意力：MHA → GQA → MLA（逐步压缩 KV Cache 开销）
- FFN 激活：GELU → SwiGLU（效果更好）
- FFN 结构：Dense → MoE（大模型降本，中小模型通常仍用 Dense）

---

## 附录：AI 概念全景分类树

> §3 的分类树聚焦"当代 ML"维度。这里补一张从历史流派到具体方法的全景图，帮助定位各概念的来龙去脉。

```
AI (Artificial Intelligence，人工智能)
├── 符号主义 (Symbolism / GOFAI)         ← 主导 1950s–1980s，现已边缘
│   ├── Expert System (专家系统)          ← MYCIN、DENDRAL、XCON
│   ├── Knowledge Graph (知识图谱)        ← 仍活跃：搜索、电商、金融风控
│   ├── Logic / Prolog (逻辑推理)
│   └── Planning & Search (规划与搜索)    ← A*、STRIPS、AlphaGo 的 MCTS (Monte Carlo Tree Search，蒙特卡洛树搜索)
│
├── 连接主义 (Connectionism)              ← 神经网络派，当下绝对主流
│   └── Machine Learning (ML，机器学习)
│       ├── 传统 ML
│       │   ├── Linear / Logistic Regression (线性 / 逻辑回归)
│       │   ├── Decision Tree (决策树) → Random Forest → GBDT (XGBoost / LightGBM)
│       │   ├── Support Vector Machine (SVM，支持向量机)
│       │   ├── Naive Bayes (朴素贝叶斯)
│       │   └── K-Means / PCA (聚类 / 降维)
│       └── Neural Network (NN，多层时称 DL - Deep Learning)
│            ├── MLP / FNN              ← 万能近似器，所有 NN 基本盘
│            ├── CNN                    ← 图像主力
│            ├── RNN / LSTM / GRU       ← 序列老兵，被 Transformer 替代
│            └── Transformer            ← 当下统治者（GPT / BERT / LLaMA …）
│                ├── Encoder-only       ← BERT 系，理解任务
│                ├── Decoder-only       ← ** GPT 系，LLM 生成主流 **
│                └── Encoder-Decoder    ← T5 / Whisper，翻译类
│            ── 其他变体概念：
│            ├── Autoencoder / VAE      ← 表示学习、生成模型源头
│            ├── GAN                    ← 生成对抗，2014–2020 图像生成主流
│            ├── Diffusion              ← 当下图像 / 视频生成主流
│            ├── MoE                    ← 稀疏激活，大参小算（DeepSeek / Mixtral）
│            ├── Mamba / SSM            ← Transformer 挑战者，长序列友好
│
├── 行为主义 (Behaviorism / Cybernetics)  ← 思想源控制论
│   ├── Reinforcement Learning (RL，强化学习)  ← 学术也归 ML 第三学习范式
│   │   ├── Value-based (Q-Learning, DQN)
│   │   ├── Policy-based (REINFORCE, PPO)
│   │   └── Actor-Critic (A3C, SAC)
│   └── Robotics (机器人学)                ← 含传统控制论
│
└── Evolutionary Computation (演化计算)   ← 小众但独立
    ├── Genetic Algorithm (GA，遗传算法)
    └── Neural Architecture Search (NAS)  ← 与 DL 结合复活
```

> 注：流派界限并非绝对，现代系统常融合多种思路（如 LLM + 知识图谱、RL + Transformer = RLHF）。本课程聚焦的是连接主义路线。

---

## 附录：术语速查表

| 术语 | 全称 | 中译 | 一句话 |
|------|------|------|--------|
| AE | Autoencoder | 自编码器 | 编码器压缩 + 解码器重建，学习数据的压缩表示 |
| Agent | AI Agent | - | 模型 + 工具调用 + 多步规划，自主完成任务 |
| BPE | Byte Pair Encoding | 字节对编码 | 子词分词算法，从字符对频率迭代合并 |
| CLM | Causal Language Modeling | 因果语言建模 | 自回归语言建模，根据前文预测下一个 token |
| CNN | Convolutional Neural Network | 卷积神经网络 | 擅长捕捉局部空间特征 |
| Continuous Batching | - | 连续批处理 | 动态拼批多请求，提升 GPU 利用率 |
| CoT | Chain of Thought | 思维链 | 让模型分步推理再给结论 |
| DPO | Direct Preference Optimization | 直接偏好优化 | 无需训练奖励模型的对齐方法 |
| DiT | Diffusion Transformer | - | 用 Transformer 替代 U-Net 做扩散模型骨干 |
| FFN | Feed-Forward Network | 前馈网络 | Transformer 内的逐位置全连接层 |
| FIM | Fill-in-Middle | 中间填充 | 给定前缀和后缀让模型补全中间内容，代码补全常用 |
| GAN | Generative Adversarial Network | 生成对抗网络 | 生成器与判别器博弈 |
| GQA | Grouped Query Attention | 分组查询注意力 | 多个 Q head 共享一组 KV，省显存 |
| GRPO | Group Relative Policy Optimization | - | 组内相对奖励策略优化，无需 Critic 模型的 RL 方法 |
| GRU | Gated Recurrent Unit | 门控循环单元 | LSTM 简化变体 |
| KV Cache | Key-Value Cache | 键值缓存 | 推理时缓存已计算的 K/V 矩阵避免重复计算 |
| LLM | Large Language Model | 大语言模型 | - |
| LoRA | Low-Rank Adaptation | 低秩适配 | 冻结原参数只训小矩阵，省显存 |
| LSTM | Long Short-Term Memory | 长短期记忆网络 | 用门控解决 RNN 长距离遗忘 |
| MCP | Model Context Protocol | - | 标准化模型与外部工具/数据源的连接协议 |
| MLA | Multi-head Latent Attention | 多头潜注意力 | 将 KV 压缩到低维潜向量，极省显存 |
| MLM | Masked Language Modeling | 掩码语言建模 | 遮住部分 token 让模型预测（BERT） |
| MLP | Multi-Layer Perceptron | 多层感知机 | 最基础的前馈全连接网络 |
| MoE | Mixture of Experts | 混合专家 | 稀疏激活降低计算量 |
| MTP | Multi-Token Prediction | 多 token 预测 | 辅助训练目标，同时预测未来多个 token |
| PPO | Proximal Policy Optimization | 近端策略优化 | RLHF 中常用的策略梯度算法 |
| QLoRA | Quantized LoRA | - | LoRA + 4-bit 量化底座，进一步省显存 |
| RAG | Retrieval-Augmented Generation | 检索增强生成 | 外挂知识库缓解幻觉 |
| RL | Reinforcement Learning | 强化学习 | - |
| RLHF | RL from Human Feedback | - | 基于人类反馈的强化学习对齐方法 |
| RMSNorm | Root Mean Square Normalization | 均方根归一化 | 比 LayerNorm 更快的归一化方案 |
| RNN | Recurrent Neural Network | 循环神经网络 | 隐状态逐步传递建模序列 |
| RoPE | Rotary Position Embedding | 旋转位置编码 | Transformer 相对位置方案 |
| SFT | Supervised Fine-Tuning | 有监督微调 | - |
| SOTA | State of the Art | 当前最优 | 某任务/基准上的最佳性能或模型 |
| Speculative Decoding | - | 推测解码 | 用小模型草拟多 token，大模型一次验证，加速推理 |
| SwiGLU | Swish-Gated Linear Unit | - | 带门控的激活函数，现代 Transformer FFN 常用 |
| VAE | Variational Autoencoder | 变分自编码器 | 潜空间连续化可采样生成 |
| VLM | Vision-Language Model | 视觉语言模型 | 图文多模态理解 |
