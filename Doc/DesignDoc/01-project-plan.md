# LLM 学习仓库 项目计划书

> 本文是所有未来要做事项的整理与拆分。以里程碑驱动，每个里程碑都有明确的可交付物。
> 配套根基文档见 `00-startup-proposal.md`。

## 0. 里程碑总览

| 里程碑 | 名称 | 关键交付物 | 预估规模 |
|---|---|---|---|
| M0 | 仓库基建 | 环境就绪（双平台 doctor 通过）、课程大纲、README、同步策略 | 小 |
| M1 | 前置知识 | PyTorch + 数学补课 + NN 基础课件 & 练习 | 中 |
| M2 | Transformer 精通 | 手写 Transformer、注意力机制课件 & 练习 | 中 |
| M3 | LLM 全链路入门 | 分词器、预训练原理、SFT 原理课件 | 中 |
| M4 | echo-mini 落地 | 能跑通 Pretrain→SFT→评测的迷你模型（full/tiny 双配置） | 大 |
| M5 | echo 落地 | 基于开源底座微调出的可对话 Echo（Win/Mac 推理双端可用） | 大 |
| M6 | 对齐与优化 | DPO / 量化 / 部署课件 + Echo 最终版 | 中 |
| M6.5 | 跨平台复现验收 | 双平台从零复现成功、陷阱清单 | 小 |
| M7 | 开源就绪 | LICENSE、贡献指南、使用文档、Demo | 小 |

---

## M0 · 仓库基建

**目标**：让仓库能 `uv sync` 一把跑起来，有清晰的入口。

### 任务

- [ ] T0.1 初始化 `pyproject.toml`，配置 optional-dependencies（模块分组 + 平台分组 `train-cuda` / `train-mps`）
- [ ] T0.2 `.python-version` 固定 3.12.10
- [ ] T0.3 配置 `.gitignore`（checkpoints/ data/ .venv/ __pycache__/ 等）
- [ ] T0.4 配置 `.gitattributes`：强制 LF、标注二进制类型
- [ ] T0.5 编写根 `README.md`：项目简介、快速开始（Win/Mac 两套命令）、目录导览
- [ ] T0.6 配置 ruff（`pyproject.toml` 内）
- [ ] T0.7 创建 `Doc/Courseware/outline.md` 骨架（章节列表先空着）
- [ ] T0.8 创建 `Echo/echo-mini` `Echo/echo` `Echo/shared` 目录与各自 README 占位
- [ ] T0.9 `Echo/shared/device.py::get_device()`：统一设备选择（cuda→mps→cpu），含 op fallback 工具
- [ ] T0.10 `scripts/doctor.py`：双平台自检脚本（见策划案 §7.5）
- [ ] T0.11 设计大文件同步方案文档（`Doc/DesignDoc/02-sync-strategy.md`），选定 HF Hub 作为 checkpoint 分发通道
- [ ] T0.12 在 Win/Mac 各自 `uv sync` + `python scripts/doctor.py` 通过

### 交付物

- 可运行的空仓（Win/Mac 双端 `doctor.py` 通过）
- 课程大纲文件（空骨架）
- 跨平台同步策略文档

---

## M1 · 前置知识

**目标**：补齐 PyTorch、浅层数学、神经网络基础，为 Transformer 做准备。

**跨平台约定**：本里程碑起，**所有练习代码必须 device 无关**（通过 `get_device()`），提交前 Win/Mac 两端都要能跑过。

### 课程章节（Doc/Courseware/ch01~ch04）

- **ch01 · 环境与工具**
  - uv 使用、PyTorch 安装（CUDA / MPS 差异）
  - Jupyter / VSCode 调试
  - 练习：环境自检脚本

- **ch02 · 必要数学**（浅层）
  - 向量、矩阵、点积、矩阵乘法
  - 梯度、链式法则直觉
  - 概率基础、softmax、交叉熵的几何意义
  - 练习：纯 NumPy 实现 softmax / 交叉熵

- **ch03 · PyTorch 入门**
  - Tensor / autograd / nn.Module
  - 数据集与 DataLoader
  - 训练循环模板
  - 练习：手写一个 MLP 分类 MNIST

- **ch04 · 神经网络与训练要素**
  - 反向传播的工程视角
  - 优化器（SGD / Adam / AdamW）
  - 学习率调度、权重初始化
  - 正则化、Dropout、BN/LN
  - 练习：对比不同优化器/初始化的收敛曲线

### 交付物

- 4 章课件 markdown
- 对应 Playground/ch01~ch04 练习代码
- 所有练习在 3060 上 1 分钟内可跑完

---

## M2 · Transformer 精通

**目标**：从注意力机制到完整 Decoder-only 架构，手撕一遍。

### 课程章节

- **ch05 · 注意力机制**
  - 从 seq2seq 到 attention 的动机
  - Q/K/V 推导、缩放点积注意力
  - 多头注意力
  - 练习：纯 PyTorch 实现单头 & 多头注意力

- **ch06 · Transformer 架构**
  - Encoder / Decoder / Decoder-only 三种拓扑
  - 位置编码（绝对 / 旋转 RoPE）
  - Pre-LN vs Post-LN
  - 残差、FFN、掩码
  - 练习：手写一个 ~1M 参数的 Decoder-only，在 tiny shakespeare 上过拟合

- **ch07 · 生成策略**
  - 贪心 / beam / top-k / top-p / temperature
  - KV cache 原理与实现
  - 练习：为 ch06 的模型加 KV cache 推理

### 交付物

- 3 章课件
- 一个 `Playground/ch06-transformer` 的完整可训练 Decoder-only 实现（将作为 echo-mini 的原型）

---

## M3 · LLM 全链路入门（理论篇）

**目标**：在动手训 echo-mini 之前，把全链路的原理讲清楚。

### 课程章节

- **ch08 · 分词器**
  - 字符 / 词 / 子词
  - BPE、WordPiece、Unigram
  - 中英混合的特殊处理
  - 练习：用 `tokenizers` 库训练一个 BPE

- **ch09 · 预训练**
  - 语言建模目标（CLM）
  - 数据构造：拼接、截断、packing
  - 训练技巧：混合精度、grad accumulation、gradient checkpointing
  - Scaling law 直觉

- **ch10 · SFT（指令微调）**
  - 对话模板、loss mask、多轮处理
  - LoRA / QLoRA 原理
  - 数据质量 > 数据数量

- **ch11 · 对齐**
  - RLHF 概览、PPO 痛点
  - DPO 原理与为何更友好
  - 简要提及 KTO / ORPO 等

- **ch12 · 评测**
  - 困惑度 PPL
  - 开源 benchmark（C-Eval / MMLU 子集）
  - 人工评测的必要性

- **ch13 · 部署**
  - 量化（int8 / int4 / GGUF）
  - llama.cpp / vLLM / Transformers 推理
  - 本地 chat UI 方案选型

### 交付物

- 6 章课件（偏理论，少量 demo）
- 每章末尾留 1~2 个思考题

---

## M4 · echo-mini 落地

**目标**：走通"数据 → 分词 → Pretrain → SFT → 评测"完整管线，产出 echo-mini。

**跨平台约定**：所有训练任务产出 `config-full.yaml`（Win 3060 生产配置）与 `config-tiny.yaml`（Mac/CPU ~100 步验证配置）两份，代码用同一入口读配置。

### 任务

- [ ] T4.1 数据：中英双语（或先英文）小规模语料采集与清洗脚本
- [ ] T4.2 训练 BPE 分词器，词表 ~16k
- [ ] T4.3 写 echo-mini 模型（参数 ~30M，Decoder-only + RoPE）
- [ ] T4.4 Pretrain 训练脚本（支持混合精度、checkpointing、断点续训）
- [ ] T4.5 产出 Pretrain `config-full.yaml` + `config-tiny.yaml`，Mac 跑 tiny 验证代码不崩
- [ ] T4.6 在 3060 上跑 Pretrain full 配置，记录 loss 曲线
- [ ] T4.7 构造小规模 SFT 对话数据（可借助公开数据集或 GPT 生成）
- [ ] T4.8 SFT 训练脚本 + full/tiny 两份配置
- [ ] T4.9 推理 CLI（带 KV cache，跨平台可用）
- [ ] T4.10 评测：PPL + 若干人工对话样例
- [ ] T4.11 Checkpoint 上传 HuggingFace Hub
- [ ] T4.12 写 `Echo/echo-mini/README.md`，记录训练配方

### 交付物

- `Echo/echo-mini` 下完整代码与配置
- 训练日志与 loss 曲线
- 一个弱但能续写的迷你模型权重（通过网盘/HF 链接分发）

---

## M5 · echo 落地

**目标**：基于开源底座做 SFT，产出真正能用的 Echo。

**跨平台约定**：SFT 脚本走配置双份化；底座推理要在 Mac 上也能跑起（MPS 或量化版）。

### 任务

- [ ] T5.1 底座选型确认（Qwen2.5-0.5B / 1.5B / 其他）
- [ ] T5.2 下载底座、在 3060 **和** Mac 上各跑通推理基线
- [ ] T5.3 整理 SFT 数据（中英对话，重点关注 Echo 人设一致性）
- [ ] T5.4 LoRA/QLoRA 微调脚本（QLoRA 走 CUDA-only 分支，Mac 用纯 LoRA）
- [ ] T5.5 产出 SFT `config-full.yaml` + `config-tiny.yaml`
- [ ] T5.6 Win 上训练并保存 adapter，上传 HF Hub
- [ ] T5.7 合并/加载 adapter，推理 CLI（Win/Mac 双端可用）
- [ ] T5.8 初步人工评测对话质量，迭代数据
- [ ] T5.9 写 `Echo/echo/README.md` 与训练配方

### 交付物

- 可对话的 Echo v1（SFT 版）
- 训练脚本与 LoRA adapter

---

## M6 · 对齐与优化

**目标**：让 Echo 更听话、更省资源。

### 任务

- [ ] T6.1 DPO 偏好数据构造（可从 SFT 样本手工挑选 chosen/rejected）
- [ ] T6.2 DPO 训练脚本
- [ ] T6.3 Echo v2（DPO 版）产出
- [ ] T6.4 int4 量化脚本（GGUF 或 bitsandbytes）
- [ ] T6.5 本地 chat UI（简单 Gradio / CLI 二选一）
- [ ] T6.6 Echo final 版本发布

### 交付物

- Echo 最终版（量化后）
- 本地可启动的 chat demo

---

## M6.5 · 跨平台复现验收

**目标**：确保外部入门者在任一平台都能复现成果，同时验证"随时切换"愿景真正达成。

### 任务

- [ ] T6.5.1 在一台干净 Win 机（或 WSL 外 CMD）上从零 bootstrap，跑完 M1–M3 所有练习
- [ ] T6.5.2 在一台干净 Mac 上同样跑通 M1–M3
- [ ] T6.5.3 Mac 端用 tiny 配置跑通 echo-mini Pretrain / SFT 脚本（不要求收敛）
- [ ] T6.5.4 Mac 端成功加载并推理 echo final（量化版）
- [ ] T6.5.5 记录"跨平台已知差异与陷阱"到 `Doc/DesignDoc/cross-platform-notes.md`

### 交付物

- 跨平台验收通过报告
- 陷阱清单文档

---

## M7 · 开源就绪

**目标**：让外部入门者能顺利复现与学习。

### 任务

- [ ] T7.1 LICENSE（推荐 MIT / Apache-2.0）
- [ ] T7.2 CONTRIBUTING.md
- [ ] T7.3 根 README 完善：学习路径图、演示 GIF、FAQ
- [ ] T7.4 课程大纲定稿、章节间跳转
- [ ] T7.5 常见坑 & 故障排查文档
- [ ] T7.6（可选）发布到 GitHub，申请加入 awesome-llm 类 list

### 交付物

- 一个可对外开源的完整仓库

---

## 执行策略

1. **严格按里程碑顺序推进**，M4 之前不动训练代码
2. 每个里程碑结束做一次**复盘文档**，落到 `Doc/DesignDoc/retro-MX.md`
3. 课件与练习**同步产出**，写完一章就配套练习
4. Echo 相关的配置、脚本、数据管线一律通过 **CLI + 配置文件** 驱动，避免硬编码
5. 大的设计变更（如 echo-mini 的架构选型）先在 `Doc/DesignDoc/` 下开专题文档讨论
6. **跨平台铁律**：每次切换机器前先 `git push`；上机第一件事 `git pull` + `scripts/doctor.py`；练习代码提交前两端都跑过；训练代码提交前至少 Mac 跑过 tiny 配置

## 当前状态

- [x] 原始需求整理：`Doc/UserDraft/repo-target-idea.md`
- [x] Startup 策划案：`Doc/DesignDoc/00-startup-proposal.md`
- [x] 项目计划书：本文
- [ ] M0 仓库基建：**下一步**
