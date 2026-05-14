# LLM 学习仓库 项目计划书

> 本文是所有未来要做事项的**框架与里程碑描述**。配套根基文档见 `00-startup-proposal.md`。
> **任务勾选清单单独维护**：见 `tasks.md`。本文不再列具体勾选项，避免高频改动污染主文。

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

**交付物**：可运行的空仓（Win/Mac 双端 `doctor.py` 通过）、课程大纲骨架、跨平台同步策略文档。

**任务清单**：见 [`tasks.md` → M0](tasks.md#m0--仓库基建)。

---

## M1 · 前置知识

**目标**：补齐 PyTorch、浅层数学、神经网络基础，为 Transformer 做准备。

**跨平台约定**：本里程碑起，**所有练习代码必须 device 无关**（通过 `get_device()`），提交前 Win/Mac 两端都要能跑过。

### 课程章节（Doc/Courseware/ch01~ch04）

- **ch01 · 环境与工具** — uv 使用、PyTorch 安装（CUDA / MPS 差异）、Jupyter / VSCode 调试。练习：环境自检脚本
- **ch02 · 必要数学（浅层）** — 向量/矩阵/点积、梯度与链式法则直觉、softmax / 交叉熵的几何意义。练习：纯 NumPy 实现 softmax / 交叉熵
- **ch03 · PyTorch 入门** — Tensor / autograd / nn.Module、DataLoader、训练循环模板。练习：MLP 分类 MNIST
- **ch04 · 神经网络与训练要素** — 反向传播工程视角、优化器（SGD/Adam/AdamW）、LR 调度、初始化、Dropout、BN/LN。练习：对比不同优化器/初始化的收敛曲线

**交付物**：4 章课件 markdown + 对应 Playground/ch01~ch04 练习代码；所有练习在 3060 上 1 分钟内可跑完。

**任务清单**：见 [`tasks.md` → M1](tasks.md#m1--前置知识)。

---

## M2 · Transformer 精通

**目标**：从注意力机制到完整 Decoder-only 架构，手撕一遍。

### 课程章节

- **ch05 · 注意力机制** — seq2seq → attention 动机、Q/K/V、缩放点积、多头。练习：纯 PyTorch 实现单/多头注意力
- **ch06 · Transformer 架构** — Encoder/Decoder/Decoder-only、位置编码（绝对/RoPE）、Pre-LN vs Post-LN、残差/FFN/掩码。练习：~1M 参数 Decoder-only 在 tiny shakespeare 上过拟合
- **ch07 · 生成策略** — 贪心/beam/top-k/top-p/temperature、KV cache 原理与实现。练习：为 ch06 的模型加 KV cache

**交付物**：3 章课件 + `Playground/ch06-transformer` 完整可训练实现（将作为 echo-mini 原型）。

**任务清单**：见 [`tasks.md` → M2](tasks.md#m2--transformer-精通)。

---

## M3 · LLM 全链路入门（理论篇）

**目标**：在动手训 echo-mini 之前，把全链路原理讲清楚。

### 课程章节

- **ch08 · 分词器** — 字符/词/子词、BPE/WordPiece/Unigram、中英混合处理。练习：用 `tokenizers` 训练 BPE
- **ch09 · 预训练** — CLM 目标、数据 packing、混合精度/grad accumulation/gradient checkpointing、scaling law 直觉
- **ch10 · SFT** — 对话模板、loss mask、多轮、LoRA / QLoRA 原理、数据质量 > 数量
- **ch11 · 对齐** — RLHF 概览、PPO 痛点、DPO 原理；KTO / ORPO 简提
- **ch12 · 评测** — PPL、开源 benchmark（C-Eval / MMLU 子集）、人工评测必要性
- **ch13 · 部署** — 量化（int8/int4/GGUF）、Ollama 上手、`llama-cpp-python`、`transformers` 原生推理对比；选型理由（不选 vLLM 详见 `02-deps-compatibility.md`）

**交付物**：6 章课件（偏理论，少量 demo），每章末尾留 1~2 思考题。

**任务清单**：见 [`tasks.md` → M3](tasks.md#m3--llm-全链路入门理论)。

---

## M4 · echo-mini 落地

**目标**：走通"数据 → 分词 → Pretrain → SFT → 评测"完整管线，产出 echo-mini。

**跨平台约定**：所有训练任务产出 `config-full.yaml`（Win 3060 生产配置）与 `config-tiny.yaml`（Mac/CPU ~100 步验证配置）两份，同一入口读配置。

**交付物**：`Echo/echo-mini` 下完整代码与配置；训练日志与 loss 曲线；弱但能续写的迷你模型权重（HF Hub 分发）。

**任务清单**：见 [`tasks.md` → M4](tasks.md#m4--echo-mini-落地)。

---

## M5 · echo 落地

**目标**：基于开源底座做 SFT，产出真正能用的 Echo。

**跨平台约定**：SFT 脚本走配置双份化；底座推理在 Mac 上也能跑起（MPS 或量化版）。

**交付物**：可对话的 Echo v1（SFT 版）、训练脚本与 LoRA adapter。

**任务清单**：见 [`tasks.md` → M5](tasks.md#m5--echo-落地)。

---

## M6 · 对齐与优化

**目标**：让 Echo 更听话、更省资源。

**交付物**：Echo 最终版（量化后）；本地可启动的 chat demo（Ollama）。

**任务清单**：见 [`tasks.md` → M6](tasks.md#m6--对齐与优化)。

---

## M6.5 · 跨平台复现验收

**目标**：确保外部入门者在任一平台都能复现成果，同时验证"随时切换"愿景达成。

**交付物**：跨平台验收通过报告；陷阱清单文档。

**任务清单**：见 [`tasks.md` → M6.5](tasks.md#m65--跨平台复现验收)。

---

## M7 · 开源就绪

**目标**：让外部入门者能顺利复现与学习。

**交付物**：可对外开源的完整仓库。

**任务清单**：见 [`tasks.md` → M7](tasks.md#m7--开源就绪)。

---

## 执行策略

1. **严格按里程碑顺序推进**，M4 之前不动训练代码
2. 每个里程碑结束做一次**复盘文档**，落到 `Doc/DesignDoc/retro-MX.md`
3. 课件与练习**同步产出**，写完一章就配套练习
4. Echo 相关的配置、脚本、数据管线一律通过 **CLI + 配置文件** 驱动，避免硬编码
5. 大的设计变更（如 echo-mini 的架构选型）先在 `Doc/DesignDoc/` 下开专题文档讨论
6. **跨平台铁律**：每次切换机器前先 `git push`；上机第一件事 `git pull` + `scripts/doctor.py`；练习代码提交前两端都跑过；训练代码提交前至少 Mac 跑过 tiny 配置

## 关联文档

- `00-startup-proposal.md`：项目根基文档
- `02-deps-compatibility.md`：依赖兼容性与部署选型
- `tasks.md`：任务勾选清单（高频改动）
