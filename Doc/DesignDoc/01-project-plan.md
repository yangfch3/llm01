# LLM 学习仓库 项目计划书

> 本文是所有未来要做事项的**框架与里程碑描述**。配套根基文档见 `00-startup-proposal.md`。
> **任务勾选清单单独维护**：见 `tasks.md`。本文不再列具体勾选项，避免高频改动污染主文。

## 0. 里程碑总览

| 里程碑 | 名称 | 关键交付物 | 预估规模 |
|---|---|---|---|
| M0 | 仓库基建 | 环境就绪（双平台 doctor 通过）、课程大纲、导论、README、同步策略 | 小 |
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

**交付物**：可运行的空仓（Win/Mac 双端 `doctor.py` 通过）、课程大纲骨架、ch00 导论课件、跨平台同步策略文档。

**任务清单**：见 [`tasks.md` → M0](tasks.md#m0--仓库基建)。

---

## M1 · 前置知识

**目标**：补齐 PyTorch、浅层数学、神经网络基础，为 Transformer 做准备。

**跨平台约定**：本里程碑起，**所有练习代码必须 device 无关**（通过 `get_device()`），提交前 Win/Mac 两端都要能跑过。

**章节列表**：见 [`Doc/Courseware/outline.md` → M1](../Courseware/outline.md#m1--前置知识)（ch01–ch04）。

**交付物**：4 章课件 markdown + 对应 Playground/ch01~ch04 练习代码；所有练习在 3060 12GB 上 1 分钟内可跑完。

**任务清单**：见 [`tasks.md` → M1](tasks.md#m1--前置知识)。

---

## M2 · Transformer 精通

**目标**：从注意力机制到完整 Decoder-only 架构，手撕一遍。

**章节列表**：见 [`Doc/Courseware/outline.md` → M2](../Courseware/outline.md#m2--transformer-精通)（ch05–ch07）。

**交付物**：3 章课件 + `Playground/ch06-transformer` 完整可训练实现（将作为 echo-mini 原型）。

**任务清单**：见 [`tasks.md` → M2](tasks.md#m2--transformer-精通)。

---

## M3 · LLM 全链路入门（理论）

**目标**：在动手训 echo-mini 之前，把全链路原理讲清楚。

**章节列表**：见 [`Doc/Courseware/outline.md` → M3](../Courseware/outline.md#m3--llm-全链路入门理论)（ch08–ch13）。

**交付物**：6 章课件（偏理论，少量 demo），每章按 `Doc/Courseware/outline.md` 章节模板组织（含自检题 + 折叠答案）。

**任务清单**：见 [`tasks.md` → M3](tasks.md#m3--llm-全链路入门理论)。

---

## M4 · echo-mini 落地

**目标**：走通"数据 → 分词 → Pretrain → SFT → 评测"完整管线，产出 echo-mini。

**跨平台约定**：所有训练任务产出 `config-full.yaml`（Win 3060 12GB 生产配置）与 `config-tiny.yaml`（Mac/CPU ~100 步验证配置）两份，同一入口读配置。

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

**验收**：满足 [`00-startup-proposal.md` §4.5 echo final 指标](00-startup-proposal.md#echo-finalm6-完量化后)（量化后衰减 ≤5%、3060 12GB ≥20 tok/s int4、Mac ≥15 tok/s GGUF Q4_K_M、`ollama run echo` 可启）。

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
2. 课件与练习**同步产出**，写完一章就配套练习；章节 README 与练习目录的具体形态、练习代码注释规范见 [`Doc/Courseware/outline.md`](../Courseware/outline.md)
3. Echo 相关的配置、脚本、数据管线一律通过 **CLI + 配置文件** 驱动，避免硬编码
4. 大的设计变更（如 echo-mini 的架构选型）先在 `Doc/DesignDoc/` 下开专题文档讨论
5. **跨平台铁律**：每次切换机器前先 `git push`；上机第一件事 `git pull` + `scripts/doctor.py`；练习代码提交前两端都跑过；训练代码提交前至少 Mac 跑过 tiny 配置

## 关联文档

- `00-startup-proposal.md`：项目根基文档
- `02-deps-compatibility.md`：依赖兼容性与部署选型
- `tasks.md`：任务勾选清单（高频改动）
