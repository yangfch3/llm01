# 项目任务清单

> 本文是 `01-project-plan.md` 的任务勾选区。**高频改动**集中在此，主文档保持稳定。
> 任务编号 `T<里程碑>.<序号>`，与计划书章节一一对应。
> 勾选规范：未做 `- [ ]`，已做 `- [x]`，进行中可在末尾追加 `（in progress）`。

<!-- M0 START -->
## M0 · 仓库基建

- [x] T0.1 初始化 `pyproject.toml`，配置 optional-dependencies（模块分组 + 平台分组 `train-cuda` / `train-mps`）
- [x] T0.2 `.python-version` 锁到 minor 3.12（patch 由本地 pyenv/uv 选）
- [x] T0.3 配置 `.gitignore`（checkpoints/ data/ .venv/ __pycache__/ 等）
- [x] T0.4 配置 `.gitattributes`：强制 LF、标注二进制类型
- [x] T0.5 编写根 `README.md`：项目简介、快速开始（Win/Mac 两套命令）、目录导览
- [x] T0.6 配置 ruff（`pyproject.toml` 内）
- [x] T0.7 创建 `Doc/Courseware/outline.md` 骨架（章节列表先空着）
- [x] T0.8 创建 `Echo/echo-mini` `Echo/echo` `Echo/shared` 目录与各自 README 占位
- [x] T0.9 `Echo/shared/device.py::get_device()`：统一设备选择（cuda→mps→cpu），含 op fallback 工具
- [x] T0.10 `scripts/doctor.py`：双平台自检脚本（见策划案 §7.5）
- [x] T0.11 编写大文件同步 SOP 文档（`Doc/DesignDoc/03-sync-strategy.md`）：策略已在策划案 §7.4 确定，本文档只补操作层（HF Hub 仓库命名规则、`huggingface-cli` 上传/下载常用命令、断点续传、数据下载脚本骨架）
- [x] T0.12 在 Win/Mac 各自 `uv sync` + `python scripts/doctor.py` 通过
- [x] T0.13 ch00 导论课件（AI 全景与概念速查）
<!-- M0 END -->

<!-- M1 START -->
## M1 · 前置知识

- [x] T1.1 ch01 课件 + Playground/ch01 环境与工具
- [x] T1.2 ch02 课件 + Playground/ch02 必要数学
- [x] T1.3 ch03 课件 + Playground/ch03 PyTorch 入门
- [x] T1.4 ch04 课件 + Playground/ch04 神经网络与训练要素
<!-- M1 END -->

<!-- M2 START -->
## M2 · Transformer 精通

- [x] T2.1 ch05 课件 + Playground/ch05 注意力机制
- [x] T2.2 ch06 课件 + Playground/ch06 Decoder-only Transformer
- [x] T2.3 ch07 课件 + Playground/ch07 生成策略与 KV cache
<!-- M2 END -->

<!-- M3 START -->
## M3 · LLM 全链路入门（理论）

- [x] T3.1 ch08 分词器
- [x] T3.2 ch09 预训练
- [x] T3.3 ch10 SFT
- [x] T3.4 ch11 对齐
- [x] T3.5 ch12 评测
- [x] T3.6 ch13 部署
<!-- M3 END -->

<!-- M4 START -->
## M4 · echo-mini 落地

- [x] T4.1 数据：中英双语（或先英文）小规模语料采集与清洗脚本
- [x] T4.2 训练 BPE 分词器，词表 ~16k
- [x] T4.3 写 echo-mini 模型（参数 ~60M，Llama 风格 Decoder-only + RoPE + GQA + SwiGLU）
- [x] T4.4 Pretrain 训练脚本（支持混合精度、checkpointing、断点续训）
- [x] T4.5 产出 Pretrain `config-full.yaml` + `config-tiny.yaml`，Mac 跑 tiny 验证代码不崩
- [x] T4.6 在 3060 12GB 上跑 Pretrain full 配置，记录 loss 曲线
- [x] T4.7 构造小规模 SFT 对话数据（可借助公开数据集或 GPT 生成）
- [x] T4.8 SFT 训练脚本 + full/tiny 两份配置
- [x] T4.9 推理 CLI（带 KV cache，跨平台可用）
- [x] T4.10 评测：PPL + 若干人工对话样例
- [ ] T4.11 Checkpoint 上传 HuggingFace Hub（可选，跳过）
- [x] T4.12 写 `Echo/echo-mini/README.md`，记录训练配方
<!-- M4 END -->

<!-- M5 START -->
## M5 · echo 落地

- [x] T5.1 底座选型确认（Qwen2.5-0.5B / 1.5B / 其他）
- [x] T5.2 下载底座、在 3060 12GB **和** Mac 上各跑通推理基线
- [x] T5.3 整理 SFT 数据（中英对话，重点关注 Echo 人设一致性）
- [x] T5.4 LoRA/QLoRA 微调脚本（QLoRA 走 CUDA-only 分支，Mac 用纯 LoRA）
- [x] T5.5 产出 SFT `config-full.yaml` + `config-tiny.yaml`
- [x] T5.6 SFT 训练并保存 adapter，（可选）上传 HF Hub
- [x] T5.7 合并/加载 adapter，推理 CLI（Win/Mac 双端可用）
- [x] T5.8 初步人工评测对话质量，迭代数据
- [x] T5.9 写 `Echo/echo/README.md` 与训练配方
<!-- M5 END -->

<!-- M6 START -->
## M6 · 对齐与优化

- [ ] T6.1 DPO 偏好数据构造（可从 SFT 样本手工挑选 chosen/rejected）
- [ ] T6.2 DPO 训练脚本
- [ ] T6.3 Echo v2（DPO 版）产出
- [ ] T6.4 int4 量化脚本（GGUF 或 bitsandbytes）
- [ ] T6.5 把量化后 GGUF 接入 Ollama，编写一条 `ollama run echo` 跑通的演示脚本
- [ ] T6.6 Echo final 版本发布
<!-- M6 END -->

<!-- M6.5 START -->
## M6.5 · 跨平台复现验收

- [x] T6.5.1 在 Win 端用全新 venv（或 WSL/Docker 容器）从零 bootstrap，跑完 M1–M3 所有练习
- [x] T6.5.2 在 Mac 端用全新 venv 同样跑通 M1–M3
- [ ] T6.5.3 Mac 端用 tiny 配置跑通 echo-mini Pretrain / SFT 脚本（不要求收敛）
- [ ] T6.5.4 Mac 端成功加载并推理 echo final（量化版）
- [ ] T6.5.5 记录"跨平台已知差异与陷阱"到 `Doc/DesignDoc/cross-platform-notes.md`
<!-- M6.5 END -->

<!-- M7 START -->
## M7 · 开源就绪

- [ ] T7.1 LICENSE（推荐 MIT / Apache-2.0）
- [ ] T7.2 CONTRIBUTING.md
- [ ] T7.3 根 README 完善：学习路径图、演示 GIF、FAQ
- [ ] T7.4 课程大纲定稿、章节间跳转
- [ ] T7.5 常见坑 & 故障排查文档
- [ ] T7.6（可选）发布到 GitHub，申请加入 awesome-llm 类 list
<!-- M7 END -->
