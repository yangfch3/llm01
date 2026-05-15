# 课程大纲

> 本文是课件总目录骨架。每章 markdown 落到 `Doc/Courseware/chNN-xxx/README.md`，配套练习落到 `Playground/chNN-xxx/`。
> 章节内容随里程碑推进逐步补全，详见 [`../DesignDoc/01-project-plan.md`](../DesignDoc/01-project-plan.md)。

## M1 · 前置知识

- [x] **ch01 · 环境与工具** — uv / PyTorch 安装（CUDA vs MPS）/ Jupyter / VSCode 调试 · 练习：环境自检脚本
- [x] **ch02 · 必要数学（浅层）** — 向量/矩阵/点积、梯度与链式法则、softmax / 交叉熵几何意义 · 练习：纯 NumPy 实现
- [x] **ch03 · PyTorch 入门** — Tensor / autograd / nn.Module / DataLoader / 训练循环模板 · 练习：MLP 分类 MNIST
- [x] **ch04 · 神经网络与训练要素** — 反向传播工程视角、优化器、LR 调度、初始化、Dropout、BN/LN · 练习：优化器对比

## M2 · Transformer 精通

- [x] **ch05 · 注意力机制** — seq2seq → attention、Q/K/V、缩放点积、多头 · 练习：手写单/多头注意力
- [x] **ch06 · Transformer 架构** — Decoder-only、位置编码（绝对/RoPE）、Pre-LN vs Post-LN、残差/FFN/掩码 · 练习：~1M Decoder-only 过拟合 tiny shakespeare
- [x] **ch07 · 生成策略** — 贪心/beam/top-k/top-p/temperature、KV cache · 练习：为 ch06 模型加 KV cache

## M3 · LLM 全链路入门（理论）

- [x] **ch08 · 分词器** — 字符/词/子词、BPE/WordPiece/Unigram、中英混合 · 练习：用 `tokenizers` 训练 BPE
- [x] **ch09 · 预训练** — CLM 目标、数据 packing、混合精度 / grad accumulation / gradient checkpointing、scaling law 直觉
- [x] **ch10 · SFT** — 对话模板、loss mask、多轮、LoRA / QLoRA 原理、数据质量 > 数量
- [x] **ch11 · 对齐** — RLHF 概览、PPO 痛点、DPO 原理、KTO / ORPO 简提
- [x] **ch12 · 评测** — PPL、开源 benchmark（C-Eval / MMLU 子集）、人工评测必要性
- [x] **ch13 · 部署** — 量化（int8/int4/GGUF）、Ollama、`llama-cpp-python`、`transformers` 原生推理对比

## 章节模板

每章 `README.md` 建议包含：

1. 学习目标（3 条以内）
2. 前置依赖（指向其它章节或外部资料）
3. 正文（伪代码 + 图示 > 完整代码）；关键节末尾可加"自检"小节
4. 练习说明（指向 `Playground/chNN-xxx/`）
5. 自检题（2–3 条）+ 折叠答案速查（必须，下方约定）
6. 参考资料

### 自检题写作约定

- 题目放在 `### 自检` 或 `## 自检` 下，紧跟正文或章节末尾
- **必须配答案速查**，用 `<details markdown="1">` 折叠（mkdocs + GitHub 双端兼容）
- 答案保持简短（每条一两句），需要展开的留作思考题或正文补充
- 折叠块内列表项之间**留空行**，避免渲染器吞行
- 模板：

  ```markdown
  ### 自检

  1. 题目一
  2. 题目二

  <details markdown="1">
  <summary>答案速查</summary>

  1. 答案一

  2. 答案二

  </details>
  ```

- 工具章（如 ch01）以"操作回顾 / 易混点辨析"为主；理论章（如 ch02）以"公式直觉 / 概念辨析"为主
- 章末"思考题"（开放式、无标准答案）可选；有标准答案的一律走自检

## 练习代码注释规范

`Playground/chNN-xxx/` 下的教学代码两类必注：

1. **外部库 API 调用** — 标注作用和参数语义
   - ✓ `rng = np.random.default_rng(0)  # NumPy 现代随机数生成器，固定 seed → 可复现`
   - ✗ `rng = np.random.default_rng(0)  # 创建生成器`

2. **复杂公式计算** — 标注算法名 + 一句话含义或公式
   - ✓ `dw = np.outer(dy, a)  # 外积 (out,)×(h2,) → (out, h2)，对应 ∂L/∂w 单样本形式`
   - ✗ `dw = np.outer(dy, a)  # 计算梯度`
