# echo-mini

从零走通"数据→分词器→Pretrain→SFT→评测→推理"全链路的迷你语言模型。教学优先，效果不强求。

## 模型规格

| 项 | 值 |
|---|---|
| 架构 | Llama-style Decoder-only (RoPE + GQA + SwiGLU) |
| 参数量 | ~57M |
| 词表 | 16,386 (BPE, 中英双语) |
| 上下文 | 1,024 tokens |
| 特殊 token | `<pad>`(0) `<bos>`(1) `<eos>`(2) `<unk>`(3) `<\|user\|>`(4) `<\|assistant\|>`(5) |

## 目录结构

```
echo-mini/
├─ configs/           训练配置（full = Win 3060 12GB, tiny = Mac/CPU 验证）
├─ data/              数据产物（.gitignore，脚本下载生成）
├─ tokenizer/         分词器（tokenizer.json .gitignore，训练脚本入仓）
├─ src/echo_mini/     模型定义、数据加载、工具函数
├─ scripts/           训练 / 数据 / 推理 / 评测脚本
├─ checkpoints/       权重（.gitignore，走 HF Hub）
├─ logs/              训练日志（.gitignore）
├─ SPEC.md            技术规格详细文档
└─ README.md          本文件
```

## 从零复现步骤

所有命令 cwd 为 `Echo/echo-mini/`。

### 1. 下载预训练语料

```bash
uv run python scripts/prepare_data.py download --config configs/pretrain-full.yaml
```

产物：`data/raw/*.txt`（FineWeb-Edu EN + Wikipedia ZH + SkyPile ZH）

### 2. 训练分词器

```bash
uv run python tokenizer/train_tokenizer.py
```

产物：`tokenizer/tokenizer.json`（vocab_size=16386）

### 3. Tokenize 语料

```bash
uv run python scripts/prepare_data.py tokenize --config configs/pretrain-full.yaml
```

产物：`data/bin/train.bin`

### 4. Pretrain

```bash
uv run accelerate launch scripts/pretrain.py --config configs/pretrain-full.yaml
```

- Win RTX 3060 12GB，约 4-5 小时（10,000 步）
- 产物：`checkpoints/pretrain/step_010000.pt`
- 日志：`logs/pretrain_loss.csv`

Mac 验证（不要求收敛）：
```bash
uv run accelerate launch scripts/pretrain.py --config configs/pretrain-tiny.yaml
```

### 5. 准备 SFT 数据

```bash
uv run python scripts/prepare_sft_data.py --config configs/sft-full.yaml
```

产物：`data/sft/train.jsonl`（~19K 条 Alpaca-GPT4 中英对话）

### 6. SFT

```bash
uv run accelerate launch scripts/sft.py --config configs/sft-full.yaml
```

- 3,000 步，约 4-5 小时
- 产物：`checkpoints/sft/step_003000.pt`
- 日志：`logs/sft_loss.csv`

### 7. 评测

```bash
uv run python scripts/evaluate.py --ckpt checkpoints/sft/step_003000.pt
```

输出 PPL + 对话样例到 `logs/eval_results.json`。

### 8. 推理

```bash
# 对话模式
uv run python scripts/chat.py --ckpt checkpoints/sft/step_003000.pt --mode chat

# 续写模式
uv run python scripts/chat.py --ckpt checkpoints/pretrain/step_010000.pt --mode complete
```

## 训练配方

### Pretrain

| 参数 | 值 |
|---|---|
| 数据 | ~160K 文档, tokenized ~67M tokens |
| Batch size | 16 × 4 grad_accum = 64 |
| Steps | 10,000 |
| LR | cosine, peak 3e-4, min 3e-5 |
| Warmup | 200 steps |
| Precision | bf16 |

### SFT

| 参数 | 值 |
|---|---|
| 数据 | ~19K 条（Alpaca-GPT4 EN 10K + ZH 10K） |
| Batch size | 8 × 4 grad_accum = 32 |
| Steps | 3,000 |
| LR | cosine, peak 2e-5, min 2e-6 |
| Warmup | 100 steps |
| Precision | bf16 |
| 初始权重 | pretrain step_010000 |

### Chat Template

```
<bos><|user|>{user_content}\n<|assistant|>{assistant_content}<eos>
```

- `<|user|>` / `<|assistant|>` 为专用特殊 token（id 16384/16385）
- SFT 只对 assistant content + eos 计算 loss

## 效果说明

echo-mini 是 57M 参数的教学模型，效果很弱。预期表现：
- 能产出语法基本通顺的短句
- 中英文能力都有限，倾向生成训练数据中高频 pattern
- 不具备真正的推理能力

这是设计预期，核心目标是**走通全链路、学懂每个环节**。

