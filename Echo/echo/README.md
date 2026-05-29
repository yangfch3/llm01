# echo

基于 `Qwen/Qwen2.5-1.5B`（base）QLoRA SFT 微调出的中英对话模型。本路线以 base 为锚点，
完整体验"教模型学 ChatML、停止行为、风格"全流程。`Qwen2.5-1.5B-Instruct` 作为对照实验保留。

文档分工：

- **本文件**：复刻命令清单 + 训练配方表 + 实测数据，回答"怎么从零跑出 echo"
- [`SPEC.md`](SPEC.md)：技术规格，回答"echo 是什么 / 为什么这么设计"
- [`stepshooting-base.md`](stepshooting-base.md) / [`stepshooting-instruct.md`](stepshooting-instruct.md)：步骤速查 + 踩坑细节
- [`../../Doc/Reading/sft-mode-collapse-findings/README.md`](../../Doc/Reading/sft-mode-collapse-findings/README.md)：mode collapse 实验报告（**理解 sweet spot 必读**）

## 模型规格

| 项 | 值 |
|---|---|
| 底座 | `Qwen/Qwen2.5-1.5B`（**非** Instruct） |
| 参数量 | ~1.5B |
| 微调方式 | QLoRA（NF4 4bit + LoRA r=128）+ `embed_tokens` 全量训练 |
| 上下文 | 2048 tokens（训练截断；底座原生 32K） |
| Chat template | Qwen2 ChatML（`<\|im_start\|>` / `<\|im_end\|>`） |
| Adapter 大小 | ~600 MB（含全量 embed_tokens） |
| 合并 bf16 权重 | ~3 GB |

## 目录结构

```
echo/
├─ configs/           SFT 配置（8g/full/tiny × base/instruct 共 6 份）
├─ data/              数据（.gitignore）
├─ src/echo/          数据加载、工具函数
├─ scripts/           数据 / 训练 / 推理 / 评测 / 合并 / 导出
├─ checkpoints/       adapter / merged 权重（.gitignore，走 HF Hub）
├─ eval/              50 题中英常识/算术评测集
├─ SPEC.md            技术规格
├─ stepshooting-base.md      base 路线步骤速查
├─ stepshooting-instruct.md  instruct 路线步骤速查
└─ README.md          本文件
```

## 从零复现（base 主线）

所有命令 cwd 为 `Echo/echo/`。下方步骤完整覆盖"数据 → 训练 → 选 ckpt → 评测 → 合并 → 部署"。

### 1. 数据准备

```bash
uv run python scripts/prepare_data.py
```

下载 `sharegpt_gpt4` + `vicgalle/alpaca-gpt4`（英）+ `silk-road/alpaca-data-gpt4-chinese`（中），
混合后输出到 `data/sft/{train,val}_aug.jsonl`（~29K 条）。

### 2. 训练前预检（强烈推荐）

```bash
uv run python scripts/preflight.py --config configs/sft-8g-base.yaml
```

5–10 分钟跑完四层检查（不写 ckpt）：config / tokenizer / 数据格式 / loss mask / 5 步微训练。
任意一层失败立即退出 + 写 `preflight-report-*.md`。

> 历史教训：第一次 base 训练 6h 后才发现"模型不会停止"——根因是数据/eos/embed
> 三层信号本可在前 5 分钟暴露。preflight 的存在就是为了不再返工。

### 3. SFT 训练

```bash
uv run python scripts/sft.py --config configs/sft-8g-base.yaml
```

- 3060 12GB，~4–5 小时（2 epoch ≈ 3560 步）
- 产物：`checkpoints/sft-base/checkpoint-{250,500,...,final}`

> sft.py 不带默认 config，需显式传。其他脚本默认值已指向 base 路线。

### 4. 选最佳 ckpt（关键步骤）

`modules_to_save: embed_tokens` 让 base 路线**记忆能力很强**，训练后期会出现
**按问题类型选择性 mode collapse**：短模板题先塌、长开放题后塌。loss 看不出来，必须人工抽测。

每个候选 ckpt 跑 6 题 × 3 次：

```bash
uv run python scripts/generate.py --adapter-dir checkpoints/sft-base/checkpoint-N
```

判定方法与三种题型见 [`stepshooting-base.md` §5.6](stepshooting-base.md)。

确定后复制为 final：

```bash
cp -r checkpoints/sft-base/checkpoint-N checkpoints/sft-base/final
```

### 5. 评测

```bash
uv run python scripts/eval.py
```

跑三项：Val PPL / 50 题正确率 / 5 条对话样例（人工查阅用）。结果存 `eval/results.json`。

### 6. 合并 Adapter

```bash
uv run python scripts/merge.py
```

底座 + adapter → 完整 bf16 权重到 `checkpoints/merged-base/`。导出 GGUF 前必须做。

### 7. 导出 GGUF + 量化

```bash
uv run python scripts/export_gguf.py             # 默认 Q4_K_M
uv run python scripts/export_gguf.py --quant Q8_0  # 几乎无损
```

依赖本地 llama.cpp（设 `LLAMA_CPP_DIR` 环境变量指向仓库根目录）。
流程：`merged-base` → f16 GGUF → Q4_K_M / Q8_0，输出到 `checkpoints/gguf-base/`。

### 8. DPO 对齐（可选，产出 v2）

SFT 之后追加一轮 DPO，目标降低 mode collapse 残留 + 提升回答信息密度。
完整流程见 [`stepshooting-base.md` §7.5](stepshooting-base.md)，命令速查：

```bash
# 8.1 偏好数据 + 训练（24GB GPU）
uv run python scripts/prepare_dpo_data.py
uv run python scripts/dpo.py --config configs/dpo-24g-base.yaml

# 8.2 抽测对比（v1 vs v2）
uv run python scripts/generate.py --merged-dir checkpoints/merged-base   # v1
uv run python scripts/generate_dpo.py                                    # v2 = merged-base + DPO adapter

# 8.3 选定 ckpt 后合并 → Echo v2
uv run python scripts/merge_dpo.py --adapter-dir checkpoints/dpo-base/checkpoint-N
# 产出 checkpoints/merged-dpo/

# 8.4 v2 GGUF
uv run python scripts/export_gguf.py \
    --merged-dir checkpoints/merged-dpo \
    --output-dir checkpoints/gguf-dpo
```

### 9. Ollama 部署

仓库提供两份 Modelfile，与 v1 / v2 一一对应：

```bash
# v1（仅 SFT）
ollama create echo -f Modelfile
ollama run echo

# v2（SFT + DPO，推荐）
ollama create echo-v2 -f Modelfile.v2
ollama run echo-v2
```

两者共存，`ollama list` 可同时看到，便于横向对比。

## 训练配方

### SFT（base 主线 · `sft-8g-base.yaml`）

| 参数 | 值 | 备注 |
|---|---|---|
| 底座 | `Qwen/Qwen2.5-1.5B` | 非 Instruct |
| 量化 | NF4 4bit + double quant | bnb |
| LoRA r / alpha | 128 / 256 | alpha = 2r |
| LoRA target | q/k/v/o + gate/up/down_proj | 7 个 proj |
| modules_to_save | `embed_tokens` | tie_word_embeddings 自动带 lm_head |
| Effective batch | 1 × 16 grad_accum | per_device=1 |
| LR | 1e-4 cosine, warmup 3% | paged_adamw_8bit |
| Epoch | 2 | **关键**：5 epoch 严重过拟合，详见 04-findings |
| Max seq length | 2048 | OOM 可降 1536 |
| 精度 | bf16 + grad_checkpointing | |
| save_steps | 250 | 密保留以便挑 sweet spot |
| 可训参数占比 | ~30% | 含 embed_tokens 全量 |

### 配置矩阵

| 配置 | 用途 | 环境 |
|---|---|---|
| **`sft-8g-base.yaml`** | **生产训练（默认）** | **Win 3060 12GB QLoRA** |
| `sft-full-base.yaml` | 大显存 + per_device=2 | 大显存 GPU QLoRA |
| `sft-tiny-base.yaml` | 代码验证 ~20 步 | Mac/CPU fp32 |
| `sft-8g-instruct.yaml` | 对照实验生产 | Win 3060 12GB QLoRA |
| `sft-full-instruct.yaml` | 对照实验大显存 | 大显存 GPU QLoRA |
| `sft-tiny-instruct.yaml` | 对照实验验证 | Mac/CPU fp32 |

### Chat Template

Qwen2 ChatML 官方格式，不自造：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{assistant_content}<|im_end|>
```

- SFT 仅对 assistant content + `<|im_end|>` 计算 loss（trl `assistant_only_loss`）
- base 路线训练时 `tokenizer.eos_token` 自动 patch 为 `<|im_end|>`，让 trl collator 用对终止符

## 实测结果（首次成功训练）

> 训练日期 2026-05-28（旧 5 epoch 配置抽测），实测最佳 ckpt 为 step 1000。
> 配置已更新为 `num_epochs=2 + save_steps=250`，下次训练 sweet spot 预期在 step 800–1200。

### 选 ckpt 抽测对照

| ckpt | epoch | 长开放题 | 短模板题 | 数据集中题 | 评价 |
|---|---|---|---|---|---|
| 500 | 0.28 | 健康（仅测笑话） | — | — | 太早 |
| **1000** | **0.56** | **健康** | **健康** | **健康** | **最佳** |
| 1500 | 0.84 | 健康 | 临界 | 健康 | 可选 |
| 2000 | 1.12 | 健康 | 已塌 | 健康 | 部分塌 |
| 2500 | 1.40 | 健康 | 已塌 | 已塌 | 部分塌 |
| 4000 | 2.24 | 已塌 | 已塌 | 已塌 | 全塌 |

### 评测数值（checkpoint-1000）

| 指标 | 数值 | 备注 |
|---|---|---|
| Val PPL | 4.09 | val_aug.jsonl |
| 50 题准确率 | 0.83 | 子串匹配判分，详见 `eval/questions.jsonl` |
| GPU 峰值显存（评测） | 5.4 GB allocated / 5.9 GB reserved | sdpa attention + bf16 compute |

### 训练资源

| 资源 | 占用 |
|---|---|
| 3060 12GB 训练时长 | ~4–5 小时（2 epoch） |
| 训练显存峰值 | ~10 GB（max_seq_length=2048） |
| Adapter 文件 | ~600 MB |
| 合并 bf16 权重 | ~3 GB |
| GGUF Q4_K_M | ~1 GB |

## 验收口径

对照 [`startup-proposal §6.2`](../../Doc/DesignDoc/00-startup-proposal.md)：

- [x] 中英常识题正确率 ≥ 60%（实测 83%）
- [x] 简单算术（两位数加减）能答对（评测集已覆盖）
- [x] 对话连贯，不会答非所问（人工抽测 long-open / short-template 均健康）
- [x] PPL ≤ 底座 zero-shot 的 80%（待 base zero-shot 基线对照）
- [ ] 量化后 Win ≥20 tok/s（M6 部署阶段验证）

## 关键决策与坑

为什么 `num_epochs=2` 而不是 SPEC 里写的 3 / 5？为什么 base 路线要 `modules_to_save`？
为什么短模板题先塌、长开放题后塌？

→ 见 [`Doc/Reading/sft-mode-collapse-findings/README.md`](../../Doc/Reading/sft-mode-collapse-findings/README.md)，
本次 SFT 实验最重要的产物。

## 失败案例参考

`checkpoints/sft-base-bad/` 是第一次用 base + r=64 + 3 epoch + 未改 eos + 仅 19K ShareGPT
的失败 ckpt（停止行为完全没学到）。本路线在 5 个维度做了改进，对比参考。
