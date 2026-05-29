# echo 训练步骤速查 · instruct 路线（对照实验）

> cwd: `Echo/echo/`，所有命令均在此目录下执行。
> 配置：`configs/sft-8g-instruct.yaml`（底座 `Qwen/Qwen2.5-1.5B-Instruct`）

本路线是与 base 主线对照的实验：用已对齐的 Instruct 底座做轻量风格漂移，
验证"已对齐底座 vs 从零教学"的训练量与效果差异。

由于 base 是项目锚点，所有脚本默认值都指向 base，**instruct 路线的每条命令都需
显式传参**。

## 关键差异 vs base 主线

| 维度 | base 主线（默认） | instruct 对照 |
|---|---|---|
| 底座 | Qwen2.5-1.5B (非 Instruct) | Qwen2.5-1.5B-Instruct |
| `is_base_model` | true | (未设, false) |
| LoRA r | 128 | 64 |
| modules_to_save | `embed_tokens` + `lm_head` | 无 |
| epoch | 2 | 2 |
| 训练数据 | `train_aug.jsonl` (~29K) | `train.jsonl` (19K) |
| adapter 输出 | `checkpoints/sft-base/` | `checkpoints/sft/` |
| 3060 训练时长 | ~4-5h | ~4h |
| adapter 大小 | ~600MB | ~150MB |

## 1. 数据准备

```bash
uv run python scripts/prepare_data.py --mode sharegpt-only
```

仅下载 `sharegpt_gpt4`，输出到 `data/sft/train.jsonl` + `data/sft/val.jsonl`（约 19K 条）。

> 默认是 `--mode aug`（base 路线增广数据），instruct 必须显式传 `sharegpt-only`。

## 2. 训练前预检（推荐）

```bash
uv run python scripts/preflight.py --config configs/sft-8g-instruct.yaml
```

详见 [stepshooting-base.md](stepshooting-base.md#2-训练前预检强烈推荐)。Instruct 路线
通常不会出现 base 那种"不会停止"的问题，但跑一下能省去后续排查时间。

## 3. SFT 训练

```bash
uv run python scripts/sft.py --config configs/sft-8g-instruct.yaml
```

代码验证（CPU/Mac）：`--config configs/sft-tiny-instruct.yaml`
断点续训：加 `--resume checkpoints/sft/checkpoint-N`

预期 3060 训练时长约 4h。产出 adapter 到 `checkpoints/sft/final`。

> instruct 路线虽无 base 路线 `modules_to_save: embed_tokens` 带来的强记忆能力，
> 但仍建议训完按 [stepshooting-base.md §5.6](stepshooting-base.md#56-选择最佳-ckpt关键步骤) 的方法
> 抽测多个 ckpt 选 sweet spot，instruct 路线的 sweet spot 可能在 step 1500~3000 区间。

## 4. 评测 Loss（可选）

```bash
uv run python scripts/eval_loss.py \
    --config configs/sft-8g-instruct.yaml \
    --adapter-dir checkpoints/sft/final \
    --val-file data/sft/val.jsonl
```

## 5. 推理验证

```bash
uv run python scripts/generate.py \
    --config configs/sft-8g-instruct.yaml \
    --adapter-dir checkpoints/sft/final
```

## 6. 评测

```bash
uv run python scripts/eval.py \
    --config configs/sft-8g-instruct.yaml \
    --adapter-dir checkpoints/sft/final \
    --val-file data/sft/val.jsonl \
    --output eval/results-instruct.json
```

## 7. 合并 Adapter

```bash
uv run python scripts/merge.py \
    --config configs/sft-8g-instruct.yaml \
    --adapter-dir checkpoints/sft/final \
    --output-dir checkpoints/merged
```

## 8. 导出 GGUF + 量化部署

```bash
uv run python scripts/export_gguf.py \
    --merged-dir checkpoints/merged \
    --output-dir checkpoints/gguf
```
