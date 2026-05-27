# echo 训练步骤速查

> cwd: `Echo/echo/`，所有命令均在此目录下执行。

## 1. 数据准备

```bash
uv run python scripts/prepare_data.py
```

从 HuggingFace 下载 sharegpt_gpt4，清洗转格式，输出 `data/sft/train.jsonl` + `data/sft/val.jsonl`。

可选参数：`--max-samples N`、`--force`（覆盖已有文件）。

## 2. SFT 训练

```bash
# 生产（QLoRA, 8GB+ GPU）
uv run python scripts/sft.py --config configs/sft-8g.yaml

# 代码验证（CPU/Mac）
uv run python scripts/sft.py --config configs/sft-tiny.yaml

# 断点续训
uv run python scripts/sft.py --config configs/sft-8g.yaml --resume checkpoints/sft/checkpoint-2000
```

产出 adapter 到 `checkpoints/sft/final`。

## 3. 评测 Loss（可选）

```bash
uv run python scripts/eval_loss.py --adapter-dir checkpoints/sft/final
uv run python scripts/eval_loss.py --adapter-dir checkpoints/sft/checkpoint-2000
```

对比不同 checkpoint 的 val loss / perplexity，选最优。

下方都假设本步骤 eval_loss 得出 checkpoint-2000 最优。

## 4. 推理验证

```bash
# 直接加载 adapter（不需要先 merge，显存友好）
uv run python scripts/generate.py --adapter-dir checkpoints/sft/checkpoint-2000
```

交互式多轮对话，验证效果。支持 `--temperature`、`--top-k`、`--top-p` 等参数。

## 5. 评测

```bash
uv run python scripts/eval.py --adapter-dir checkpoints/sft/checkpoint-2000
```

PPL + 常识题（需 `eval/questions.jsonl`，无则自动跳过）+ 对话样例生成。结果存 `eval/results.json`。

## 6. 合并 Adapter

```bash
# 默认 final
uv run python scripts/merge.py

# 显示指定中间 ckpt
uv run python scripts/merge.py --adapter-dir checkpoints/sft/checkpoint-2000
```

底座 + adapter → 完整 bf16 权重，输出到 `checkpoints/merged/`。导出 GGUF 前必须做。

## 7. 导出 GGUF + 量化部署

```bash
uv run python scripts/export_gguf.py
```

依赖本地 llama.cpp。流程：merged → f16 GGUF → Q4_K_M。支持 `--quant Q8_0` 等。
