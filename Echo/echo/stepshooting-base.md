# echo 训练步骤速查 · base 路线（默认）

> cwd: `Echo/echo/`，所有命令均在此目录下执行。
> 配置：`configs/sft-8g-base.yaml`（底座 `Qwen/Qwen2.5-1.5B`，**非** Instruct）

本路线是 echo 项目的默认主线：用 base 从零跑通 SFT 全链路，教模型学 ChatML、
停止行为、风格。所有脚本默认值都指向本路线，下方命令无需任何额外参数即可运行。

## 关键设计

| 维度 | 取值 | 理由 |
|---|---|---|
| 底座 | `Qwen/Qwen2.5-1.5B` (非 Instruct) | 锚点：教模型从零学 ChatML |
| `is_base_model` 标志 | true | sft.py 自动 patch `tokenizer.eos_token = <\|im_end\|>` |
| LoRA r | 128 | 给新行为（停止信号、特殊 token embedding）足够容量 |
| modules_to_save | `embed_tokens` + `lm_head` | 让 `<\|im_end\|>` 等 special token 真正学到 embedding |
| epoch | 5 | base 学新格式 + 停止信号需要更多曝光 |
| 训练数据 | `train_aug.jsonl` (~29K) | ShareGPT 19K + 短问答 10K，强化 `<\|im_end\|>` 信号密度 |
| adapter 输出 | `checkpoints/sft-base/` | 与对照 instruct 路线物理隔离 |
| 3060 训练时长 | ~10-12h | r=128 + modules_to_save + 5 epoch + 数据 1.5× |
| adapter 大小 | ~600MB | 含全量训练的 embed/lm_head |

## 1. 数据准备

```bash
uv run python scripts/prepare_data.py
```

下载 `sharegpt_gpt4` + `vicgalle/alpaca-gpt4` (英) + `silk-road/alpaca-data-gpt4-chinese` (中)，
混合后输出到 `data/sft/train_aug.jsonl` + `data/sft/val_aug.jsonl`。

可选参数：`--short-qa-en N`、`--short-qa-zh N`、`--force`。

## 2. 训练前预检（强烈推荐）

```bash
uv run python scripts/preflight.py --config configs/sft-8g-base.yaml
```

5-10 分钟跑完四层检查（不写 ckpt）：
- Layer 1 静态：config 字段、tokenizer special token、数据格式 + 长度分布、chat_template 渲染
- Layer 2 模型：加载到 device、PEFT 包装、tie_word_embeddings 状态、trainable 参数比例
- Layer 3 单步：loss mask 可视化（看 `<|im_end|>` 是否进入 loss 计算）+ 5 步微训练看 loss 下降
- Layer 4 生成：底座 generate 试探，验证 prompt 形态与 stop_ids 配置

任意一层失败立即退出 + 写报告 `preflight-report-sft-8g-base.md`。

> 历史教训：第一次 base 训练 6h 后才发现"模型不会停止"——根因是数据/eos/embed
> 三层信号本可在前 5 分钟暴露。preflight 的存在就是为了不再返工。

## 3. SFT 训练

```bash
uv run python scripts/sft.py --config configs/sft-8g-base.yaml
```

> sft.py 不带默认 config，需显式传。其他脚本默认值已指向 base 路线。

产出 adapter 到 `checkpoints/sft-base/final`（中间 ckpt 在 `checkpoints/sft-base/checkpoint-*`）。

断点续训：

```bash
uv run python scripts/sft.py --config configs/sft-8g-base.yaml \
    --resume checkpoints/sft-base/checkpoint-N
```

> 显存观察：modules_to_save 把 embed/lm_head 升到 bf16 全精度训练，3060 12GB 上余量较紧。
> 若 OOM，先把 `max_seq_length` 从 2048 调到 1536。

## 4. 评测 Loss（可选）

```bash
# 默认即评 final
uv run python scripts/eval_loss.py

# 指定中间 ckpt
uv run python scripts/eval_loss.py --adapter-dir checkpoints/sft-base/checkpoint-N
```

下方假设 `checkpoint-N` 为最优，按实际结果替换。

## 5. 推理验证

```bash
# 默认即加载 checkpoints/sft-base/final
uv run python scripts/generate.py

# 用中间 ckpt
uv run python scripts/generate.py --adapter-dir checkpoints/sft-base/checkpoint-N
```

交互式多轮对话。**重点检查停止行为**：模型回答完应自然停止，不应自问自答。

## 5.5 裸 base 对照（可选，推荐做一次）

```bash
# 纯续写模式：看 base 原始倾向（输入会被当文章接着写）
uv run python scripts/generate_base.py

# 伪 ChatML 模式：手拼 ChatML 包装，看 base 能否仅靠 prompt 工程出对话能力
uv run python scripts/generate_base.py --mode chatml
```

仅加载裸 `Qwen/Qwen2.5-1.5B`，不挂 adapter。用同样的 prompt 跑一遍 base + 跑一遍 SFT 后
（第 5 节的 generate.py），直观对比"SFT 把推理范式从续写扭成对话"的效果。

> 注意：instruct 路线不需要这个对照——instruct 底座本身已学会对话和停止，
> raw 模式无意义，chatml 模式接近 SFT 后。要对比直接拿 instruct 底座
> vs SFT 后 instruct 比即可。

## 6. 评测

```bash
uv run python scripts/eval.py --adapter-dir checkpoints/sft-base/checkpoint-N
```

PPL + 常识题（需 `eval/questions.jsonl`，无则自动跳过）+ 对话样例生成。结果存 `eval/results.json`。

## 7. 合并 Adapter

```bash
# 默认 final → checkpoints/merged-base/
uv run python scripts/merge.py

# 指定中间 ckpt
uv run python scripts/merge.py --adapter-dir checkpoints/sft-base/checkpoint-N
```

底座 + adapter → 完整 bf16 权重。导出 GGUF 前必须做。

## 8. 导出 GGUF + 量化部署

```bash
uv run python scripts/export_gguf.py
```

依赖本地 llama.cpp。流程：`merged-base` → f16 GGUF → Q4_K_M，输出到 `checkpoints/gguf-base/`。

## 失败案例参考

`checkpoints/sft-base-bad/` 是第一次用 base + r=64 + 3 epoch + 未改 eos + 仅 19K ShareGPT
的失败 ckpt（停止行为完全没学到）。本路线在 5 个维度做了改进，对比参考。
