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
| epoch | 2 | 实测 5 epoch 严重过拟合，sweet spot 在 epoch 0.5~1.0；详见 §5.6 |
| 训练数据 | `train_aug.jsonl` (~29K) | ShareGPT 19K + 短问答 10K，强化 `<\|im_end\|>` 信号密度 |
| adapter 输出 | `checkpoints/sft-base/` | 与对照 instruct 路线物理隔离 |
| 3060 训练时长 | ~4-5h | r=128 + modules_to_save + 2 epoch + 数据 1.5× |
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

## 5.6 选择最佳 ckpt（关键步骤）

base 路线开 `modules_to_save: embed_tokens` 后**记忆能力很强**，训练后期会出现
**按问题类型选择性 mode collapse**：短模板题（晚安祝福 / 周末计划）可能 epoch 1 末就塌，
长开放题撑到 epoch 2 后才塌。loss 单一指标看不出来，必须人工抽测生成。

**测试方法**：每个候选 ckpt 跑 6 题 × 3 次，覆盖三种类型：

```bash
uv run python scripts/generate.py --adapter-dir checkpoints/sft-base/checkpoint-N
```

| 类型 | 示例 prompt | 健康标志 |
|---|---|---|
| 长开放题 | "推荐一本书"、"如何看待 AI 替代人类工作" | 3 次内容/角度/例子明显不同 |
| 短模板题 | "写一句晚安祝福"、"周末计划推荐" | 3 次至少有词序/句式差异 |
| 数据集中题 | "讲个笑话"、"你最喜欢哪种颜色？" | 3 次至少 2 种不同梗/答案 |

**判定**：

- 全部 3 类都有多样性 → 健康，可选
- 仅长题健康，短题塌 → 部分塌缩，看接受度
- 全部塌缩 → 已 collapse，不可用

**首次实测结果（2026-05-28，旧 5 epoch 配置）**：

| ckpt | epoch | 长题 | 短题 | 数据集中题 | 评价 |
|---|---|---|---|---|---|
| 500 | 0.28 | 健康（仅测笑话） | — | — | 太早 |
| **1000** | **0.56** | **健康** | **健康** | **健康** | **本次最佳** |
| 1500 | 0.84 | 健康 | 临界 | 健康 | 可选 |
| 2000 | 1.12 | 健康 | 已塌 | 健康 | 部分塌 |
| 2500 | 1.40 | 健康 | 已塌 | 已塌 | 部分塌 |
| 4000 | 2.24 | 已塌 | 已塌 | 已塌 | 全塌 |

> 配置已更新为 `num_epochs=2 + save_steps=250`，下次训练 sweet spot 预期在 step 800~1200。

确定最佳 ckpt 后复制为 final，后续脚本默认路径就指向它：

```bash
cp -r checkpoints/sft-base/checkpoint-N checkpoints/sft-base/final
```

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

## 8. 导出 GGUF + 量化（Linux 机器）

> 当前 llama.cpp 仅在 Linux 机器上就绪，本节命令在该机器执行。
> Win/Mac 后续打通后命令一致。merged-base → GGUF 文件的机器间同步走 scp/rsync 自行处理。

### 8.1 一次性环境（Linux）

```bash
# 1. 克隆 llama.cpp（首次）
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp

# 2. 编译（CPU 版即可量化；如需 CUDA 加速量化加 -DGGML_CUDA=ON）
cd ~/llama.cpp
cmake -B build
cmake --build build -j --config Release

# 3. 为 convert_hf_to_gguf.py 建独立 venv，避免污染 echo venv
#    （requirements 里钉了 torch==2.11，会覆盖 echo 的 CUDA torch）
uv venv ~/llama.cpp/.venv-convert --python 3.12
# requirements 含多个 --extra-index-url（pytorch cpu + nightly），
# uv 默认 first-match 会找不到 transformers 等非 torch 包；
# 用 unsafe-best-match 让 uv 在所有索引里找最佳版本。
uv pip install --python ~/llama.cpp/.venv-convert/bin/python \
    --index-strategy unsafe-best-match \
    -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# 4. 导出环境变量给 export_gguf.py 用（或每次跑时传 --convert-python）
#    LLAMA_CPP_DIR 仅当 llama.cpp 不在 ~/llama.cpp、~/repos/llama.cpp、
#    /opt/llama.cpp 这三个默认位置之一时才需要设。
export LLAMA_CPP_DIR=~/llama.cpp                       # 按实际路径改，例：/data2/llama.cpp
export LLAMA_CPP_PYTHON=$LLAMA_CPP_DIR/.venv-convert/bin/python
```

> 为何不纳入 echo 的 pyproject.toml：requirements 含 `torch==2.11.0` 钉版本，
> 与 echo 的 CUDA torch（cu124, 2.4/2.5）冲突；gguf/numpy/sentencepiece 等版本
> 跟 llama.cpp 仓库走，单独 venv 隔离最干净。

### 8.2 转换 + 量化

```bash
# 默认：merged-base → checkpoints/gguf-base/{echo-f16.gguf, echo-Q4_K_M.gguf}
uv run python scripts/export_gguf.py

# 改量化档（可选）
uv run python scripts/export_gguf.py --quant Q8_0

# 只转 f16，不量化（调试 convert 阶段）
uv run python scripts/export_gguf.py --no-quantize
```

产出（Q4_K_M 主线）：

- `checkpoints/gguf-base/echo-f16.gguf` —— f16 中间产物，~3GB（1.5B × 2byte）
- `checkpoints/gguf-base/echo-Q4_K_M.gguf` —— int4 部署产物，~1GB

## 9. Ollama 部署 + demo（Win / Mac）

> 跨平台命令一致。前提：`checkpoints/gguf-base/echo-Q4_K_M.gguf` 已同步到当前机器。

### 9.1 安装 Ollama（一次性）

- Win：<https://ollama.com/download/windows>，桌面应用安装后自动起守护进程
- Mac：`brew install ollama` 或 <https://ollama.com/download/mac>；启动 `Ollama.app` 或 `ollama serve`

### 9.2 一键 demo

```bash
cd Echo/echo
uv run python scripts/run_demo.py
```

脚本会：

1. 检查 `ollama` 可执行 + 守护进程
2. 检查 GGUF 文件存在
3. `ollama create echo -f Modelfile`（首次）
4. 跑内置 5 条中英 prompt 验收

```bash
# Modelfile 调过参数后强制重建
uv run python scripts/run_demo.py --recreate

# 仅环境检查，不跑 prompt
uv run python scripts/run_demo.py --check-only
```

### 9.3 交互式聊天

```bash
ollama run echo
```

### 9.4 速度验收

参照 `Doc/DesignDoc/00-startup-proposal.md` §4.5：

- Win 3060 12GB int4 ≥ 20 tok/s
- Mac Apple Silicon Q4_K_M ≥ 15 tok/s

`ollama run` 末尾会打印 `eval rate: X tokens/s`，即为生成速度。

## 失败案例参考

`checkpoints/sft-base-bad/` 是第一次用 base + r=64 + 3 epoch + 未改 eos + 仅 19K ShareGPT
的失败 ckpt（停止行为完全没学到）。本路线在 5 个维度做了改进，对比参考。
