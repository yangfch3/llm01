# echo-mini 训前准备

## 已完成

- `src/echo_mini/config.py` — 模型配置 dataclass
- `src/echo_mini/model.py` — Llama-style 模型 (RMSNorm/RoPE/GQA/SwiGLU), 验证 54.8M params
- `src/echo_mini/data.py` — mmap binary DataLoader
- `src/echo_mini/utils.py` — lr scheduler, checkpoint, config loader
- `tokenizer/train_tokenizer.py` — BPE 分词器训练脚本
- `scripts/prepare_data.py` — 数据下载 + tokenize 双命令
- `scripts/pretrain.py` — Accelerate 手写 training loop
- `configs/pretrain-full.yaml` — Win 3060 12GB 配置
- `configs/pretrain-tiny.yaml` — Mac/CPU 100 步验证配置

## 待操作（按顺序）

工作目录：`Echo/echo-mini/`

```bash
# 1. 下载原始文本
uv run python scripts/prepare_data.py download --config configs/pretrain-full.yaml

# 2. 训练分词器
uv run python tokenizer/train_tokenizer.py

# 3. Tokenize → binary
uv run python scripts/prepare_data.py tokenize --config configs/pretrain-full.yaml

# 4. 启动预训练（按显存选配置）
#    12GB 显卡 (RTX 3060 12GB):
uv run accelerate launch --mixed_precision bf16 scripts/pretrain.py --config configs/pretrain-full.yaml
#    8GB 显卡:
uv run accelerate launch --mixed_precision bf16 scripts/pretrain.py --config configs/pretrain-full-8g.yaml
#    CPU/Mac 快速验证:
uv run accelerate launch scripts/pretrain.py --config configs/pretrain-tiny.yaml
```

### 配置说明

| 配置文件 | 显存 | batch × accum | 等效 batch |
|---|---|---|---|
| `pretrain-full.yaml` | 12GB | 16 × 4 | 64 |
| `pretrain-full-8g.yaml` | 8GB | 8 × 8 | 64 |
| `pretrain-tiny.yaml` | CPU/MPS | 4 × 1 | 4 |

> `--mixed_precision bf16` 必须通过 accelerate launch 命令行传入，否则会被 launch 默认值覆盖为 fp32。tiny 配置不需要（已设 `"no"`）。
