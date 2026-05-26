# echo-mini 技术规格

> echo-mini 全链路开发与训练的第一参考资料。决策冲突时本文优先于 README，但低于 `Doc/DesignDoc/00-startup-proposal.md`。

## 1. 定位

- 教学产物，走通"数据→分词器→Pretrain→SFT→评测→推理"全链路
- 效果不强求，**学懂每个环节**是核心目标
- 产出弱但能续写/简单对话的迷你模型

## 2. 模型架构

**风格**：Llama 风格 Decoder-only Transformer

| 组件 | 选型 |
|---|---|
| Normalization | RMSNorm (Pre-Norm) |
| 位置编码 | RoPE (Rotary Position Embedding) |
| FFN | SwiGLU (gate + up + down) |
| 注意力 | GQA (Grouped Query Attention) |
| 激活函数 | SiLU (SwiGLU 内含) |

### 2.1 超参规格（~60M 参数）

| 超参 | 值 | 说明 |
|---|---|---|
| `vocab_size` | 16,386 | BPE 分词器词表（含 6 个特殊 token） |
| `d_model` | 512 | 隐藏层维度 |
| `n_layers` | 16 | Transformer 层数 |
| `n_heads` | 8 | 注意力头数 |
| `n_kv_heads` | 4 | KV 头数 (GQA, ratio=2) |
| `d_ff` | 1,376 | FFN 中间维度 (≈ 2.67 × d_model, SwiGLU 修正) |
| `max_seq_len` | 1,024 | 最大序列长度 |
| `dropout` | 0.0 | Pretrain 不用 dropout |
| `rope_theta` | 10,000 | RoPE base frequency |

> 参数量估算：embedding(16K×512) + 16层×(attn+ffn+norm) ≈ 8.4M + 48M ≈ **~57M**。
> 若需微调到正好 60M 可调整 d_ff 或 n_layers，训练前最终确认。

### 2.2 权重初始化

- Embedding: N(0, 0.02)
- Linear layers: N(0, 0.02 / √(2 × n_layers))
- RMSNorm: weight 全 1

## 3. 分词器

| 项 | 规格 |
|---|---|
| 算法 | BPE (HuggingFace `tokenizers` 库训练) |
| 词表大小 | 16,386 |
| 语言覆盖 | 中英双语 |
| 特殊 token | `<pad>`(0), `<bos>`(1), `<eos>`(2), `<unk>`(3), `<|user|>`(4), `<|assistant|>`(5) |
| 训练语料 | 与 Pretrain 语料同源（取子集即可） |

产物目录：`Echo/echo-mini/tokenizer/`（模型文件 .gitignore，训练脚本入仓）

## 4. 数据

### 4.1 Pretrain 语料

| 来源 | 语言 | 预估量 | 获取方式 |
|---|---|---|---|
| HuggingFaceFW/fineweb-edu | EN | ~300-500MB | HF datasets streaming 切片 |
| Wikipedia (zh subset) | ZH | ~200-300MB | HF datasets |
| SkyPile-150B 子集 | ZH | ~100-200MB | HF datasets 切片 |

- 总量：~500MB-1GB 原始文本，tokenize 后约 **100-250M tokens**
- 中英比例：约 **4:6**（英文略多）
- 数据处理：去重、长度过滤、质量过滤，输出为预分词的 binary 格式（token ids）

### 4.2 SFT 对话数据

| 项 | 规格 |
|---|---|
| 来源 | HuggingFace 现成对话数据集（如 ShareGPT 子集 / Alpaca 翻译版等） |
| 规模 | ~5K-20K 条对话 |
| 格式 | multi-turn 对话，统一为 `[{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]` |
| 语言 | 中英混合 |

## 5. 训练

### 5.1 Pretrain

| 项 | 规格 |
|---|---|
| 目标 | Next Token Prediction (Causal LM) |
| 框架 | Accelerate (手写 training loop) |
| 精度 | bf16 混合精度 |
| 优化器 | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| 学习率 | peak 3e-4, cosine decay, warmup 2% steps |
| Batch size | config-full: 根据显存最大化; config-tiny: 4 |
| Gradient accumulation | 按需（等效 batch size 目标 ~64-128 seqs） |
| Gradient checkpointing | 可选开启（60M 下大概率不需要） |
| Checkpoint 策略 | 每 N steps 存一次，支持断点续训 |
| 预计训练时长 (full) | 3060 12GB 上 3-8 小时（视最终数据量和 batch） |

### 5.2 SFT

提供两版实现，对比学习：

| 版本 | 框架 | 训练方式 |
|---|---|---|
| SFT-v1 | Accelerate 手写 loop | Full-param fine-tuning |
| SFT-v2 | trl `SFTTrainer` | LoRA (rank=16, alpha=32) |

两版共享同一份数据处理流程和配置格式。

### 5.3 配置双份化

所有训练任务提供：

- `config-full.yaml`：Win 3060 12GB 生产配置
- `config-tiny.yaml`：Mac/CPU ~100 步验证配置

同一入口脚本通过 `--config` 参数切换。

## 6. 推理

| 项 | 规格 |
|---|---|
| 实现 | 手写推理 CLI（`scripts/generate.py`） |
| KV Cache | 支持（ch07 已学） |
| 采样策略 | temperature + top-k + top-p |
| 跨平台 | CUDA / MPS / CPU 均可，走 `get_device()` |

## 7. 评测

| 指标 | 方法 |
|---|---|
| PPL | 留出验证集计算 perplexity |
| 生成质量 | 给定 prompt 生成样例，人工 spot check |

不设量化验收标准。能续写连贯句子、loss 收敛即视为成功。

## 8. 目录结构

```
Echo/echo-mini/
├── SPEC.md              本文档
├── README.md            训练配方与结果记录（训练完成后补）
├── configs/
│   ├── pretrain-full.yaml
│   ├── pretrain-tiny.yaml
│   ├── sft-full.yaml
│   └── sft-tiny.yaml
├── data/                (.gitignore) 数据存放
├── tokenizer/
│   ├── train_tokenizer.py   分词器训练脚本
│   └── (产物 .gitignore)
├── src/echo_mini/
│   ├── __init__.py
│   ├── model.py         模型定义
│   ├── config.py        模型配置 dataclass
│   ├── data.py          数据加载与处理
│   └── utils.py         训练工具函数
├── scripts/
│   ├── prepare_data.py  数据下载与预处理
│   ├── pretrain.py      预训练入口
│   ├── sft.py           SFT 入口 (Accelerate 版)
│   ├── sft_trl.py       SFT 入口 (trl 版)
│   ├── generate.py      推理 CLI
│   └── eval.py          评测脚本
└── checkpoints/         (.gitignore) 权重，走 HF Hub
```

## 9. 开发顺序

```
T4.1  prepare_data.py → 下载、清洗、存储语料
T4.2  train_tokenizer.py → 训练 BPE 分词器
T4.3  model.py + config.py → 模型实现
T4.4  pretrain.py → Pretrain 训练脚本
T4.5  configs/ → full/tiny 双配置
T4.6  执行 Pretrain (Win full)
T4.7  SFT 数据处理
T4.8  sft.py + sft_trl.py → SFT 训练
T4.9  generate.py → 推理 CLI
T4.10 eval.py → PPL + 生成样例
T4.11 上传 HF Hub
T4.12 补 README.md 训练配方
```

## 10. 共用基础设施 (Echo/shared/)

| 模块 | 用途 |
|---|---|
| `device.py` | 统一设备选择 (已有) |
| 后续按需新增 | 日志、评测工具、通用 data utils |

新增 shared 模块时遵循：echo-mini 和 echo 都能复用才下沉。

## 11. 约束与铁律

- 禁止硬编码设备，走 `get_device()`
- 路径一律 `pathlib.Path`
- 大文件不入 Git，走 HF Hub
- 训练脚本必须提供 full/tiny 双配置
- 代码风格遵循 ruff 配置
- 文件编码 UTF-8，换行 LF，末尾留空行
