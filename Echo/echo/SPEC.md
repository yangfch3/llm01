# echo 技术规格

> echo 全链路开发与训练的第一参考资料。决策冲突时本文优先于 README，但低于 `Doc/DesignDoc/00-startup-proposal.md`。

## 1. 定位

- 实用产物，基于开源底座微调，目标"初中生水平"中英基础对话
- 产出可本地运行、可通过 Ollama 一行启动的对话模型
- 验收标准见 `Doc/DesignDoc/00-startup-proposal.md` §4.5

## 2. 底座模型

| 项 | 规格 |
|---|---|
| 模型 | Qwen2.5-1.5B (`Qwen/Qwen2.5-1.5B`) |
| 参数量 | ~1.5B |
| 架构 | Decoder-only Transformer, GQA, RoPE, SwiGLU |
| 上下文 | 32,768 tokens（训练时截断，见 §5） |
| 词表 | 151,665 (Qwen2 tokenizer) |
| 许可证 | Apache 2.0 |

选型理由：
- 中英双语原生支持，无需额外适配
- 1.5B 在 3060 12GB 上 QLoRA 训练余量充足
- 社区生态完善，transformers / peft / trl 原生适配
- Apache 2.0 可商用可开源

## 3. 微调方式

**QLoRA (4-bit NormalFloat + LoRA)**

| 项 | 规格 |
|---|---|
| 量化 | NF4 (bitsandbytes 4bit) |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| LoRA dropout | 0.05 |
| 可训参数占比 | ~2-3% |

底座以 4bit 加载（~1GB 显存），仅训练 LoRA adapter 参数。

## 4. 数据

### 4.1 SFT 对话数据

| 项 | 规格 |
|---|---|
| 来源 | 公开中英对话数据集（ShareGPT 子集 / Alpaca 翻译版 / 其他 HF 数据集） |
| 规模 | ~10K-20K 条对话 |
| 语言 | 中英混合 |
| 格式 | multi-turn，统一为 Qwen2 chat template |
| 筛选标准 | 去重、去低质、长度过滤（单轮 ≤ 2048 tokens） |

### 4.2 数据格式

统一为 JSON Lines，每条记录：

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

system prompt 统一使用简短通用描述，不设特殊人设。

### 4.3 数据处理流程

```
下载原始数据集 (HF datasets)
    → 格式统一 (转为 messages 格式)
    → 质量过滤 (去重/去短/去乱码)
    → 应用 chat template (Qwen2 格式)
    → tokenize + 截断
    → 保存为 Arrow/datasets 格式
```

## 5. 训练

### 5.1 SFT 训练

| 项 | 规格 |
|---|---|
| 框架 | trl `SFTTrainer` + peft |
| 目标 | Causal LM，仅计算 assistant 部分 loss |
| 精度 | 底座 NF4，LoRA 参数 bf16 |
| 优化器 | paged_adamw_8bit |
| 学习率 | 2e-4, cosine decay, warmup 3% steps |
| Max seq length | 2048 tokens |
| Batch size (full) | 根据显存最大化 (预估 per_device=4, grad_accum=4 → 等效 16) |
| Epochs | 3 |
| 预计训练时长 (full) | 3060 12GB 上 1-3 小时（视数据量） |

### 5.2 配置方案

| 配置文件 | 用途 | 环境 |
|---|---|---|
| `config-full.yaml` | 生产训练 | Win 3060 12GB, QLoRA 4bit |
| `config-tiny.yaml` | 代码验证 | Mac/CPU, bf16 小 batch ~20 步 |

同一入口脚本通过 `--config` 参数切换。

### 5.3 Checkpoint 策略

- 每 epoch 结束存一次 adapter
- 训练结束保存 best adapter（按 eval loss）
- 支持断点续训（resume from checkpoint）
- adapter 权重走 HF Hub 分发

## 6. 合并与导出

训练完成后：

```
LoRA adapter + 底座 → merge → 完整 bf16 权重
    → GGUF 转换 (llama.cpp)
    → 量化 (Q4_K_M / Q8_0)
    → Ollama Modelfile → ollama run echo
```

| 步骤 | 工具 |
|---|---|
| Merge adapter | peft `merge_and_unload()` |
| 转 GGUF | llama.cpp `convert_hf_to_gguf.py` |
| 量化 | llama.cpp `llama-quantize` |
| 部署 | Ollama |

## 7. 推理

| 项 | 规格 |
|---|---|
| 开发阶段 | transformers `pipeline` 或手写 generate loop |
| 部署阶段 | Ollama (GGUF) |
| KV Cache | 依赖框架/推理引擎自带 |
| 采样策略 | temperature + top-k + top-p + repetition_penalty |
| 跨平台 | CUDA / MPS / CPU 均可 |

## 8. 评测

| 指标 | 方法 | 目标 |
|---|---|---|
| PPL | 留出验证集 | ≤ 底座 zero-shot 的 80% |
| 常识/算术 | 自建 50 题评测集 | ≥ 60% 正确率 |
| 对话连贯性 | 人工 30 轮对话 | ≥3 轮不跑题 |
| 推理速度 (量化后) | 实测 tokens/s | Win ≥20 tok/s (int4), Mac ≥15 tok/s (Q4_K_M) |

评测集与脚本放 `Echo/echo/eval/`。

## 9. 目录结构

```
Echo/echo/
├── SPEC.md              本文档
├── README.md            训练配方与结果记录
├── configs/
│   ├── sft-full.yaml    生产训练配置
│   └── sft-tiny.yaml    代码验证配置
├── data/                (.gitignore) 数据存放
│   └── scripts/         数据下载与处理脚本
├── src/echo/
│   ├── __init__.py
│   ├── data.py          数据加载与处理
│   └── utils.py         工具函数
├── scripts/
│   ├── prepare_data.py  数据下载与预处理
│   ├── sft.py           SFT 训练入口
│   ├── merge.py         合并 adapter 到底座
│   ├── export_gguf.py   导出 GGUF（调用 llama.cpp）
│   ├── generate.py      推理 CLI
│   └── eval.py          评测脚本
├── eval/                评测集
├── checkpoints/         (.gitignore) adapter / 合并权重，走 HF Hub
└── Modelfile            Ollama 模型定义文件
```

## 10. 开发顺序

```
T5.1  prepare_data.py → 下载、清洗、格式化 SFT 数据
T5.2  data.py → 数据加载、chat template 应用
T5.3  sft.py → QLoRA SFT 训练脚本
T5.4  configs/ → full/tiny 双配置
T5.5  执行 SFT (Win full)
T5.6  merge.py → 合并 adapter
T5.7  generate.py → 推理 CLI 验证效果
T5.8  eval.py → 评测
T5.9  export_gguf.py + Modelfile → 量化部署
T5.10 上传 HF Hub
T5.11 补 README.md 训练配方
```

## 11. 依赖（已在 pyproject.toml [echo] 分组）

核心：`torch`, `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`, `accelerate`, `huggingface-hub`

部署阶段追加：`llama-cpp-python`（`[deploy-llamacpp]` 分组）

## 12. 约束与铁律

- 禁止硬编码设备，走 `get_device()`
- 路径一律 `pathlib.Path`
- 大文件不入 Git，走 HF Hub
- 训练脚本必须提供 full/tiny 双配置
- 代码风格遵循 ruff 配置
- 文件编码 UTF-8，换行 LF，末尾留空行
- 底座模型通过 HF model id 引用，不在仓库里存权重
- chat template 使用 Qwen2 官方格式，不自造
