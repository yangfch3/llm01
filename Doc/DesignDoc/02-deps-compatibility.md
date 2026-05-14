# 依赖兼容性与部署选型说明

> 本文记录本项目对 Python 3.12 + 双平台（Win 3060 / Mac Apple Silicon）下的依赖兼容性评估，
> 以及由此推导出的部署路线选型。M0 选定依赖版本、M5/M6 部署阶段需对照本文。
>
> 本文是 **持续追加** 的活文档，踩坑后请追加到 §5。

## 1. Python 3.12 主流依赖兼容性总览

### 1.1 全绿（直接用，无坑）

| 库 | 最低推荐版本 | 用途 |
|---|---|---|
| `torch` | ≥2.2 | 核心，CUDA 12.x / MPS 双端 wheel 齐全 |
| `numpy` | ≥1.26 | 1.26 起原生支持 3.12 |
| `transformers` | ≥4.36 | 模型加载 / 推理 / 训练 |
| `tokenizers` | 最新 | 分词器（Rust 实现） |
| `datasets` | ≥2.16 | 数据集管线 |
| `accelerate` | ≥0.25 | 训练加速封装 |
| `peft` | ≥0.7 | LoRA 微调 |
| `trl` | ≥0.7 | SFT / DPO 训练 |
| `safetensors` | 最新 | 权重存储格式 |
| `huggingface-hub` | 最新 | 权重 / 数据分发 |
| `sentencepiece` | ≥0.2.0 | 部分底座的分词器依赖 |
| `ruff` / `pyyaml` / `tqdm` / `rich` | 最新 | 通用工具 |
| `evaluate` | 最新 | 评测 |
| `llama-cpp-python` | 最新 | 跨平台量化推理 |

### 1.2 有坑但可控

#### `bitsandbytes`（QLoRA 量化训练用）

- **历史**：长期只支 Linux，Win 靠社区 fork
- **现状**：**0.43+ 官方支持 Windows + CUDA 12.x**
- **Mac**：完全不支持
- **应对**：
  - 锁定 `bitsandbytes>=0.43.0`，放 `train-cuda` 分组
  - `doctor.py` 增加 `import bitsandbytes; bnb.cuda_setup.main.evaluate_cuda_setup()` 自检
  - 失败降级：用纯 LoRA（不量化），3060 12GB 跑 0.5B 仍可

#### `flash-attn`（注意力加速）

- **状态**：官方只发 Linux wheel；Win 编译极难（CUDA toolkit + 对应 MSVC + 十几 GB 中间产物）
- **应对**：
  - **不强依赖**，全部代码走 PyTorch 2.x 自带的 `scaled_dot_product_attention`
  - 性能差距对 echo-mini（~30M）和 0.5B 底座可忽略
  - 真碰到瓶颈再单独评估

### 1.3 不用（明确避开）

| 库 | 原因 | 替代 |
|---|---|---|
| `vllm` | 不支持 Windows native（要 WSL），不支持 Mac | **Ollama** + `llama-cpp-python` |
| `auto-gptq` / `autoawq` | Win 编译坑多，跨平台体验差 | GGUF 量化（llama.cpp 系） |
| `xformers` | 与 PyTorch 版本耦合紧，Win 麻烦 | PyTorch 原生 SDPA |

## 2. 部署路线选型

### 2.1 选型结论

| 通道 | 用途 | 跨平台 |
|---|---|---|
| **Ollama** | 终端/桌面体验、demo 演示、最简上手路径 | Win / Mac / Linux 全支持 |
| **`llama-cpp-python`** | 程序化调用、自定义脚本、本地服务 | Win / Mac / Linux 全支持 |
| `transformers` 原生 | 训练后第一时间推理验证、调试 | Win (CUDA) / Mac (MPS) |

三者底层关系：
- Ollama / `llama-cpp-python` 共底层 `llama.cpp`，都跑 **GGUF 格式**量化模型
- `transformers` 跑 **HF 原生权重**（safetensors）

工作流：
```
训练（transformers + peft）
   │
   ▼
合并 LoRA → HF 权重
   │
   ▼
量化导出 GGUF（M6 量化阶段）
   │
   ├──▶ Ollama 加载（demo）
   └──▶ llama-cpp-python 加载（程序化）
```

### 2.2 不选 vLLM 的理由

- 双平台不兼容（Win/Mac native 都不支持）
- 优势在高并发吞吐，本项目是单用户本地玩，吞吐不是瓶颈
- 学习成本 > 收益

### 2.3 概念澄清：vLLM vs Ollama

很多人会混淆，简记：

- **vLLM**：推理引擎库，目标是高并发线上服务，类比 TGI / TensorRT-LLM
- **Ollama**：本地模型管理 CLI 工具，类比"LLM 界的 Docker"，底层是 llama.cpp
- 不是同一层东西，本项目场景下 Ollama 完胜

## 3. 依赖分组建议（M0 落 `pyproject.toml` 时参考）

```toml
[project.optional-dependencies]
# 核心（跨平台通用）
echo-mini = [
  "torch>=2.2",
  "transformers>=4.36",
  "tokenizers",
  "datasets>=2.16",
  "accelerate>=0.25",
  "peft>=0.7",
  "trl>=0.7",
  "safetensors",
  "huggingface-hub",
  "sentencepiece>=0.2",
  "pyyaml",
  "tqdm",
  "rich",
]

echo = [...]  # 与 echo-mini 大量重叠，可复用

# 平台专属
train-cuda = [
  "bitsandbytes>=0.43",
]
train-mps = [
  # 暂无强制额外依赖，留扩展位
]

# 部署（跨平台）
deploy = [
  "llama-cpp-python",
]

# 开发工具
dev = [
  "ruff",
  "pytest",
]
```

安装命令：
```bash
# Win
uv sync --extra dev --extra echo-mini --extra train-cuda --extra deploy

# Mac
uv sync --extra dev --extra echo-mini --extra train-mps --extra deploy
```

Ollama 不通过 pip 装，是独立 CLI 工具，README 单独说明安装方法。

## 4. Python 3.12 vs 3.11 取舍记录

- 选 3.12 的理由：用户已就绪、性能略好（~5%）、主流库全支持
- 不选 3.11 的代价：遇到边缘库（极个别）不支持时要 fallback
- 真碰到 3.12 不兼容的库，单独建子 venv 隔离，不动主环境

## 5. 踩坑记录（持续追加）

> 格式：日期 · 平台 · 现象 · 根因 · 解决

（M0 启动后补充）
