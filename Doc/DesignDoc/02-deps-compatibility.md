# 依赖兼容性与部署选型说明

> 本文记录本项目对 Python 3.12 + 双平台（Win 3060 / Mac Apple Silicon）下的依赖兼容性评估，
> 以及由此推导出的部署路线选型。M0 选定依赖版本、M5/M6 部署阶段需对照本文。
>
> **决策性文档，改动频率低**。具体踩坑细节迁到 [`troubleshooting.md`](troubleshooting.md)。

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
| `llama-cpp-python` | 最新 | 跨平台量化推理（Win 可能要编译，见 §1.2） |

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

#### `llama-cpp-python`（量化推理）

- **状态**：跨平台支持，但 Win + Python 3.12 经常没现成 wheel，触发本地编译
- **本地编译要求**：MSVC、CMake、（CUDA 加速时）CUDA toolkit 多方版本对齐，新 VS 版本下踩坑率高
- **应对**：
  - 拆到独立 extras `deploy-llamacpp`，主流程默认不装
  - M6 部署阶段再 `uv sync --extra deploy-llamacpp`
  - 编译失败时回退到预编译 wheel：<https://github.com/abetlen/llama-cpp-python/releases>
  - 实在装不上：跳过 `llama-cpp-python`，部署只走 Ollama（本项目主路径）

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
- 概念区分：**vLLM** 是高并发推理引擎（类比 TGI / TensorRT-LLM），**Ollama** 是本地模型管理 CLI（类比 LLM 界的 Docker，底层 llama.cpp），两者不是同一层东西

## 3. 依赖分组建议（M0 落 `pyproject.toml` 时参考）

> **真源在仓库根 `pyproject.toml`，下表仅示意分组意图，不保证字段实时同步**。
> 当真源与本表冲突，以 `pyproject.toml` 为准。

| 分组 | 维度 | 关键依赖 | 备注 |
|---|---|---|---|
| `courseware` | 模块 | torch、numpy、matplotlib、jupyter、mkdocs* | 课件与 Playground 练习 |
| `echo-mini` | 模块 | torch、transformers、tokenizers、datasets、accelerate、peft、trl、safetensors、huggingface-hub、sentencepiece | 全链路从零训练 |
| `echo` | 模块 | 同 echo-mini（高度重叠） | M5 启动后按底座需求增减 |
| `train-cuda` | 平台 | bitsandbytes>=0.43（带 win32/linux marker） | Win/Linux 训练加速 |
| `train-mps` | 平台 | （留空占位） | Mac 端预留 |
| `deploy` | 部署 | （留空占位） | 跨平台轻量部署依赖预留 |
| `deploy-llamacpp` | 部署 | llama-cpp-python | M6 部署阶段按需，Win 易触发本地编译 |
| `dev` | 开发 | ruff、pytest | 通用开发工具 |

`torch` 在 Win/Linux 走 `pytorch-cu124` 源（`[tool.uv.sources]` + `[[tool.uv.index]]`），Mac 走默认 PyPI。
`train-cuda` 与 `train-mps` 在 `[tool.uv].conflicts` 中互斥声明。

安装命令：
```bash
# Win
uv sync --extra dev --extra courseware --extra echo-mini --extra train-cuda

# Mac
uv sync --extra dev --extra courseware --extra echo-mini --extra train-mps

# M6 部署阶段（任一平台）按需追加：
uv sync --extra deploy-llamacpp
```

Ollama 不通过 pip 装，是独立 CLI 工具，README 单独说明安装方法。

## 4. Python 3.12 vs 3.11 取舍记录

- 选 3.12 的理由：用户已就绪、性能略好（~5%）、主流库全支持
- 不选 3.11 的代价：遇到边缘库（极个别）不支持时要 fallback
- 真碰到 3.12 不兼容的库，单独建子 venv 隔离，不动主环境

## 5. 踩坑记录

踩坑记录已迁出本文档，集中维护在 [`troubleshooting.md`](troubleshooting.md)（活文档，按日期倒序追加）。

本文只保留**决策性**内容（兼容性表、部署选型、依赖分组建议）。具体踩坑细节请去 troubleshooting 查。
