# ch13 · 部署（Deployment）

> 训完一个模型，HF 权重躺在硬盘上，怎么让它**真正能用**？
> 
> 个人角度：能在普通显卡跑、能在 Mac 跑、能快速返回第一个 token、能塞进 Ollama 等桌面部署工具让别人 `ollama run echo` 一句话启用。
> 
> 本章 = 个人 Demo 小模型部署 ≠ 工业级部署

## 学习目标

1. 理解 fp16 → int8 → int4 显存与速度的量级变化，以及精度衰减的来源
2. 能解释 GGUF 是什么、为什么 llama.cpp 系把它当统一格式
3. 知道在不同场景（开发调试 / demo 演示 / 程序化调用）该选哪条部署链路
4. 能复述把 "训完的 HF 权重 → Ollama 可跑" 完整流程

## 前置依赖

- ch09（混合精度）、ch10（LoRA / adapter 概念）
- 部署属 "工程链路"，没有难数学，但坑特别多

---

## 1. 为什么要量化

### 1.1 显存账

模型权重的显存占用（粗算）：

```
显存(字节) = 参数数 × 每参数字节数
```

| 精度 | 每参数字节 | 0.5B 模型 | 7B 模型 | 70B 模型 |
|---|---|---|---|---|
| fp32 | 4 | 2.0 GB | 28 GB | 280 GB |
| fp16 / bf16 | 2 | 1.0 GB | 14 GB | 140 GB |
| int8 | 1 | 0.5 GB | 7 GB | 70 GB |
| int4 | 0.5 | 0.25 GB | 3.5 GB | 35 GB |

推理时还要加 KV cache（与 batch、序列长度成正比）和激活值，实际占用比表格略高。但权重通常是大头。

如 12GB 显存跑 7B fp16 模型显存就紧巴巴了；int4 后只占 3.5 GB，连 13B 都能塞下。**量化是把 "勉强能跑" 变成 "舒服能跑" 的关键**。

### 1.2 推理速度

量化不只是省显存，**通常也变快**：

- decode 阶段（逐 token 自回归生成）显存带宽（显存的读写速率）是瓶颈：每生成一个 token 都要把整组权重读一遍
- int4 权重比 fp16 小 4 倍 → 显存读取量 1/4 → 带宽瓶颈显著缓解
- 假如用 CPU 推理，int4 的整数运算指令也比浮点快

> prefill 阶段（处理 prompt 那一遍长 forward）是 compute-bound，瓶颈在算力而非带宽，量化加速效果不如 decode 明显。
> 
> 但日常对话生成长度通常 >> prompt，整体还是 decode 占大头，memory-bound。

实际上 12GB 显存跑 7B 模型：

- fp16 权重 14GB 已超 12GB 显存，必须 CPU offload 一部分，吞吐通常掉到 10—15 tok/s（tokens per second，每秒生成 token 数）
- int4 GGUF（GGML Universal File Format，llama.cpp 的模型存储格式，见 §2.2）通过 llama.cpp 全程显卡跑，约 60 tok/s
- **fp16 在 12GB 卡上跑 7B 实际不可行**，量化在这个量级是必选项

### 1.3 精度损失从哪来

权重原本是 fp16 的连续浮点数（如 `0.0123, -0.4567, ...`）。
量化是把它们**离散化**到一组有限的整数级别上：

```
int8: 256 个离散级别（-128 到 127）
int4:  16 个离散级别（-8 到 7）
```

每个权重要找最近的离散级别 → **舍入误差**。权重越小、分布越集中，舍入相对误差越大。
推理时这些小误差会**沿着多层网络累积**，最终输出概率分布会与 fp16 略有偏移。

上面的 "-128 到 127"、"-8 到 7" 是**朴素对称量化**的形式，方便理解原理。真实部署里 GGUF 的 K-quant、bitsandbytes 的 NF4 等都用**非均匀离散**级别（针对正态分布权重做了优化），精度显著优于朴素方案——见 §2.2。

### 自检

1. fp16 → int4 显存省 4 倍，速度也快 ~2 倍 —— "那为什么不直接 int2、int1？"
2. 量化损精度，但为什么实测下游任务分数衰减通常 < 5%？

<details markdown="1">
<summary>答案速查</summary>

1. 离散级别太少 → 舍入误差太大 → 多层累积后输出严重失真。int2 已基本不可用；int1（二值化）只在小模型 / 特殊架构上可行。社区当前下限是 int3（如 GPTQ-3bit），低于此通常崩

2. ① 现代量化算法（GPTQ / AWQ / GGUF 的 K-quant 系列）做了 "重要权重保留高精度 + 不重要权重激进量化" ② LLM 的输出概率分布对小扰动有冗余（top-k 概率最高的 token 不会因为权重小变动就被挤出） ③ 多选题打分对小数概率差不敏感，只要排序不变就答案不变

</details>

---

## 2. 量化的几种主流路线

### 2.1 训练时量化 vs 训练后量化

| 类别 | 代表 | 何时做 | 难度 |
|---|---|---|---|
| **PTQ**（Post-Training Quantization） | GGUF / GPTQ / AWQ | 训完后离线量化 | 低，社区主流 |
| **QAT**（Quantization-Aware Training） | PyTorch `torch.ao.quantization` 等 | 训练过程插入 fake quant 节点感知量化误差 | 高，少用 |
| **QLoRA** | bitsandbytes 4bit + LoRA | 把底座 4bit 冻结 + 训 LoRA | 中，省显存训练（NF4 / double quantization 细节见 ch10 §4.4） |

> echo-mini/echo 项目用 PTQ：训练阶段保持 fp16/bf16，训完后导出 GGUF int4 部署。QLoRA 用在 M5 微调底座阶段（省显存训练，不是部署）。

> 另有一类做法是直接以指定精度（如 fp8）训练（如 DeepSeek-V3 低精 fp8 版），权重天生低精度，不经过 PTQ 步骤。此类属前沿大厂方案，本课程不涉及。

### 2.2 GGUF + llama.cpp：跨平台事实标准

**llama.cpp**（ggerganov 开源）是用 C++ 写的纯 CPU/GPU 通用推理引擎。**GGUF** 是它的模型存储格式，特点：

- **单文件**：权重 + 词表 + 元数据打包成一个 `.gguf` 文件（chat template 在新版 convert 脚本下也会写入元数据，但部署时是否生效另说，见 §4）
- **跨平台**：Windows / Mac / Linux / Android 都能跑
- **多种量化级别**：
  - `Q8_0`：8-bit ≈ int8 量化，几乎无衰减，文件接近 fp16 一半
  - `Q5_K_M`：5-bit，常用平衡点
  - `Q4_K_M`：4-bit + K-quant ≈ int4 量化，**社区默认推荐**，质量与体积平衡最好
  - `Q4_0`：老式 4-bit ≈ int4 量化，质量略差于 Q4_K_M
  - `Q3_K_M`：3-bit，明显衰减但仍可用
  - `Q2_K`：2-bit，质量明显退化，仅紧凑场景

为什么不是我们之前学过的 fp8, int8, int4 等概念？

是因为 GGUF 格式有自己的量化方案命名体系，它们本质上就是 int8、int4 等，只是使用的不是标准 IEEE 格式 / 基本不使用硬件原生支持的格式 / 自行编解码。

> **K-quant 是什么**：传统 int4 是 "每个权重独立量化"，K-quant 是 "按 block 分组 + 每组单独 scale + 重要 block 用更高 bit"，质量显著优于朴素 int4

了解了 GGUF 与 K-quant 后，看它在工具链中的位置：

```
        llama.cpp (C++ 引擎，跑 GGUF)
                │
   ┌─────────────┼────────────┐
   ▼            ▼           ▼
llama-cpp     Ollama      LM Studio
-python      (CLI/服务)    (桌面 GUI)
(Python 绑定)
```

### 2.3 GGUF 之外

GGUF 不是唯一的量化格式，下面列表详细介绍。

| 格式 | 用在哪 | 备注 |
|---|---|---|
| **GPTQ**（Generative Pre-trained Transformer Quantization，针对 GPT 系的量化算法） | `auto-gptq` / vLLM | GPU 推理向，文件小，仅 CUDA 友好 |
| **AWQ**（Activation-aware Weight Quantization，激活感知权重量化） | `autoawq` / vLLM | GPU 推理，质量略好于 GPTQ |
| **bitsandbytes 4bit** | 训练阶段（QLoRA）/ HF 推理 | 不存盘，运行时即时量化 |
| **GGUF** | llama.cpp 系 | **跨平台首选**，本项目主路径 |

> DeepSeek-V3 提供的 671B fp8 精度推荐部署方案：SGLang / vLLM
>
> HF 上的 Qwen 2.5 原始权重：bf16 safetensors，全精度，无量化；也提供量化版：`Qwen2.5-7B-Instruct-GPTQ-Int4`

### 自检

1. 同一个模型 Q4_K_M 与 Q4_0 都是 4-bit，体积差不多，为什么推荐 Q4_K_M？
2. 公司服务器上要服务 1000 并发请求，量化方案该选 GGUF 吗？

<details markdown="1">
<summary>答案速查</summary>

1. K-quant 用"分组量化 + 重要 block 高精度"策略，下游任务衰减明显小于朴素 Q4_0。体积上 Q4_K_M 略大（多了 scale 表），但精度优势远大于代价

2. 不该。GGUF / llama.cpp 是**单实例本地推理**优化的，不擅长高并发。高并发场景应用 vLLM + AWQ/GPTQ + 多卡。本项目走 GGUF 是因为目标是"单用户本地玩 + 跨平台 demo"

</details>

---

## 3. 三条部署通道对比

### 3.1 `transformers` 原生

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("path/to/echo")
model = AutoModelForCausalLM.from_pretrained("path/to/echo", torch_dtype="auto", device_map="auto")
out = model.generate(tok("你好", return_tensors="pt").input_ids.to(model.device), max_new_tokens=64)
print(tok.decode(out[0]))
```

- **优点**：训练后第一时间能跑，与训练代码同栈，方便调试
- **缺点**：跑得慢（Python 调度开销、无 kernel 融合、无 continuous batching）、显存占用高（fp16）、Mac 上 MPS 路径偶有 op fallback
- **适用**：开发调试、训练后立即推理验证

### 3.2 `llama-cpp-python`

```python
from llama_cpp import Llama
llm = Llama(model_path="echo-q4_k_m.gguf", n_ctx=2048)
out = llm("你好", max_tokens=64)
print(out["choices"][0]["text"])
```

- **优点**：
  - 跑 GGUF 量化模型，跨平台，速度比 transformers 快很多
  - 可自定义采样 —— ch07 提到的输出策略
- **缺点**：
  - Win + Python 3.12 经常没现成 wheel，触发本地编译（详见 `02-deps-compatibility.md`）
  - 编译要 MSVC / CMake / 可能 CUDA toolkit
- **适用**：程序化调用、写本地服务、自定义采样策略

### 3.3 Ollama

llama.cpp 生态衍生出多款桌面部署工具（LM Studio、Jan、Ollama 等），把 "手动编译 + 命令行参数" 打包成开箱即用的体验。本项目选 **Ollama** 作为 M6 主路径 —— CLI 优先、轻量、自带 REST API 方便程序化调用。

```bash
# 一次性导入 GGUF
ollama create echo -f Modelfile
# 跑
ollama run echo
```

`Modelfile` 长这样（类似 Dockerfile）：

```
FROM ./echo-q4_k_m.gguf
TEMPLATE """<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER temperature 0.7
PARAMETER stop "<|im_end|>"
```

- **优点**：装一个独立 CLI（Command Line Interface，命令行界面）就能用，自带 REST API（Representational State Transfer + Application Programming Interface，符合 REST 风格的 HTTP 接口）（端口 11434），跨平台体验最好
- **缺点**：底层还是 llama.cpp，能力受 GGUF 限制；模型管理是 Ollama 自己的 "私有 registry" 格式
- **适用**：demo 演示、桌面应用、最简上手路径
- **Python 调用**：通过 REST API（`POST http://localhost:11434/api/generate`）或官方 `ollama` Python 包，见思考题 3

### 3.4 选型矩阵

| 场景 | 推荐 |
|---|---|
| 训完立刻看效果 | transformers 原生 |
| 写 Python 服务 / 自定义采样 | llama-cpp-python |
| 给别人演示 / 桌面 chat | **Ollama**（M6 主路径） |
| 高并发服务 | 不在本项目范围（用 vLLM） |

> **本章 ≠ 工业级部署**
>
> 本章讲的是"单用户、本地、低延迟"的个人/demo 部署链路。工业级在线服务（多用户并发、SLA 保障、弹性扩缩）是完全不同的技术栈：vLLM / TensorRT-LLM / continuous batching / KV cache 分页 / 多卡 tensor parallel。那条线不在本项目范围内，但知道它存在很重要 —— 本章介绍的都不是搬上生产的手段。

---

## 4. 训完到部署的完整链路

> 这是 echo 项目 M6 阶段会真做的事。本章先把流程讲清，落地代码到 M6。

```
[训练结束]
   │
   │  HF safetensors 权重 + LoRA adapter
   ▼
[1] 合并 LoRA → 得到完整 HF 模型
   │  peft.merge_and_unload() 把 LoRA 合并回 base
   ▼
[2] 转 GGUF
   │  llama.cpp 仓库的 convert_hf_to_gguf.py
   │  输出：echo-fp16.gguf（中间产物，未量化）
   ▼
[3] 量化
   │  llama-quantize echo-fp16.gguf echo-q4_k_m.gguf Q4_K_M
   ▼
[4] 写 Modelfile + Ollama 导入
   │  ollama create echo -f Modelfile
   ▼
[5] 跑 / 分发
   │  ollama run echo
   │  ollama push <user>/echo  （上传到 ollama.com）
```

> 哪一步后可发布到 HF：peft.merge_and_unload() 把 LoRA adapter 合并回 base 模型后，调用 model.save_pretrained() + tokenizer.save_pretrained() 保存出来的就是标准 HF 格式（safetensors + config.json + tokenizer 文件），直接能 push 到 HuggingFace Hub

**几个工程坑**（M6 会踩，先有数）：

- **chat template 必须在 Modelfile 里显式写对**：较新版本的 `convert_hf_to_gguf.py` 会把 HF `tokenizer_config.json` 的 `chat_template` 写入 GGUF 元数据，但 Ollama 的 Modelfile `TEMPLATE` 字段会**覆盖** GGUF 内置模板。所以即使 GGUF 自带模板，Modelfile 写错（或漏写）依然会导致对话被拼坏、模型答非所问
- **stop token**：必须在 Modelfile `PARAMETER stop` 里加，否则模型停不下来
- **量化级别选择**：先 Q4_K_M 跑通，质量不够再 Q5_K_M / Q8_0 升级，体积换质量
- **测试要全链路**：HF 推理通过 ≠ GGUF 推理通过 ≠ Ollama 推理通过，每一步都要验

### 自检

1. 为什么转 GGUF 后输出乱码 / 答非所问，第一件该检查的事是什么？
2. 假设 echo 在 HF transformers 下答得很好，转 Q4_K_M GGUF 后明显变差，可能原因？

<details markdown="1">
<summary>答案速查</summary>

1. 检查 **chat template 与 stop token**：即使 GGUF 内置了从 HF 转过来的 chat_template，Ollama Modelfile 的 `TEMPLATE` 会覆盖它，所以 Modelfile 必须自己写对。多数 "乱码 / 答非所问" 是模板拼错或 stop token 缺失（模型说完话又开始说下一轮的 "用户" 内容）

2. 候选：① 量化级别太激进（Q4 衰减）→ 升 Q5/Q8 再测 ② chat template 写错 → 对照 HF 的 `tokenizer.apply_chat_template` 输出 ③ tokenizer 转换有差异（罕见 BPE 边界 case） ④ KV cache / 采样参数（temperature / top_p）默认值与训练评测时不一致

</details>

---

## 5. 性能基准的"读法"

社区里常见的性能数字（仅供数量级直觉，会因硬件、上下文长度、batch 而变）：

| 平台 | 模型 | 量化 | tok/s（单 batch，纯 GPU/Metal） |
|---|---|---|---|
| RTX 3060 12GB | 7B | Q4_K_M | ~60 |
| M2 Pro Mac | 7B | Q4_K_M (Metal) | ~25 |
| M2 Pro Mac | 7B | Q8_0 (Metal) | ~18 |
| 纯 CPU (i7) | 7B | Q4_K_M | ~6 |

> **关于 3060 12GB + 7B fp16**：约 14GB 权重显存占用，超 12GB 显存，必须做 CPU offload，瓶颈从 GPU 显存带宽变成 PCIe（Peripheral Component Interconnect Express，外设互联标准）传输，吞吐通常掉到 10—15 tok/s，且严格说不算"GPU 原生推理"——所以没列进表。**这正是为什么 7B 在 3060 12GB 上量化是必选项，而非"想省显存才量化"**。
>
> 上面的数字为假设短上下文（≤ 2k）、无 batch、未启用 Flash Attention 类加速。长上下文（如 8k+）tok/s 通常掉一半。

**echo final 验收**（出自 `00-startup-proposal.md`）：

- 量化后效果衰减 ≤ 5%
- 3060 12GB ≥ 20 tok/s（int4）
- Mac ≥ 15 tok/s（GGUF Q4_K_M）

### 自检

1. 同一模型同一量化级别，序列长度 4k 和 32k 时 tok/s 差很多。为什么？
2. "纯 CPU 跑 7B Q4 也有 6 tok/s" —— 这数字看着挺实用，那为什么开源界还在卷 GPU 推理？

<details markdown="1">
<summary>答案速查</summary>

1. 自回归生成时每一步都要 "看完" 前面所有 token 的 KV cache。序列越长，KV cache 越大，每步要做的 attention 计算量与显存读取量越大，tok/s 自然下降。这也是为什么超长上下文模型推理慢

2. ① "6 tok/s" 是单用户、短对话、小模型场景，多人 / 长对话立刻拖到 1 tok/s ② 大模型（70B+）CPU 跑不动 ③ 真实产品需要并发，CPU 没法堆并发 ④ batch 推理 GPU 优势倍数级。CPU 跑 LLM 是 "够用底线"，GPU 是 "产品基线"

</details>

---

## 6. 练习

落到 `Playground/ch13-deploy/`：

| 脚本 | 内容 |
|---|---|
| `01_quant_simulation.py` | 玩具版量化：手动把一组 fp32 权重量化到 int8 / int4，看舍入误差分布；模拟 "权重越多层叠加误差越大" 现象 |
| `02_inference_compare.py` | 用 GPT-2 small 对比 fp32 vs fp16 vs int8（动态量化，PyTorch 内置）的输出差异与单 token 推理耗时；说明 "量化省的不只是空间，还有时间" |

跑法：

```bash
uv run python Playground/ch13-deploy/01_quant_simulation.py
uv run python Playground/ch13-deploy/02_inference_compare.py
```

练习只用 PyTorch + transformers，不依赖 llama.cpp / Ollama（M6 才装）。
真实 GGUF 量化与 Ollama 集成放到 M6 的 `Echo/echo` 工程代码里。

## 思考题

1. 你要把 echo 部署到一个朋友的笔记本（无显卡，CPU i5，16GB 内存）。从模型选型、量化级别、部署通道三方面给出方案。
2. 同样一份 echo GGUF Q4_K_M，在 M2 Pro Mac 上 25 tok/s，在 i7 桌面 CPU 上只有 6 tok/s。你能想到的差异来源至少 3 条。
3. Ollama 走 REST API（端口 11434）后，前端做一个简易 chat web 页面需要哪几步？（不用真写代码，列出步骤）

## 参考资料

- **llama.cpp**：<https://github.com/ggerganov/llama.cpp>（GGUF 与量化生态源头）
- **Ollama**：<https://ollama.com/>（CLI + 模型 registry）
- **llama-cpp-python**：<https://github.com/abetlen/llama-cpp-python>
- **GGUF 格式说明**：<https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
- **bitsandbytes**：<https://github.com/TimDettmers/bitsandbytes>（QLoRA / 训练时量化）
- **GPTQ 论文**：Frantar et al., "GPTQ: Accurate Post-Training Quantization for GPT" (2022)
- **AWQ 论文**：Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression" (2023)
- 项目内：[`02-deps-compatibility.md`](../../DesignDoc/02-deps-compatibility.md) §1.2 / §2 部署选型记录
