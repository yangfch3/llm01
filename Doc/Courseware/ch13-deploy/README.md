# ch13 · 部署（Deployment）

> 训完一个模型，HF 权重躺在硬盘上，怎么让它**真正能用**？
> "能用"包括：能在普通显卡跑、能在 Mac 跑、能 1 秒出第一个 token、能塞进 Ollama 让别人 `ollama run echo` 一句话起。
>
> 本章讲清三件事：
> 1. 量化（int8/int4）省什么、损什么
> 2. GGUF / llama.cpp 生态为什么是当下"跨平台本地推理"的事实标准
> 3. 三条主流部署通道（`transformers` 原生 / `llama-cpp-python` / Ollama）的取舍

## 学习目标

1. 理解 fp16 → int8 → int4 显存与速度的量级变化，以及精度衰减的来源
2. 能解释 GGUF 是什么、为什么 llama.cpp 系把它当统一格式
3. 知道在不同场景（开发调试 / demo 演示 / 程序化调用）该选哪条部署链路
4. 能复述把"训完的 HF 权重 → Ollama 可跑"完整流程

## 前置依赖

- ch09（混合精度）、ch10（LoRA / adapter 概念）
- 部署属"工程链路"，没有难数学，但坑特别多

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

> 推理时还要加 KV cache（与 batch、序列长度成正比）和激活值，实际占用比表格略高。但权重通常是大头。

3060 12GB 跑 7B fp16 模型显存就紧巴巴了；int4 后只占 3.5 GB，连 13B 都能塞下。**量化是把"勉强能跑"变成"舒服能跑"的关键**。

### 1.2 推理速度

量化不只是省显存，**通常也变快**：

- 显存带宽是大模型推理的瓶颈（每生成一个 token 都要把整组权重读一遍）
- int4 权重比 fp16 小 4 倍 → 内存读取量 4 倍 → 带宽瓶颈显著缓解
- CPU 推理时 int4 的整数运算指令也比浮点快

实测上 3060 跑 7B：fp16 约 30 tok/s，int4 GGUF 通过 llama.cpp 约 60 tok/s。

### 1.3 精度损失从哪来

权重原本是 fp16 的连续浮点数（如 `0.0123, -0.4567, ...`）。
量化是把它们**离散化**到一组有限的整数级别上：

```
int8: 256 个离散级别（-128 到 127）
int4:  16 个离散级别（-8 到 7）
```

每个权重要找最近的离散级别 → **舍入误差**。
权重越小、分布越集中，舍入相对误差越大。
推理时这些小误差会**沿着多层网络累积**，最终输出概率分布会与 fp16 略有偏移。

### 自检

1. fp16 → int4 显存省 4 倍，速度也快 ~2 倍——"那为什么不直接 int2、int1？"
2. 量化损精度，但为什么实测下游任务分数衰减通常 < 5%？

<details markdown="1">
<summary>答案速查</summary>

1. 离散级别太少 → 舍入误差太大 → 多层累积后输出严重失真。int2 已基本不可用；int1（二值化）只在小模型 / 特殊架构上可行。社区当前下限是 int3（如 GPTQ-3bit），低于此通常崩

2. ① 现代量化算法（GPTQ / AWQ / GGUF 的 K-quant 系列）做了"重要权重保留高精度 + 不重要权重激进量化" ② LLM 的输出概率分布对小扰动有冗余（top-k 概率最高的 token 不会因为权重小变动就被挤出） ③ 多选题打分对小数概率差不敏感，只要排序不变就答案不变

</details>

---

## 2. 量化的几种主流路线

### 2.1 训练时量化 vs 训练后量化

| 类别 | 代表 | 何时做 | 难度 |
|---|---|---|---|
| **PTQ**（Post-Training Quantization） | GGUF / GPTQ / AWQ | 训完后离线量化 | 低，社区主流 |
| **QAT**（Quantization-Aware Training） | bitsandbytes 的 8bit/4bit 训练 | 训练过程感知量化误差 | 高，少用 |
| **QLoRA** | bitsandbytes 4bit + LoRA | 把底座 4bit 冻结 + 训 LoRA | 中，省显存训练 |

> echo 项目用 PTQ：训练阶段保持 fp16/bf16，训完后导出 GGUF int4 部署。QLoRA 用在 M5 微调底座阶段（省显存训练，不是部署）。

### 2.2 GGUF + llama.cpp：跨平台事实标准

**llama.cpp**（ggerganov 开源）是用 C++ 写的纯 CPU/GPU 通用推理引擎。
**GGUF** 是它的模型存储格式，特点：

- **单文件**：权重 + 分词器 + 元数据全打包成一个 `.gguf` 文件
- **跨平台**：Windows / Mac / Linux / Android 都能跑
- **多种量化级别**：
  - `Q8_0`：8-bit，几乎无衰减，文件接近 fp16 一半
  - `Q5_K_M`：5-bit，常用平衡点
  - `Q4_K_M`：4-bit + K-quant，**社区默认推荐**，质量与体积平衡最好
  - `Q4_0`：老式 4-bit，质量略差于 Q4_K_M
  - `Q3_K_M`：3-bit，明显衰减但仍可用
  - `Q2_K`：2-bit，质量明显退化，仅紧凑场景

> **K-quant 是什么**：传统 int4 是"每个权重独立量化"，K-quant 是"按 block 分组 + 每组单独 scale + 重要 block 用更高 bit"，质量显著优于朴素 int4。

llama.cpp 的下游生态：

```
        llama.cpp (C++ 引擎，跑 GGUF)
              │
   ┌──────────┼─────────────┐
   ▼          ▼             ▼
llama-cpp   Ollama        LM Studio
-python    (CLI/服务)      (桌面 GUI)
(Python 绑定)
```

### 2.3 GGUF 之外

| 格式 | 用在哪 | 备注 |
|---|---|---|
| **GPTQ** | `auto-gptq` / vLLM | GPU 推理向，文件小，仅 CUDA 友好 |
| **AWQ** | `autoawq` / vLLM | GPU 推理，质量略好于 GPTQ |
| **bitsandbytes 4bit** | 训练阶段（QLoRA）/ HF 推理 | 不存盘，运行时即时量化 |
| **GGUF** | llama.cpp 系 | **跨平台首选**，本项目主路径 |

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
- **缺点**：跑得慢（无 KV cache 优化时）、显存占用高（fp16）、Mac 上 MPS 路径偶有 op fallback
- **适用**：开发调试、训练后立即推理验证

### 3.2 `llama-cpp-python`

```python
from llama_cpp import Llama
llm = Llama(model_path="echo-q4_k_m.gguf", n_ctx=2048)
out = llm("你好", max_tokens=64)
print(out["choices"][0]["text"])
```

- **优点**：跑 GGUF 量化模型，跨平台，速度比 transformers 快很多
- **缺点**：
  - Win + Python 3.12 经常没现成 wheel，触发本地编译（详见 `02-deps-compatibility.md`）
  - 编译要 MSVC / CMake / 可能 CUDA toolkit
- **适用**：程序化调用、写本地服务、自定义采样策略

### 3.3 Ollama

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

- **优点**：装一个独立 CLI 就能用，自带 REST API（端口 11434），跨平台体验最好
- **缺点**：底层还是 llama.cpp，能力受 GGUF 限制；模型管理是 Ollama 自己的"私有 registry"格式
- **适用**：demo 演示、桌面应用、最简上手路径

### 3.4 选型矩阵

| 场景 | 推荐 |
|---|---|
| 训完立刻看效果 | transformers 原生 |
| 写 Python 服务 / 自定义采样 | llama-cpp-python |
| 给别人演示 / 桌面 chat | **Ollama**（M6 主路径） |
| 高并发服务 | 不在本项目范围（用 vLLM） |

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

**几个工程坑**（M6 会踩，先有数）：

- **chat template 必须显式写在 Modelfile**：转 GGUF 不会自动带 HF 的 `chat_template`，没写的话 Ollama 把对话拼错，模型答非所问
- **stop token**：必须在 Modelfile `PARAMETER stop` 里加，否则模型停不下来
- **量化级别选择**：先 Q4_K_M 跑通，质量不够再 Q5_K_M / Q8_0 升级，体积换质量
- **测试要全链路**：HF 推理通过 ≠ GGUF 推理通过 ≠ Ollama 推理通过，每一步都要验

### 自检

1. 为什么转 GGUF 后输出乱码 / 答非所问，第一件该检查的事是什么？
2. 假设 echo 在 HF transformers 下答得很好，转 Q4_K_M GGUF 后明显变差，可能原因？

<details markdown="1">
<summary>答案速查</summary>

1. 检查 **chat template 与 stop token**：HF tokenizer 自带的 `chat_template` 不会自动迁到 GGUF/Ollama，要手动写到 Modelfile。多数"乱码 / 答非所问"是模板拼错或 stop token 缺失（模型说完话又开始说下一轮的"用户"内容）

2. 候选：① 量化级别太激进（Q4 衰减）→ 升 Q5/Q8 再测 ② chat template 写错 → 对照 HF 的 `tokenizer.apply_chat_template` 输出 ③ tokenizer 转换有差异（罕见 BPE 边界 case） ④ KV cache / 采样参数（temperature / top_p）默认值与训练评测时不一致

</details>

---

## 5. 性能基准的"读法"

社区里常见的性能数字（仅供数量级直觉，会因硬件、上下文长度、batch 而变）：

| 平台 | 模型 | 量化 | tok/s（单 batch） |
|---|---|---|---|
| RTX 3060 12GB | 7B | Q4_K_M | ~60 |
| RTX 3060 12GB | 7B | fp16 | ~30 |
| M2 Pro Mac | 7B | Q4_K_M (Metal) | ~25 |
| M2 Pro Mac | 7B | Q8_0 (Metal) | ~18 |
| 纯 CPU (i7) | 7B | Q4_K_M | ~6 |

**echo final 验收**（出自 `00-startup-proposal.md`）：
- 量化后衰减 ≤ 5%
- 3060 ≥ 20 tok/s（int4）
- Mac ≥ 15 tok/s（GGUF Q4_K_M）

### 自检

1. 同一模型同一量化级别，序列长度 4k 和 32k 时 tok/s 差很多。为什么？
2. "纯 CPU 跑 7B Q4 也有 6 tok/s"——这数字看着挺实用，那为什么开源界还在卷 GPU 推理？

<details markdown="1">
<summary>答案速查</summary>

1. 自回归生成时每一步都要"看完"前面所有 token 的 KV cache。序列越长，KV cache 越大，每步要做的 attention 计算量与显存读取量越大，tok/s 自然下降。这也是为什么超长上下文模型推理慢

2. ① "6 tok/s"是单用户、短对话、小模型场景，多人 / 长对话立刻拖到 1 tok/s ② 大模型（70B+）CPU 跑不动 ③ 真实产品需要并发，CPU 没法堆并发 ④ batch 推理 GPU 优势倍数级。CPU 跑 LLM 是"够用底线"，GPU 是"产品基线"

</details>

---

## 6. 练习

落到 `Playground/ch13-deploy/`：

| 脚本 | 内容 |
|---|---|
| `01_quant_simulation.py` | 玩具版量化：手动把一组 fp32 权重量化到 int8 / int4，看舍入误差分布；模拟"权重越多层叠加误差越大"现象 |
| `02_inference_compare.py` | 用 GPT-2 small 对比 fp32 vs fp16 vs int8（动态量化，PyTorch 内置）的输出差异与单 token 推理耗时；说明"量化省的不只是空间，还有时间" |

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
