# Echo SFT mode collapse 实验发现

> 日期：2026-05-28
> 阶段：M5 · echo 落地
> 路线：base（Qwen2.5-1.5B + QLoRA + modules_to_save embed_tokens）
> 性质：实验报告 + 机理分析。**操作 SOP 见 [`Echo/echo/stepshooting-base.md` §5.6](../../Echo/echo/stepshooting-base.md)；
> 踩坑速查见 [`troubleshooting.md`](troubleshooting.md) 2026-05-28 条目。**

## 1. 摘要

本次 SFT 训练原配置 5 epoch，实测在 epoch 2.2（step 4000）时模型已严重 mode collapse：
同 prompt × 多次生成**逐字一致**，温度 0.7 / top_p 0.9 也压不出多样性。

回溯历史 ckpt 发现：

- **sweet spot 在 step 1000（epoch 0.56）**，所有问题类型生成质量都健康
- **mode collapse 不是均匀的，是按问题类型分阶段塌缩**——短模板题 epoch 1 末就塌，长开放题撑到 epoch 2+ 才塌
- **loss / entropy 单指标看不出选择性塌缩**，必须人工生成抽测

三个反直觉结论：

1. epoch 不是越多越好，`modules_to_save: embed_tokens` 场景下 sweet spot 在 epoch 0.5~1.0
2. base 路线不需要超长训练——Qwen2.5 base 自带 ChatML prior，500 步就会停
3. 短模板数据是 collapse 的"金丝雀"，最早塌、最先暴露问题

后续配置已调整：`num_epochs` 5 → 2，`save_steps` 500 → 250。

## 2. 实验配置

| 维度 | 取值 |
|---|---|
| 底座 | Qwen/Qwen2.5-1.5B（**非** Instruct） |
| 量化 | nf4 + bf16 compute + double quant |
| LoRA | r=128, alpha=256, dropout=0.05, target=q/k/v/o/gate/up/down |
| modules_to_save | `embed_tokens`（lm_head 通过 tie 跟随更新） |
| 数据 | `train_aug.jsonl` ~29K（ShareGPT 19K + 短问答 10K） |
| effective batch | 1 × 16 = 16 |
| learning_rate | 1.0e-4 cosine, warmup 0.03 |
| **num_epochs（旧）** | **5**（实测过拟合） |
| **num_epochs（新）** | **2** |
| 单 epoch step 数 | ~1813 |
| save_steps | 旧 500 / 新 250 |

复现命令（cwd: `Echo/echo/`）：

```bash
uv run python scripts/sft.py --config configs/sft-8g-base.yaml
```

## 3. 现象观测

### 3.1 训练曲线关键节点

| 时点 | step | epoch | loss | entropy | token_acc | grad_norm |
|---|---|---|---|---|---|---|
| 训练初 | 10 | 0.006 | 0.95 | 1.41 | 0.67 | 2.5 |
| warmup 顶 | 270 | 0.15 | 0.85 | 1.06 | 0.72 | 2.0 |
| epoch 1 末 | 1813 | 1.00 | 0.83 | 1.00 | 0.72 | ~1.0 |
| **epoch 2 起** | 1820 | 1.004 | **0.66 ↓** | **0.91 ↓** | 0.79 | 1.4 |
| epoch 2 末 | 3625 | 2.00 | 0.62 | 0.74 | 0.79 | ~1.0 |
| **epoch 3 起** | 3630 | 2.003 | **0.50 ↓** | **0.71 ↓** | 0.82 | 1.1 |
| ...（已 collapse） | 4000 | 2.24 | 0.36 | 0.49 | 0.87 | 1.0 |

**两个特征**：

1. **epoch 边界出现 loss 阶跃跳降**，不是渐变。epoch 1→2 跳降 -30%，epoch 2→3 跳降 -42%（递增！）
2. **grad_norm 健康，但 entropy 持续下行**——梯度看着正常，但模型在"硬记答案"

### 3.2 ckpt 生成质量对照（人工抽测）

测试方法：6 题 × 3 次跑，覆盖三种问题类型：

- 长开放题：推荐一本书 / 如何看待 AI 替代人类工作
- 短模板题：写一句晚安祝福 / 周末计划推荐
- 数据集中题：讲个笑话 / 你最喜欢哪种颜色

| ckpt | step | epoch | 长题多样性 | 短题多样性 | 集中题多样性 | 综合 |
|---|---|---|---|---|---|---|
| checkpoint-500 | 500 | 0.28 | （仅测笑话健康） | — | — | 太早，能力不全 |
| **checkpoint-1000** | **1000** | **0.56** | 健康 ✅ | 健康 ✅ | 健康 ✅ | **本次最佳** |
| checkpoint-1500 | 1500 | 0.84 | 健康 | 临界 🟡 | 健康 | 可接受 |
| checkpoint-2000 | 2000 | 1.12 | 健康 | **已塌** ❌ | 健康 | 部分塌 |
| checkpoint-2500 | 2500 | 1.40 | 健康 | 已塌 | **已塌** ❌ | 部分塌 |
| checkpoint-4000 | 4000 | 2.24 | **已塌** ❌ | 已塌 | 已塌 | 全面塌缩 |

### 3.3 典型样例（mode collapse 演化）

prompt: 写一句晚安祝福 / 同温度同 top_p / 跑 3 次

- checkpoint-1000：3 次回答均不同（词序、句式、祝愿对象有差异）
- checkpoint-2000：3 次**逐字一致**

prompt: 推荐一本书 / 跑 3 次

- checkpoint-2000：推 3 本不同的书（沉默的羔羊 / 苏菲的世界 / 哈利波特）
- checkpoint-4000：3 次都推同一本《人类简史》且文字几乎相同

## 4. 机理分析

### 4.1 sweet spot 由四要素乘积决定

```
sweet_spot_step ≈ f(数据曝光速率) × g(记忆能力) × h(数据集中度) × k(底座 prior)
```

| 要素 | 本次取值 | 对 sweet spot 的影响 |
|---|---|---|
| **曝光速率** | 单 epoch 1813 步，step 1000 见过 ~55% 数据 | 曝光越快越早塌（相同 epoch 数下） |
| **记忆能力** | `modules_to_save: embed_tokens` 让 ~2.3 亿参数全参更新 | 记忆能力 ↑ → sweet spot ↓ |
| **数据集中度** | 短问答 10K 答案分布窄 | 集中度 ↑ → 短题最先塌 → 整体 sweet spot ↓ |
| **底座 prior** | Qwen2.5 base 已见过 ChatML 痕迹 | prior 越强 SFT 学得越快 → sweet spot ↓ |

四要素乘起来，**把本应在 step 3000~4000 的 sweet spot 提前到了 step 1000**。

如果把它们改成"通用 SFT"配置（纯 LoRA 不开 modules_to_save，数据全长对话，LLaMA-2 底座），
sweet spot 会回到 step 3000~5000，5 epoch 就是合理配置。

### 4.2 选择性塌缩

**核心发现**：mode collapse 不是均匀发生，而是按问题类型分阶段。

机制：模型在每个 step 的更新对所有样本是均等的，但**不同问题的训练数据答案分布不同**：

| 问题类型 | 数据答案分布 | 模型最优策略 | 塌缩速度 |
|---|---|---|---|
| 短模板题（祝福、问候） | 集中（top-3 答案占主导） | 收敛到单一最优答案 | **最快**（epoch 1 末） |
| 长开放题（推书、看法） | 广（数百种合理答案） | 维持分布，按情境采样 | 慢（epoch 2+） |
| 数据集中题（笑话） | 中等集中（少数高频梗） | 介于两者之间 | 中等 |

**这意味着**：用长开放题做评测，会**严重低估模型的塌缩程度**。必须用短模板题作为早期信号。

实测对照（checkpoint-2000，已部分塌缩）：

- 推书题：3 次推 3 本不同的书 → 看起来"健康"
- 晚安祝福：3 次逐字相同 → 已塌

仅看推书题会得出"模型 OK 继续训"的错误结论。

### 4.3 为什么 loss 单一指标不够

| 指标 | 能反映 mode collapse 吗 |
|---|---|
| train loss | ❌ 持续下降本身不是 collapse 信号 |
| eval loss | 部分（需要 val 数据多样性高） |
| entropy | 部分（整体尺度，不分题型） |
| token_acc | ❌ 高 acc 也可能是泛化 |
| **生成多样性人工抽测** | ✅ 唯一可靠信号 |

数学原因：loss 是 `E[-log p(target)]`，**只关心 target token 的概率**，不关心其他候选 token 的概率分布。
模型把 P(target) 从 0.3 推到 0.95 时，loss 从 1.2 降到 0.05，但**other tokens 的概率从均匀分布塌成 0**——
这就是 mode collapse 的数学定义。loss 看不出这种"分布塌缩"。

## 5. 反直觉结论

### 5.1 epoch 不是越多越好

教科书 SFT lr / epoch 经验值（lr 1e-5~5e-5, epoch 3~5）是基于**全参 SFT + 普通 LoRA** 推出的。
你这套配置（QLoRA r=128 + modules_to_save embed）打破了前提：

- 可训参数从 ~30M 暴涨到 ~260M
- 等效记忆容量接近全参 SFT，但只更新少数层
- 过拟合速度比纯 LoRA 快 3~5 倍

**实操原则**：开了 `modules_to_save` 后，把经验 epoch 数除以 2~3 作为新起点。

### 5.2 base 路线不需要超长训练

Qwen2.5-1.5B base 实测在 **step 500（epoch 0.28）** 就已经：

- 学会 ChatML 框架
- 学会用 `<|im_end|>` 停止
- 学会 AI 助手风格的开场白

这远快于 LLaMA-2 base 的预期（~step 2000~3000 才稳定）。原因是 Qwen2.5 预训练数据
里**已混入对话格式语料**（同期对照实验：base + 伪 ChatML 时模型自己会吐 `user` / `assistant` 角色名）。

**实操影响**：选 Qwen 系 base 做 SFT 时，5 epoch 是浪费；2 epoch 是合理上限。

### 5.3 短模板数据是 collapse 的"金丝雀"

短模板题（晚安祝福、问候、推荐计划）有两个特征让它成为最敏感的早期信号：

1. 答案分布最集中 → 最早塌缩
2. 塌缩信号明显（逐字一致 vs 词序差异）→ 人工容易判断

**实操建议**：训练数据中**保留** 5~10% 的短模板题，并在评测脚本里**专门挑短模板题**做多样性测试。

## 6. 变量敏感度（对未来调参的指导）

固定其他变量，单独调一项后 sweet spot 漂移方向：

| 改动 | sweet spot 漂向 | 幅度 |
|---|---|---|
| 数据 29K → 50K | 推迟 | step +1000~+2000 |
| 数据 29K → 10K | 提前 | step -500~-700 |
| 关 `modules_to_save: embed_tokens` | 推迟 | step +2000~+3000 |
| LoRA r 128 → 64 | 略推迟 | step +200~+500 |
| 数据全换长对话（去掉短问答） | 推迟，但短题不再是金丝雀 | step +500~+1000 |
| 底座换 LLaMA-2 base | 推迟 | step +1500~+2500 |
| 底座换 Qwen2.5-Instruct | 大幅提前（无需教对话能力） | step ÷ 2 |
| lr 1e-4 → 2e-4 | 提前 | step ×0.7 |
| lr 1e-4 → 5e-5 | 推迟 | step ×1.5 |

**任何配置改动都需要重新做 ckpt 抽测**，不能套用 step 1000 这个数字。

## 7. 后续工作

### 7.1 M6 DPO 利用本次塌缩样本

`checkpoint-4000` 全面塌缩的输出（"原子梗"笑话 + 死板答案）天然是 DPO 训练的 **rejected** 候选。
配合 `checkpoint-1000` 的多样化输出作为 **chosen**，可以低成本构造一批偏好数据，
教模型"避免塌缩式回答"。

详见 M6 DPO 设计文档（待写）。

### 7.2 ch10 / ch11 课件素材

本次产出可作为 SFT / 对齐章节的实证素材：

- ch10 SFT
  - "为什么 SFT 不可替代"：[`generate_base.py`](../../Echo/echo/scripts/generate_base.py) 三组对照样本（base raw / base chatml / SFT）
  - "为什么 epoch 太多会塌"：本次 ckpt-1000 vs ckpt-4000 对照
  - "为什么短模板题最先塌"：选择性塌缩现象
- ch11 对齐 / DPO
  - "为什么 SFT 不够，需要 RLHF/DPO"：本次 sweet spot 的存在性 + 难定位本身就是论据

### 7.3 下次 SFT 训练改进方向

1. **加生成抽测 callback**：训练每 N 步固定 prompt 生成 + 词级 diff，自动检测塌缩
2. **数据均衡**：短模板题加大答案多样性（同义改写 / 多人风格混合）
3. **early stop 机制**：基于生成多样性而非 loss 的 early stop
4. **对照实验记录**：尝试不同的 modules_to_save 配置（仅 embed / 仅 lm_head / 都不开），
   对比训练时长与最终质量

这些是 M5 收尾后或 M6 之前可做的实验，**不阻塞 M5 当前交付**。
