# ch11 · 对齐（Alignment）

> SFT 之后的模型已经能 "按格式回答"，但回答的**好坏**还没人教过它。
> "好"是个主观概念 —— 同一个问题两个回答，A 比 B 好在哪？这种偏好信号 SFT 学不到。
> 
> **对齐（alignment）** 就是把 "人类评委的偏好" 塞进模型的过程。
>
> 本章以理论 + 公式推导为主，echo 项目 M5 对齐阶段会回来把 DPO 落地。

## 学习目标

1. 能讲清 RLHF 三段式各自的作用与产物
2. 能解释 PPO 在 LLM 对齐中为什么难训（奖励 hacking、高方差、显存开销）
3. 能写出 DPO loss 公式，并解释 β、参考模型 π_ref 的角色
4. 知道在不同数据形态（成对偏好 / 单点打分 / 二元接受）下选哪种对齐算法

## 前置依赖

- ch10（SFT、loss mask、LoRA）；ch09 的 CLM loss
- 对 "概率模型对一段序列的对数似然" `log P(y|x) = Σ log P(y_t|x, y_<t)` 要熟

---

## 1. 为什么需要对齐

### 1.1 SFT 的天花板

SFT 数据形如 `(prompt, response)`，是一种 "模仿学习" —— 告诉模型**这样答**，但**不告诉模型为什么这样比那样好**。

实际场景里同一个 prompt 可以有许多合理回答：

```
Q: "怎么学 Python？"
A1: 推荐《Python Crash Course》。               ← 简洁
A2: 学 Python 没有标准答案，要看你目标...        ← 详细
A3: 先 print('hello world')                    ← 偷懒
```

哪个最好？看场景、看用户、看心情。SFT 数据里每条 prompt 通常只配一个 response，模型学到的只是 "某种合理回答的样本"，不是 "哪种回答更被偏好"。

### 1.2 偏好信号比 "金标准" 便宜

让人写一条理想的 response 很贵（要写、要校）；让人在两条候选里**选一个更好**很便宜（点击一下）。所以对齐数据通常是 **成对偏好**：

```
(prompt, chosen, rejected)
```

人不需要写答案，只需要在两个候选间选边。OpenAI / Anthropic 都靠这种数据规模化做对齐。

### 自检

1. 已经做了 SFT 的模型为什么还可能 "礼貌但不正确" 或 "胡乱讨好"？对齐能解什么、不能解什么？

<details markdown="1">
<summary>答案速查</summary>

1. SFT 学的是 "格式 + 风格"，质量上限取决于训练数据中的 response 质量。对齐通过偏好信号让模型在多个候选里**选更好的方向**，能改善 "哪种回答更好" 的判断。但若偏好数据本身有偏（如标注员偏爱长回答、偏爱讨好语气），模型便会学到这些偏好。**对齐不能创造新知识，只能重新加权已有能力**

</details>

---

## 2. RLHF 三段式

RLHF —— Reinforcement Learning from Human Feedback，中译「基于人类反馈的强化学习」。Christiano 2017 / OpenAI InstructGPT 2022 把 RLHF 推到主流。结构清晰但工程复杂。

RLHF 把流程拆成三段：

- SFT 打底获得基本对话能力（π_SFT）
- 用人类标注的偏好对比数据基于 π_SFT 训出一个 Reward Model（RM）来量化 "好坏"
- 最后用 RL 算法 PPO（Proximal Policy Optimization，近端策略优化；一种通过限制每步更新幅度来保证训练稳定的强化学习算法）以 RM 打分为奖励信号，驱动 SFT 模型持续改进输出质量。

```
阶段 1: SFT
  数据: (prompt, response)
  目标：见 ch10
  产出: π_SFT —— 一个"会按格式说话" 的模型

阶段 2: RM（reward model）
  数据: (prompt, chosen, rejected)
  目标: 好回答得分 > 差回答得分，即 RM(prompt, chosen) > RM(prompt, rejected)
  产物：RM —— 能对任意 (prompt, response) 输出标量分数的评分模型

阶段 3: PPO
  从 π_SFT 的拷贝（π_ref）出发，以 RM 打分为奖励信号，用 PPO 持续优化
  目标: maximize E[RM(p, π_RL(p))] - β · KL(π_RL || π_SFT)
                    ↑ 让 π_RL 的回答拿更高分      ↑ 不让 π_RL 偏离 π_SFT 太远
  产出：π_RL —— 经过 RL 优化后的对话模型
```

其中，对于阶段 2 的主要技术细节有：

- RM 为基于 π_SFT 改造：把 lm_head（输出词表预测分布，见 ch06）换成 scalar head（只输出一个标量分数）
- loss: -log σ(RM(p, c) - RM(p, r))

  σ = sigmoid；两个分数做差过 sigmoid 再取负对数 → pairwise ranking loss
  
  理论来源：Bradley-Terry 模型（经典的成对比较概率模型）

其中，阶段 3 的拆解如下：

- `E[RM(p, π_RL(p))]`: π_RL 生成的回答平均能拿多高的 RM 分
  - `π_RL(p)` = 当前模型对 prompt 生成的回答
  - `E[RM(...)]` = 在很多 prompt 上的回答的 RM 得分期望（平均）
- `β · KL(π_RL || π_SFT)`：π_RL 偏离 π_SFT 越多，扣分（惩罚）越多
  - KL 散度 = 衡量 π_RL 和 π_SFT 两个分布的差异程度
  - β = 控制惩罚力度的系数

### 2.1 PPO 的 KL 惩罚

KL（Kullback-Leibler divergence，KL 散度）是信息论中衡量两个概率分布差异的经典度量。

$$KL(P \| Q) = \sum_x P(x) \cdot \log \frac{P(x)}{Q(x)}$$

在 RLHF 语境下：P = π_RL 的 token 分布，Q = π_SFT 的 token 分布。对每个位置计算这个值，衡量 π_RL 输出偏离 π_SFT 多少。

PPO 优化 reward 时模型会**钻 RM 的漏洞**（reward hacking）—— RM 从有限数据训出，评分能力有盲区，存在 "RM 打高分但人类觉得差" 的输出模式（如堆砌套话、重复讨好句式）。模型一旦发现这类捷径，没有拉回策略的话会迅速收敛到这种退化输出。

KL 项把策略约束在 π_SFT 附近，飘越远施加惩罚越大，抑制大幅偏离，也间接 **"缓解"** 了 reward hacking。β 控制约束强度：β 大 → 偏保守；β 小 → 偏激进。

### 2.2 PPO 的问题

PPO 的问题有哪些呢？

| 问题 | 来源 |
|---|---|
| 显存爆炸 | 训练时同时活着 4 个模型：π_RL、π_ref、RM、value head |
| 超参敏感 | β / clip_eps / lr / KL 目标值，调一组花几天 |
| 训练不稳 | RL 信号高方差，loss 曲线像心电图，崩盘频繁 |
| 实现复杂 | rollout、advantage 估计、ratio clip、value loss 全要写对 |
| Reward hacking | 模型钻 RM 漏洞，输出讨好但实质差 |

> **PPO 内部术语简介：**：
>
> - **π_ref**：π_SFT 的冻结副本，专门用来算 KL 散度，训练中不更新
> - **value head**：挂在 base model 上的 scalar 输出头（与 lm_head 并列共享 backbone），用来估计每个状态的期望回报，算 advantage 时当 baseline
> - **clip_eps**：PPO 裁剪阈值，限制单步策略更新幅度，防止一步跨太大崩盘
> - **rollout**：让 π_RL 对一批 prompt 生成回答，收集（prompt, response, reward）作为训练数据
> - **advantage 估计**：reward − value（实际回报 − baseline），衡量 "这个回答比平均水平好多少"
> - **ratio clip**：新旧策略输出同一 token 的概率之比，裁剪到 [1-ε, 1+ε] 范围内，是 PPO 保证稳定的核心机制

"学术界的 PPO 论文很美，工业界的 PPO 训练日志很丑。"

### 自检

1. RM 训练用的是分类还是回归？loss 公式直觉是什么？
2. 没有 KL 惩罚 PPO 会怎样？把 β 调到 0 试试？

<details markdown="1">
<summary>答案速查</summary>

1. **打分回归 + 排序约束**。形式上 RM 输出标量，但 loss 是 `-log σ(s_chosen - s_rejected)`，本质是让 chosen 比 rejected 分数高的二元交叉熵（Bradley-Terry / pairwise logistic）。RM 学的是"哪个更好"的相对排序，不是绝对分数

2. 模型会快速学会 reward hacking 模式（重复某些讨好短语 / 特定格式），输出虽然 RM 打分很高但完全失去语言能力。β=0 等价于"无约束最大化 RM"，必然崩

</details>

---

## 3. DPO

Rafailov et al. 2023。把 RLHF 三段简化成一段：**RM 不要了，PPO 不要了，直接对偏好数据做有监督训练** —— DPO（Direct Preference Optimization）

### 3.1 关键洞察

回忆一下上面 RLHF 优化的目标是：

```
# r = RM, x = p, y = π_RL(p), π = π_RL
max E[r(x, y)]  - β · KL(π || π_ref)
```

目标里 `r` 想把概率推向高 reward 的 y，KL 想把概率拉回 π_ref，二者平衡解就是"以 π_ref 为先验，按 reward 指数加权重新归一化"。

数学上这个优化问题有**闭式最优解**：

```
# 式 1
π*(y|x) ∝ π_ref(y|x) · exp(r(x, y) / β)
```

> **（数科的）直觉**：这是带 KL 约束最大化问题的标准结论（与最大熵 / 玻尔兹曼分布同源），完整推导见 DPO 论文附录 A.1。

反过来，如果已知 π*（理论最优模型）和 π_ref 就能反推 reward：

```
# 式 2
r(x, y) = β · log(π*(y|x) / π_ref(y|x)) + const(x)
```

把这个 r 代回 RM 的 Bradley-Terry loss `-log σ(r_chosen - r_rejected)`，`const(x)` 抵消了：

```
# 式 3
# π = 当前训练中的模型，目标是逼近 π*
L_DPO = -E[ log σ( β · (log π(y_c|x)/π_ref(y_c|x) - log π(y_r|x)/π_ref(y_r|x)) ) ]
                       └─ chosen 的 logp ratio ──┘   └─ rejected 的 logp ratio ─┘
```

- β 来自原 RLHF 目标里的 KL 系数 —— 它**同时是隐式 KL 约束强度**：β 大 → 约束强，π 留在 π_ref 附近；β 小 → 约束弱，π 可以漂很远
- 训练时 minimize L_DPO，让 π 往设想的 π* 逼近
- **不再需要显式 RM 了**

### 3.2 公式直觉

DPO loss 想让模型干的事：

- **拉高** `log π(chosen|x) - log π_ref(chosen|x)` —— 比参考模型更倾向输出 chosen
- **压低** `log π(rejected|x) - log π_ref(rejected|x)` —— 比参考模型更不倾向输出 rejected

整个 loss 形如二元交叉熵，**完全监督学习**，没有采样、没有 RL 信号、没有 advantage 估计。

DPO 把 RL 问题降级成了普通的监督学习问题，工程上和训 SFT 一样简单，一个常规的 forward + backward 即可。

### 3.3 必须冻结的 π_ref

π_ref 是 "参考点" —— 通常就是 SFT 后的模型。它的作用：

- **防漂移**：DPO 隐式包含 KL 约束，π 跑离 π_ref 太远会被惩罚
- **基准**：`log π(y) - log π_ref(y)` 衡量的是 "相对参考模型，π 在 y 上变了多少"

实现上 π_ref 冻结，前向产出 logp 供 loss 计算但不参与反向传播（梯度只流过 π）。代价：训练时显存里要同时放两个模型副本。

LoRA（见 ch09）场景下可省掉 —— base 权重 W 本身就是 SFT 后的状态且冻结不动，关闭 LoRA 旁路走纯 W 即为 π_ref，打开旁路走 W+BA 即为 π，同一份 base 按需切换身份，零额外显存。

### 3.4 训练示例

训练用的样本大致格式：

```
样本: { "prompt": "...", "chosen": "...", "rejected": "..." }
```

计算流程中，需要先把 chosen / rejected 各拼成完整对话送进模型，分别算 m=π/π_ref 的 `log m(chosen|prompt)` 与 `log m(rejected|prompt)`：

```python
# 伪代码
for batch in loader:
    # π 与 π_ref 都吃同样的输入
    logp_pi_c    = sequence_logp(model,    batch.prompt + batch.chosen)
    logp_pi_r    = sequence_logp(model,    batch.prompt + batch.rejected)
    logp_ref_c   = sequence_logp(ref_model, batch.prompt + batch.chosen)
    logp_ref_r   = sequence_logp(ref_model, batch.prompt + batch.rejected)

    pi_ratio_c  = logp_pi_c  - logp_ref_c    # chosen 的 logp 比例
    pi_ratio_r  = logp_pi_r  - logp_ref_r    # rejected 的 logp 比例
    loss = -F.logsigmoid(beta * (pi_ratio_c - pi_ratio_r)).mean()
```

> `sequence_logp(model, ids)` 做的事：forward 拿 logits → `log_softmax` 沿 vocab 维 → 用 `gather` 取出真实 token 位置的 logp → 沿 response 部分 `sum`。结果是单个标量（每条样本一个），代表"模型生成这条 response 的对数概率之和"。完整实现见练习 `01_dpo_loss.py`。prompt 与 chat template 部分不计入（与 SFT 的 loss mask 同源）。

### 3.5 DPO 优势与代价

| 维度 | RLHF (PPO) | DPO |
|---|---|---|
| 阶段数 | 3（SFT + RM + PPO） | 2（SFT + DPO） |
| 训练稳定性 | 心电图 | 平稳，类似常规 SFT |
| 显存 | 4 个模型 | 2 个模型（π + π_ref），LoRA 下 1 个 |
| 超参敏感度 | 高 | 低（主要就 β，0.1–0.5） |
| 数据复用率 | RM 训完丢，PPO 用 RM 在线打分 | 偏好对直接训，离线即可 |
| 理论上限 | RM 是真 reward 的近似，PPO 能逼近 RM 最优 | 直接拟合最优解，但只用了静态数据，无法在线探索 |

实务上 DPO **几乎完全替代了 PPO** 在开源界的位置。Llama-3、Qwen、Yi 系列正式版的对齐都是 DPO 或其变体。

### 自检

1. DPO 公式里 β 调大调小分别会怎样？
2. 为什么 DPO 要冻结 π_ref 而不是动态更新？

<details markdown="1">
<summary>答案速查</summary>

1. β 是 DPO 隐式 KL 约束的强度系数。
   - β 大 → sigmoid 饱和更快（margin 略大就趋零梯度）→ 模型不需要把 margin 推得很大就能让 loss 趋零 → 收敛时 π 更靠近 π_ref（约束强、保守）
   - β 小 → sigmoid 平缓，需要 margin 很大才能压低 loss → π 漂得远（约束弱、激进）

   典型值 0.1–0.5。注意 01 验证 3 测的是"同一 margin 下"的单步梯度，那是公式里的线性放大；这里讨论的是"训练收敛终点"的偏移，由 sigmoid 饱和速度主导，二者不矛盾

2. π_ref 是隐式 KL 约束的"原点"。如果 π_ref 跟着 π 一起更新，约束失效（π 永远等于 π_ref，`logp - logp_ref` 始终 0），整个 loss 退化为常数

</details>

---

## 4. KTO / ORPO 简提

DPO 解决了 PPO 的工程难题，但仍依赖**成对偏好数据**（来自用户真实反馈或标注员）。许多场景这种数据不易得：

- 用户日志通常是 "点赞 / 不点赞"（二元单点信号）
- 真实标注员有时只能说 "这个还行" 或 "这个不行"（不在/难以在两个间选）

后续工作针对不同数据形态扩展：

| 算法 | 数据形态 | 一句话 |
|---|---|---|
| DPO | (prompt, chosen, rejected) | 成对偏好 |
| **KTO** | (prompt, response, ±1) | 单点二元信号，每条样本独立打 "好/坏" |
| **ORPO** | (prompt, chosen, rejected) | DPO + SFT 合并成一阶段，无需 π_ref |
| **IPO**（Identity Preference Optimization，恒等偏好优化） | (prompt, chosen, rejected) | DPO 的 loss 改良，缓解过拟合偏好对 |
| **SimPO**（Simple Preference Optimization，简化偏好优化） | (prompt, chosen, rejected) | 也无需 π_ref，把 logp **平均**（除以 response 长度）当作隐式 reward —— 平均化让模型自己充当"长度归一"的参考点，省掉 π_ref |

以 7B 量级模型的对齐，工程选型决策树如下：

```
有 (chosen, rejected) 偏好对？
├─ 是 ─── 显存够（>20GB）？
│         ├─ 是 ─── DPO（首选，社区最稳）
│         └─ 否 ─── ORPO 或 SimPO（无 π_ref，省一份显存）
└─ 否 ─── 数据是"赞/踩"二元？
          ├─ 是 ─── KTO
          └─ 否 ─── 回去标偏好对，或回到 SFT 多迭代数据
```

工业界几乎不再考虑 PPO（除非你是 OpenAI/Anthropic 这种有专门 RL 团队的）。学术界 PPO 仍有研究价值（探索 + 在线学习）。

> echo-mini / echo 的对齐路线：先用 DPO 跑通主流程，需要时再切 ORPO / SimPO 省显存。

### 自检

1. ORPO 把 SFT 与对齐合并到一个阶段，听上去更省事，为什么 DPO 仍是事实标准？
2. KTO 的"二元点赞"数据有什么坑？

<details markdown="1">
<summary>答案速查</summary>

1. ORPO 把 SFT loss 与偏好 loss 加权混在一起，超参（两个 loss 的权重）需要调，且**没有独立 SFT 阶段**意味着模型还没学会基本格式就要学偏好，对底座质量要求更高。DPO 流程更分离、更可控、社区实践更多

2. 单点信号方差大、噪声大，且"全是点赞"或"全是踩"的数据集有严重 class imbalance；好回答与差回答之间没有显式比较关系，模型只能学"绝对好/绝对差"的边界，远不如成对信号信息量大

</details>

---

## 5. 练习

落到 `Playground/ch11-alignment/`：

| 脚本 | 内容 |
|---|---|
| `01_dpo_loss.py` | 手写 DPO loss 公式，用玩具 logits 验证：chosen logp 上升、rejected logp 下降；β 影响梯度幅值 |
| `02_beta_effect.py` | 固定一对 (chosen, rejected) 数据，扫不同 β 训若干步，看 loss 与 logp ratio 的演化曲线 |

跑法：

```bash
uv run python Playground/ch11-alignment/01_dpo_loss.py
uv run python Playground/ch11-alignment/02_beta_effect.py
```

不依赖 trl / transformers，纯 PyTorch 玩具数据，跑几秒钟，看清公式行为即可。

## 思考题

1. DPO 不显式训 RM，是不是说 reward 模型在 DPO 里就消失了？从 §3.1 的推导看，π 与 r 的关系是什么？
2. 假设你的 SFT 数据本身有偏（如全是过度礼貌的回答），DPO 能纠正吗？需要什么样的偏好数据？
3. 为什么 LoRA 与 DPO 是天然搭档？（提示：π_ref 的处理）

## 参考资料

- **Christiano et al., "Deep Reinforcement Learning from Human Preferences"** (2017)：RLHF 起源
- **Ouyang et al., "Training language models to follow instructions with human feedback"** (InstructGPT, 2022)：RLHF 三段式范本
- **Rafailov et al., "Direct Preference Optimization"** (DPO, 2023)：DPO 原论文，§4 推导值得细读
- **Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization"** (2024)
- **Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model"** (2024)
- **HuggingFace TRL**：<https://huggingface.co/docs/trl> — DPO/KTO/ORPO 全有官方实现
