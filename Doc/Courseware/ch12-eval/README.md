# ch12 · 评测（Evaluation）

> "我们训了个模型，效果挺好" —— 业界最不可信的一句话。
> "好"是声明，不是结论。结论需要**指标 + 数据集 + 复现配方**。
>
> 本章讲清三件事：
> 1. PPL（perplexity）测什么、什么时候有用、什么时候没用
> 2. 开源 benchmark（C-Eval / MMLU 等）是怎么"考"模型的
> 3. 为什么自动指标都跑完了还得做人工评测

## 学习目标

1. 能写出 PPL 公式与 CLM loss 的关系，知道两个模型能不能直接比 PPL
2. 理解 MMLU/C-Eval 这类多选题 benchmark 的"打分"机制（loglikelihood scoring）
3. 知道开源 benchmark 的常见坑：污染、prompt 敏感、多选 vs 生成
4. 能为自己的模型设计一份 minimal 评测套件（PPL + 小 benchmark + 人工抽样）

## 前置依赖

- ch09（CLM loss、teacher forcing）
- ch10（chat template、loss mask）

---

## 1. PPL：最便宜也最容易被误用的指标

### 1.1 公式

PPL = `exp(average negative log-likelihood per token)` = `exp(CLM_loss)`

```
PPL = exp( -1/N · Σ_t log P(x_t | x_<t) )
```

直觉：模型对下一个 token "平均要在多少个候选间犹豫"。

- PPL = 1 → 完美预测，每步都 100% 命中
- PPL = V（词表大小）→ 等同均匀随机
- PPL = 50 → "平均每步像在 50 个候选里挑"

> **重要等价**：训练 loss（按 token 平均的 CE）取 `exp` 就是 PPL。它**不是**新指标，只是 loss 换了个看起来更直观的单位。

### 1.2 PPL 怎么算（工程视角）

```python
# 伪代码：在某个评测语料上算 PPL
# 假设 dataloader 的 batch.input_ids / batch.labels 已按 ch09 §1.2 准备好：
#   labels 是 input_ids 左移一位的"下一个 token"目标，pad 位填 -100
total_nll, total_tokens = 0.0, 0
for batch in eval_loader:
    with torch.no_grad():
        logits = model(batch.input_ids)        # (B, L, V)
        loss = F.cross_entropy(
            logits.reshape(-1, V),
            batch.labels.reshape(-1),
            ignore_index=-100,                 # pad / 不算 loss 的位置
            reduction="sum",                   # 别用 mean，先求 sum 再合
        )
    total_nll += loss.item()
    total_tokens += (batch.labels != -100).sum().item()
ppl = math.exp(total_nll / total_tokens)
```

> 关键是 `reduction="sum"` 后**手动除以总有效 token 数**。如果用 `mean` 然后对 batch 求平均，遇到 batch 间长度差异时会算错（每个 batch 的"分母 token 数"不同）。

### 1.3 PPL 什么时候有用

- ✓ **同一模型训练过程中追踪进度**：loss 下降 = PPL 下降，监控收敛
- ✓ **同一架构 + 同一分词器**下比较两个 checkpoint 哪个更好
- ✓ 评估底座的 **领域适应** 程度（在金融语料 PPL 是否比通用语料低）

### 1.4 PPL 什么时候骗人

- ✗ **跨分词器比较**：词表大小 / token 粒度不同，PPL 不可比。
  极端例子：把"中国"分成 `[中, 国]` 两个 token vs `[中国]` 一个 token，CLM loss 的"每 token 平均"就完全不是同一回事
- ✗ **跨语种**：分词器对中文/英文/代码的切分密度不同，PPL 数值差很大
- ✗ **下游任务能力**：PPL 低 ≠ 推理能力强、对话能力强、数学能力强。一个在新闻语料上 PPL 极低的模型可能完全不会做数学题
- ✗ **对齐后**：DPO / SFT 后的模型在原始通用语料上 PPL 通常**变差**（因为分布漂到对话格式上了），但下游对话质量明显变好

> 一句话：**PPL 是训练监控指标，不是模型能力指标**。两个不同模型比 PPL 决出高下，多半是错的比法。

### 自检

1. 模型 A 用词表 32k 的 BPE，模型 B 用词表 8k 的 BPE，A 在某语料上 PPL=20，B 在同语料上 PPL=15。能说 B 更好吗？
2. SFT 后模型在通用网络爬虫上 PPL 升高了，是不是训坏了？

<details markdown="1">
<summary>答案速查</summary>

1. 不能。分词粒度不同 → 平均每 token 承载的信息量不同 → PPL 数值不可比。要比应该换成"按字符 / 字节平均"的 BPC（bits per character），或干脆改用下游任务指标

2. 不一定。SFT 把模型分布拉向"chat 模板 + 助理风格"，与原始 web 文本分布天然有差距，通用语料 PPL 升高很常见。判断好坏要看**对话质量**（人工或对话 benchmark），不是 PPL

</details>

---

## 2. 开源 Benchmark：把"模型能力"打分量化

### 2.1 主流 benchmark 一览

| Benchmark | 内容 | 任务形式 | 难度 |
|---|---|---|---|
| **MMLU** | 57 个学科多选（高中—专家） | 4 选 1 | 中—难 |
| **C-Eval** | 中文 52 学科多选 | 4 选 1 | 中—难 |
| **CMMLU** | 中文 67 学科多选 | 4 选 1 | 中 |
| **GSM8K** | 小学数学题 | 生成数字答案 | 中 |
| **HumanEval** | Python 代码补全 | 生成函数体 | 中 |
| **MT-Bench** | 两轮对话（一问一追问） | 生成 + GPT-4 打分 | 综合 |
| **AlpacaEval** | 单轮指令 | 生成 + GPT-4 对比 | 综合 |

> M3 阶段你只需要认识它们；M5/M6 落 echo 时，最少跑 **C-Eval 子集 + MT-Bench 中文子集**。

### 2.2 多选题怎么"考" LLM：loglikelihood scoring

直觉做法是让模型自由生成"A"/"B"/"C"/"D"再解析，但模型可能输出 `"A."`、`" A"`、`"我觉得是 A"` 等无数变体，解析脆弱。
主流做法换个角度：把"选哪个"转成"4 个候选答案谁的条件概率最高"——**loglikelihood scoring**：

```
prompt: "下列哪个是哺乳动物？\nA. 鲨鱼  B. 海豚  C. 章鱼  D. 海星\n答案："

候选: " A" / " B" / " C" / " D"

对每个候选 c，算 log P(c | prompt) → 取最大那个作为预测
```

实现伪代码：

```python
scores = {}
for letter in ["A", "B", "C", "D"]:
    full = tokenize(prompt + " " + letter)
    logits = model(full[:-1])
    # 简化前提：候选只占 1 个 token（如 " A" 在 GPT-2 BPE 下确实如此）
    # 多 token 候选（例如候选是整句话 "Dolphin is a mammal"）需把候选所有位置的
    # logp 求和：sum(log P(c_i | prompt, c_<i)) for i in choice_tokens
    logp = log_softmax(logits[-1], dim=-1)[full[-1]]
    scores[letter] = logp.item()
predicted = max(scores, key=scores.get)
```

> **关键细节**：候选答案前面的空格、是否带 `.`、letter 后面是否带换行——都会影响分词结果，从而影响打分。这就是为什么 benchmark 要严格规定 prompt 模板。

### 2.3 Few-shot：给几个例子再考

zero-shot：直接问。few-shot：先给 N 个题目+答案的"示范"再问目标题。

```
[Example 1]
问: ...  答: A

[Example 2]
问: ...  答: C

[Target]
问: ...  答:        ← 模型在这里续写
```

MMLU 标配是 5-shot；C-Eval 也是 5-shot。few-shot 让模型"看懂格式"再答题，分数通常显著高于 zero-shot。

> **同一个模型 zero-shot 和 5-shot 分数差 5~15 个点很常见**。所以看 benchmark 数字必须问"几 shot"。

### 2.4 生成式题目（GSM8K / HumanEval）

不是选答案，是真生成：

- **GSM8K**：让模型解小学数学题，正则提取最后的数字与标准答案比对
- **HumanEval**：让模型补全函数体，跑单元测试看 pass 率

这类任务直接考 **生成能力 + 推理能力**，比多选题更接近真实使用，但也更贵（要真的 generate 而非只算 4 个候选 logp）。

### 自检

1. MMLU 在打分时为什么不让模型自己生成 "A"/"B"/"C"/"D"，而是算 loglikelihood？
2. 同一个模型，zero-shot MMLU 35 分，5-shot 50 分，是模型变强了吗？

<details markdown="1">
<summary>答案速查</summary>

1. 自由生成时模型可能输出 "A.", " A", "A、", "我觉得是 A" 等无数变体，解析容易错；loglikelihood scoring 把"选哪个"转成"4 个候选谁概率高"，**确定性、可复现、低成本**（不用真的采样生成）

2. 模型本身没变，只是 5-shot 示例帮它"理解了答题格式"。说明模型 zero-shot 下"知识在但不会按格式答"。这两个数字不能跨 shot 比

</details>

---

## 3. Benchmark 的常见坑

### 3.1 数据污染（contamination）

预训练语料里**已经包含了 benchmark 题目和答案**，模型不是在做题，而是在背答案。

- 来源：MMLU/C-Eval 的题目早被人搬运到各种博客、知乎、GitHub 上
- 现象：模型在常见 benchmark 上分数很高，换一个新 benchmark 立刻崩
- 应对：
  - 看 benchmark 的发布日期 vs 模型的预训练数据截止日期
  - 用 **去污染检测**（n-gram 命中、embedding 相似度）
  - 用 **新 benchmark / 持续刷新的 benchmark**（如 LiveBench）

### 3.2 Prompt 敏感

同一个模型，prompt 模板换一下分数差 3—10 个点。所以：

- 跨模型比 benchmark 必须用**同一个 prompt 模板**和**同一份评测代码**
- 业内事实标准：**lm-evaluation-harness**（EleutherAI）、**OpenCompass**（中文圈）

### 3.3 多选 ≠ 真能力

模型在 4 个固定选项里挑对的，不代表自由问它能给对答案。
真实使用是开放生成，benchmark 是选择题，**两者评估的能力维度有偏移**。

### 3.4 Benchmark 通胀

新模型层出不穷，benchmark 分数被刷得越来越高（MMLU 早期 50 分是 SOTA，现在 90 分起步）。
高分越来越难区分模型，需要不断推出更难的 benchmark（如 MMLU-Pro、GPQA、GAIA）。

> **底线**：benchmark 分数是参考，不是结论。看到"我们在 MMLU 上 85 分"先问"几 shot、什么 harness、训练数据有无污染"。

---

## 4. 人工评测：自动指标补不齐的部分

### 4.1 自动指标看不见什么

- **风格 / 调性**：助理是讨好还是严谨，benchmark 不测
- **多轮一致性**：上下文是否漂移、人设是否稳定，多选题不测
- **拒绝合理性**：该拒绝的拒绝、不该拒绝的别瞎拒绝，benchmark 不测
- **事实性 vs 流畅性的权衡**：流畅但胡编 vs 朴素但准确，多选题打不出区别
- **长尾错误**：罕见话题、边缘 case、对抗性输入

### 4.2 实务做法

| 方式 | 成本 | 适用 |
|---|---|---|
| **自己写一组 prompt 集（10—50 条）每次手测** | 低 | 训练迭代期"快速 sanity check" |
| **MT-Bench / AlpacaEval 用 GPT-4 当裁判** | 中 | 单轮/多轮对话能力，自动化 |
| **A/B 盲测**：两个模型回答打乱顺序，标注员选偏好 | 高 | 关键发布节点 |
| **真实用户上线 + 反馈日志** | 高 | 长期迭代 |

> **echo（M5+）的最小评测套件**：
> - 自建 20 条中文 prompt（人设、对话、知识、拒绝场景各 5 条）每次训完手测
> - C-Eval 验证集子集（500 题）跑 lm-evaluation-harness
> - MT-Bench 中文版（80 题）用 GPT-4 当裁判
> - 上线前最终一次 A/B 盲测对比上一版

### 4.3 GPT-4 当裁判靠谱吗

- 部分靠谱：相关性高，但有偏：
  - **风格偏**：偏好长回答、偏好礼貌语气、偏好 GPT 风格
  - **position bias**：把同一对回答的顺序调换，裁判结果常常跟着变。最被研究、最实操要紧的偏。**必须做 swap**（A/B 与 B/A 各跑一遍取一致结论），否则裁判分数被先后顺序干扰
- 必须配合人工抽检（10% 样本）校准
- 别迷信单一裁判，多裁判聚合（GPT-4 + Claude + 人工抽样）更稳

### 自检

1. 你的模型 MMLU 50 分，竞品 MMLU 55 分，能说竞品更适合做对话助理吗？
2. 为什么不能只靠 benchmark 决定模型发不发布？

<details markdown="1">
<summary>答案速查</summary>

1. 不能。MMLU 测的是学科知识，对话助理需要的是指令跟随、多轮一致、风格、拒绝判断等。MMLU 高的可能是"知识好但不会聊天"的底座，MMLU 低的可能是"知识一般但对话顺滑"的对齐版。两者评的不是同一回事

2. ① benchmark 数字易被污染、易被 prompt 操纵 ② 自动指标对风格 / 多轮 / 拒绝场景近似失明 ③ 真实用户体验需要在真实分布上测，benchmark 只能 proxy。结论：benchmark 是必要不充分条件

</details>

---

## 5. 工具栈速查

| 工具 | 用途 | 备注 |
|---|---|---|
| **lm-evaluation-harness** | 跑 MMLU / C-Eval 等多选 benchmark | EleutherAI 出品，社区事实标准 |
| **OpenCompass** | 同上，中文圈用得多 | 上海 AI Lab 出品 |
| **MT-Bench** | 两轮对话 + GPT-4 裁判 | LMSYS 出品 |
| **AlpacaEval** | 单轮指令对比 | Stanford 出品 |
| 自己写脚本算 PPL | 训练监控 | 几十行 Python 即可，见 §1.2 |

> M5 阶段先把"自写 PPL + lm-eval-harness 跑 C-Eval 子集 + 手工 20 条"这套最小流程跑起来即可，别一上来就追求大而全。

---

## 6. 练习

落到 `Playground/ch12-eval/`：

| 脚本 | 内容 |
|---|---|
| `01_ppl.py` | 用 GPT-2 small（或本地小模型）在一段莎士比亚 / 中文新闻上算 PPL；同模型不同语料对比 PPL，验证"跨语料不可直接比强弱"直觉 |
| `02_loglikelihood_mcq.py` | 手写 loglikelihood scoring：构造一道 4 选 1 题，用小模型分别算 4 个选项的条件 logp，看模型选哪个；改 prompt 模板看分数怎么变 |

跑法：

```bash
uv run python Playground/ch12-eval/01_ppl.py
uv run python Playground/ch12-eval/02_loglikelihood_mcq.py
```

依赖 `transformers`（加载 HF 上的小模型），首次运行会下载几百 MB 权重；离线/无网时脚本里也提供了纯 PyTorch 玩具回退路径。

## 思考题

1. 你训了一个 echo-mini，怎么设计一个"3 分钟跑完的迭代期评测"和一个"3 小时跑完的发布评测"？各包含什么？
2. 假设你的模型在 C-Eval 上 60 分，竞品 65 分。在不动模型的前提下，列举至少 3 种"提分"手法（含合规与不合规的）。
3. 为什么 benchmark 排行榜（如 OpenLLM Leaderboard）的"第 1 名"通常不等于"最好用"？

## 参考资料

- **EleutherAI lm-evaluation-harness**：<https://github.com/EleutherAI/lm-evaluation-harness>
- **OpenCompass**：<https://github.com/open-compass/opencompass>
- **MMLU 论文**：Hendrycks et al., "Measuring Massive Multitask Language Understanding" (2020)
- **C-Eval 论文**：Huang et al., "C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite" (2023)
- **MT-Bench / Chatbot Arena**：Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023)
- **数据污染综述**：Sainz et al., "NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination" (2023)
