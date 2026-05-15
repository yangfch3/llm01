# ch04 · 神经网络与训练要素

> ch03 把"前向 + 反向 + 优化"模板背下来了。本章把模板里**每一颗螺丝**单独拧一拧：
> 初始化、优化器、学习率调度、Dropout、归一化（BN/LN）。
> 这些是后面 Transformer 训练能不能稳定收敛的决定性因素。

## 学习目标

1. 理解为什么"初始化错了 → 训不动"，能区分 Xavier vs He vs 朴素初始化
2. 区分 SGD / SGD+momentum / Adam / AdamW 的核心差异，知道何时用哪个
3. 理解 LR 调度的"为什么"，能解释 cosine / step / warmup 的动机
4. 解释 Dropout 在训/推阶段的差异；理解为什么 LLM 全用 LayerNorm 而非 BatchNorm

## 前置依赖

- ch02（链式法则 / 梯度直觉）、ch03（PyTorch 训练循环模板）

---

## 1. 反向传播的工程视角

ch02 把反向传播当数学讲。本节把它当**工程问题**看：每个工程细节背后都有一个"如果不这样做就翻车"的故事。

### 1.1 计算图复用与 retain_graph

PyTorch 的计算图**默认 backward 一次就被释放**（省显存）。再 backward 会报：

```
RuntimeError: Trying to backward through the graph a second time
```

99% 训练场景一次足够。需要多次反传同一张图（比如算高阶梯度、某些 RL 算法）才用 `loss.backward(retain_graph=True)`。

### 1.2 zero_grad 必须在 backward 之前

ch03 §2.2 已强调。再补一个细节：`set_to_none=True`（PyTorch 1.7+ 默认就是这个）。

```python
optimizer.zero_grad()                # 等价于 optimizer.zero_grad(set_to_none=True)
```

`set_to_none=True` 直接把 `.grad` 设回 `None`，比"填零"更省一次显存写入；唯一区别是某些自定义优化器需要先判 `if p.grad is not None`。

### 1.3 梯度爆炸 / 消失：用数值看

不做任何归一化的深网络，前向激活值会指数级炸或衰减。最快诊断方法：每隔几步打印各层梯度范数：

```python
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"{name}: grad_norm={p.grad.norm().item():.3e}")
```

经验阈值：

- `< 1e-7` → 梯度消失（学不动）
- `> 1e3` → 梯度爆炸（很快变 nan）
- `nan / inf` → 已经爆了，回退到上一个 ckpt + 减小 lr / 加 grad clip

**Gradient clipping** 是常用救命药：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 在 step() 之前调
```

意思：把所有参数梯度看成一个长向量，范数超过 1.0 就等比例缩放回 1.0。

---

## 2. 参数初始化

> 一句话：**初始化 ≠ 锦上添花，是网络能不能训起来的前提**。
> 直觉：每层输出方差应保持稳定，否则前向几层就炸/消，反向梯度同理。

### 2.1 三档对照

| 方案 | 公式（fan_in 是输入维度） | 配什么激活 |
|---|---|---|
| 朴素正态 $\mathcal{N}(0, 1)$ | std = 1 | **不要用**，几层就炸 |
| Xavier（Glorot） | std = $\sqrt{1/\mathrm{fan\_in}}$ | sigmoid / tanh |
| Kaiming（He） | std = $\sqrt{2/\mathrm{fan\_in}}$ | ReLU 系（含 GELU / SiLU） |

ReLU 把负半轴砍掉，输出方差减半，所以 He 比 Xavier 多了个 $\sqrt{2}$ 系数补回来。

### 2.2 PyTorch 默认初始化

`nn.Linear` 默认走 **Kaiming 均匀**（uniform 版的 He）。`nn.Conv2d` 同。所以**普通 MLP/CNN 你不动它就对了**。

需要手动初始化的场景：

```python
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)  # 递归遍历所有 submodule
```

LLM 训练里更常见的是 GPT-2 / LLaMA 风格初始化（小标准差如 0.02），M2 ch06 会再讲。

### 自检

1. 用朴素 $\mathcal{N}(0, 1)$ 初始化一个 10 层 ReLU MLP，前向第几层激活会爆？
2. 为什么 He 比 Xavier 多个 $\sqrt{2}$？

<details markdown="1">
<summary>答案速查</summary>

1. 大约 4–5 层后激活值数量级开始失控。可以用 `01_init_compare.py` 实测

2. ReLU 砍掉负半轴 → 输出方差变成原来的 1/2 → 要把方差扩大 2 倍补回来 → std 乘 $\sqrt{2}$

</details>

---

## 3. 优化器

### 3.1 四档主流对照

| 优化器 | 一句话特性 | 何时用 |
|---|---|---|
| **SGD** | 纯梯度下降 | 简单凸问题、教学；现代深网络极少单独用 |
| **SGD + momentum** | 累积"惯性方向"，跨过窄峡谷 | 视觉任务（ResNet 时代）、LLM 预训练偶见 |
| **Adam** | 一阶矩 + 二阶矩自适应学习率 | NLP / Transformer 默认 |
| **AdamW** | Adam 的 weight decay 修正版 | **LLM 预训练 / SFT 的事实标准** |

### 3.2 momentum 的直觉

```
普通 SGD：       w ← w - lr · g
SGD + momentum：v ← β·v + g
                w ← w - lr · v
```

`v` 是梯度的指数滑动平均（β 通常 0.9）。**累积同向梯度**让步子变大、抵消反向噪声。直觉：滚下山的小球积攒动量，能冲过小坑。

### 3.3 Adam = momentum + 自适应学习率

Adam 同时维护两个滑动平均：

- 一阶矩 $m$（梯度均值，类似 momentum）
- 二阶矩 $v$（梯度平方均值，估计每个参数的"波动幅度"）

更新规则（省略 bias correction）：

\[
w \leftarrow w - \mathrm{lr} \cdot \frac{m}{\sqrt{v} + \epsilon}
\]

**核心创新**：每个参数有**自己的学习率**，由 $\sqrt{v}$ 倒数缩放。波动大的参数自动减小步长，波动小的自动放大。这让 Adam 几乎不用调 lr 也能跑。

### 3.4 AdamW：weight decay 的正解

L2 正则原本是在 loss 里加 $\frac{\lambda}{2}\|w\|^2$。Adam 把它和梯度一起塞进 $\sqrt{v}$ 缩放，等价于"波动大的参数被较少正则化"——**与正则化初衷相反**。

AdamW 的修法：weight decay **不进梯度**，直接在参数上扣：

```
w ← w - lr · (Adam 更新方向) - lr · λ · w
```

LLM 预训练几乎一律 AdamW。PyTorch 一行：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
```

### 自检

1. 同一个网络用 SGD 和 Adam，初始 lr 一般谁大？为什么？
2. 为什么 LLM 预训练用 AdamW 而不是 Adam？

<details markdown="1">
<summary>答案速查</summary>

1. SGD 大（典型 1e-2 ~ 1e-1），Adam 小（典型 1e-4 ~ 1e-3）。Adam 自带 $1/\sqrt{v}$ 自适应放大，名义 lr 已被隐式放大，需要手动调小

2. Adam 的 weight decay 实现等价于"波动大的参数被较少正则化"，与正则化目的相反；AdamW 把 weight decay 与梯度更新解耦，对大模型泛化和稳定性更友好

</details>

---

## 4. 学习率调度

固定 lr 的两个问题：

- 前期太大 → 震荡 / 发散；前期太小 → 收敛慢
- 后期太大 → loss 在最小值附近徘徊；后期太小 → 早期就这样

解法：**lr 随训练动态变化**。

### 4.1 三种常见 schedule

| 名称 | 形状 | 用法 |
|---|---|---|
| **StepLR** | 阶梯式衰减（每 N 步乘 0.1） | 简单 / 视觉任务老牌 |
| **CosineAnnealing** | 余弦曲线从 max → min | LLM 预训练默认 |
| **Warmup + Cosine** | 前 K 步线性升到 max，再 cosine 衰减到 min | LLM 预训练事实标准 |

### 4.2 为什么 LLM 都要 warmup

训练前期参数随机，梯度方差极大。直接上高 lr 容易把参数推到"再也回不来"的位置。warmup（前几百到几千步线性升 lr）让网络先"探探路"再放开。

伪代码骨架：

```python
def get_lr(step: int, warmup_steps: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * step / warmup_steps                              # 线性 warmup
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))  # cosine
```

PyTorch 提供 `torch.optim.lr_scheduler.CosineAnnealingLR`、`OneCycleLR` 等开箱方案，**M4 训 echo-mini 时手写一份**就行。

### 自检

1. warmup 期可以省吗？跳过会怎样？
2. cosine 衰减到的 `lr_min` 一般取多少？为什么不直接到 0？

<details markdown="1">
<summary>答案速查</summary>

1. 小模型 / 小数据可以省。大模型一旦省了，前几百步极易 loss 飞到 nan，没救只能从头来。代价比加几行 schedule 高得多

2. 通常取 `lr_max` 的 1/10（如 max=3e-4 → min=3e-5）。直接到 0 后期等于不学，浪费算力且有时反而 loss 反弹

</details>

---

## 5. Dropout

### 5.1 训练时随机丢神经元

```python
nn.Dropout(p=0.5)                                # 训练时以概率 p 把激活置 0
```

直觉：强迫网络不要依赖某些"明星神经元"，提升泛化（类似 ensemble 多个子网络）。

### 5.2 训/推不一致的关键细节

训练丢一半神经元，推理时全保留——**激活值期望就翻倍了**。Dropout 在训练时主动**乘 1/(1-p)** 补偿（叫 inverted dropout），推理时什么都不用做。

```python
model.train()                                    # Dropout 生效
model.eval()                                     # Dropout 关闭，全保留
```

**没切 eval 直接推理 = bug**（指标偏低且每次结果不一样）。

### 5.3 LLM 里 Dropout 用得少

GPT-2/3、LLaMA 系列预训练阶段 Dropout 一般设为 0 或极小（0.0–0.1）。原因：

- 数据足够多时 Dropout 几乎没收益
- 与 Pre-LN 结构、warmup、AdamW 一起用时它的正则化作用被替代
- SFT / 微调阶段会重新启用（数据量小，需要正则）

---

## 6. 归一化：BatchNorm vs LayerNorm

> 归一化的共同目的：把每层激活值拉回均值 0、方差 1 附近，让训练更稳。

### 6.1 BatchNorm：沿 batch 维统计

输入 `(N, C, H, W)`，BN 对每个 channel 在 `(N, H, W)` 维上求均值/方差。

```python
nn.BatchNorm2d(num_features=C)
```

**BN 的问题**：

- batch 太小（< 8）时统计量噪声大，反而拖累训练
- 推理时用训练阶段的 running statistics（不是当前 batch），训推不一致
- 在 RNN / 序列长度可变的场景几乎不能用——**包括 Transformer**

### 6.2 LayerNorm：沿特征维统计

输入 `(N, ..., D)`，LN 对每个样本在最后一维（feature dim）上求均值/方差。

```python
nn.LayerNorm(normalized_shape=D)
```

**LN 的优点**：

- 与 batch size 无关，batch=1 也能用
- 训推一致（同一种统计公式）
- 序列每个位置独立归一化，天然适配变长序列

**所以 LLM 全用 LN（或它的变种 RMSNorm）**。BN 和 LLM 几乎绝缘。

### 6.3 Pre-LN vs Post-LN（M2 详讲）

Transformer 里 LN 放在残差**之前**还是**之后**直接影响训练稳定性。简短结论：

- **Pre-LN**（LN 在残差前）：训稳，主流
- **Post-LN**（LN 在残差后，原版 Transformer）：需精细 warmup，不稳

ch06 会画图详解，这里先记结论。

### 自检

1. batch_size=2 训练含 BN 的网络有什么问题？换成 LN 呢？
2. 为什么 Transformer 不能用 BN？

<details markdown="1">
<summary>答案速查</summary>

1. BN：batch 太小，每个 batch 的均值/方差噪声大，running stats 估计偏差，loss 震荡甚至无法收敛。LN：完全没影响，每个样本独立算，与 batch size 解耦

2. (a) 序列长度可变，BN 的 channel 统计无意义；(b) 推理时常 batch=1 或长序列流式，BN running stats 与训练分布严重不一致；(c) attention 让不同位置 token 之间相互影响，BN 的"位置独立"假设被破坏

</details>

---

## 7. 练习

落到 `Playground/ch04-nn-training/`：

| 脚本 | 内容 |
|---|---|
| `01_init_compare.py` | 朴素 / Xavier / He 初始化在 10 层 ReLU MLP 上的激活方差对比 |
| `02_optimizer_compare.py` | SGD / Momentum / Adam / AdamW 在合成数据上的 loss 曲线 |
| `03_lr_schedule.py` | 固定 / Step / Cosine / Warmup+Cosine 四种 lr 曲线可视化（数值打印） |
| `04_dropout_bn_ln.py` | Dropout 训/推差异；小 batch 下 BN 翻车 vs LN 稳定 |

跑法同 ch03。所有脚本不依赖外部数据，3060 / Mac / CPU 都秒级跑完。

## 思考题

1. 如果你的 Transformer loss 在第 200 步突然 nan，你会按什么顺序排查？（涉及本章 lr / clip / init / 数据）
2. AdamW 的 `weight_decay=0.1` 在 LLM 预训练里很常见。0.1 看起来不大，为什么对几亿/几十亿参数模型有显著影响？
3. 假设你设计一个新的归一化算子，希望同时具备 LN 的"训推一致"和 BN 的"feature 维白化"——你会怎么设计统计量？（这不是无聊的脑洞，是 GroupNorm / RMSNorm 的来源）

## 参考资料

- **Kaiming He et al., "Delving Deep into Rectifiers"**：He 初始化原论文
- **Diederik Kingma, "Adam: A Method for Stochastic Optimization"**：Adam 原论文
- **Loshchilov & Hutter, "Decoupled Weight Decay Regularization"**：AdamW 原论文
- **Ba et al., "Layer Normalization"**：LN 原论文
- **GPT-2 / LLaMA 模型卡 / 训练配方**：现代 LLM 实际使用的 init / optimizer / lr 配置参考
