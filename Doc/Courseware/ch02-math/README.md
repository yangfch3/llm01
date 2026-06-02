# ch02 · 必要数学（浅层）

> 本章补的数学知识目标是 "够用"，不是 "完备"。本章只覆盖 LLM 训练实战会反复出现的几个数学点。
> 
> 每个公式后面都跟一个小数字例子，看不懂公式时**先看例子再回头看公式**。

## 学习目标

1. 看到向量/矩阵运算时知道**形状**怎么对齐、**含义**是什么
2. 理解链式法则在反向传播里的角色，能手算两层网络的**梯度**
3. 理解 softmax + 交叉熵为什么是分类问题的 "标配组合"，数值稳定要怎么做

## 前置依赖

- ch01 课件 + 练习
- 高中 / 大一线性代数残留记忆即可

---

## 1. 向量、矩阵、点积

### 1.1 形状是第一公民

本节用最朴素的**标量、向量、矩阵**来讲，配套练习用 NumPy `ndarray` 多维数组演示。

- 0 维：标量（一个数），如 `loss = 0.83`
- 1 维数组 - shape(x, )：向量（一行数），如词向量 `[0.1, -0.2, 0.5, ...]`
- 2 维数组 - shape(x, y)：矩阵（x 行 × y 列），如一批样本 `(batch, dim)`

向量与矩阵的运算中，shape 需要满足一定的数学合法性，养成先在脑里/注释里写出 shape 的习惯：

```python
# x: (batch=2, dim_in=3)
# w: (dim_in=3, dim_out=4)
# y = x @ w  -> (batch=2, dim_out=4)
```

形状对了，语义大概率对；形状错了立刻报错，例如 NumPy 下的：

```
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0,
            with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is different from 3)
```

**这是初学最高频的报错，没有之一**。养成习惯：写向量/矩阵（以及 ch03 会讲到的 tensor）操作时**先注释 shape**，写完代码再核对一遍。

> **tensor**: 可以先简单理解为 PyTorch 框架对多维数组更高级的包装 + 额外封装了利于模型训练的信息存储。

### 1.2 点积的两种理解

向量 $a, b \in \mathbb{R}^d$ 的点积：

\[
a \cdot b = \sum_{i=1}^{d} a_i b_i = \|a\|\,\|b\|\cos\theta
\]

**代数视角**：逐元素相乘求和。

```
a = [1, 2, 3]
b = [4, 5, 6]
a · b = 1·4 + 2·5 + 3·6 = 4 + 10 + 18 = 32
```

**几何视角**：$\|a\|\|b\|\cos\theta$，衡量 "方向有多像"。

- 完全同向 $\theta = 0$ → $\cos\theta = 1$ → 点积最大
- 垂直 $\theta = 90°$ → $\cos\theta = 0$ → 点积为 0
- 反向 $\theta = 180°$ → $\cos\theta = -1$ → 点积最小（负）

```
 y
 ↑
 │     ↗ a       ← 向量 a 从原点斜向上
 │   ↗
 │ ↗ ────→ b    ← 向量 b 从原点向右
 O────────→ x
                  ← a, b 都从原点 O 出发，θ 是它们之间的夹角
                    θ 小（方向接近）→ cos θ 接近 1 → 点积大
```

> **这是后面要学的 attention 机制的核心相似度算子**：算 query 和每个 key 的点积，值大表示 "方向接近 + 模长可观" —— 也就是 "应该多关注这个位置"。LLM 里反复出现的 $QK^\top$，每一项就是某个 token 的 query 向量和某个 token 的 key 向量的点积。

### 1.3 矩阵乘 = 批量点积

\[
(AB)_{ij} = \sum_k A_{ik} B_{kj}
\]

公式释义：$AB$ 第 $i$ 行第 $j$ 列的元素 = $A$ **第 $i$ 行**和 $B$ **第 $j$ 列**的点积。

**最小数字例子**：

```
A = [[1, 2],     B = [[5, 6],
     [3, 4]]         [7, 8]]

AB[0][0] = A 第 0 行 · B 第 0 列 = [1,2]·[5,7] = 1·5 + 2·7 = 19
AB[0][1] = A 第 0 行 · B 第 1 列 = [1,2]·[6,8] = 1·6 + 2·8 = 22
AB[1][0] = A 第 1 行 · B 第 0 列 = [3,4]·[5,7] = 3·5 + 4·7 = 43
AB[1][1] = A 第 1 行 · B 第 1 列 = [3,4]·[6,8] = 3·6 + 4·8 = 50

AB = [[19, 22],
      [43, 50]]
```

**形状口诀**：内维相同，外维保留。

\[
(m, k) \times (k, n) \to (m, n)
\]

```
(2, 3) × (3, 4) = (2, 4)   ✓ 内维 3 = 3
(2, 3) × (4, 3) = ?        ✗ 内维 3 ≠ 4，报错
```

**Linear 层就是矩阵乘**：`nn.Linear(in_features=3, out_features=4)` 概念上等价于右乘一个 $(3, 4)$ 的权重矩阵（转置已合并进去），输入 $(N, 3)$ 出来 $(N, 4)$。

### 自检

- $A$ 是 $(5, 8)$，$B$ 是 $(8, 3)$，$AB$ 形状是？$BA$ 能算吗？
- 点积 $a \cdot b = 0$ 说明 $a$ 和 $b$ 啥关系？

<details markdown="1">
<summary>答案速查</summary>

- $AB$ 形状是 $(5, 3)$（外维保留）；$BA$ 不能算，内维 $3 \neq 5$

- $a$ 和 $b$ 互相**正交**（夹角 90°）。注意零向量也满足 $a \cdot b = 0$，是边界情况

</details>

---

## 2. 梯度与链式法则

### 2.1 导数与梯度

回忆一下我们学过导数：导数是函数的局部性质。一个函数 $f(x)$ 在某一点 $x$ 的导数就是该函数所代表的曲线在这一点上的切线斜率，描述了这个函数在这一点附近的**变化率**。

一元函数 $f(x)$ 的导数 $f'(x)$ 表示：$x$ 增加一点，$f$ 增加多少。

对于一个多表达式复合而成的多元函数 $L(w_1, w_2, \ldots, w_d)$，其**梯度** $\nabla L$ 是个 $d$ 维向量，每个分量是对应方向的偏导：

\[
\nabla_w L = \left(\tfrac{\partial L}{\partial w_1}, \tfrac{\partial L}{\partial w_2}, \ldots, \tfrac{\partial L}{\partial w_d}\right)
\]

**几何含义**：站在 $L$ 这座山的某一点，梯度向量指向**最陡的上山方向**，模长是陡度。

```
          ↑ ∇L  最陡上山方向
        ╱
       ╱      L(w) 的等高线
   ───●───    我们站在这里
       ╲
        ╲
          ↓ -∇L  最陡下山方向
```

### 2.2 梯度下降："学习" 的本质

机器学习（ML）的本质是：建立一个复杂的用于评估预测质量的拟合模型，简单描述为函数 `loss L = fn(x)`，不断调整 `fn` 使其最终预测更为精准 + 符合预期。

- 输入 x
- 预测值 y
- 预期值 t
- loss L：预测值 y 与正确值 t 的误差表示（回忆一下线性回归建模）
- fn 内部有许多参数与机制：统一简单抽象为 $w$

所谓的训练的过程，本质是调整 $w$，使我们的预测误差逐步缩小收敛。上一节我们学到了**梯度**的几何含义，联想一下，有何灵感？

我们要最小化 loss $L$，那就找到下山的方向，**沿 -∇L 方向**小步走，这就是**梯度下降**。每走一小步，对应微调 $w$，以期未来预测的误差变小，便是模型 "学习" 的本质所在。

\[
w \leftarrow w - \eta \cdot \nabla_w L
\]

$\nabla_w L$ 是 2.1 中提到的梯度 —— 下山的方向，$\eta$ 是学习率，控制 "下山" 步子的大小。

> 学习率太大会冲过头（loss 震荡甚至爆炸），太小走得慢，合适的学习率才能让模型面对海量、复杂的学习语料，以合适的速率趋势往小收敛。

到这里，你已经理解了机器 **"学习"**：

- 拿一批数据前向算出 loss → 算梯度
- 按上式反向调整 $w$
- 海量样本，如此反复

loss 一点点降，$w$ 一点点变好。所谓 "训练一个模型"，本质上就是**海量不同的输入，重复执行前向计算预测 + 反向调整**。

但这里还差一块：公式里的 $\nabla_w L$ 怎么算？或者说复合函数怎么求导？单层、单参数好说，可实际中建模多是多层堆叠的复合函数，$w$ 中的 $w_i$ 离 $L$ 可能隔了许多层运算，没法直接写出导数。

### 2.3 链式法则：反向传播的灵魂

链式法则解决的问题：**复合函数怎么求导** —— 也就是 2.2 末尾留下的 "$\nabla_w L$ 在多层堆叠里怎么算" 的问题。

设 $L = f(g(w))$，其中 $w$ 经过 $g$ 变成中间量 $u = g(w)$，再经过 $f$ 变成 $L$。问 $L$ 对 $w$ 怎么变？

\[
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial w}
\]

**口语化**：变化沿计算图**一节一节传递**，每一节乘上当地的 "放大倍率"。

> 计算图（computation graph）= 把一个公式拆成若干最基本的运算节点，按 "谁产生谁" 的关系连成图。相比于链（直线），图还可以包含分支、汇合等。

**最小例子**：

```
w ──[平方]──→ u ──[乘 3]──→ L
      u = w²       L = 3u

求 ∂L/∂w：
  ∂L/∂u = 3       （L 是 u 的 3 倍，u 增 1，L 增 3）
  ∂u/∂w = 2w      （u = w²，u 对 w 的导数）
  ∂L/∂w = 3 · 2w = 6w

验证：L = 3u = 3w²，直接求导 ∂L/∂w = 6w ✓
```

**反向传播 = 沿计算图反向应用链式法则**。从 loss 出发，一层一层把梯度 "传" 回每个参数。

### 2.4 重点：两层堆叠反向手算

本节是后面所有反向传播相关知识的母题，将以一个两层堆叠的复合函数为例进行讲解。**强烈建议拿张纸和我一起算一遍**。

**前向**：

```
x ──[× w1]──→ h ──[ReLU]──→ a ──[× w2]──→ y ──[(y-t)²/2]──→ L
```

公式（标量版，简化掉 batch）：

\[
h = w_1 \cdot x,\quad
a = \mathrm{ReLU}(h) = \max(h, 0),\quad
y = w_2 \cdot a,\quad
L = \tfrac{1}{2}(y - t)^2
\]

$\mathrm{ReLU}$：层（Linear）间激活控制函数，后面章节会详细介绍。

$L = \tfrac{1}{2}(y - t)^2$：**MSE（Mean Squared Error，均方误差）** 的单样本版——回归任务最常用的 loss。

> 多样本时取平均：$\tfrac{1}{N}\sum_i (y_i - t_i)^2$。前面的 $\tfrac{1}{2}$ 是为了求导后系数干净（$\partial L / \partial y = y - t$，没有多余的 2），不影响优化方向。

**反向**（从 L 出发，反向走一遍）：

**Step 1**：$L$ 对 $y$ 的梯度。$L = \tfrac{1}{2}(y-t)^2$，求导：

\[
\frac{\partial L}{\partial y} = y - t \quad\text{（误差本身）}
\]

**Step 2**：$L$ 对 $w_2$。 $y = w_2 \cdot a$，所以 $\partial y / \partial w_2 = a$。链式：

\[
\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w_2} = (y - t) \cdot a
\]

**Step 3**：$L$ 对 $a$。 $\partial y / \partial a = w_2$，链式：

\[
\frac{\partial L}{\partial a} = (y - t) \cdot w_2
\]

**Step 4**：$L$ 对 $h$。ReLU 的导数：$h > 0$ 时为 1，$h \leq 0$ 时为 0（$h = 0$ 处约定取 0 即可）。

\[
\frac{\partial L}{\partial h} = \frac{\partial L}{\partial a} \cdot \mathbb{1}[h > 0]
\]

**Step 5**：$L$ 对 $w_1$。 $h = w_1 \cdot x$，$\partial h / \partial w_1 = x$，链式：

\[
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial h} \cdot x
\]

**带数字过一遍**（设 $x = 2$，$w_1 = 0.5$，$w_2 = 3$，$t = 5$）：

```
前向：
  h = 0.5 · 2 = 1
  a = ReLU(1) = 1
  y = 3 · 1 = 3
  L = (3-5)² / 2 = 2            ← 预测 3，实际值 5

反向：
  ∂L/∂y  = 3 - 5 = -2
  ∂L/∂w2 = -2 · 1 = -2          ← w2 增加 → L 减小，梯度为负
  ∂L/∂a  = -2 · 3 = -6
  ∂L/∂h  = -6 · 1 = -6          ← h=1>0，ReLU 导数为 1
  ∂L/∂w1 = -6 · 2 = -12         ← w1 增加 → L 减小，梯度为负

更新（学习率 0.01）：
  w1 ← 0.5 - 0.01·(-12) = 0.62  ← 增大 w1
  w2 ← 3   - 0.01·(-2)  = 3.02  ← 增大 w2
```

更新 w1, w2 的实际意义：修正 w1, w2 以期让下次前向预测误差更小。

PyTorch 的 `loss.backward()` + `optimizer.step()` 自动做的就是这件事。但**至少手算一次**，你才会真懂 "梯度是什么、它怎么传"。

**从标量到向量**：练习 `03_gradient_chain_rule.py` 把上面的标量版推广到**向量版**，$x$ 是向量、$w_1$ 是矩阵 $(h, d_{in})$、$h$ 是向量。推导思路完全一样，只是上面例子每一步的运算符变成了矩阵运算：

- 标量版：$\partial L / \partial w_1 = (\partial L / \partial h) \cdot x$
- 向量版：$\partial L / \partial w_1 = (\partial L / \partial h) \otimes x$（外积，`np.outer(dh, x)`）

形状对一下就明白：$\partial L / \partial w_1$ 必须和 $w_1$ 同形 $(h, d_{in})$，而 $dh$ 是 $(h,)$、$x$ 是 $(d_{in},)$ → 唯一能凑出 $(h, d_{in})$ 的就是外积。这个 "形状反推" 是大多数人手推矩阵反向传播的实战技巧。

### 2.5 MLP 与神经网络

顺便给上面的多层堆叠案例一个名字：`x → h → a → y` 这种 **「Linear → 激活 → Linear」** 的两层结构，就是最小的 **MLP（Multi-Layer Perceptron，多层感知机）**。把这个 pattern 堆 N 遍（每层之间夹激活）就是 N 层 MLP。

MLP 与**神经网络（Neural Network, NN）**的关系：神经网络是一个大类，凡是「若干层可学习参数 + 层间非线性 + 端到端梯度下降训练」的模型都算，MLP 只是其中**最基础的一种**。

神经网络按照层结构的分类：

- **MLP**（多层感知机）
- **CNN**（卷积网络，擅长图像）
- **RNN / LSTM**（循环网络，早期处理序列的主力）
- **Transformer**（本课程的主线，LLM 的底座）

不同结构的差别在于 "层" 长什么样（卷积层、循环层、注意力层……），但**训练方式完全一致**：前向算 loss → 反向传梯度 → 梯度下降更新参数 $w_i$ 。本章手算的这套链式法则，对所有 NN 都通用。

既然 Transformer 是 LLM 的底座了，那 MLP 是不是已经被淘汰的老古董呢？—— 不是的。

Transformer 内部每个 block 都嵌了一个 MLP 子层（FFN，Feed-Forward Network，见后续章节），他们的关系是：`LLM -> Transformer -> MLP 部件`

ch03 会用 PyTorch 把 MLP 实现一遍，ch04 则以 MLP 为最简载体，讲解适用于所有神经网络的通用训练技术（初始化、优化器、学习率调度、归一化等）。把 MLP 这个最朴素的结构吃透，后面学 Transformer 时才不会在最基础、最核心的地方卡住。

### 2.6 数值梯度：调试的救命稻草

自己写反向传播容易出错。**用数值梯度做 sanity check**：

\[
\frac{\partial L}{\partial w_i} \approx \frac{L(w + \epsilon e_i) - L(w - \epsilon e_i)}{2\epsilon}
\]

意思：把 $w_i$ 微微抬一点（加 $\epsilon$）算 $L$，再微微压一点算 $L$，差除以 $2\epsilon$ 就是斜率近似。

- $\epsilon$ 取 $10^{-5}$ 量级（太小被浮点精度吞掉——`float32` 精度约 $10^{-7}$，$\epsilon < 10^{-6}$ 时 $L(w+\epsilon)$ 和 $L(w)$ 的差被舍入误差淹没；太大则一阶近似不准）
- 和你手算的解析梯度比较，最大**相对误差** $< 10^{-5}$ → 实现正确（用相对误差而非绝对差，是因为梯度量级本身可能很大或很小，绝对差阈值意义不稳定）

练习里 `03_gradient_chain_rule.py` 就跑这个对照。

### 自检

- 为什么是 $w \leftarrow w - \eta \nabla L$ 而不是 $w \leftarrow w + \eta \nabla L$？
- 链式法则告诉我们：如果 $\partial L / \partial a$ 已知，要算 $\partial L / \partial w_1$，还需要知道什么？
- 上面手算例子里，如果 $h = -1$（ReLU 把它截成 0），$\partial L / \partial w_1$ 等于多少？这意味着什么？

<details markdown="1">
<summary>答案速查</summary>

- 我们要**最小化** loss，梯度指向上山方向，所以减号往下山方向走。加号会让 loss 越来越大

- 还需要 $\partial a / \partial h$（这里 ReLU 的导数）和 $\partial h / \partial w_1$（这里是 $x$）

- $\partial L / \partial w_1 = 0$。意味着 ReLU 把 $h$ 截成 0 后，这条路径"梯度断了"——$w_1$ 这次更新拿不到梯度信号。这就是 **dying ReLU** 问题的根源

</details>

---

## 3. softmax 与交叉熵

§1 §2 给的是**通用工具**（线代 + 微分）。本节把它们用在第一个**具体场景**：分类。

这套组合（梯度、链式法则 + 本节的新概念）是 LLM 训练的核心算子：每个 token 位置都在做 "下一词分类"，词表多大那下一个词就有多少种分类可能。

> **logits**：LLM 的输出层是一个 **长度 = 词表大小 $V$ **的向量（logits）。常见 $V$ 在 3–15 万（GPT-2 是 5 万、LLaMA-2 是 3.2 万、Qwen2.5 是 15 万）。
> 
> **logits 向量 -> 概率选取**：从 $V$ 个候选里挑 1 个 = 一道 $V$ 类的选择题，走的就是 §3 这套 softmax → 概率 → CE（Cross-Entropy，交叉熵）。

### 3.1 为什么需要新概念

来看一个 3 类分类任务的网络：

```
        前两节的工具            |         ← 本节要补的两块 →       |
input ─[Linear/MLP]─→ logits ─[softmax]─→ probs ─[CE with label]─→ loss
                      (3 个                (3 个                   (1 个
                       实数)                概率)                   标量)
```

到 logits 这步用的是前文的矩阵乘 + 链式法则。但**还差两块**：

**缺口 1：logits 不是概率**

Linear 出来的是任意实数，可能是 `[2.3, -1.5, 0.8]`。我们想要的是 "模型认为选取各类别的概率"，即 3 个**正数 + 总和为 1** 的数。需要一个函数把任意实数挤进 $[0, 1]$ 还要总和归一，**softmax** 就是干这个活的。

**缺口 2：分类问题用 MSE 不友好**

回归问题（房价预测之类）用 MSE loss $\tfrac{1}{2}(y - t)^2$ 没问题。但分类问题如果硬套 MSE，会出毛病。

设 3 类，真值是第 1 类（onehot $[0, 1, 0]$，即 `probs[1]` 该接近 1）。看两个错法不同的预测：

```
                类0    类1   类2
模型 A：probs = [0.49, 0.49, 0.02]   ← 在"类0"和"类1（正确）"之间纠结
模型 B：probs = [0.02, 0.49, 0.49]   ← 在"类1（正确）"和"类2"之间纠结

loss MSE_A = ((0.49-0)² + (0.49-1)² + (0.02-0)²) / 3 ≈ 0.167
loss MSE_B = ((0.02-0)² + (0.49-1)² + (0.49-0)²) / 3 ≈ 0.167
```

两个 loss 一样：MSE **看不出两种错法的差异**。

> 即使换成不对称的例子（比如 A 把概率分散在 5 个分类、B 集中错给 1 个分类），MSE 也常常给不出 **"哪种错更糟"** 的合理排序，它把分类问题**当回归对待**，对 "错得不对称" 这件事不敏感。

我们想要一种 loss 满足：

- **正确答案概率高 → loss 小**
- **正确答案概率低 → loss 大**（且越自信地错惩罚越重）
- 梯度形式优雅，方便和 softmax 串在一起求导

这就是**交叉熵**的活。

>
> - 3.2 讲 softmax 怎么补缺口 1
> - 3.3 讲数值稳定 trick
> - 3.4 讲交叉熵怎么补缺口 2
> - 3.5 揭示 softmax+CE 合并求导后梯度优雅到不可思议——而这个优雅的推导，**就是 §2 链式法则在这个具体复合函数上的应用**。

---

### 3.2 softmax：实数 → 概率分布

分类网络最后输出 $C$ 个数（$C$ 是类别数），叫 **logits**，可以是任意实数。但我们想要 "模型认为每个类别的概率"，那便需要把 logits -> [ 正数 & 总和为 1 ]。

softmax 干这事：

\[
\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

**为什么用 $e^{z}$ 而不是直接归一化**？

- $z_i$ 可能为负，直接 $z_i / \sum z_j$ 会出现负概率
- $e^{z}$ 永远 $> 0$，且**单调**：大的更大、小的更小，可放大差距
- $e$ 求导优雅 $\frac{d}{dz}e^z = e^z$（链式法则不爆炸）

**最小数字例子**：

```
z = [2, 1, 0]             ← 3 个类别的 logits

e^z = [e², e¹, e⁰] ≈ [7.39, 2.72, 1.00]
sum = 11.11
softmax = [7.39/11.11, 2.72/11.11, 1.00/11.11]
        ≈ [0.665, 0.245, 0.090]   ← 总和 = 1.0 ✓
```

观察：logits 相邻位置差 1 的时候，softmax 概率的**比值**是 $e \approx 2.7$ 倍（如 $0.665 / 0.245 \approx 2.71$）。**logits 的差距决定概率的比例**。

### 3.3 防溢出 trick：减最大值

logits 大的时候，softmax 计算过程中 $e^z$ 会溢出（`float32` 上 $e^{89}$ 就 overflow）。

观察一个事实：**softmax 减常数等价**。

\[
\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i}/e^c}{\sum_j e^{z_j}/e^c} = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

分子分母同乘 $e^c$ 抵消，结果不变。**取 $c = \max(z)$**：

```
z = [1000, 1001, 1002]    ← 直接 e^z 三个都 inf，结果是 nan/nan

减最大值后：
z' = z - 1002 = [-2, -1, 0]
e^z' = [0.135, 0.368, 1.000]
sum = 1.503
softmax = [0.090, 0.245, 0.665]   ← 数值稳定，结果完全正确
```

**铁律**：手写 softmax 永远先减 max。PyTorch 内置 `F.softmax` 已经做了。

### 3.4 交叉熵：分布之间的"距离"

模型预测 $p$（softmax 输出），真值是 onehot $y$（如 3 类、第 1 类正确就是 $[0, 1, 0]$）。**交叉熵**衡量两者多接近：

\[
\mathrm{CE}(y, p) = -\sum_i y_i \log p_i
\]

onehot 情况下只剩**正确类**那一项有贡献：

\[
\mathrm{CE} = -\log p_{\text{target}}
\]

直觉：**模型给正确答案的概率越高，loss 越小**。

```
3 类分类，真值是第 1 类（onehot = [0, 1, 0]）

模型 A：p = [0.10, 0.85, 0.05]   ← 自信且正确
  CE_A = -log(0.85) ≈ 0.163

模型 B：p = [0.10, 0.55, 0.35]   ← 正确但不太自信
  CE_B = -log(0.55) ≈ 0.598

模型 C：p = [0.10, 0.05, 0.85]   ← 自信但错了
  CE_C = -log(0.05) ≈ 2.996     ← 惩罚极重
```

**注意 C**：模型自信地错，loss 比 B 高 5 倍。这正是我们想要的：交叉熵 **重罚自信的错误**。

### 3.5 softmax + CE 的"组合拳"

理论上你可以：先 softmax 拿 $p$，再算 $-\log p_y$。但**两件事合在一起求导才优雅**：

\[
\frac{\partial \mathrm{CE}}{\partial z_i} = p_i - y_i
\]

**梯度形式简洁到不可思议**：预测概率减真实 onehot。

<details markdown="1">
<summary>为什么是 $p_i - y_i$？最小推导（点开看）</summary>

设真值类别是 $k$（即 $y_k = 1$，其它 $y_j = 0$）。CE 此时退化为：

\[
\mathrm{CE} = -\log p_k = -\log \frac{e^{z_k}}{\sum_j e^{z_j}} = -z_k + \log \sum_j e^{z_j}
\]

对任意 $z_i$ 求偏导：

- $-z_k$ 这一项：$i = k$ 时贡献 $-1$，否则 $0$ → 合起来就是 $-y_i$
- $\log \sum_j e^{z_j}$ 这一项：$\frac{\partial}{\partial z_i} \log \sum_j e^{z_j} = \frac{e^{z_i}}{\sum_j e^{z_j}} = p_i$

合并：

\[
\frac{\partial \mathrm{CE}}{\partial z_i} = -y_i + p_i = p_i - y_i
\]

CE 的 $-\log$ 和 softmax 的 $e^{z}/\sum$ 在求导时**互相抵消**，只剩下"预测减真实"。
</details>

直觉：

- 正确类（$y_i = 1$）：梯度 $p_i - 1$ 是负数 → 增大对应 $z_i$ → 让模型下次更确信这是对的
- 错误类（$y_i = 0$）：梯度 $p_i$ 是正数 → 减小对应 $z_i$ → 让模型下次别那么确信

**为什么 PyTorch `nn.CrossEntropyLoss` 直接吃 logits**？

把 softmax 和 log 合并实现成一个算子（`log_softmax`），既数值稳定（不会先 softmax 出极小值再 log 爆精度），又走简洁梯度路径。**所以你永远不要在 `CrossEntropyLoss` 之前手动 softmax**——会算两次，且数值更差。

```python
# ✗ 错
probs = F.softmax(logits, dim=-1)
loss = F.cross_entropy(probs, target)   # 形状对，但语义错

# ✓ 对
loss = F.cross_entropy(logits, target)  # 直接吃 logits
```

**和 §2 链式法则的呼应**：上面那个简洁梯度不是天上掉的，正是 $\partial L / \partial z = (\partial L / \partial p) \cdot (\partial p / \partial z)$ 在这个特定复合函数上算出来的结果。这种"复杂前向 + 简洁梯度"的优雅结构在深度学习里随处可见——**不是巧合，是有人精心选了组合**。

### 自检

- softmax 减最大值为什么不改变结果？用一行算式说明。
- 真值 onehot = [0, 1, 0]，预测 p = [0.3, 0.4, 0.3]，CE 是多少？
- $\partial\mathrm{CE} / \partial z_i = p_i - y_i$ 这个梯度告诉优化器"该往哪走"——具体怎么走？

<details markdown="1">
<summary>答案速查</summary>

- $\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i}/e^c}{\sum_j e^{z_j}/e^c} = \frac{e^{z_i}}{\sum_j e^{z_j}}$，分子分母同乘 $e^c$ 抵消

- $\mathrm{CE} = -\log(0.4) \approx 0.916$（只有正确类那一项贡献）

- $z_{\text{正确类}}$ 加（梯度为负，按 $w \leftarrow w - \eta \nabla L$ 是加），所有错误类的 $z$ 减。**等价于"把概率从错的类挪到对的类"**

</details>

---

## 4. 练习

落到 `Playground/ch02-math/`，**全部纯 NumPy**，目的是把 PyTorch 帮我们做的事看清楚：

| 脚本 | 内容 |
|---|---|
| `01_vector_matrix.py` | 点积、矩阵乘、形状练习；与 NumPy 内置对照 |
| `02_softmax_cross_entropy.py` | 数值稳定版 softmax + CE，验证梯度 = $p - y$ |
| `03_gradient_chain_rule.py` | 两层网络解析梯度 vs 数值梯度对照 |
| `04_mlp_numpy.py` | **综合 §1–§3 全部内容**：把手算版扩展成可训练的完整分类网络（ch03 PyTorch 版的对照） |

通过标准：每个脚本独立跑通，最后一行打印 `PASS`。

## 思考题（不一定有标准答案，写下来你的想法）

1. 如果 softmax 不减最大值，$z = [1000, 1001, 1002]$ 会发生什么？亲手在 Python 里试试，观察 `naive` 输出的 `nan` 是怎么来的。
2. 为什么 `nn.CrossEntropyLoss` 要求传 logits 而不是 probabilities？传 probabilities 会怎样？
3. 矩阵乘 $(m, k) \times (k, n)$ 的计算量是 $O(mkn)$。一个 $(B, T, D) \times (D, D)$ 的 batched 线性变换（$B$=batch, $T$=序列长, $D$=hidden dim）总共多少次乘加？这就是为什么 LLM 训练要 GPU 的原因。

## 参考资料

- **3Blue1Brown** 神经网络系列 4 集（YouTube / B 站）：梯度、链式法则的几何直觉，**强烈推荐先看再回来读这章**
- **CS231n notes** "Backpropagation, Intuitions"：<https://cs231n.github.io/optimization-2/>
- **PyTorch 文档** `nn.CrossEntropyLoss`：<https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>
- **Distill.pub**：<https://distill.pub/>（深度学习概念可视化解读，进阶选读，可跳过）
