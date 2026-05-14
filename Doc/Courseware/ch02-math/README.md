# ch02 · 必要数学（浅层）

> 目标是"够用"，不是"完备"。本章只覆盖 LLM 训练实战会反复出现的几个数学点。
> 每个公式后面都跟一个小数字例子，看不懂公式时**先看例子再回头看公式**。

## 学习目标

1. 看到向量/矩阵运算时知道**形状**怎么对齐、**含义**是什么
2. 理解链式法则在反向传播里的角色，能手算两层网络的梯度
3. 理解 softmax + 交叉熵为什么是分类问题的"标配组合"，且数值稳定要怎么做

## 前置依赖

- ch01 课件 + 练习
- 高中 / 大一线性代数残留记忆即可

---

## 1. 向量、矩阵、点积

### 1.1 形状是第一公民

写每一行 tensor 操作前，先在脑里/注释里写出 shape：

```python
# x: (batch=2, dim_in=3)
# w: (dim_in=3, dim_out=4)
# y = x @ w  -> (batch=2, dim_out=4)
```

形状对了，语义大概率对；形状错了 PyTorch 立刻报：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 4x3)
```

**这是初学最高频的报错，没有之一**。养成习惯：写 tensor 操作时**先注释 shape**，写完代码再核对一遍。

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

**几何视角**：$\|a\|\|b\|\cos\theta$，衡量"方向有多像"。

- 完全同向 $\theta = 0$ → $\cos\theta = 1$ → 点积最大
- 垂直 $\theta = 90°$ → $\cos\theta = 0$ → 点积为 0
- 反向 $\theta = 180°$ → $\cos\theta = -1$ → 点积最小（负）

```
b
↑
│   .a       ← a 和 b 都在第一象限，θ 小，点积大
│  /
│ /
└─────→
```

**这就是 attention 的本质**：算 query 和每个 key 的方向相似度。LLM 里反复出现的 $QK^\top$，每一项就是某个 token 的 query 向量和某个 token 的 key 向量的点积——值大表示"应该多关注这个位置"。

### 1.3 矩阵乘 = 批量点积

\[
(AB)_{ij} = \sum_k A_{ik} B_{kj}
\]

意思：$AB$ 第 $i$ 行第 $j$ 列的元素 = $A$ **第 $i$ 行**和 $B$ **第 $j$ 列**的点积。

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

**Linear 层就是矩阵乘**：`nn.Linear(in_features=3, out_features=4)` 概念上等价于乘一个 $(3, 4)$ 矩阵，输入 $(N, 3)$ 出来 $(N, 4)$。

> 注：PyTorch 实际把权重存为 $(\text{out}, \text{in}) = (4, 3)$，前向算的是 $y = x W^\top + b$。`print(linear.weight.shape)` 看到的是 $(4, 3)$，别慌——和上面"概念形状"互为转置，乘出来结果一样。这个存储约定是为了行优先访存友好。

### 自检

- $A$ 是 $(5, 8)$，$B$ 是 $(8, 3)$，$AB$ 形状是？$BA$ 能算吗？
- 点积 $a \cdot b = 0$ 说明 $a$ 和 $b$ 啥关系？

---

## 2. 梯度与链式法则

### 2.1 梯度：多变量函数的"上山方向"

一元函数 $f(x)$ 的导数 $f'(x)$ 表示"$x$ 增加一点，$f$ 增加多少"。

多元函数 $L(w_1, w_2, \ldots, w_d)$ 的**梯度** $\nabla L$ 是个 $d$ 维向量，每个分量是对应方向的偏导：

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

**梯度下降**：训练就是下山。我们要最小化 loss $L$，所以**沿 -∇L 方向**走一小步：

\[
w \leftarrow w - \eta \cdot \nabla_w L
\]

$\eta$ 是学习率，控制步子大小。太大冲过头（loss 震荡甚至爆炸），太小走得慢。

### 2.2 链式法则：反向传播的灵魂

链式法则解决的问题：**复合函数**怎么求导。

设 $L = f(g(w))$，其中 $w$ 经过 $g$ 变成中间量 $u = g(w)$，再经过 $f$ 变成 $L$。问 $L$ 对 $w$ 怎么变？

\[
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial w}
\]

**口语化**：变化沿计算图**一节一节传递**，每一节乘上当地的"放大倍率"。

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

**反向传播 = 沿计算图反向应用链式法则**。从 loss 出发，一层一层把梯度"传"回每个参数。

### 2.3 两层网络手算（关键节）

这是后面所有反向传播题的母题。**强烈建议拿张纸和我一起算一遍**。

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
  L = (3-5)² / 2 = 2

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

PyTorch 的 `loss.backward()` + `optimizer.step()` 自动做的就是这件事。但**至少手算一次**，你才会真懂"梯度是什么、它怎么传"。

### 2.4 数值梯度：调试的救命稻草

自己写反向传播容易出错。**用数值梯度做 sanity check**：

\[
\frac{\partial L}{\partial w_i} \approx \frac{L(w + \epsilon e_i) - L(w - \epsilon e_i)}{2\epsilon}
\]

意思：把 $w_i$ 微微抬一点（加 $\epsilon$）算 $L$，再微微压一点算 $L$，差除以 $2\epsilon$ 就是斜率近似。

- $\epsilon$ 取 $10^{-5}$ 量级（太小被浮点精度吞掉，太大近似不准）
- 和你手算的解析梯度比较，最大绝对差 $< 10^{-6}$ → 实现正确

练习里 `gradient_chain_rule.py` 就跑这个对照。

### 自检

- 为什么是 $w \leftarrow w - \eta \nabla L$ 而不是 $w \leftarrow w + \eta \nabla L$？
- 链式法则告诉我们：如果 $\partial L / \partial a$ 已知，要算 $\partial L / \partial w_1$，还需要知道什么？
- 上面手算例子里，如果 $h = -1$（ReLU 把它截成 0），$\partial L / \partial w_1$ 等于多少？这意味着什么？

---

## 3. softmax 与交叉熵

### 3.1 softmax：实数 → 概率分布

分类网络最后输出 $C$ 个数（$C$ 是类别数），叫 **logits**，可以是任意实数。但我们想要"模型认为每个类别的概率"——需要把 logits 变成正数 + 总和为 1。

softmax 干这事：

\[
\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

**为什么用 $e^{z}$ 而不是直接归一化**？

- $z_i$ 可能为负，直接 $z_i / \sum z_j$ 会出现负概率
- $e^{z}$ 永远 $> 0$，且**单调**——大的更大、小的更小（"放大差距"）
- $e$ 求导优雅 $\frac{d}{dz}e^z = e^z$（链式法则不爆炸）

**最小数字例子**：

```
z = [2, 1, 0]             ← 3 个类别的 logits

e^z = [e², e¹, e⁰] ≈ [7.39, 2.72, 1.00]
sum = 11.11
softmax = [7.39/11.11, 2.72/11.11, 1.00/11.11]
        ≈ [0.665, 0.245, 0.090]   ← 总和 = 1.0 ✓
```

观察：logits 差 1 的时候，softmax 概率差大概是 $e \approx 2.7$ 倍。**logits 的差距决定概率的比例**。

### 3.2 数值稳定 trick：减最大值

logits 大的时候 $e^z$ 会溢出（`float32` 上 $e^{89}$ 就 overflow）。

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

### 3.3 交叉熵：分布之间的"距离"

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
  CE_C = -log(0.05) ≈ 2.996      ← 惩罚极重
```

**注意 C**：模型自信地错，loss 比 B 高 5 倍。这正是我们想要的——交叉熵**重罚自信的错误**。

### 3.4 softmax + CE 的"组合拳"

理论上你可以："先 softmax 拿 $p$，再算 $-\log p_y$"。但**两件事合在一起求导才优雅**：

\[
\frac{\partial \mathrm{CE}}{\partial z_i} = p_i - y_i
\]

**梯度形式简洁到不可思议**：预测概率减真实 onehot。

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

### 自检

- softmax 减最大值为什么不改变结果？用一行算式说明。
- 真值 onehot = [0, 1, 0]，预测 p = [0.3, 0.4, 0.3]，CE 是多少？
- $\partial\mathrm{CE} / \partial z_i = p_i - y_i$ 这个梯度告诉优化器"该往哪走"——具体怎么走？

---

## 4. 练习

落到 `Playground/ch02-math/`，**全部纯 NumPy**，目的是把 PyTorch 帮我们做的事看清楚：

| 脚本 | 内容 |
|---|---|
| `vector_matrix.py` | 点积、矩阵乘、形状练习；与 NumPy 内置对照 |
| `softmax_cross_entropy.py` | 数值稳定版 softmax + CE，验证梯度 = $p - y$ |
| `gradient_chain_rule.py` | 两层网络解析梯度 vs 数值梯度对照 |
| `mlp_numpy.py` | 完整两层 MLP 在小合成数据集上训练（ch03 PyTorch 版的对照） |

通过标准：每个脚本独立跑通，最后一行打印 `PASS`。

## 思考题（不一定有标准答案，写下来你的想法）

1. 如果 softmax 不减最大值，$z = [1000, 1001, 1002]$ 会发生什么？亲手在 Python 里试试，观察 `naive` 输出的 `nan` 是怎么来的。
2. 为什么 `nn.CrossEntropyLoss` 要求传 logits 而不是 probabilities？传 probabilities 会怎样？
3. 矩阵乘 $(m, k) \times (k, n)$ 的计算量是 $O(mkn)$。一个 $(B, T, D) \times (D, D)$ 的 batched 线性变换（$B$=batch, $T$=序列长, $D$=hidden dim）总共多少次乘加？这就是为什么 LLM 训练要 GPU 的原因。

## 参考资料

- **3Blue1Brown** 神经网络系列 4 集（YouTube / B 站）：梯度、链式法则的几何直觉，**强烈推荐先看再回来读这章**
- **CS231n notes** "Backpropagation, Intuitions"：<https://cs231n.github.io/optimization-2/>
- **PyTorch 文档** `nn.CrossEntropyLoss`：<https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>
- **Distill** "Visualizing the Hessian"（进阶可读）：<https://distill.pub/>
