# ch02 · 必要数学（浅层）

> 目标是"够用"，不是"完备"。本章只覆盖 LLM 训练实战会反复出现的几个数学点。

## 学习目标

1. 看到向量/矩阵运算时知道**形状**怎么对齐、**含义**是什么
2. 理解链式法则在反向传播里的角色，能手算两层网络的梯度
3. 理解 softmax + 交叉熵为什么是分类问题的"标配组合"，且数值稳定要怎么做

## 前置依赖

- ch01 课件 + 练习
- 高中 / 大一线性代数残留记忆即可

## 1. 向量、矩阵、点积

### 1.1 形状是第一公民

写每一行带 tensor 操作的代码前，先在脑里/注释里写出 shape：

```python
# x: (batch, dim_in)
# w: (dim_in, dim_out)
# y = x @ w  -> (batch, dim_out)
```

形状对了，语义大概率对；形状错了，PyTorch 报 `RuntimeError: mat1 and mat2 shapes cannot be multiplied`。

### 1.2 点积的两种几何理解

向量 $a, b \in \mathbb{R}^d$ 的点积 $a \cdot b = \sum_i a_i b_i$：

- **代数视角**：逐元素相乘求和
- **几何视角**：$a \cdot b = \|a\| \|b\| \cos\theta$，衡量"方向相似度"

LLM 里 attention 的 $QK^\top$ 就是把"query 和每个 key 的相似度"批量算出来。

### 1.3 矩阵乘 = 批量点积

$(AB)_{ij} = \sum_k A_{ik} B_{kj}$，即 $A$ 第 $i$ 行与 $B$ 第 $j$ 列的点积。

记忆口诀："**内维相同，外维保留**"：$(m, k) @ (k, n) \to (m, n)$。

## 2. 梯度与链式法则

### 2.1 梯度是什么

标量函数 $L: \mathbb{R}^d \to \mathbb{R}$ 在点 $w$ 处的梯度 $\nabla_w L$ 是个 $d$ 维向量，方向指向 $L$ 增长最快方向，模长是增长率。**梯度下降**：往反方向走一步，$w \leftarrow w - \eta \nabla_w L$。

### 2.2 链式法则（标量版）

$L = f(g(w))$，则

\[
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial g} \cdot \frac{\partial g}{\partial w}
\]

反向传播本质就是把这个法则**沿计算图反向**做一遍。

### 2.3 两层网络手算示例

```text
x → [linear w1] → h → [relu] → a → [linear w2] → y → [mse with target t] → L
```

设：

- $h = w_1 x$，$a = \mathrm{relu}(h)$，$y = w_2 a$，$L = \tfrac{1}{2}(y - t)^2$

反向：

\[
\frac{\partial L}{\partial y} = y - t,\quad
\frac{\partial L}{\partial w_2} = (y - t)\,a,\quad
\frac{\partial L}{\partial a} = (y - t)\,w_2
\]

\[
\frac{\partial L}{\partial h} = \frac{\partial L}{\partial a}\,\mathbb{1}[h > 0],\quad
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial h}\,x
\]

PyTorch `loss.backward()` 帮你做的就是这件事，但**至少手算一次**才知道为什么 ReLU 在 $h=0$ 处梯度需要约定（一般取 0 或 1，影响极小）。

### 2.4 数值梯度做 sanity check

调试自定义反向传播必备：

\[
\frac{\partial L}{\partial w_i} \approx \frac{L(w + \epsilon e_i) - L(w - \epsilon e_i)}{2\epsilon}
\]

$\epsilon$ 取 $10^{-5}$ 量级。和解析梯度比较，相对误差 $< 10^{-5}$ 即认为实现正确。

## 3. softmax 与交叉熵

### 3.1 softmax：把任意实数变成概率分布

\[
\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

性质：每个分量 $\in (0, 1)$，总和为 1。

**数值稳定 trick**：$z_i$ 大时 $e^{z_i}$ 溢出。先减最大值不改变结果：

\[
\mathrm{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
\]

### 3.2 交叉熵：分布之间的距离

真实标签 onehot 为 $y$，预测分布为 $p = \mathrm{softmax}(z)$。交叉熵：

\[
\mathrm{CE}(y, p) = -\sum_i y_i \log p_i
\]

onehot 情况下退化成 $-\log p_{\text{target}}$，即"模型给正确类别多大概率"。

### 3.3 为什么 softmax + CE 是组合拳

直接对 $\log p_{\text{target}}$ 求梯度会因 $\log$ + $\exp$ 同时出现而数值不稳。把两步合并求导：

\[
\frac{\partial \mathrm{CE}}{\partial z_i} = p_i - y_i
\]

**梯度形式极其简洁**：预测概率减真实 onehot。这也是 PyTorch `nn.CrossEntropyLoss` 内部直接吃 logits（不要先 softmax）的原因。

## 4. 练习

落到 `Playground/ch02-math/`，全部纯 NumPy 实现：

| 脚本 | 内容 |
|---|---|
| `vector_matrix.py` | 点积、矩阵乘、形状练习；与 NumPy 内置对照 |
| `softmax_cross_entropy.py` | 手写 softmax（数值稳定版）+ CE，验证梯度 = $p - y$ |
| `gradient_chain_rule.py` | 两层网络解析梯度 vs 数值梯度对照 |
| `mlp_numpy.py` | 完整两层 MLP 在小合成数据集上训练（ch03 PyTorch 版的对照组） |

## 思考题

1. 如果 softmax 不减最大值，$z = [1000, 1001, 1002]$ 会发生什么？亲手试试。
2. 为什么 `nn.CrossEntropyLoss` 要求传 logits 而不是 probabilities？传 probabilities 会怎样？

## 参考资料

- 3Blue1Brown《深度学习》系列前 4 集（向量、矩阵、链式法则的几何直觉）
- CS231n notes: <https://cs231n.github.io/optimization-2/>
- PyTorch 文档 `nn.CrossEntropyLoss`：<https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>
