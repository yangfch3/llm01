# Playground/ch02-math

配套课件：[`Doc/Courseware/ch02-math/README.md`](../../Doc/Courseware/ch02-math/README.md)

全部纯 NumPy，不依赖 PyTorch，目的是把 ch03 起 PyTorch 帮我们做的事看清楚。

| 脚本 | 内容 |
|---|---|
| `01_vector_matrix.py` | 手写点积、矩阵乘 vs NumPy 内置；形状错误演示 |
| `02_softmax_cross_entropy.py` | 数值稳定 softmax + CE；解析梯度 vs 数值梯度 |
| `03_gradient_chain_rule.py` | 两层网络解析反向 vs 数值梯度对照 |
| `04_mlp_numpy.py` | 纯 NumPy 两层 MLP，过拟合二维双高斯团（ch03 PyTorch 版的对照） |

## 通过标准

每个脚本独立跑通，最后一行打印 `PASS`。

```bash
uv run python Playground/ch02-math/01_vector_matrix.py
uv run python Playground/ch02-math/02_softmax_cross_entropy.py
uv run python Playground/ch02-math/03_gradient_chain_rule.py
uv run python Playground/ch02-math/04_mlp_numpy.py
```
