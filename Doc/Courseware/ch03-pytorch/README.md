# ch03 · PyTorch 入门

> ch02 用 NumPy 把"前向 + 反向 + 优化"硬撕了一遍。本章把同一件事**搬到 PyTorch**，看 framework 替我们省了什么、带来了什么新约定。
> 
> 主线：Tensor → autograd → nn.Module → DataLoader → 训练循环 → MNIST（Modified National Institute of Standards and Technology database，手写数字数据集，深度学习入门标配）综合实战。

## 学习目标

1. 能用 PyTorch 表达 ch02 的 MLP，理解 `nn.Module` / `autograd` / `optimizer` 三件套各自的职责
2. 写出标准训练循环模板：`forward → loss → zero_grad → backward → step`
3. 用 `Dataset / DataLoader` 喂数据，跑通 MNIST 分类（3060 上 1 分钟内 ≥ 97% 测试集准确率）

## 前置依赖

- ch01（`get_device()` / uv 命令）
- ch02（softmax / CE / 链式法则；`04_mlp_numpy.py` 是本章 PyTorch 版的对照组）

---

## 1. Tensor：带梯度的多维数组

`torch.Tensor` 概念上 = `np.ndarray` + 三件附加品：

| 属性 | 含义 | NumPy 有吗 |
|---|---|---|
| `device` | 数据所在设备（cpu/cuda/mps） | ✗ |
| `dtype` | 元素类型（float32/float16/int64...） | ✓（但 PyTorch 的 dtype 自成体系） |
| `requires_grad` | 是否参与 autograd 计算图 | ✗ |

### 1.1 创建与设备

```python
import torch
from Echo.shared.device import get_device

device = get_device()                            # cuda → mps → cpu

x = torch.zeros(3, 4)                            # CPU 上的 (3, 4) 全零
x = torch.zeros(3, 4, device=device)             # 直接建在目标设备
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])       # 从 list 建
x = torch.from_numpy(np_array)                   # 与 numpy 共享内存（CPU 上）

x = torch.tensor([1, 2, 3], dtype=torch.float32) # 显式指定 dtype
x = x.float()                                    # 等价转换；.long() / .to(torch.float32) 同理
```

**铁律**：业务代码**禁止**写 `.cuda()` / `.to("cuda")`。一律 `.to(device)`，`device` 来自 `get_device()`。

### 1.2 形状操作三连

ch02 反复强调"形状是第一公民"。PyTorch 提供三个高频形状算子：

```python
x = torch.arange(12)               # shape (12,)
x.view(3, 4)                       # (3, 4)，要求内存连续；不连续时报错
x.reshape(3, 4)                    # 同上但不连续会自动 copy；更稳但偶有性能损失
x.permute(1, 0)                    # 维度重排，不改数据只改 stride
x.transpose(0, 1)                  # 交换两个维度，permute 的两元特例
x.unsqueeze(0)                     # 加一维：(N,) → (1, N)
x.squeeze(0)                       # 去一维：(1, N, 1) → (N, 1)
```

`view` vs `reshape` 是初学常见困惑：**优先 `view`，遇到 stride 报错再换 `reshape`**。

### 1.3 与 NumPy 的桥

```python
t = torch.tensor([1.0, 2.0, 3.0])
a = t.numpy()                      # CPU 上零拷贝；GPU 上必须先 .cpu()
t2 = torch.from_numpy(a)           # 同样零拷贝；改 a 会改 t2
```

GPU tensor 转 numpy 必须 `t.cpu().numpy()`，且如果 `t.requires_grad=True` 还要 `.detach()`：

```python
arr = t.detach().cpu().numpy()     # 三连：脱离计算图 → 搬回 CPU → 转 numpy
```

### 自检

1. `torch.tensor([1, 2, 3])` 和 `torch.tensor([1.0, 2.0, 3.0])` 的 `dtype` 分别是什么？后续做矩阵乘哪个会出错？
2. 一个 `requires_grad=True` 的 GPU tensor 怎么安全地转成 numpy？

<details markdown="1">
<summary>答案速查</summary>

1. 前者 `int64`，后者 `float32`。神经网络权重默认 `float32`，整型 tensor 不能和浮点权重做矩阵乘，会报 `RuntimeError: expected scalar type Float but found Long`

2. `t.detach().cpu().numpy()`。`detach()` 切断计算图避免改 numpy 反向传播污染，`cpu()` 搬回主存，`numpy()` 才合法

</details>

---

## 2. autograd：把 ch02 的链式法则交给框架

ch02 我们手算了 $\partial L / \partial w_1 = \partial L / \partial h \cdot x$。PyTorch 把这事**自动化**：你写 forward，它在背后画一张计算图，`loss.backward()` 沿图反向把梯度填到每个 leaf tensor 的 `.grad` 上。

### 2.1 最小可运行例子

```python
w = torch.tensor(0.5, requires_grad=True)        # 标记"要算梯度"
x = torch.tensor(2.0)                            # 普通 tensor，不参与 autograd
y_true = torch.tensor(3.0)

y = w * x                                         # forward：建图节点 1，y = 0.5·2 = 1
loss = (y - y_true) ** 2                          # forward：建图节点 2，loss = (1-3)² = 4

loss.backward()                                   # 沿图反向，填 w.grad
print(w.grad)                                     # tensor(-8.) ← dL/dw = 2(y-y_true)·x = 2·(1-3)·2
```

> 注：ch02 写的是 $(y-t)^2/2$，那个 $1/2$ 是为了**手推梯度时**抵消平方求导出来的 2，让结果干净（$\partial L/\partial y = y-t$）。ch03 起交给 autograd 自动求导，多不多那个系数都一样能反传，去掉反而让 loss 数值（4）和梯度链路（$-4 \to -8$）更直观，故本章不再带 $1/2$。

**对照 ch02**：你不再手写 `dw = (y-t) * x * 2`，PyTorch 自动算出来。代价是它要在 forward 时**保留中间张量**做反向用—— **所以训练比纯前向（推理）显存占用大几倍**。

### 2.2 三个最易踩的坑

**坑 1：梯度会累加**

```python
loss.backward()                # w.grad = -4.0
loss2.backward()               # w.grad = -4.0 + 新梯度，**不是覆盖**
```

PyTorch 设计成累加是为了支持 grad accumulation（小显存模拟大 batch）。代价是**普通训练每步必须 `optimizer.zero_grad()` 清零**，否则梯度爆炸。

**坑 2：只能对 leaf tensor 拿 `.grad`**

> **leaf tensor**（叶子张量）：直接 `torch.tensor(...)` 创建的、不是由其他 tensor 运算产生的 tensor。模型权重都是 leaf；前向中间结果不是。

```python
w = torch.tensor(0.5, requires_grad=True)        # leaf
y = w * 2                                         # 非 leaf（由运算产生）
loss = y ** 2
loss.backward()
print(w.grad)                                     # ✓ 有值
print(y.grad)                                     # ✗ None + warning
```

中间结果想拿梯度要 `y.retain_grad()`。但 99% 场景你只关心权重梯度，不需要。

**坑 3：推理时关 autograd**

```python
with torch.no_grad():                            # 上下文内不建图，省显存 + 提速
    logits = model(x)
```

或者整个推理函数加 `@torch.inference_mode()` 装饰器（比 `no_grad` 更激进）。

### 自检

1. 为什么训练循环里每步都要 `optimizer.zero_grad()`？删掉会怎样？
2. `with torch.no_grad():` 和 `tensor.detach()` 都能阻止梯度回传，区别在哪？

<details markdown="1">
<summary>答案速查</summary>

1. PyTorch 梯度累加而非覆盖。删掉的话每步梯度都加上历史，等价于学习率被无限放大，loss 立刻爆炸成 nan

2. `no_grad()` 是上下文管理器，**作用域内所有运算都不建图**；`detach()` 只对单个 tensor 切断计算图，作用域外的运算照常建图。前者用于推理整段，后者用于"我想用这个值但不想反传"

</details>

---

## 3. nn.Module：参数管理 + forward 模板

ch02 我们用一个 dict 装 `W1/b1/W2/b2`。PyTorch 把这件事标准化成 `nn.Module`：

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        # nn.Linear 内部就是 ch02 的 W·x + b，权重默认 Kaiming 均匀初始化
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        return self.fc2(h)                       # 返回 logits，不要在这 softmax

model = MLP(784, 128, 10).to(device)
logits = model(x)                                # 直接调 model() 等价于 model.forward(x)
```

### 3.1 nn.Module 替你做了什么

1. **参数自动注册**：所有 `nn.Linear` / `nn.Conv2d` 之类的 submodule，它们的 `weight` / `bias` 自动出现在 `model.parameters()` 里
2. **`.to(device)` 一键搬全家**：所有子参数和 buffer 都跟着搬
3. **`train() / eval()` 切模式**：影响 Dropout、BN（BtachNorm，批归一化） 等"训推不一致"算子（详见 ch04）
4. **`state_dict()` 标准持久化**：保存/加载权重的 lingua franca

```python
for name, p in model.named_parameters():
    print(name, p.shape, p.requires_grad)
# fc1.weight torch.Size([128, 784]) True   ← 注意 (out, in)，ch02 提过的转置约定
# fc1.bias   torch.Size([128]) True
# fc2.weight torch.Size([10, 128]) True
# fc2.bias   torch.Size([10]) True
```

### 3.2 Sequential：堆叠的语法糖

简单串联结构可以省掉自定义类：

```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

**何时用 Sequential，何时写自定义类**：分支 / 残差 / 多输入 → 自定义类；纯线性串联 → Sequential。LLM 里 Transformer block 一定是自定义类（残差 + 注意力 + FFN 多路）。

> **顺带一提**：loss 函数也是 `nn.Module`，常用的有 `nn.CrossEntropyLoss`（分类）、`nn.MSELoss`（回归），用法 `loss = loss_fn(pred, target)`。§5 训练循环就用它。

### 3.3 保存与加载

```python
torch.save(model.state_dict(), "ckpt.pt")        # 只存权重 dict（推荐）

model2 = MLP(784, 128, 10)                       # 重建结构
model2.load_state_dict(torch.load("ckpt.pt"))    # 灌权重
```

**禁止** `torch.save(model, ...)`：会序列化整个类定义，跨版本/跨机器加载常出错。

**state_dict 的三种用法**（本章只演示第 1 种，后两种留到 M5 部署章）：

- 训练中途存档：连 `optimizer.state_dict()` 一起存，崩了能续训
- 最终权重发布：只存 `model.state_dict()`，对应 HuggingFace 上的 `pytorch_model.bin` / `model.safetensors`
- 跨环境部署：再转 ONNX / TorchScript / GGUF，目标是脱离 PyTorch 运行

> 配套演示见 `Playground/ch03-pytorch/03_nn_module.py` 末尾：训完模型 → 存权重 → 重建空壳 → 加载 → 断言输出一致。

### 自检

1. `nn.Linear(784, 128)` 的 `weight.shape` 是什么？为什么不是 `(784, 128)`？
2. 写 `forward()` 时，最后一层之后**不要**手动 `softmax`——为什么？

<details markdown="1">
<summary>答案速查</summary>

1. `(128, 784)`，即 `(out, in)`。PyTorch 内部算的是 $y = x W^\top + b$，行优先访存友好。ch02 已埋过这个伏笔

2. `nn.CrossEntropyLoss` 内部已经合并了 `log_softmax + NLL`，外面再 softmax 等于做两遍且数值更差。ch02 §3.5 详述

</details>

---

## 4. Dataset 与 DataLoader：把数据喂进模型

### 4.1 为什么需要 batching

ch02 的 `04_mlp_numpy.py` 一次性把 200 个样本全塞进网络（"全 batch"）。真实场景：

- MNIST 6 万张图、ImageNet 128 万张、LLM 语料几十亿 token
- 全塞 → 显存爆炸；一条一条 → 梯度噪声大、GPU 利用率低

折中：**mini-batch**，每次取 32 / 64 / 128 个样本算梯度。

> **MNIST 的形状约定**（§5/`05_mnist_mlp.py` 会用到）：`torchvision.datasets.MNIST` 配 `transforms.ToTensor()` 拿到的单样本是 `(1, 28, 28)` 的 float tensor，DataLoader 堆完 batch 是 `(B, 1, 28, 28)`。喂 MLP 前需要 `x = x.view(B, -1)` 拍平成 `(B, 784)`；喂 CNN（convolutional neural network, 卷积神经网络） 则保留四维。

### 4.2 PyTorch 抽象

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]          # 返回单样本，DataLoader 负责堆 batch

loader = DataLoader(
    MyDataset(x, y),
    batch_size=64,
    shuffle=True,                                # 训练集每 epoch 重新打乱
    num_workers=0,                               # Win 上建议 0；Linux/Mac 可调高
    pin_memory=True,                             # GPU 训练加速 host→device 传输
)

for batch_x, batch_y in loader:                  # 每次拿 (64, ...) 的 batch
    batch_x = batch_x.to(device, non_blocking=True)
    batch_y = batch_y.to(device, non_blocking=True)
    ...
```

### 4.3 Windows 的 `num_workers` 坑

Win 上 `num_workers > 0` + 没用 `if __name__ == "__main__":` 守卫 → multiprocessing 会**无限递归 fork**。两条规避：

1. 教学代码 / 单文件脚本：`num_workers=0`，省心
2. 真要并行：脚本入口必须 `if __name__ == "__main__": main()`

Mac/Linux 用 fork（不 spawn），没这问题。

### 自检

1. `shuffle=True` 为什么只对训练集设？验证/测试集设了会怎样？
2. `pin_memory=True` 在纯 CPU 训练时有用吗？

<details markdown="1">
<summary>答案速查</summary>

1. 训练打乱避免模型记住样本顺序、提升泛化。验证/测试集只评指标不学习，打乱不影响结果但会让"按 batch 看错例"等调试动作不可复现，所以一般 `False`

2. 没用。`pin_memory` 把数据钉在不可换页内存里，加速 CPU→GPU DMA 传输，纯 CPU 训练里目标设备就是 CPU 自己，多此一举且略费内存

</details>

---

## 5. 训练循环模板（背下来）

```python
model = MLP(...).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    # ---- 训练 ----
    model.train()                                # 切训练模式（影响 Dropout/BN；本章 MLP 没这俩，写上是肌肉记忆，ch04 详解）
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)                        # forward
        loss = loss_fn(logits, y)                # CE 直接吃 logits（ch02 §3.5）

        optimizer.zero_grad()                    # 必须，否则梯度累加（§2.2 坑 1）
        loss.backward()                          # autograd 反向
        optimizer.step()                         # 按梯度更新参数

    # ---- 验证 ----
    model.eval()                                 # 切推理模式
    correct = 0
    with torch.no_grad():                        # 关 autograd 省显存
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=-1)
            correct += (pred == y).sum().item()  # .item() 把 0 维 tensor 转 Python int，避免循环里堆一串小 tensor
    print(f"epoch {epoch}  val_acc={correct / len(val_set):.4f}")
```

**这五步就是后续 echo-mini Pretrain / SFT 的骨架**，差别只是 loss、数据、模型变复杂。

> **`optimizer` 是什么**：持有 `model.parameters()` 的引用 + 实现 `step()`。`step()` 读每个参数的 `.grad` 按各自规则更新参数；`zero_grad()` 清空 `.grad`。所以三件套的分工是：`backward()` 算梯度填 `.grad` → `optimizer.step()` 用 `.grad` 改参数 → `zero_grad()` 清场准备下一轮。
>
> **`non_blocking=True` 提示**：配合 `pin_memory=True` 让 host→device 传输与 GPU 计算重叠，单独写没意义。教学脚本里写或不写都行。

### 自检

1. 把 `optimizer.zero_grad()` 挪到 `optimizer.step()` 之后会怎样？
2. 验证阶段为什么既要 `model.eval()` 又要 `torch.no_grad()`？只用其中一个行不行？

<details markdown="1">
<summary>答案速查</summary>

1. 第一个 batch 没问题；第二个 batch 的 `loss.backward()` 会把梯度加在第一个 batch 已 step 过的"残留梯度"上，等价于隐式 grad accumulation 但你没意识到，loss 会偏高/震荡。**位置写在 `backward()` 之前最稳**

2. 不行，两者管的事不同。`eval()` 切 Dropout/BN 等算子的训推模式；`no_grad()` 关计算图、省显存提速。少前者会导致 Dropout 仍随机丢神经元、BN 用 batch 统计而非 running 统计，指标偏差；少后者只是慢且费显存，结果还是对的

</details>

---

## 6. 练习

落到 `Playground/ch03-pytorch/`：

| 脚本 | 内容 |
|---|---|
| `01_tensor_basics.py` | Tensor 创建 / 形状 / 设备 / 与 numpy 互操作 |
| `02_autograd.py` | `requires_grad` / `backward()`，与 ch02 解析梯度对照 |
| `03_nn_module.py` | 用 `nn.Module` 重写 `04_mlp_numpy.py`，对比代码量 |
| `04_dataloader.py` | 自定义 `Dataset` + `DataLoader`，演示 batch / shuffle |
| `05_mnist_mlp.py` | **综合实战**：MNIST 分类，~3 epoch 跑到 97%+ |

跑法：

```bash
uv run python Playground/ch03-pytorch/01_tensor_basics.py
# ... 依次 02 ~ 05
```

通过标准：每个脚本独立跑通，最后打印 `PASS`。`05_mnist_mlp.py` 首次运行会自动下载 MNIST 到 `./data/mnist/`（约 11MB，已被 `.gitignore`），3060 上约 30 秒、Mac M 系列约 1 分钟、纯 CPU 约 2 分钟。

## 思考题

1. ch02 的 `04_mlp_numpy.py` 和本章 `03_nn_module.py` 行数大约比例多少？PyTorch 省的主要是哪一块（数据 / forward / 反向 / 优化）？
2. `model.parameters()` 返回的是生成器，传给 `Adam(...)` 后 Adam 怎么知道哪些参数该更新？如果新增一个 `nn.Parameter` 但**不**注册到 `nn.Module` 上，会发生什么？
3. MNIST 用 MLP 能到 97%+，但用 CNN 能到 99%+。差距来自哪里？这给我们什么启示——为什么 LLM 不用 CNN？

## 参考资料

- **PyTorch 官方 60 分钟入门**：<https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html>
- **PyTorch autograd 机制**：<https://pytorch.org/docs/stable/notes/autograd.html>
- **`nn.Module` 源码导读**（进阶）：<https://pytorch.org/docs/stable/generated/torch.nn.Module.html>
