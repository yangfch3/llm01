# ch03 · PyTorch 入门

> ch02 用 NumPy 把"前向 + 反向 + 优化"硬撕了一遍。本章把同一件事**搬到 PyTorch**，看 framework 替我们省了什么、带来了什么新约定。
> 
> 主线：Tensor → autograd → nn.Module → DataLoader → 训练循环 → MNIST（Modified National Institute of Standards and Technology database，手写数字数据集，深度学习入门标配）综合实战。

## 学习目标

1. 能用 PyTorch 表达 ch02 的 MLP，理解 `nn.Module` / `autograd` / `optimizer` 三件套各自的职责
2. 写出标准训练循环模板：`forward → loss → zero_grad → backward → step`
3. 用 `Dataset / DataLoader` 喂数据，跑通 MNIST 分类（3060 12GB 上 1 分钟内 ≥ 97% 测试集准确率）

## 前置依赖

- ch01（`get_device()` / uv 命令）
- ch02（softmax / CE / 链式法则；`04_mlp_numpy.py` 是本章 PyTorch 版的对照组）

---

## 1. Tensor：带梯度的多维数组

**tensor（张量）= 带梯度等信息的多维数组**，是 PyTorch 里所有数据的统一形式。维度数（叫 "阶 / rank"）从低到高：

- 0 维：标量（一个数），如 `loss = 0.83`
- 1 维：向量（一行数），如词向量 `[0.1, -0.2, 0.5, ...]`
- 2 维：矩阵（行 × 列），如一批样本 `(batch, dim)`
- 3 维及以上：高维 tensor，如一批序列 `(batch, seq_len, dim)`、一批图像 `(batch, channel, H, W)`

LLM 里的中间结果几乎都是 3D/4D tensor，所以 ch02 已经学过的 **"形状"** 会成为你 debug 时最常看的东西。

`torch.Tensor` 概念简单理解：NumPy `.ndarray` + 下表三件附加品

| 属性 | 含义 | NumPy 有吗 |
|---|---|---|
| `device` | 数据所在设备（cpu/cuda/mps） | ✗ |
| `dtype` | 元素类型（float32/float16/int64...） | ✓（但 PyTorch 的 dtype 自成体系） |
| `requires_grad` | 是否参与 autograd 计算图 | ✗ |

### 1.1 创建与设备

tensor 必须绑一个设备：CPU tensor 数据在主存，GPU tensor 数据在显存（独立物理芯片）。算子（矩阵乘等）是**设备特定 kernel**：CUDA kernel 只读显存、CPU kernel 只读主存，没法也不应该同时摸两块物理内存的数据。

不同 device 类型的 tensor 其实可以使用 `.to(device)` 互相转换，转换后到同一硬件类型上即可进行算子操作。不在同一硬件类型上进行运算 PyTorch 会抛出 `Expected all tensors to be on the same device` 异常。

所以建 tensor 时就得想清楚去哪。`get_device()` 在 ch01 已统一出口（cuda → mps → cpu），业务代码只认它返回的 `device`，不轻易写死 `"cuda"`/`"cpu"`/`"mps"`。

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

### 1.2 shape 与 stride

ch02 反复强调 "形状是第一公民"，**shape** 是逻辑视图层面的概念，而 tensor 在内存中其实是一维连续数组，需要一个叫 **stride** 的步长信息告知 PyTorch 怎么从逻辑坐标 `(i, j, k)` 算出一维数组中对应的下标。

```
逻辑视图：
[[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11]]

内存里：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]   ← 一维连续

stride = (4, 1)
          ↑  ↑
          │  └─ 列方向走一步，内存跳 1 个元素（next col = 下一个）
          └──── 行方向走一步，内存跳 4 个元素（next row = 跳过一整行）
```

不难想到：不同长度的一维数组 + 不同的 stride 可以灵活组合为不同的 shape。

PyTorch 形状算子按职责分三类 —— 变形、换轴、增删维，每类挑一个常用的就够日常 debug：

```python
x = torch.arange(12)               # shape (12,)

# 变形：元素总数不变，重排成新 shape
x.view(3, 4)                       # (3, 4)，要求内存连续；不连续时报错
x.reshape(3, 4)                    # 同上但不连续会自动 copy；更稳但偶有性能损失

# 换轴：不改数据只改 stride，维度重新排列
x2 = torch.zeros(2, 3, 4)
x2.permute(2, 0, 1)                # (2,3,4) → (4,2,3)，任意维度重排
x2.transpose(0, 1)                 # 交换两个维度，permute 的两元特例

# 增删维：长度为 1 的维度加/去
x.unsqueeze(0)                     # 加一维：(N,) → (1, N)
torch.zeros(1, 5, 1).squeeze()     # 去掉所有长度 1 的维：(1,5,1) → (5,)
```

`view` vs `reshape` 是初学常见困惑：**优先 `view`，遇到 stride 报错再换 `reshape`**。

### 1.3 PyTorch 与 NumPy 的桥

实战中常需要 numpy ↔ tensor 互转：matplotlib 可视化、sklearn metric、读写 `.npy`、对照 ch02 numpy 实现 debug。

NumPy 数组永远在 CPU 主存，而 PyTorch 的 tensor：

- 可能在 CPU 主存
- 也可能在 GPU 显存
- `requires_grad=True` 时还挂在 autograd 计算图上

对都在 CPU 的数据，下面两个接口零拷贝（共享同一块内存，改一边另一边跟着变）：

```python
t = torch.tensor([1.0, 2.0, 3.0])
a = t.numpy()                      # tensor → numpy，CPU 上零拷贝
t2 = torch.from_numpy(a)           # numpy → tensor，同样零拷贝；改 a 会改 t2
```

但如果 `t` 的 `requires_grad=True`，零拷贝拿到的 `a` 一旦做 in-place 写（如 `a[0] = 999`、`a += 1`），就会绕过 autograd 直接改掉计算图里记录的输入数据，导致后续 `backward()` 静默算错梯度。所以 PyTorch 直接拦截，要求先 `.detach()` 切断计算图再 `.numpy()`。

加上训练/推理时 tensor 经常在 GPU 上（numpy 读不到显存，必须先 `.cpu()`），实战里默认用兼容所有情况的三连写法：

```python
# 三连：脱离计算图 → 搬回 CPU → 转 numpy
arr = t.detach().cpu().numpy()
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

## 2. autograd：链式法则交给框架

ch02 我们手算了 $\partial L / \partial w_1 = \partial L / \partial h \cdot x$。PyTorch 把这事**自动化**：你写 forward，它在背后画一张**计算图**（DAG），`loss.backward()` 沿图反向把梯度算出来。

计算图里的张量分两类，搞清楚区别 autograd 才不会踩坑：

- **leaf tensor**（叶子）：用户直接创建的张量，如 `w = torch.tensor(0.5, requires_grad=True)`。**模型权重都是 leaf**。反向后梯度填到 `w.grad`，优化器读这里更新参数。
- **中间张量**（非 leaf）：由运算产生的张量，如 `y = w * x`、`loss = (y-t)**2`。它们是反向链式法则的过路节点，PyTorch 把它们钩在图上、**需等 backward 用完才释放**，默认不保留 `.grad`。

一句话：leaf 是你关心的 "参数端点"，中间张量是计算的 "过路节点"。

### 2.1 最小可运行例子

```python
w = torch.tensor(0.5, requires_grad=True)        # leaf，标记"要算梯度"
x = torch.tensor(2.0)                            # leaf，但 requires_grad=False，不参与 autograd
y_true = torch.tensor(3.0)

y = w * x                                         # 中间张量，y = 0.5·2 = 1
loss = (y - y_true) ** 2                          # 中间张量，loss = (1-3)² = 4

loss.backward()                                   # 沿图反向，填 w.grad
print(w.grad)                                     # tensor(-8.) ← dL/dw = 2(y-y_true)·x = 2·(1-3)·2
```

**对照 ch02**：

- 去掉了 ch02 方便直观手算求导的 MSE 系数 $1/2$，现在起交给 autograd 自动求导
- 你不再手写 `dw = (y-t) * x * 2`，PyTorch 自动算出来。代价是只要 `requires_grad=True` 链路上的运算，PyTorch 就要在 forward 时把每个算子的输入张量**钩在计算图上**等 backward 用——**所以训练比纯前向（推理）显存占用大几倍**。

### 2.2 三个需牢记的点

**梯度会累加**

```python
loss1.backward()
print(w.grad)                  # tensor(-8.)
loss2.backward()               # 不是覆盖，而是 w.grad += 新梯度
print(w.grad)                  # tensor(-16.)（假设新梯度也是 -8）
```

所以普通训练每步必须 `optimizer.zero_grad()` 清零，否则梯度越积越大、loss 直接飞掉。

> 为啥设计成累加而不是覆盖？为了支持 **grad accumulation**：显存装不下大 batch 时，把一个大 batch 拆成 N 个小 batch，每个小 batch `backward()` 后**不清零**，攒够 N 步再 `step()`，等价于用大 batch 训练。这是大模型训练的常用技巧，ch04 会用到。

**只能对 leaf tensor 拿 `.grad`**

```python
w = torch.tensor(0.5, requires_grad=True)        # leaf
y = w * 2                                         # 中间张量
loss = y ** 2
loss.backward()
print(w.grad)                                     # ✓ 有值
print(y.grad)                                     # ✗ None + warning
```

中间结果想拿梯度要 `y.retain_grad()`。但 99% 场景你只关心权重梯度（leaf tensor 梯度），不需要。

**推理时关 autograd**

训练时需 forward 和 backward，但推理时权重已固定，只需 forward 即可。

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

## 3. nn.Module：模块化封装

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

### 3.1 关于 weight.shape

对于上方的示例 MLP，我们如果执行下方的调试代码，会看到如注释的 log：

```python
for name, p in model.named_parameters():
    print(name, p.shape, p.requires_grad)
# fc1.weight torch.Size([128, 784]) True   ← (out, in)，见上方说明
# fc1.bias   torch.Size([128]) True
# fc2.weight torch.Size([10, 128]) True
# fc2.bias   torch.Size([10]) True
```

观察到：`weight.shape` 是 `(out, in)` 而不是 `(in, out)`**，为什么呢？

ch02 §1.3 把 Linear 层概念上等价于 "右乘 $(in, out)$ 权重矩阵"——输入 $(N, in)$ × 权重 $(in, out)$ → 输出 $(N, out)$。

但 PyTorch 实际把权重**存为 $(out, in)$**，前向算的是 $y = x W^\top + b$。两种写法乘出来结果一样，存储约定选 $(out, in)$ 是为了**行优先**访存友好。

### 3.2 nn.Module 替你做了什么

1. **参数自动注册**：所有 `nn.Linear` / `nn.Conv2d` 之类的 submodule，它们的 `weight` / `bias` 自动出现在 `model.parameters()` 里
2. **`.to(device)` 一键搬全家**：所有子参数和 buffer 都跟着搬
3. **`train() / eval()` 切模式**：影响 Dropout、BN（BtachNorm，批归一化） 等 "训推不一致" 算子（详见 ch04）
4. **`state_dict()` 标准持久化**：保存/加载权重


### 3.3 Sequential：堆叠的语法糖

简单串联结构可以省掉自定义类：

```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

**何时用 Sequential，何时写自定义类**：

- 分支 / 残差 / 多输入 → 自定义类
- 纯线性串联 → Sequential

LLM 里 Transformer block 一定是自定义类（残差 + 注意力 + FFN 多路）。

> **顺带一提**：loss 函数也是 `nn.Module`，常用的有 `nn.CrossEntropyLoss`（分类）、`nn.MSELoss`（回归），用法 `loss = loss_fn(pred, target)`。§5 训练循环就用它。

### 3.4 保存与加载

模型训完总要存盘 —— 下次接着用、发布给别人、部署到生产，中途也可能要定期存盘。PyTorch 的标准做法是**只存权重**（一个 `name → tensor` 的字典，叫 **state_dict**），不存代码：

```python
torch.save(model.state_dict(), "ckpt.pt")        # 只存权重 dict（推荐）

model2 = MLP(784, 128, 10)                       # 重建结构
model2.load_state_dict(torch.load("ckpt.pt"))    # 灌权重
```

**禁止** `torch.save(model, ...)`：会用 pickle 把"类的导入路径"一起记下，加载时必须能 import 到同名同路径的类，模块改名或重构就加载失败，还和 PyTorch 版本耦合。

**两种典型场景**（本章只演示第 1 种，第 2 种续训留到训练章）：

1. **最终权重发布**：只存 `model.state_dict()`，对应 HuggingFace 上的 `model.safetensors`（主流，安全）或老格式 `pytorch_model.bin`（pickle，逐步淘汰）。
2. **训练断点续训**：长训练任务定期存档是常态。光存 model 不够，Adam 的 momentum / variance、step 计数、scheduler 进度都得一起存，否则恢复后 lr / 更新方向都乱：

   ```python
   torch.save({
       "epoch": epoch,
       "step": global_step,
       "model": model.state_dict(),
       "optimizer": optimizer.state_dict(),
       "scheduler": scheduler.state_dict(),
   }, "ckpt.pt")
   ```

> **关于跨环境部署**（ONNX / TorchScript / GGUF）：那是脱离 PyTorch 运行的"格式转换"，不是 state_dict 的另一种用法，目标完全不同，留到 M5 部署章再讲。

> 配套演示见 `Playground/ch03-pytorch/03_nn_module.py` 末尾：训完模型 → 存权重 → 重建空壳 → 加载 → 断言输出一致。

### 自检

1. `nn.Linear(784, 128)` 的 `weight.shape` 是什么？为什么不是 `(784, 128)`？
2. 写 `forward()` 时，最后一层之后**不要**手动 `softmax`——为什么？

<details markdown="1">
<summary>答案速查</summary>

1. `(128, 784)`，即 `(out, in)`。PyTorch 内部算的是 $y = x W^\top + b$，行优先访存友好。详见 §3.1 的说明块

2. `nn.CrossEntropyLoss` 内部已经合并了 `log_softmax + NLL`，外面再 softmax 等于做两遍且数值更差。ch02 §3.5 详述

</details>

---

## 4. Dataset 与 DataLoader

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

### 4.3 Win 的 `num_workers` 坑

Win 上 `num_workers > 0` + 没用 `if __name__ == "__main__":` 守卫 → multiprocessing 会**无限递归 fork**。两条规避：

1. 教学代码 / 单文件脚本：`num_workers=0`，省心
2. 真要并行：脚本入口必须 `if __name__ == "__main__": main()`

Linux 默认 fork，没这问题。Mac Python 3.8+ 同样默认 spawn，但不需要 `if __name__` 守卫即可正常运行（multiprocessing 安全机制与 Win 不同）。

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

1. 初始 `.grad` 为 `None`，第一步 `backward()` 直接赋值（碰巧正确）；但第二步起 `backward()` 会把新梯度加在上一步 `step()` 已用过的残留梯度上，等价于隐式 grad accumulation，loss 会偏高/震荡。**位置写在 `backward()` 之前最稳**

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

通过标准：每个脚本独立跑通，最后打印 `PASS`。`05_mnist_mlp.py` 首次运行会自动下载 MNIST 到 `./data/mnist/`（约 11MB，已被 `.gitignore`），3060 12GB 上约 30 秒、Mac M 系列约 1 分钟、纯 CPU 约 2 分钟。

## 思考题

1. ch02 的 `04_mlp_numpy.py` 和本章 `03_nn_module.py` 行数大约比例多少？PyTorch 省的主要是哪一块（数据 / forward / 反向 / 优化）？
2. `model.parameters()` 返回的是生成器，传给 `Adam(...)` 后 Adam 怎么知道哪些参数该更新？如果新增一个 `nn.Parameter` 但**不**注册到 `nn.Module` 上，会发生什么？
3. MNIST 用 MLP 能到 97%+，但用 CNN 能到 99%+。差距来自哪里？这给我们什么启示——为什么 LLM 不用 CNN？

## 参考资料

- **PyTorch 官方 60 分钟入门**：<https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html>
- **PyTorch autograd 机制**：<https://pytorch.org/docs/stable/notes/autograd.html>
- **`nn.Module` 源码导读**（进阶）：<https://pytorch.org/docs/stable/generated/torch.nn.Module.html>
