# Playground/ch03-pytorch

ch03 课件配套练习。课件见 [`Doc/Courseware/ch03-pytorch/README.md`](../../Doc/Courseware/ch03-pytorch/README.md)。

| 脚本 | 内容 | 跑法 |
|---|---|---|
| `01_tensor_basics.py` | Tensor 创建 / 形状 / 设备 / numpy 桥 / dtype 易错 | `uv run python Playground/ch03-pytorch/01_tensor_basics.py` |
| `02_autograd.py` | autograd 与 ch02 解析梯度对照、梯度累加、`no_grad` | 同上 |
| `03_nn_module.py` | 用 `nn.Module` 重写 ch02 的 `mlp_numpy.py` | 同上 |
| `04_dataloader.py` | 自定义 `Dataset` + `DataLoader`，演示 batch / shuffle / drop_last | 同上 |
| `05_mnist_mlp.py` | **综合实战**：MNIST 分类，3 epoch ≥ 97% | 同上，首次会下载 MNIST 到 `data/mnist/`（≈ 11MB） |

通过标准：每个脚本独立跑通，最后一行打印 `PASS`。

## 备注

- `05_mnist_mlp.py` 首次需联网下载数据。`data/` 已在 `.gitignore`。
- 所有脚本头部都有 `sys.path.insert(0, REPO_ROOT)`，让 `from Echo.shared.device import get_device` 可用（ch01 自检 3）。
- 性能参考（batch=128，3 epoch）：3060 ≈ 30s，Mac M 系列 ≈ 1min，纯 CPU ≈ 2min。
