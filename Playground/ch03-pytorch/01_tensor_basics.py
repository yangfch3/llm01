"""ch03 练习 1：Tensor 基础。

走 5 个最小例子覆盖：创建、形状、设备、与 numpy 互操作、dtype 易错点。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # 让 `from Echo.shared...` 能 import（ch01 自检 3）

import numpy as np
import torch

from Echo.shared.device import get_device


def demo_create() -> None:
    print("\n--- 1. 创建 ---")
    a = torch.zeros(2, 3)  # 全零 (2,3)，dtype 默认 float32
    b = torch.ones(2, 3)
    c = torch.arange(6).reshape(2, 3)  # arange 默认 int64
    d = torch.randn(2, 3)  # 标准正态采样，常用于权重初始化
    print(f"zeros: {a.shape} {a.dtype}")
    print(f"ones:  {b.shape} {b.dtype}")
    print(f"arange:{c.shape} {c.dtype}")
    print(f"randn: {d.shape} {d.dtype}")
    assert a.dtype == torch.float32
    assert c.dtype == torch.int64


def demo_shape() -> None:
    print("\n--- 2. 形状操作 ---")
    x = torch.arange(12)  # (12,)
    print(f"原始:    {x.shape}")
    print(f"view:    {x.view(3, 4).shape}")  # 不改数据只改 stride 视图，要求内存连续
    print(f"reshape: {x.reshape(3, 4).shape}")  # 不连续时自动 copy，更稳
    print(f"unsqueeze(0): {x.unsqueeze(0).shape}")  # (12,) → (1,12) 加 batch 维常用
    print(f"transpose:    {x.view(3, 4).transpose(0, 1).shape}")  # (3,4) → (4,3)


def demo_device() -> None:
    print("\n--- 3. 设备 ---")
    device = get_device()  # cuda → mps → cpu，铁律：禁止硬编码字符串
    print(f"使用设备: {device}")
    x = torch.randn(2, 3, device=device)  # 直接建在目标设备，省一次 host→device 拷贝
    y = torch.randn(2, 3).to(device)  # 先建 CPU 再搬，等价但多一步
    z = x + y  # 同设备运算 OK
    print(f"x.device={x.device}  z.device={z.device}")
    assert z.device.type == device.type


def demo_numpy_bridge() -> None:
    print("\n--- 4. NumPy 桥 ---")
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = torch.from_numpy(a)  # 零拷贝共享内存（CPU 上）
    a[0] = 99.0
    print(f"改 numpy 后 tensor: {t}")  # tensor[0] 也会变成 99，验证零拷贝
    assert t[0].item() == 99.0

    # 反向：tensor → numpy
    t2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    # 三连：detach() 切计算图 → cpu() 搬回主存（已经在 CPU 也安全）→ numpy()
    arr = t2.detach().cpu().numpy()
    print(f"detach→cpu→numpy: {arr}")


def demo_dtype_pitfall() -> None:
    print("\n--- 5. dtype 易错点 ---")
    int_tensor = torch.tensor([1, 2, 3])  # 推断成 int64
    float_tensor = torch.tensor([1.0, 2.0, 3.0])  # 推断成 float32
    print(f"int_tensor.dtype  = {int_tensor.dtype}")
    print(f"float_tensor.dtype= {float_tensor.dtype}")

    # 神经网络权重默认 float32，整型 tensor 不能直接做矩阵乘
    w = torch.randn(3, 2)  # float32
    try:
        _ = int_tensor @ w  # 故意触发：Long 不能和 Float 做矩阵乘
    except RuntimeError as e:
        print(f"预期报错: {type(e).__name__}: {str(e).splitlines()[0]}")

    # 修法：显式转 float
    ok = int_tensor.float() @ w
    print(f"转 float 后 OK: {ok.shape}")


def main() -> None:
    demo_create()
    demo_shape()
    demo_device()
    demo_numpy_bridge()
    demo_dtype_pitfall()
    print("\nPASS")


if __name__ == "__main__":
    main()
