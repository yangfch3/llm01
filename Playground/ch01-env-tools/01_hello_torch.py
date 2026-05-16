"""ch01 练习 1：用 PyTorch 跑一次前向 + 反向。

目标：
1. 走 `get_device()` 拿到设备（Win→cuda / Mac→mps / 否则 cpu）
2. 建张量、做矩阵乘、求 sum 当作 loss
3. 反向传播，打印梯度

跑法：
    uv run python Playground/ch01-env-tools/01_hello_torch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Echo/ 大写不是标准 Python 包名，需手动把仓库根加到 sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Echo.shared.device import enable_mps_fallback, get_device  # noqa: E402


def main() -> None:
    # MPS 算子 fallback：Mac 用户友好（Win/Linux 上是 no-op）
    enable_mps_fallback()

    device = get_device()
    print(f"device = {device}")

    # requires_grad=True 让 PyTorch 跟踪计算图
    x = torch.randn(3, 4, device=device, requires_grad=True)
    w = torch.randn(4, 2, device=device, requires_grad=True)

    # 前向：y = x @ w，然后求和当 loss
    y = x @ w
    loss = y.sum()
    print(f"loss = {loss.item():.4f}")

    # 反向：梯度算到 x.grad / w.grad 上
    loss.backward()

    print(f"x.grad shape = {tuple(x.grad.shape)}, w.grad shape = {tuple(w.grad.shape)}")
    print(f"x.grad[0] = {x.grad[0].tolist()}")
    print(f"w.grad mean = {w.grad.mean().item():.4f}")


if __name__ == "__main__":
    main()
