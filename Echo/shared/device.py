"""统一设备选择。

优先级：CUDA → MPS → CPU。
禁止业务代码硬编码设备字符串，一律走 `get_device()`。
"""

from __future__ import annotations

import os

import torch


def is_mps_available() -> bool:
    """MPS 是否可用（Apple Silicon + PyTorch MPS backend）。"""
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def get_device(prefer: str | None = None) -> torch.device:
    """按 CUDA → MPS → CPU 优先级返回可用设备。

    Args:
        prefer: 显式指定 "cuda" / "mps" / "cpu"，不可用时降级到下一档。
                None 时按默认优先级走。

    Returns:
        torch.device

    Note:
        - MPS 上部分算子未实现，建议在脚本入口设置：
          `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")`
          让 PyTorch 自动 fallback 到 CPU 执行未支持算子。
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and is_mps_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")

    # 默认优先级
    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def mps_safe_to(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """把 tensor 搬到目标设备，MPS 不支持的 dtype 自动降级。

    MPS 已知不支持 float64 / complex 类型，遇到则转 float32。
    """
    if device.type == "mps":
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        elif tensor.dtype.is_complex:
            raise NotImplementedError("MPS 不支持复数张量")
    return tensor.to(device)


def enable_mps_fallback() -> None:
    """显式开启 MPS 算子 fallback，建议训练/推理脚本入口调用一次。"""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
