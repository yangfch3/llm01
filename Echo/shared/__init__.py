"""Echo 双产物共用工具包。"""

from .device import get_device, is_mps_available, mps_safe_to

__all__ = ["get_device", "is_mps_available", "mps_safe_to"]
