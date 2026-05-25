"""echo 工具函数。"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
