"""ch01 练习 2：uv 常用命令速查。

本脚本不执行 shell，只把命令以注释形式列出，配合本章 README §1 阅读。

跑法（仅打印一段提示）：
    uv run python Playground/ch01-env-tools/hello_uv.py
"""

from __future__ import annotations

# === 安装 / 同步依赖 ===
# Win:
#     uv sync --extra dev --extra courseware --extra echo-mini --extra train-cuda
# Mac:
#     uv sync --extra dev --extra courseware --extra echo-mini --extra train-mps
#
# `uv sync` 读 pyproject.toml + uv.lock，确保锁文件一致。
# 改了依赖后，uv.lock 也会变，记得一起 commit。

# === 运行脚本 ===
# 推荐：
#     uv run python Playground/ch01-env-tools/hello_torch.py
# 等价：进入 venv 后 `python xxx.py`，但 `uv run` 不要求显式激活。

# === 添加 / 移除依赖 ===
#     uv add numpy
#     uv add --optional courseware matplotlib   # 加到 optional 组
#     uv remove numpy

# === 一次性工具（不污染项目） ===
#     uvx ruff check .
#     uvx ruff format .

# === 锁文件操作 ===
#     uv lock              # 重新解算锁文件（依赖 spec 改了用）
#     uv lock --upgrade    # 升级所有依赖到最新兼容版本


def main() -> None:
    print("uv 常用命令请见本文件注释 + Doc/Courseware/ch01-env-tools/README.md §1")
    print("无可执行逻辑。")


if __name__ == "__main__":
    main()
