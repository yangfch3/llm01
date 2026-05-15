"""本地预览 mkdocs 站点。

封装 mkdocs serve，自动用 Misc/mkdocs/.venv-docs 里的隔离环境，无需 cd。
首次跑前需先建好 docs venv（见下方提示）。

用法：
    uv run python scripts/docs_serve.py            # 默认 0.0.0.0:8000
    uv run python scripts/docs_serve.py --help     # 透传 mkdocs serve 全部参数

注意：
  - mkdocs 不在主 .venv，因此不能 `uv run mkdocs serve`
  - docs 依赖独立由 Misc/mkdocs/requirements-docs.txt 管，与 CI 一致
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MKDOCS_DIR = REPO_ROOT / "Misc" / "mkdocs"
VENV_DIR = MKDOCS_DIR / ".venv-docs"
REQ_FILE = MKDOCS_DIR / "requirements-docs.txt"

# 平台差异：Win 在 Scripts/，*nix 在 bin/
MKDOCS_BIN = (
    VENV_DIR / "Scripts" / "mkdocs.exe"
    if os.name == "nt"
    else VENV_DIR / "bin" / "mkdocs"
)


def ensure_venv() -> None:
    """docs venv 不存在就提示用户手动建，不自动建以避免静默卡住。"""
    if MKDOCS_BIN.exists():
        return
    print(
        f"[docs_serve] 未找到 docs venv：{VENV_DIR}\n"
        f"请先执行（一次性）：\n"
        f"    cd {MKDOCS_DIR}\n"
        f"    uv venv .venv-docs\n"
        f"    uv pip install --python .venv-docs -r requirements-docs.txt\n",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> int:
    ensure_venv()
    # 透传命令行参数给 mkdocs serve
    cmd = [str(MKDOCS_BIN), "serve", "-f", str(MKDOCS_DIR / "mkdocs.yml"), *sys.argv[1:]]
    print(f"[docs_serve] {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
