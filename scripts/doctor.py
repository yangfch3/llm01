"""双平台环境自检脚本。

上机第一件事跑一遍：
    uv run python scripts/doctor.py

检查项：
  - Python 版本满足 3.12
  - 平台与预期依赖组一致（Win 应有 train-cuda、Mac 应有 train-mps）
  - torch 可用 + 设备可见（cuda/mps）
  - 关键依赖 import 不报错
  - Git 换行符配置（core.autocrlf=false + .gitattributes）
  - 打印当前平台推荐训练入口
"""

from __future__ import annotations

import contextlib
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 把仓库根加到 sys.path，让 `from Echo.shared.device import ...` 可用
# Echo/ 目录是大写驼峰命名，不是标准 Python 包，需手动加路径
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Win 已知坑：torch 与 pyarrow 都带 OpenMP runtime，同进程加载会踩
# (ACCESS_VIOLATION 0xC0000005)。允许重复加载即可。必须在 import torch 前设。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 行缓冲：避免 native crash 时 stdout 截断丢日志
with contextlib.suppress(AttributeError, OSError):
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# 关键依赖列表（import 名 → 是否必需）
CORE_DEPS: list[tuple[str, bool]] = [
    ("torch", True),
    ("numpy", True),
    ("yaml", True),  # pyyaml
    ("tqdm", True),
    ("rich", True),
]

ECHO_DEPS: list[tuple[str, bool]] = [
    ("transformers", False),
    ("tokenizers", False),
    ("datasets", False),
    ("accelerate", False),
    ("peft", False),
    ("trl", False),
    ("safetensors", False),
    ("huggingface_hub", False),
    ("sentencepiece", False),
]

# 状态符号
OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"


def header(title: str) -> None:
    print(f"\n=== {title} ===")


def check_python() -> bool:
    header("Python 版本")
    v = sys.version_info
    print(f"实际：{v.major}.{v.minor}.{v.micro}  ({platform.python_implementation()})")
    if (v.major, v.minor) != (3, 12):
        print(f"{FAIL} 项目要求 Python 3.12（minor 锁定）")
        return False
    print(f"{OK} 满足 3.12")
    return True


def check_platform() -> str:
    header("平台")
    sysname = platform.system()
    machine = platform.machine()
    print(f"系统：{sysname}  架构：{machine}")
    if sysname == "Windows":
        plat = "win"
    elif sysname == "Darwin":
        plat = "mac"
    elif sysname == "Linux":
        plat = "linux"
    else:
        plat = "unknown"
    print(f"识别为：{plat}")
    return plat


def check_torch_device(plat: str) -> bool:
    header("PyTorch 与设备")
    try:
        import torch
    except ImportError as e:
        print(f"{FAIL} torch import 失败：{e}")
        return False

    print(f"torch 版本：{torch.__version__}")

    cuda_ok = torch.cuda.is_available()
    mps_ok = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    print(f"CUDA 可用：{cuda_ok}")
    print(f"MPS  可用：{mps_ok}")

    if cuda_ok:
        print(f"  CUDA 设备：{torch.cuda.get_device_name(0)}")
        print(f"  CUDA 版本：{torch.version.cuda}")

    # 平台预期校验
    if plat == "win":
        if not cuda_ok:
            print(f"{WARN} Win 平台未检测到 CUDA，训练 full 配置将不可用")
        else:
            print(f"{OK} Win + CUDA")
    elif plat == "mac":
        if not mps_ok:
            print(f"{WARN} Mac 平台未检测到 MPS（仅 CPU 可用，将极慢）")
        else:
            print(f"{OK} Mac + MPS")
    else:
        print(f"{WARN} 平台 {plat} 不在主线支持范围（仅 Win/Mac 验证过）")

    # 设备 smoke test
    try:
        from Echo.shared.device import get_device

        dev = get_device()
        x = torch.randn(2, 3).to(dev)
        y = (x @ x.T).sum().item()
        print(f"{OK} 设备 smoke test 通过（device={dev}, value={y:.4f}）")
    except Exception as e:
        print(f"{FAIL} 设备 smoke test 失败：{e}")
        return False

    return True


def check_imports(deps: list[tuple[str, bool]], group_name: str) -> bool:
    """主进程内检查（适用核心依赖）。"""
    header(f"依赖 import · {group_name}")
    all_ok = True
    for name, required in deps:
        try:
            importlib.import_module(name)
            print(f"{OK} {name}")
        except ImportError as e:
            tag = FAIL if required else WARN
            print(f"{tag} {name}: {e}")
            if required:
                all_ok = False
    return all_ok


def check_imports_subprocess(deps: list[tuple[str, bool]], group_name: str) -> bool:
    """子进程隔离检查（适用 ECHO_DEPS：pyarrow/pandas 等与 torch 同进程会
    在 Win 触发 ACCESS_VIOLATION，必须独立子进程跑）。"""
    header(f"依赖 import · {group_name}（子进程隔离）")
    all_ok = True
    for name, required in deps:
        result = subprocess.run(
            [sys.executable, "-c", f"import {name}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if result.returncode == 0:
            print(f"{OK} {name}")
        else:
            tag = FAIL if required else WARN
            err = (result.stderr or result.stdout or "").strip().splitlines()
            tail = err[-1] if err else f"exit code {result.returncode}"
            print(f"{tag} {name}: {tail}")
            if required:
                all_ok = False
    return all_ok


def check_platform_extras(plat: str) -> None:
    header("平台专属依赖")
    if plat == "win":
        # 子进程隔离：bitsandbytes 启动会调 nvidia-smi 等工具，输出含 GBK
        # 字节，主进程读 stdout 可能解码失败。同时也避免污染主进程 DLL。
        result = subprocess.run(
            [sys.executable, "-c", "import bitsandbytes"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if result.returncode == 0:
            print(f"{OK} bitsandbytes 已安装")
            try:
                import torch

                if not torch.cuda.is_available():
                    print(f"{WARN} CUDA 不可用，bitsandbytes 实际跑不起来")
            except ImportError:
                pass
        else:
            print(f"{WARN} bitsandbytes 未安装或 import 失败。QLoRA 训练需要：")
            print(
                "       uv sync --extra dev --extra courseware --extra echo-mini --extra train-cuda"
            )
    elif plat == "mac":
        # train-mps 暂无强制依赖
        print("Mac 平台 train-mps 当前无强制额外依赖")
    else:
        print(f"平台 {plat} 跳过")


def check_git_config() -> None:
    header("Git 换行符配置")
    try:
        result = subprocess.run(
            ["git", "config", "--get", "core.autocrlf"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        autocrlf = (result.stdout or "").strip() or "(unset)"
        print(f"core.autocrlf = {autocrlf}")
        if autocrlf == "true":
            print(f"{WARN} 建议设为 false（仓库已用 .gitattributes 强制 LF）：")
            print("       git config --global core.autocrlf false")
        else:
            print(f"{OK} 不会自动转换 CRLF")
    except FileNotFoundError:
        print(f"{WARN} 未检测到 git 命令")
    except subprocess.TimeoutExpired:
        print(f"{WARN} git 命令超时")

    gitattr = REPO_ROOT / ".gitattributes"
    if gitattr.exists():
        print(f"{OK} .gitattributes 存在")
    else:
        print(f"{FAIL} .gitattributes 缺失")


def print_recommendation(plat: str) -> None:
    header("当前平台推荐入口")
    if plat == "win":
        print(
            "依赖安装：uv sync --extra dev --extra courseware --extra echo-mini --extra train-cuda"
        )
        print("训练脚本：使用 config-full.yaml（生产配置）")
        print("（M6 部署阶段需要 llama-cpp-python 时再加 --extra deploy-llamacpp）")
    elif plat == "mac":
        print(
            "依赖安装：uv sync --extra dev --extra courseware --extra echo-mini --extra train-mps"
        )
        print("训练脚本：使用 config-tiny.yaml（仅验证代码正确性）")
        print("（M6 部署阶段需要 llama-cpp-python 时再加 --extra deploy-llamacpp）")
    else:
        print("非主线支持平台，自行参考策划案 §6")


def main() -> int:
    print(f"llm01 doctor · 仓库根：{REPO_ROOT}")
    ok = True
    ok &= check_python()
    plat = check_platform()
    ok &= check_torch_device(plat)
    ok &= check_imports(CORE_DEPS, "核心")
    check_imports_subprocess(ECHO_DEPS, "Echo 训练相关（缺失仅 WARN）")
    check_platform_extras(plat)
    check_git_config()
    print_recommendation(plat)

    print()
    if ok:
        print(f"{OK} doctor 全部必需检查通过")
        return 0
    print(f"{FAIL} doctor 存在必需项失败，请按上面提示处理")
    return 1


if __name__ == "__main__":
    sys.exit(main())
