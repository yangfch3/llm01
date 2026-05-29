"""echo · Ollama 部署 demo 脚本。

跨平台调用 `ollama` CLI：
    1. 检查 ollama 可执行 + 守护进程
    2. 检查 GGUF 文件存在
    3. ollama create <name> -f Modelfile（若模型未注册）
    4. 对内置 prompt 列表逐个 ollama run，打印响应

用法：
    # base 路线（默认）
    uv run python scripts/run_demo.py

    # 强制重建模型（Modelfile 改动后）
    uv run python scripts/run_demo.py --recreate

    # 自定义模型名
    uv run python scripts/run_demo.py --name echo-v1

    # 仅检查环境，不真正跑 prompt
    uv run python scripts/run_demo.py --check-only

依赖：
    - 系统已安装 Ollama (https://ollama.com)
    - 已完成 export_gguf.py，产出 checkpoints/gguf-base/echo-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

console = Console()

DEFAULT_MODEL_NAME = "echo"
DEFAULT_MODELFILE = Path("Modelfile")
DEFAULT_GGUF = Path("checkpoints/gguf-base/echo-Q4_K_M.gguf")

DEMO_PROMPTS = [
    "你好，请简单介绍一下你自己。",
    "用一句话写晚安祝福。",
    "推荐一本适合周末读的书，并说说理由。",
    "Explain what a transformer is in two sentences.",
    "讲一个冷笑话。",
]


def check_ollama() -> str:
    """检查 ollama 可执行是否存在，返回完整路径。"""
    exe = shutil.which("ollama")
    if not exe:
        console.print("[bold red]Error:[/bold red] 未找到 ollama 命令")
        console.print("  Win:  https://ollama.com/download/windows")
        console.print("  Mac:  brew install ollama  或  https://ollama.com/download/mac")
        raise SystemExit(1)
    console.print(f"[bold]ollama:[/bold] {exe}")
    return exe


def check_daemon(exe: str) -> None:
    """ollama list 同时验证守护进程是否在运行。"""
    result = subprocess.run(
        [exe, "list"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        console.print("[bold red]Error:[/bold red] ollama 守护进程未运行")
        console.print("  Win:  Ollama 桌面应用启动后自动起服务")
        console.print("  Mac:  ollama serve  （或启动 Ollama.app）")
        console.print(f"  stderr: {result.stderr.strip()}")
        raise SystemExit(1)


def model_exists(exe: str, name: str) -> bool:
    """检查 ollama 仓库里是否已有 <name>:latest。"""
    result = subprocess.run(
        [exe, "list"], capture_output=True, text=True, check=False
    )
    for line in result.stdout.splitlines()[1:]:  # 跳过表头
        first = line.split()[0] if line.split() else ""
        # ollama list 的第一列形如 "echo:latest"
        if first.split(":")[0] == name:
            return True
    return False


def create_model(exe: str, name: str, modelfile: Path) -> None:
    """ollama create <name> -f <Modelfile>。"""
    console.print(f"[bold]Creating ollama model:[/bold] {name}  (from {modelfile})")
    result = subprocess.run(
        [exe, "create", name, "-f", str(modelfile)], check=False
    )
    if result.returncode != 0:
        console.print("[bold red]ollama create failed[/bold red]")
        raise SystemExit(1)
    console.print(f"[bold green]Created:[/bold green] {name}")


def run_prompt(exe: str, name: str, prompt: str) -> None:
    """单条 prompt 走 ollama run（非交互、读 stdin）。"""
    console.print(f"\n[bold cyan]>>>[/bold cyan] {prompt}")
    # ollama run 支持 stdin 输入；--nowordwrap 让输出不被终端宽度切断
    result = subprocess.run(
        [exe, "run", name],
        input=prompt,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        console.print("[bold red]ollama run failed[/bold red]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Echo Ollama demo runner")
    parser.add_argument("--name", default=DEFAULT_MODEL_NAME, help="ollama 模型名")
    parser.add_argument(
        "--modelfile", type=Path, default=DEFAULT_MODELFILE, help="Modelfile 路径"
    )
    parser.add_argument(
        "--gguf", type=Path, default=DEFAULT_GGUF, help="GGUF 文件路径（仅做存在性检查）"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="无论是否已存在都重新 ollama create"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="仅做环境与文件检查，不跑 demo prompt"
    )
    args = parser.parse_args()

    # 1. 环境检查
    exe = check_ollama()
    check_daemon(exe)

    # 2. GGUF 文件检查
    if not args.gguf.exists():
        console.print(f"[bold red]Error:[/bold red] GGUF 不存在: {args.gguf}")
        console.print("  请先运行: uv run python scripts/export_gguf.py")
        raise SystemExit(1)
    console.print(f"[bold]GGUF:[/bold] {args.gguf}  ({args.gguf.stat().st_size / 1e9:.2f} GB)")

    # 3. Modelfile 检查
    if not args.modelfile.exists():
        console.print(f"[bold red]Error:[/bold red] Modelfile 不存在: {args.modelfile}")
        raise SystemExit(1)

    # 4. 注册模型
    if args.recreate or not model_exists(exe, args.name):
        create_model(exe, args.name, args.modelfile)
    else:
        console.print(f"[dim]ollama 模型 {args.name} 已存在，跳过 create（--recreate 强制重建）[/dim]")

    if args.check_only:
        console.print("\n[bold green]Check passed.[/bold green]")
        return

    # 5. 跑 demo prompt
    console.print(f"\n[bold]Running {len(DEMO_PROMPTS)} demo prompts on {args.name}:[/bold]")
    for prompt in DEMO_PROMPTS:
        run_prompt(exe, args.name, prompt)

    console.print("\n[bold green]Demo complete.[/bold green]")
    console.print(f"  交互式继续聊：[bold]ollama run {args.name}[/bold]")


if __name__ == "__main__":
    main()
