"""导出合并模型为 GGUF 格式并量化。

依赖：需要本地克隆 llama.cpp 仓库（用其 convert_hf_to_gguf.py 和 llama-quantize）。

默认走 base 路线：merged-base → gguf-base。instruct 路线需显式传参。

convert_hf_to_gguf.py 的 Python 依赖（gguf / numpy / sentencepiece / protobuf /
torch==2.11 等）建议装在 llama.cpp **独立 venv** 里，避免覆盖 echo venv 的
CUDA torch。通过 --convert-python 或环境变量 LLAMA_CPP_PYTHON 指定该
venv 的解释器；不指定则回退到当前 sys.executable（即 echo venv）。

用法：
    # base 路线（默认，用独立 convert venv）
    uv run python scripts/export_gguf.py \\
        --convert-python ~/llama.cpp/.venv-convert/bin/python

    # 或导出环境变量后简化
    export LLAMA_CPP_PYTHON=~/llama.cpp/.venv-convert/bin/python
    uv run python scripts/export_gguf.py

    # instruct 路线
    uv run python scripts/export_gguf.py --merged-dir checkpoints/merged \\
        --output-dir checkpoints/gguf

    # 指定量化方式
    uv run python scripts/export_gguf.py --quant Q8_0

    # 只转换不量化
    uv run python scripts/export_gguf.py --no-quantize

环境变量：
    LLAMA_CPP_DIR:    llama.cpp 仓库路径（默认 ~/llama.cpp）
    LLAMA_CPP_PYTHON: convert_hf_to_gguf.py 用的 Python 解释器
                      （默认 sys.executable，即当前 echo venv）
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

console = Console()

DEFAULT_MERGED = Path("checkpoints/merged-base")
DEFAULT_OUTPUT = Path("checkpoints/gguf-base")
DEFAULT_QUANT = "Q4_K_M"


def find_llama_cpp() -> Path:
    """定位 llama.cpp 目录。"""
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p

    # 常见默认位置
    candidates = [
        Path.home() / "llama.cpp",
        Path.home() / "repos" / "llama.cpp",
        Path("/opt/llama.cpp"),
    ]
    for c in candidates:
        if c.exists():
            return c

    console.print("[bold red]Error:[/bold red] 找不到 llama.cpp 目录")
    console.print("  设置环境变量 LLAMA_CPP_DIR 或克隆到 ~/llama.cpp：")
    console.print("  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
    raise SystemExit(1)


def resolve_convert_python(cli_value: str | None) -> str:
    """决定用哪个 Python 跑 convert_hf_to_gguf.py。

    优先级：CLI 参数 > 环境变量 LLAMA_CPP_PYTHON > sys.executable（兜底，会污染 echo venv）。
    """
    candidate = cli_value or os.environ.get("LLAMA_CPP_PYTHON")
    if not candidate:
        console.print(
            "[yellow]Warning:[/yellow] 未指定 --convert-python / LLAMA_CPP_PYTHON，"
            "将用当前 sys.executable 跑 convert_hf_to_gguf.py。"
        )
        console.print(
            "  若该 venv 未装 gguf/numpy/sentencepiece 等依赖会失败；"
            "在 echo venv 直接装这些依赖会覆盖 CUDA torch。"
        )
        console.print(
            "  推荐在 llama.cpp 仓库下建独立 venv："
            "uv venv ~/llama.cpp/.venv-convert && "
            "uv pip install --python ~/llama.cpp/.venv-convert/bin/python "
            "--index-strategy unsafe-best-match -r "
            "~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
        )
        return sys.executable

    p = Path(candidate).expanduser()
    if not p.exists():
        console.print(f"[bold red]Error:[/bold red] convert python 不存在: {p}")
        raise SystemExit(1)
    return str(p)


def convert_to_gguf(
    merged_dir: Path, output_dir: Path, llama_cpp: Path, convert_python: str
) -> Path:
    """调用 convert_hf_to_gguf.py 转换为 f16 GGUF。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "echo-f16.gguf"

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        console.print(f"[bold red]Error:[/bold red] {convert_script} not found")
        raise SystemExit(1)

    cmd = [
        convert_python, str(convert_script),
        str(merged_dir),
        "--outfile", str(output_file),
        "--outtype", "f16",
    ]

    console.print(f"[bold]Converting to GGUF:[/bold] {output_file}")
    console.print(f"  python: {convert_python}")
    console.print(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        console.print("[bold red]Conversion failed![/bold red]")
        raise SystemExit(1)

    console.print(f"[bold green]Done:[/bold green] {output_file} ({output_file.stat().st_size / 1e9:.2f} GB)")
    return output_file


def quantize_gguf(input_file: Path, output_dir: Path, quant_type: str, llama_cpp: Path) -> Path:
    """调用 llama-quantize 量化。"""
    output_file = output_dir / f"echo-{quant_type}.gguf"

    # 查找 quantize 二进制
    quantize_bin = None
    for name in ["llama-quantize", "quantize"]:
        for subdir in ["build/bin", "build", ""]:
            candidate = llama_cpp / subdir / name
            if candidate.exists():
                quantize_bin = candidate
                break
            # Windows .exe
            candidate_exe = candidate.with_suffix(".exe")
            if candidate_exe.exists():
                quantize_bin = candidate_exe
                break
        if quantize_bin:
            break

    if not quantize_bin:
        console.print("[bold red]Error:[/bold red] llama-quantize 二进制未找到")
        console.print("  请先编译 llama.cpp：cd ~/llama.cpp && make")
        raise SystemExit(1)

    cmd = [str(quantize_bin), str(input_file), str(output_file), quant_type]

    console.print(f"[bold]Quantizing ({quant_type}):[/bold] {output_file}")
    console.print(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        console.print("[bold red]Quantization failed![/bold red]")
        raise SystemExit(1)

    console.print(f"[bold green]Done:[/bold green] {output_file} ({output_file.stat().st_size / 1e9:.2f} GB)")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Export merged model to GGUF")
    parser.add_argument("--merged-dir", type=Path, default=DEFAULT_MERGED, help="Merged model dir")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="GGUF output dir")
    parser.add_argument("--quant", type=str, default=DEFAULT_QUANT, help="Quantization type (Q4_K_M, Q8_0, etc.)")
    parser.add_argument("--no-quantize", action="store_true", help="Only convert, skip quantization")
    parser.add_argument(
        "--convert-python",
        type=str,
        default=None,
        help="Python interpreter for convert_hf_to_gguf.py (overrides LLAMA_CPP_PYTHON)",
    )
    args = parser.parse_args()

    merged_dir = Path(args.merged_dir)
    if not merged_dir.exists():
        console.print(f"[bold red]Error:[/bold red] merged 目录不存在: {merged_dir}")
        console.print("  请先运行: uv run python scripts/merge.py")
        raise SystemExit(1)

    llama_cpp = find_llama_cpp()
    console.print(f"[bold]llama.cpp:[/bold] {llama_cpp}")

    convert_python = resolve_convert_python(args.convert_python)

    # 1. 转 GGUF (f16)
    f16_file = convert_to_gguf(merged_dir, Path(args.output_dir), llama_cpp, convert_python)

    # 2. 量化
    if not args.no_quantize:
        quantize_gguf(f16_file, Path(args.output_dir), args.quant, llama_cpp)

    console.print("\n[bold green]Export complete![/bold green]")
    console.print(f"  GGUF files in: {args.output_dir}/")


if __name__ == "__main__":
    main()
