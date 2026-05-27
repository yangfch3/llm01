"""导出合并模型为 GGUF 格式并量化。

依赖：需要本地克隆 llama.cpp 仓库（用其 convert_hf_to_gguf.py 和 llama-quantize）。

用法：
    # 默认流程：merged → GGUF f16 → Q4_K_M
    uv run python scripts/export_gguf.py

    # 指定路径和量化方式
    uv run python scripts/export_gguf.py --merged-dir checkpoints/merged --quant Q8_0

    # 只转换不量化
    uv run python scripts/export_gguf.py --no-quantize

环境变量：
    LLAMA_CPP_DIR: llama.cpp 仓库路径（默认 ~/llama.cpp）
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

DEFAULT_MERGED = Path("checkpoints/merged")
DEFAULT_OUTPUT = Path("checkpoints/gguf")
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


def convert_to_gguf(merged_dir: Path, output_dir: Path, llama_cpp: Path) -> Path:
    """调用 convert_hf_to_gguf.py 转换为 f16 GGUF。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "echo-f16.gguf"

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        console.print(f"[bold red]Error:[/bold red] {convert_script} not found")
        raise SystemExit(1)

    cmd = [
        sys.executable, str(convert_script),
        str(merged_dir),
        "--outfile", str(output_file),
        "--outtype", "f16",
    ]

    console.print(f"[bold]Converting to GGUF:[/bold] {output_file}")
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
    args = parser.parse_args()

    merged_dir = Path(args.merged_dir)
    if not merged_dir.exists():
        console.print(f"[bold red]Error:[/bold red] merged 目录不存在: {merged_dir}")
        console.print("  请先运行: uv run python scripts/merge.py")
        raise SystemExit(1)

    llama_cpp = find_llama_cpp()
    console.print(f"[bold]llama.cpp:[/bold] {llama_cpp}")

    # 1. 转 GGUF (f16)
    f16_file = convert_to_gguf(merged_dir, Path(args.output_dir), llama_cpp)

    # 2. 量化
    if not args.no_quantize:
        quantize_gguf(f16_file, Path(args.output_dir), args.quant, llama_cpp)

    console.print("\n[bold green]Export complete![/bold green]")
    console.print(f"  GGUF files in: {args.output_dir}/")


if __name__ == "__main__":
    main()
