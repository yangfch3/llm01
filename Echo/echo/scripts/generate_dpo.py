"""DPO 抽测 CLI — 在 merged-base 上叠加 DPO adapter，验证 v2 效果。

T6.3 流程中"合并 merged-dpo 之前"的人工抽测专用脚本。merge_dpo.py 跑完后
直接用 ``scripts/generate.py --merged-dir checkpoints/merged-dpo`` 即可，
本脚本届时退役。

为什么单独成一个脚本而不挤进 generate.py：
    - generate.py 当前两模式（仅 adapter / 仅 merged）逻辑清晰，
      硬塞 "merged + adapter 叠加" 第三模式会让分支膨胀
    - DPO 抽测仅在 v2 合并前用一两次，是阶段性场景
    - chat_template 优先级在叠加场景下与单独场景相反（DPO adapter 优先），
      单独成脚本不易踩坑

用法（cwd: Echo/echo/）：
    # 默认：merged-base + dpo-base/final
    uv run python scripts/generate_dpo.py

    # 测中间 ckpt
    uv run python scripts/generate_dpo.py --adapter-dir checkpoints/dpo-base/checkpoint-200

抽测建议参考 stepshooting-base.md §7.5.4：重点测 SFT 阶段塌缩过的短模板题
（如"写一句晚安祝福"），看 DPO 是否恢复多样性。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from echo.utils import CHATML_INFER_TEMPLATE  # noqa: E402

console = Console()

DEFAULT_MERGED = Path("checkpoints/merged-base")
DEFAULT_ADAPTER = Path("checkpoints/dpo-base/final")


def load_model_and_tokenizer(args: argparse.Namespace):
    """加载 merged-base + DPO adapter，并选取正确的 tokenizer。

    Tokenizer 优先级：adapter > merged。DPO 训练时把 chat_template 写到 adapter
    目录（见 dpo-base/checkpoint-*/chat_template.jinja），优先用它能保证推理
    模板与 DPO 训练阶段对齐；merged-base 是 SFT 阶段的 tokenizer，作为兜底。
    """
    merged_dir = Path(args.merged_dir)
    adapter_dir = Path(args.adapter_dir)

    if not merged_dir.exists():
        console.print(f"[bold red]Error:[/bold red] merged 目录不存在: {merged_dir}")
        raise SystemExit(1)
    if not adapter_dir.exists():
        console.print(f"[bold red]Error:[/bold red] adapter 目录不存在: {adapter_dir}")
        raise SystemExit(1)

    # Tokenizer：先试 adapter，失败回落 merged
    tokenizer = None
    for kind, src in [("adapter", adapter_dir), ("merged", merged_dir)]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)
            console.print(f"[load_tokenizer] loaded from {kind}: {src}")
            break
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]tokenizer load failed from {kind} ({src}): {e}[/yellow]")
    if tokenizer is None:
        raise RuntimeError("tokenizer 加载失败：adapter / merged 均不可用")

    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = CHATML_INFER_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 模型：merged 完整权重 → 贴 adapter
    console.print(f"[bold]Merged base:[/bold] {merged_dir}")
    console.print(f"[bold]DPO adapter:[/bold] {adapter_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer


def chat_loop(model, tokenizer, args: argparse.Namespace) -> None:
    """交互式多轮对话循环（与 generate.py 行为一致）。"""
    device = next(model.parameters()).device
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    console.print("\n[bold green]Echo Chat (DPO adapter)[/bold green] (输入 /quit 退出, /clear 清空历史)\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "/quit":
            break
        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            console.print("[dim]History cleared.[/dim]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        stop_token_ids = [
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        stop_token_ids = [
            tid for tid in stop_token_ids if tid is not None and tid >= 0
        ]
        if not stop_token_ids:
            stop_token_ids = [tokenizer.eos_token_id]

        print("Echo: ", end="", flush=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.temperature > 0,
                eos_token_id=stop_token_ids,
                pad_token_id=tokenizer.pad_token_id,
                streamer=streamer,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        assistant_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": assistant_text})
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Echo DPO 抽测 CLI (merged-base + DPO adapter)"
    )
    parser.add_argument(
        "--merged-dir", type=Path, default=DEFAULT_MERGED, help="SFT merged base dir"
    )
    parser.add_argument(
        "--adapter-dir", type=Path, default=DEFAULT_ADAPTER, help="DPO adapter dir"
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args)
    chat_loop(model, tokenizer, args)


if __name__ == "__main__":
    main()
