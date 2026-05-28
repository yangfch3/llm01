"""echo SFT 预检脚本 — 训练前 5-10 分钟内暴露所有可在静态/单步层面发现的问题。

四层检查（任意一层失败立即退出）：
  1. 静态：config 字段、tokenizer special token、数据格式与分布、chat_template 渲染
  2. 模型：加载到 device、PEFT 包装、tie_word_embeddings 状态、trainable 参数比例
  3. 单步：跑 1 条样本前向 + loss mask 可视化、跑 5 个 micro step 看 loss 下降
  4. 生成：底座 + adapter（若存在）跑一次 generate，看 prompt 形态与停止行为

不写 ckpt、不动训练数据。报告输出到 ``preflight-report-<config-stem>.md``。

用法：
    # base 主线（默认 config）
    uv run python scripts/preflight.py --config configs/sft-8g-base.yaml

    # instruct 对照
    uv run python scripts/preflight.py --config configs/sft-8g-instruct.yaml

    # 跳过最慢的第 4 层 generate（只做静态 + 单步）
    uv run python scripts/preflight.py --config configs/sft-8g-base.yaml --skip-generate
"""

from __future__ import annotations

# datasets 必须在 torch 之前 (Windows pyarrow DLL 冲突)
import datasets  # noqa: F401, I001

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))

from device import get_device
from echo.data import load_sft_data
from echo.utils import CHATML_TRAIN_TEMPLATE, load_config

console = Console()

# 单步前向用的样本数
SAMPLE_N = 3
# 微步训练数（验证 loss 下降）
MICRO_STEPS = 5
# 数据分布抽样上限（避免遍历全量耗时）
DIST_SAMPLE = 2000


# ------------------------------------------------------------------
# 通用工具
# ------------------------------------------------------------------


class PreflightFail(RuntimeError):
    """预检失败的标记异常，主流程捕获后写报告并退出。"""


# 严重警告（不抛异常但要阻断"通过"判定）累积容器
critical_warnings: list[str] = []


def section(title: str) -> None:
    console.print(Panel.fit(f"[bold cyan]{title}[/bold cyan]"))


def ok(msg: str) -> None:
    console.print(f"  [green]✓[/green] {msg}")


def warn(msg: str) -> None:
    """轻量警告：建议关注，不阻断。"""
    console.print(f"  [yellow]⚠[/yellow] {msg}")


def critical(msg: str) -> None:
    """严重警告：配置层面错误，会让训练结果偏离预期，阻断"全部通过"。"""
    console.print(f"  [red]⚠ CRITICAL[/red] {msg}")
    critical_warnings.append(msg)


def fail(msg: str) -> None:
    console.print(f"  [red]✗[/red] {msg}")
    raise PreflightFail(msg)


# ------------------------------------------------------------------
# 第 1 层：静态检查
# ------------------------------------------------------------------


def check_config(cfg: dict, report: list[str]) -> None:
    section("Layer 1.1 · Config 字段")
    required_top = ["model", "lora", "data", "training"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        fail(f"config 缺顶层字段: {missing}")
    if "model_id" not in cfg["model"]:
        fail("config.model 缺 model_id")
    is_base = cfg["model"].get("is_base_model", False)
    mts = (cfg.get("lora") or {}).get("modules_to_save")
    ok(f"model_id={cfg['model']['model_id']}, is_base_model={is_base}")
    ok(f"lora.r={cfg['lora'].get('r')}, modules_to_save={mts}")
    ok(f"data.train_file={cfg['data'].get('train_file')}")
    ok(f"training.output_dir={cfg['training'].get('output_dir')}")

    # base + lm_head 同时在 modules_to_save 是 Qwen2.5 tie 冲突的常见踩坑
    if is_base and mts and "lm_head" in mts and "embed_tokens" in mts:
        warn(
            "base 模式同时把 embed_tokens + lm_head 写进 modules_to_save，"
            "Qwen2.5-1.5B tie_word_embeddings=true，建议只留 embed_tokens"
        )
        report.append("- ⚠ base + 双模块 modules_to_save 可能破坏 tie")

    report.append(
        f"- model_id: `{cfg['model']['model_id']}`\n"
        f"- is_base_model: `{is_base}`\n"
        f"- lora.r: `{cfg['lora'].get('r')}`\n"
        f"- modules_to_save: `{mts}`\n"
        f"- train_file: `{cfg['data'].get('train_file')}`\n"
        f"- output_dir: `{cfg['training'].get('output_dir')}`"
    )


def check_tokenizer_special_tokens(tokenizer, is_base: bool, report: list[str]) -> None:
    """检查 special token + 应用 base patch（与 sft.py 训练时行为一致）。

    分两段打印：
      - 原始状态（底座加载后立即看到的 tokenizer 状态）
      - patch 后状态（base 模式下 sft.py 实际训练时的状态）
    """
    section("Layer 1.2 · Tokenizer special tokens")
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    if im_end_id is None or im_end_id < 0:
        fail("tokenizer 找不到 <|im_end|>，底座非 Qwen2.5 系列?")
    ok(f"<|im_start|> id={im_start_id}")
    ok(f"<|im_end|>   id={im_end_id}")
    ok(f"<|endoftext|> id={eot_id}")

    # 原始状态
    console.print(f"  [dim]原始 tokenizer:[/dim]")
    ok(f"  eos_token={tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    ok(f"  pad_token={tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")

    # base 模式：模拟 sft.py 的 patch
    if is_base:
        tokenizer.eos_token = "<|im_end|>"
        console.print(f"  [dim]Base patch 后（与 sft.py 训练时一致）:[/dim]")
        ok(f"  eos_token={tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
        ok(
            f"  pad_token={tokenizer.pad_token!r} (id={tokenizer.pad_token_id}) "
            f"故意保留 <|endoftext|>，与 eos 解耦"
        )

    report.append(
        f"- `<|im_start|>` id: `{im_start_id}`\n"
        f"- `<|im_end|>` id: `{im_end_id}`\n"
        f"- `<|endoftext|>` id: `{eot_id}`\n"
        f"- 训练时 eos_token: `{tokenizer.eos_token}` (id `{tokenizer.eos_token_id}`)\n"
        f"- 训练时 pad_token: `{tokenizer.pad_token}` (id `{tokenizer.pad_token_id}`)"
    )


def check_data(data_path: Path, tokenizer, report: list[str], max_seq_length: int) -> list[dict]:
    """检查数据格式 + 长度分布；返回前 SAMPLE_N 条样本供后续层用。"""
    section("Layer 1.3 · 数据格式 + 分布")
    if not data_path.exists():
        fail(f"数据文件不存在: {data_path}")

    # 行数 + 格式校验：抽前 DIST_SAMPLE 条
    samples: list[dict] = []
    role_counter: Counter[str] = Counter()
    assistant_lens: list[int] = []
    im_end_per_sample: list[int] = []
    rendered_token_lens: list[int] = []  # 渲染后整条样本 token 长度
    bad = 0
    total = 0
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            total += 1
            if i >= DIST_SAMPLE:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            msgs = rec.get("messages")
            if not msgs or not isinstance(msgs, list):
                bad += 1
                continue
            for m in msgs:
                role_counter[m.get("role", "?")] += 1
                if m.get("role") == "assistant":
                    text = m.get("content", "")
                    assistant_lens.append(len(text))
            # 渲染后看 <|im_end|> 出现次数 + 整条 token 长度
            try:
                rendered = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                im_end_per_sample.append(rendered.count("<|im_end|>"))
                # token 长度（不截断，看真实长度分布）
                token_ids = tokenizer(rendered, add_special_tokens=False).input_ids
                rendered_token_lens.append(len(token_ids))
            except Exception:  # noqa: BLE001
                bad += 1
            if len(samples) < SAMPLE_N:
                samples.append(rec)

    if bad:
        warn(f"无效记录: {bad} / {min(total, DIST_SAMPLE)}")
    ok(f"总行数: {total}（抽样 {min(total, DIST_SAMPLE)} 条做分布统计）")
    ok(f"role 分布: {dict(role_counter)}")
    if assistant_lens:
        avg = sum(assistant_lens) / len(assistant_lens)
        p50 = sorted(assistant_lens)[len(assistant_lens) // 2]
        p95 = sorted(assistant_lens)[int(len(assistant_lens) * 0.95)]
        ok(f"assistant 字符长度: avg={avg:.0f}, p50={p50}, p95={p95}")
    if im_end_per_sample:
        avg_imend = sum(im_end_per_sample) / len(im_end_per_sample)
        ok(f"渲染后每条样本 <|im_end|> 出现次数: avg={avg_imend:.2f}")
        if avg_imend < 2:
            warn("<|im_end|> 信号密度偏低，base 路线可能学不会停止")

    # token 长度分布 + 截断风险评估（关键：超 max_seq 的样本末尾 <|im_end|> 会被砍掉）
    truncate_ratio = 0.0
    if rendered_token_lens:
        sorted_lens = sorted(rendered_token_lens)
        n = len(sorted_lens)
        avg_t = sum(sorted_lens) / n
        p50_t = sorted_lens[n // 2]
        p95_t = sorted_lens[int(n * 0.95)]
        max_t = sorted_lens[-1]
        truncated = sum(1 for x in sorted_lens if x > max_seq_length)
        truncate_ratio = truncated / n
        ok(
            f"渲染后 token 长度: avg={avg_t:.0f}, p50={p50_t}, "
            f"p95={p95_t}, max={max_t} (max_seq={max_seq_length})"
        )
        msg = f"超 max_seq 比例: {truncate_ratio * 100:.1f}% ({truncated}/{n})"
        if truncate_ratio > 0.10:
            critical(msg + "；末尾 <|im_end|> 会被 right truncation 砍掉，停止信号大量丢失")
        elif truncate_ratio > 0.03:
            warn(msg + "；少量样本末尾停止信号会丢失，可接受")
        else:
            ok(msg)

    report.append(
        f"- 总行数: `{total}`（抽样 `{min(total, DIST_SAMPLE)}` 条）\n"
        f"- 无效记录: `{bad}`\n"
        f"- role 分布: `{dict(role_counter)}`\n"
        f"- assistant 字符长度 avg: `{sum(assistant_lens) / max(len(assistant_lens), 1):.0f}`\n"
        f"- 每条样本 `<|im_end|>` 平均次数: "
        f"`{sum(im_end_per_sample) / max(len(im_end_per_sample), 1):.2f}`\n"
        f"- 渲染后 token 长度 avg: "
        f"`{sum(rendered_token_lens) / max(len(rendered_token_lens), 1):.0f}` "
        f"(max_seq=`{max_seq_length}`, 超长比例 `{truncate_ratio * 100:.1f}%`)"
    )

    if not samples:
        fail("数据集采样结果为空")
    return samples


def check_chat_template_render(samples: list[dict], tokenizer, report: list[str]) -> None:
    section("Layer 1.4 · chat_template 渲染")
    # 用训练模板（含 {% generation %}）来检查
    saved_template = tokenizer.chat_template
    tokenizer.chat_template = CHATML_TRAIN_TEMPLATE
    try:
        rendered = tokenizer.apply_chat_template(
            samples[0]["messages"], tokenize=False, add_generation_prompt=False
        )
    finally:
        tokenizer.chat_template = saved_template

    head = rendered[:500]
    tail = rendered[-200:] if len(rendered) > 700 else ""
    ok(f"渲染长度: {len(rendered)} 字符")

    # 关键 token 必须存在
    for tok in ("<|im_start|>system", "<|im_start|>user", "<|im_start|>assistant", "<|im_end|>"):
        if tok in rendered:
            ok(f"含 `{tok}`")
        else:
            fail(f"渲染结果缺 `{tok}`")

    console.print("\n[dim]--- 渲染样本（前 500 字符）---[/dim]")
    console.print(head)
    if tail:
        console.print("[dim]--- 末尾 200 字符 ---[/dim]")
        console.print(tail)

    report.append(
        "### 渲染样本（首条）\n```\n" + head + ("\n...\n" + tail if tail else "") + "\n```"
    )


# ------------------------------------------------------------------
# 第 2 层：模型加载
# ------------------------------------------------------------------


def load_base_model(cfg: dict, force_cpu: bool):
    """加载底座（4bit 或全精度）。返回 (model, dtype_str)。"""
    model_id = cfg["model"]["model_id"]
    quant_cfg = cfg.get("quantization") or {}
    use_4bit = quant_cfg.get("enabled", False) and not force_cpu

    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        return model, "4bit-nf4"
    else:
        device = "cpu" if force_cpu else get_device()
        train_cfg = cfg.get("training") or {}
        if train_cfg.get("bf16"):
            dtype = torch.bfloat16
        elif train_cfg.get("fp16"):
            dtype = torch.float16
        else:
            dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        return model, str(dtype).replace("torch.", "")


def check_model_load(cfg: dict, force_cpu: bool, report: list[str]):
    section("Layer 2.1 · 模型加载")
    console.print(f"  loading {cfg['model']['model_id']} ...")
    model, dtype_str = load_base_model(cfg, force_cpu=force_cpu)
    ok(f"loaded ({dtype_str})")

    # tie 状态（PEFT 包装前）
    embed_w = model.get_input_embeddings().weight
    head_w = model.get_output_embeddings().weight
    is_tied_pre = embed_w.data_ptr() == head_w.data_ptr()
    ok(f"PEFT 前 tie_word_embeddings: {is_tied_pre}")

    report.append(
        f"- 加载精度: `{dtype_str}`\n"
        f"- PEFT 前 embed/lm_head tied: `{is_tied_pre}`"
    )
    return model, is_tied_pre


def check_peft_wrap(cfg: dict, model, is_tied_pre: bool, report: list[str]):
    section("Layer 2.2 · PEFT 包装 + LoRA")
    quant_cfg = cfg.get("quantization") or {}
    if quant_cfg.get("enabled", False):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg["training"].get("gradient_checkpointing", False)
        )

    lora_cfg = cfg["lora"]
    lora_kwargs = dict(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        modules_to_save=lora_cfg.get("modules_to_save"),
        bias="none",
    )
    if lora_cfg.get("ensure_weight_tying", True):
        lora_kwargs["ensure_weight_tying"] = True
    lora = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora)
    if cfg["training"].get("gradient_checkpointing", False) and hasattr(
        model, "enable_input_require_grads"
    ):
        model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    ok(f"trainable: {trainable:,} / {total:,} ({pct:.2f}%)")

    # tie 状态（PEFT 包装后）
    peft_base = model.base_model.model
    embed_w = peft_base.get_input_embeddings().weight
    head_w = peft_base.get_output_embeddings().weight
    is_tied_post = embed_w.data_ptr() == head_w.data_ptr()
    if is_tied_pre and not is_tied_post:
        critical(
            "tie 被 PEFT 破坏：embed/lm_head 不再共享权重。"
            "应在 LoraConfig 加 ensure_weight_tying=True，或显式 tie_word_embeddings=False"
        )
    else:
        ok(f"PEFT 后 tie_word_embeddings: {is_tied_post}")

    # base 路线参数占比理论值校验
    # base 路线（modules_to_save=embed_tokens + LoRA r=128 7 proj）实测 ~30%
    # instruct 路线（仅 LoRA r=64 7 proj）实测 ~2%
    # 偏离区间 → 配置出错（target_modules 拼写错 / modules_to_save 没生效 / tie 拆开）
    is_base = cfg["model"].get("is_base_model", False)
    if is_base:
        if pct < 15:
            critical(
                f"base 路线 trainable 占比 {pct:.2f}% 偏低（预期 20-40%），"
                "modules_to_save=embed_tokens 可能未生效；继续训会变成纯 LoRA SFT，"
                "<|im_end|> embedding 学不到，停止信号失败"
            )
        elif pct > 50:
            critical(
                f"base 路线 trainable 占比 {pct:.2f}% 偏高（预期 20-40%），"
                "可能 tie 已断开 lm_head 也被独立训练，adapter 体积会翻倍"
            )
        else:
            ok(f"trainable 占比 {pct:.2f}% 在 base 路线预期区间 (20-40%)")
    else:
        if pct < 1.0:
            critical(
                f"instruct 路线 trainable 占比 {pct:.2f}% 偏低（预期 1.5-4%），"
                "target_modules 可能拼写错没命中 LoRA"
            )
        elif pct > 5:
            warn(
                f"instruct 路线 trainable 占比 {pct:.2f}% 偏高（预期 1.5-4%），"
                "确认 modules_to_save 是否误开"
            )
        else:
            ok(f"trainable 占比 {pct:.2f}% 在 instruct 路线预期区间 (1.5-4%)")

    report.append(
        f"- trainable: `{trainable:,}` / `{total:,}` (`{pct:.2f}%`)\n"
        f"- PEFT 后 embed/lm_head tied: `{is_tied_post}`"
    )
    return model


# ------------------------------------------------------------------
# 第 3 层：单步前向 + loss mask 可视化 + 微步训练
# ------------------------------------------------------------------


def visualize_loss_mask(samples: list[dict], tokenizer, report: list[str]) -> None:
    """模拟 trl 的 assistant_only_loss：用 return_assistant_tokens_mask 拿到 mask 并可视化。"""
    section("Layer 3.1 · loss mask 可视化")
    saved = tokenizer.chat_template
    tokenizer.chat_template = CHATML_TRAIN_TEMPLATE
    try:
        result = tokenizer.apply_chat_template(
            samples[0]["messages"],
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
    finally:
        tokenizer.chat_template = saved

    input_ids = result["input_ids"]
    mask = result.get("assistant_masks") or result.get("assistant_tokens_mask")
    if mask is None:
        fail(
            "tokenizer 未返回 assistant_masks，可能 chat_template 缺 {% generation %} 标记 "
            "或 transformers 版本过低"
        )

    n_total = len(input_ids)
    n_loss = sum(mask)
    pct = 100 * n_loss / n_total if n_total else 0
    ok(f"总 token: {n_total}, 计入 loss 的 assistant token: {n_loss} ({pct:.1f}%)")

    if pct < 5:
        warn(f"loss 覆盖率仅 {pct:.1f}%，assistant 段是否过短或 chat_template 标记错位？")

    # 验证最后一个 assistant 段尾部的 <|im_end|> 也进入 loss
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_end_in_loss = sum(
        1 for tid, m in zip(input_ids, mask) if tid == im_end_id and m == 1
    )
    if im_end_in_loss > 0:
        ok(f"<|im_end|> 进入 loss 计算的次数: {im_end_in_loss}（停止信号会被学到 ✓）")
    else:
        warn("<|im_end|> 未进入 loss！模型学不到停止行为")

    # 可视化：每个 token 标记 [L] (loss) / [_] (masked)，前 80 个
    table = Table(title="前 80 个 token 的 loss mask", show_header=True)
    table.add_column("idx", style="dim", width=4)
    table.add_column("tok_id", width=8)
    table.add_column("repr", width=20)
    table.add_column("mask", style="bold", width=4)
    for i, (tid, m) in enumerate(zip(input_ids[:80], mask[:80])):
        flag = "[green]L[/green]" if m else "[dim]_[/dim]"
        repr_str = tokenizer.decode([tid]).replace("\n", "\\n")[:18]
        table.add_row(str(i), str(tid), repr_str, flag)
    console.print(table)

    report.append(
        f"- 总 token: `{n_total}`\n"
        f"- 计入 loss: `{n_loss}` (`{pct:.1f}%`)\n"
        f"- `<|im_end|>` 进入 loss 次数: `{im_end_in_loss}`"
    )


def run_micro_steps(
    model, tokenizer, samples: list[dict], cfg: dict, report: list[str]
) -> None:
    section("Layer 3.2 · 微步训练（5 step，看 loss 是否下降）")
    saved = tokenizer.chat_template
    tokenizer.chat_template = CHATML_TRAIN_TEMPLATE
    device = next(model.parameters()).device

    # 构造 batch
    batches: list[dict] = []
    try:
        for s in samples:
            enc = tokenizer.apply_chat_template(
                s["messages"],
                tokenize=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
                return_dict=True,
                truncation=True,
                max_length=cfg["training"].get("max_seq_length", 2048),
            )
            ids = torch.tensor(enc["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
            mask_list = enc.get("assistant_masks") or enc.get("assistant_tokens_mask")
            mask = torch.tensor(mask_list, dtype=torch.long).unsqueeze(0).to(device)
            labels = ids.clone()
            labels[mask == 0] = -100
            batches.append({"input_ids": ids, "labels": labels})
    finally:
        tokenizer.chat_template = saved

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"].get("learning_rate", 2e-4),
    )

    losses: list[float] = []
    model.train()
    for step in range(MICRO_STEPS):
        batch = batches[step % len(batches)]
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(float(loss.item()))
        ok(f"step {step + 1}: loss = {losses[-1]:.4f}")

    if losses[-1] < losses[0]:
        ok(f"loss 下降: {losses[0]:.4f} → {losses[-1]:.4f}（训练通路 OK）")
    else:
        warn(f"loss 未下降: {losses[0]:.4f} → {losses[-1]:.4f}（学习率 / 数据 / 模型 排查）")

    report.append(
        "### 微步训练 loss 序列\n```\n"
        + "\n".join(f"step {i + 1}: {v:.4f}" for i, v in enumerate(losses))
        + "\n```"
    )


# ------------------------------------------------------------------
# 第 4 层：generate 试探（底座未训，仅看 prompt 形态 + 停止行为）
# ------------------------------------------------------------------


def check_generate(model, tokenizer, cfg: dict, report: list[str]) -> None:
    section("Layer 4 · 底座 generate 试探（输出会乱，看形态即可）")
    # 注：tokenizer 的 base patch 已在 Layer 1.2 应用，此处直接用即可
    is_base = cfg["model"].get("is_base_model", False)

    # 用推理模板拼 prompt
    from echo.utils import CHATML_INFER_TEMPLATE

    saved = tokenizer.chat_template
    tokenizer.chat_template = CHATML_INFER_TEMPLATE
    try:
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "你好"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    finally:
        tokenizer.chat_template = saved

    ok(f"prompt 末尾: ...{text[-80:]!r}")
    if not text.rstrip().endswith("<|im_start|>assistant"):
        warn("prompt 末尾不是 <|im_start|>assistant，chat_template 渲染可能错位")

    inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    stop_ids = [
        tid
        for tid in (
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        )
        if tid is not None and tid >= 0
    ]

    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=False)
    last_tok_id = gen_ids[-1] if gen_ids else None
    stopped_naturally = last_tok_id in stop_ids

    ok(f"生成 token 数: {len(gen_ids)} / 64 max")
    ok(f"末尾 token id: {last_tok_id} (in stop_ids? {stopped_naturally})")
    console.print("\n[dim]--- 生成内容（含 special token）---[/dim]")
    console.print(decoded[:400])

    if not stopped_naturally and len(gen_ids) >= 64:
        if is_base:
            ok("base 未训 + 不自然停止是预期（这正是 SFT 要解决的）")
        else:
            warn("instruct 模型未自然停止？检查 chat_template / stop_ids 配置")

    report.append(
        f"- 生成 token 数: `{len(gen_ids)}`\n"
        f"- 末尾 token id: `{last_tok_id}` (in stop_ids: `{stopped_naturally}`)\n"
        "### 生成片段\n```\n" + decoded[:400] + "\n```"
    )


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="echo SFT 预检")
    parser.add_argument("--config", type=Path, required=True, help="YAML config 路径")
    parser.add_argument("--cpu", action="store_true", help="强制 CPU（绕开 CUDA / 量化）")
    parser.add_argument("--skip-generate", action="store_true", help="跳过 Layer 4")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="报告输出路径（默认 preflight-report-<config-stem>.md）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    report: list[str] = [
        f"# Preflight Report · {args.config.name}",
        "",
        "## 1. Config",
        "",
    ]

    # 加载 tokenizer。base 模式的 eos_token patch 推迟到 Layer 1.2 内部应用
    # （那里会先打印原始状态再打印 patch 后状态，便于诊断）。
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["model_id"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    is_base = cfg["model"].get("is_base_model", False)

    try:
        # Layer 1
        check_config(cfg, report)
        report.append("\n## 2. Tokenizer\n")
        check_tokenizer_special_tokens(tokenizer, is_base, report)

        report.append("\n## 3. 数据\n")
        max_seq_length = cfg["training"].get("max_seq_length", 2048)
        samples = check_data(
            Path(cfg["data"]["train_file"]), tokenizer, report, max_seq_length
        )

        report.append("\n## 4. Chat template 渲染\n")
        check_chat_template_render(samples, tokenizer, report)

        # Layer 2
        report.append("\n## 5. 模型加载\n")
        model, is_tied_pre = check_model_load(cfg, args.cpu, report)
        report.append("\n## 6. PEFT 包装\n")
        model = check_peft_wrap(cfg, model, is_tied_pre, report)

        # Layer 3
        report.append("\n## 7. Loss mask 可视化\n")
        visualize_loss_mask(samples, tokenizer, report)

        report.append("\n## 8. 微步训练\n")
        run_micro_steps(model, tokenizer, samples, cfg, report)

        # Layer 4
        if not args.skip_generate:
            report.append("\n## 9. 底座 generate 试探\n")
            check_generate(model, tokenizer, cfg, report)

        if critical_warnings:
            report.append("\n---\n\n## 严重警告\n")
            for w in critical_warnings:
                report.append(f"- ⚠ {w}")
            report.append(
                f"\n**预检未通过 ✗**：存在 {len(critical_warnings)} 个严重警告，"
                "训练结果会偏离预期，请先修复后再跑 sft.py。"
            )
            console.print(
                Panel.fit(
                    f"[bold red]预检未通过 ✗ · {len(critical_warnings)} 个严重警告[/bold red]\n"
                    + "\n".join(f"  - {w}" for w in critical_warnings)
                )
            )
            report_path = args.report or Path(f"preflight-report-{args.config.stem}.md")
            report_path.write_text("\n".join(report), encoding="utf-8")
            console.print(f"报告写入: {report_path}")
            raise SystemExit(2)

        report.append("\n---\n\n**预检通过 ✓** 可进入正式训练。")
        console.print(Panel.fit("[bold green]预检全部通过 ✓[/bold green]"))
    except PreflightFail as e:
        report.append(f"\n---\n\n**预检失败 ✗**: {e}")
        console.print(Panel.fit(f"[bold red]预检失败: {e}[/bold red]"))
        report_path = args.report or Path(f"preflight-report-{args.config.stem}.md")
        report_path.write_text("\n".join(report), encoding="utf-8")
        console.print(f"报告写入: {report_path}")
        raise SystemExit(1)

    report_path = args.report or Path(f"preflight-report-{args.config.stem}.md")
    report_path.write_text("\n".join(report), encoding="utf-8")
    console.print(f"\n报告写入: [bold]{report_path}[/bold]")


if __name__ == "__main__":
    main()
