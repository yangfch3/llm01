"""echo 工具函数。"""

from __future__ import annotations

from pathlib import Path

import yaml
from transformers import AutoTokenizer

# 推理专用 ChatML 模板（无 trl 训练标记 `{% generation %}`）。
# 仅在 adapter / merged / base 三处都拿不到 chat_template 时兜底使用，
# 与 sft.py 的训练模板逻辑对齐，确保 prompt 拼接形态与训练一致。
CHATML_INFER_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message.role == 'user' or (message.role == 'system' and not loop.first) %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)

# 训练专用 ChatML 模板：在 assistant 内容外包 {% generation %} 标记，
# 让 trl 的 assistant_only_loss 能识别哪些 token 算 loss。
# sft.py / preflight.py 共用。
CHATML_TRAIN_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message.role == 'user' or (message.role == 'system' and not loop.first) %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{% generation %}"
    "{{- message.content + '<|im_end|>\\n' }}"
    "{% endgeneration %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


def load_config(config_path: Path) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_inference_tokenizer(
    *,
    adapter_dir: Path | None = None,
    merged_dir: Path | None = None,
    base_model_id: str | None = None,
):
    """统一的推理 tokenizer 加载逻辑。

    优先级：merged_dir > adapter_dir > base_model_id。
    chat_template 缺失时回落到 ``CHATML_INFER_TEMPLATE``，pad_token 自动补 eos。

    至少需提供其中一个来源。
    """
    sources: list[tuple[str, str]] = []
    if merged_dir is not None:
        sources.append(("merged", str(merged_dir)))
    if adapter_dir is not None:
        sources.append(("adapter", str(adapter_dir)))
    if base_model_id is not None:
        sources.append(("base", base_model_id))
    if not sources:
        raise ValueError("至少需提供 adapter_dir / merged_dir / base_model_id 之一")

    last_err: Exception | None = None
    tokenizer = None
    loaded_kind: str | None = None
    loaded_src: str | None = None
    for kind, src in sources:
        try:
            tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
            loaded_kind, loaded_src = kind, src
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    if tokenizer is None:
        raise RuntimeError(f"全部 tokenizer 来源加载失败: {last_err}")

    # 静默降级是历史踩坑点（merged_dir 缺 tokenizer 时回落到 base，用户难察觉）。
    # 显式打印实际加载源，避免误判。
    print(f"[load_inference_tokenizer] loaded from {loaded_kind}: {loaded_src}")

    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = CHATML_INFER_TEMPLATE

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
