# Git Code Review Report

**审查范围**：本地未提交变更（Echo/echo SFT 双路线改造：base 主线 + instruct 对照）
**审查日期**：2026-05-28

---

## 变更概览

| 类型 | 文件 |
|------|------|
| 修改 .py | `scripts/eval.py` `scripts/eval_loss.py` `scripts/export_gguf.py` `scripts/generate.py` `scripts/merge.py` `scripts/prepare_data.py` `scripts/run_eval_pipeline.py` `scripts/sft.py` `src/echo/utils.py` |
| 删除 .yaml | `configs/sft-8g.yaml` `configs/sft-full.yaml` `configs/sft-tiny.yaml` `configs/sft-titanxp.yaml` |
| 新增 .yaml | `configs/sft-{8g,full,tiny}-{base,instruct}.yaml` 共 6 份 |
| 文档（不在 review 范围） | `SPEC.md` `stepshooting*.md` |

主线改动一句话：**引入 base 路线为项目锚点（默认配置）、instruct 路线降级为对照实验**，所有脚本默认值切到 `sft-base/` checkpoint 与 `train_aug.jsonl` 数据。

---

## 1. `scripts/sft.py` — base 模式补丁

### ✅ 正面评价

- `is_base_model` 分支显式校验 `<|im_end|>` token 存在，校验失败直接 `RuntimeError`，提示明确。
- `enable_input_require_grads()` 调用条件清晰、注释解释了 4bit 路径已处理 + 非量化兜底的取舍，避免了 PEFT 常见的 "element 0 of tensors does not require grad" 报错。
- 收尾保存时切换到 `CHATML_INFER_TEMPLATE`（推理版，无 `{% generation %}`），让下游 `generate / eval` 直接 `from_pretrained(adapter_dir)` 就拿到正确模板，闭环漂亮。

### ⚠️ 问题

**[中等] `modules_to_save` 同时含 `embed_tokens` + `lm_head` 与 Qwen2.5-1.5B 的 `tie_word_embeddings=True` 冲突**

Qwen2.5-1.5B（含 base / instruct）`config.tie_word_embeddings = True`，即 lm_head 与 embed_tokens 共享同一权重张量。PEFT 的 `modules_to_save` 会为列出的每个模块创建独立的可训练副本（`ModulesToSaveWrapper`），**两次包装会破坏 tie**：训练后 lm_head 与 embed_tokens 不再共享权重，adapter 体积从 ~300MB 翻到 ~600MB（与注释里写的 600MB 吻合，但代价是无意义的副本）。

文件位置：`configs/sft-8g-base.yaml:37-39`、`configs/sft-full-base.yaml:27-29`、`configs/sft-tiny-base.yaml:24-26`

建议二选一：

- **推荐**：只保留 `embed_tokens`（lm_head 通过 tie 自动跟随更新）。验证方法：`print(model.base_model.model.lm_head.weight.data_ptr() == model.base_model.model.model.embed_tokens.weight.data_ptr())` 看是否仍 tie。
- 或者 显式 `model.config.tie_word_embeddings = False` 后再都训，但这才是真正"两份独立权重"的语义，需自行权衡。

注：PEFT 在某些版本（≥0.7）对 tied weights 处理有改进，但行为依赖 transformers 版本。建议加载完 `get_peft_model` 后打印一行 `trainable_params` 与 `lm_head` 是否仍 tie，避免静默翻倍。

**[低] `tokenizer.pad_token = tokenizer.eos_token` 在 base 模式下顺序问题**

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # 此时 eos 还是 <|endoftext|>
if model_cfg.get("is_base_model", False):
    tokenizer.eos_token = im_end                # 改 eos 后，pad_token 仍指向 <|endoftext|> 字符串/id
```

这里 pad 与 eos 在 base 模式下变成不同的 token，反而是想要的（pad 用 `<|endoftext|>`、eos 用 `<|im_end|>`，避免 pad 与对话终止符混淆）。但此意图未写注释，未来维护者容易误以为是 bug 把它"修正"。建议补一行注释明确"pad 故意保留 `<|endoftext|>`，与 eos 解耦"。

---

## 2. `src/echo/utils.py` — `load_inference_tokenizer` 抽取

### ✅ 正面评价

- 多源 fallback（merged > adapter > base）结合 `for/else` 写法干净；`chat_template` / `pad_token` 兜底集中在一处，消除了 5 个脚本里的重复初始化代码。
- `CHATML_INFER_TEMPLATE` 与 `sft.py` 的训练模板对齐（仅去掉 `{% generation %}`），保证 prompt 拼接形态训练-推理一致。

### ⚠️ 问题

**[低] fallback 语义不一定符合直觉**

当用户传 `merged_dir=X`（X 存在但 tokenizer 文件缺失）时，函数会**静默回落到 base_model_id**，导致用户以为加载了 merged 的 tokenizer，实际上拿到的是底座原始 tokenizer（chat_template 也是底座的）。

这种场景在生产中确实可能踩坑：例如 export_gguf 中途产物。建议至少在每次成功加载时打印一行 `console.print(f"tokenizer loaded from {kind}: {src}")`，避免静默降级。

**[低] `except Exception` 过宽**

`except Exception as e: ...` 会吞掉 `KeyboardInterrupt` 之外的所有异常（包括磁盘错误、权限错误等真实问题）。建议至少缩到 `except (OSError, ValueError, EnvironmentError)`，让真正的环境问题尽快暴露。当前实现可接受，属于"先求通后求精"。

---

## 3. `scripts/prepare_data.py` — 数据增广

### ✅ 正面评价

- 分函数：`load_sharegpt` / `load_short_qa` / `split_and_write`，主流程一目了然。
- `--mode aug | sharegpt-only` 二选一，输出文件名分离（`train_aug.jsonl` vs `train.jsonl`），base / instruct 路线物理隔离。
- `short_qa_filter` 在 `quality_filter` 基础上加 assistant 长度上限，防止短问答数据集里混入长输出污染风格。

### ⚠️ 问题

**[中等] 中文数据集字段自适应的 `ds_zh[0]` 可能 KeyError**

```python
sample = ds_zh[0]
zh_keys = ("instruction_zh", "input_zh", "output_zh")
en_keys = ("instruction", "input", "output")
keys = zh_keys if all(k in sample for k in zh_keys) else en_keys
```

`silk-road/alpaca-data-gpt4-chinese` 的实际字段是 `instruction_zh / input_zh / output_zh`（你的注释也这么说）。但若上游数据集结构变更（HF 数据集偶尔会动），`en_keys` 兜底也可能不存在 → `convert_alpaca_to_messages` 里 `example.get(...)` 全返回空 → 全部被过滤 → 输出"采样: 0"，用户难以察觉。

建议在选定 `keys` 后打印 sample 一行实际字段并校验：

```python
missing = [k for k in keys if k not in sample]
if missing:
    raise RuntimeError(f"中文短问答数据集字段缺失: {missing}, 实际字段: {list(sample.keys())}")
```

**[低] `--max-samples` 仅作用于 ShareGPT，命名歧义**

参数注释已改为 "ShareGPT 最多保留条数"，但 `--mode aug` 下，短问答两路有自己的 `--short-qa-en/zh` 默认 5000+5000，而 ShareGPT 默认 20000。三者拼起来是 30000 总量，与"max-samples"字面含义有 gap。可接受，但建议把英文/中文短问答的默认值也写进 docstring 用法示例，避免用户调一个参数以为限制了总量。

---

## 4. `scripts/eval.py` / `eval_loss.py` / `generate.py` / `merge.py` / `export_gguf.py` / `run_eval_pipeline.py` — 默认值切换

### ✅ 正面评价

- 所有脚本默认 config 统一切到 `sft-8g-base.yaml`、checkpoint 切到 `sft-base/final`、val 文件切到 `val_aug.jsonl`，与"base 路线为锚点"的设计一致。
- 顶部 docstring 都补了"base 路线（默认）/ instruct 对照"两段示例，user-facing 文档同步更新。
- `eval.py` / `generate.py` 中 `_get_stop_token_ids` 加了 `tid is not None and tid >= 0` 过滤 + `eos_token_id` 兜底，正面应对 `convert_tokens_to_ids` 对未注册 token 返回 unk_id 的边界。

### ⚠️ 问题

**[低] `merge.py` 的 `DEFAULT_OUTPUT` 改到 `merged-base/`，但 `export_gguf.py` 仍写死 `merged-base → gguf-base`，instruct 路线需要传 `--merged-dir checkpoints/merged --output-dir checkpoints/gguf` 双参数**

属于设计选择（base 路线为默认），但 `export_gguf.py` 用法注释里 instruct 的命令行较长。可接受，无修改建议；提及只为让用户知晓。

**[低] `run_eval_pipeline.py` 的 `eval_single` 改用 `load_inference_tokenizer(adapter_dir=adapter_dir, base_model_id=model_id)`**

每个 checkpoint 循环都加载一次 tokenizer。adapter_dir 加载失败回落到 base，但 base tokenizer 在循环内重复加载是浪费（虽然走的是 HF 缓存，第二次很快）。优化优先级低，仅在 ckpt 数 ≥ 10 时可见，可暂不处理。

---

## 5. 配置文件（新增 6 份 yaml）

### ✅ 正面评价

- base / instruct 命名约定清晰：`sft-{显存档}-{路线}.yaml`，6 份配置矩阵完整。
- `sft-8g-base.yaml` 头部注释列出"关键差异 vs sft-8g-instruct"，对照点（is_base_model / r 翻倍 / modules_to_save / num_epochs / output_dir）一目了然。
- `sft-tiny-base.yaml` 保留了 `modules_to_save` 与 `is_base_model`，确实在验证 base 代码分支完整性，符合 tiny 配置的"代码验证而非效果验证"定位。

### ⚠️ 问题

**[中等] `sft-full-base.yaml` 的 `gradient_checkpointing: false` + `modules_to_save` 显存评估存疑**

`per_device_batch_size=2` + `seq_len=2048` + Qwen2.5-1.5B + LoRA r=128 + `embed_tokens/lm_head` 全量训练（参数量 ~233M），关掉 grad ckpt 后激活值显存：

- 1.5B base 4bit ≈ 0.9GB
- LoRA r=128 fp32 grad+optim ≈ ~3-4GB
- modules_to_save 233M fp32 grad+optim（paged AdamW 8bit 略减）≈ ~1.5GB
- 激活值（bs=2, seq=2048, 28 层 × ~hidden=1536）grad ckpt 关闭 ≈ 4-6GB

总和接近 10-12GB，**RTX 3060 12GB 边界相当紧**。`sft-8g-base.yaml` 用 grad_ckpt + bs=1 是稳妥的；`sft-full-base.yaml` 关 grad_ckpt + bs=2 可能 OOM，建议在头部注释里明确"先试跑 100 步再放训"，或备一个 fallback 注释（`若 OOM，切回 bs=1 + grad_ckpt=true`）。

**[低] `output_dir` 命名不一致**

- `sft-tiny-base.yaml` → `checkpoints/sft-tiny-base`
- `sft-tiny-instruct.yaml` → `checkpoints/sft-tiny`（仍是旧名）
- `sft-8g-base.yaml` / `sft-full-base.yaml` → 都是 `checkpoints/sft-base`（两个 base 配置共用一个目录，可能互相覆盖 final/）
- `sft-8g-instruct.yaml` / `sft-full-instruct.yaml` → 都是 `checkpoints/sft`

base / instruct 主路线 8g 和 full 共用同名 output_dir 是有意设计（同一路线 8g 验证完了换 full 继续训），但 tiny 路径命名两边不对称（一个带 `-base` 后缀一个不带）。建议统一为 `sft-tiny-base` / `sft-tiny-instruct`，或都不带后缀按 train_file 区分。

---

## 总结

| 级别 | 数量 | 说明 |
|------|------|------|
| 🔴 阻断 | 0 | — |
| 🟡 中等 | 3 | tied weights 重复包装、中文数据集字段防御、full-base 显存评估 |
| 🟢 低 | 6 | 注释补充、命名一致性、fallback 日志、异常缩窄等 |

**总体评价**：

这次 base/instruct 双路线改造结构清晰，default 切换覆盖到位，`load_inference_tokenizer` 抽取消除了 5 个脚本的重复代码，`CHATML_INFER_TEMPLATE` 训练-推理对齐的设计闭环漂亮。新增的 6 份 yaml 命名规整、对照清晰，注释也充分讲明了"为什么这么调参"。

主要风险点集中在 **modules_to_save 与 Qwen2.5 tied weights 的潜在冲突**：建议在第一次跑 base 路线训练前，主动验证一下 `embed_tokens` / `lm_head` 是否仍共享权重，必要时只保留 embed_tokens 即可省一半 adapter 体积。其余都是注释/防御性问题，不阻塞合并。

建议合并前先 push 到 feature 分支，跑一次 `sft-tiny-base.yaml` 20 步验证（同时验 base 模式补丁与 modules_to_save tied 行为），再上 8g/full。
