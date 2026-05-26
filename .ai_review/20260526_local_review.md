# Git Code Review Report

**审查范围**：本地未提交变更
**审查日期**：2026-05-26

---

## 总览

本次变更核心意图：
1. 将重复的推理代码（load_model / generate）抽取为 `echo_mini.inference` 模块，chat.py / evaluate.py 复用
2. 移除未使用的 `dropout` 配置字段
3. 训练脚本引入 `loss_count` 变量（意图更健壮地计算 avg_loss）
4. prepare_data tokenize 阶段在每篇文档末尾追加 `<eos>` token
5. 统一使用 `shared/device.py::get_device()` 替代硬编码设备选择
6. 小幅代码清理（import 排序、注释、noqa）

---

## ⚠️ 问题

### 🔴 [阻断] `loss_count` 引入但未使用 — avg_loss 计算仍除以 `log_interval`

**文件**：`pretrain.py` L167, `sft.py` L223

新增了 `loss_count` 累加器并在每次 log 后归零，但 `avg_loss` 的计算仍为：
```python
avg_loss = total_loss / log_interval
```
而非 `total_loss / loss_count`。

**影响分析**：
- 若 `grad_accumulation_steps > 1`，每个 step 实际执行了 `accumulation_steps` 次 forward，但 `loss.item()` 在每次 micro-step 都累加 → `loss_count` 会大于 `log_interval`，二者不等。
- 如果原意是保持旧行为（log_interval == 实际 loss 采样次数），则 `loss_count` 完全冗余，应删除。
- 如果原意是修正 grad accumulation 下的统计，则 `avg_loss` 应改为 `total_loss / loss_count`。

**当前状态**：`loss_count` 是死代码，不会引入 Bug，但传达了错误的设计意图。属于"改了一半"的状态，建议二选一完成。

**严重性**：降级为 🟡（不会崩溃，但 loss 日志在 grad_accum > 1 时不精确——与变更前行为一致，不算新引入 Bug）。

---

### 🟡 [中等] 移除 `dropout` 字段后旧 YAML 兼容性

**文件**：`config.py`（删除 `dropout` 字段），`pretrain.py` L41 / `sft.py` L47

`EchoMiniConfig(**cfg.get("model", {}))` 使用 dataclass kwargs 构造。若有人使用仍含 `dropout` 键的旧配置文件，会得到：
```
TypeError: EchoMiniConfig.__init__() got an unexpected keyword argument 'dropout'
```

**本次 diff 内所有 YAML 已同步删除**，所以 git 跟踪的配置不会出错。但：
- 用户本地可能有未入库的自定义 YAML 副本
- 如果以后想向前兼容，可考虑在 `load_config` 或构造前 `pop` 未知字段

**建议**：无需立即修复（风险可控），但可在 `EchoMiniConfig` 上加文档注释说明"dropout 已移除"，方便排查。

---

### 🟡 [中等] `inference.py` 内 `sys.path.insert` 指向 `shared/` 可能在包导入时意外触发

**文件**：`inference.py` L16

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
```

`inference.py` 被 `__init__.py` 导入 → 任何 `import echo_mini` 都会执行这行，包括训练脚本。训练脚本自身也做了 `sys.path.insert(shared/)`，不会出错，但这个全局副作用在库包模块里不太干净。

**影响**：功能正常，不会 Bug。属于代码卫生问题。

---

### 🟢 [低] `evaluate.py` 中 `compute_ppl` 的 `model` 参数类型标注从 `EchoMini` 改为裸 `model`

**文件**：`evaluate.py` L69

```python
def compute_ppl(model, tokenizer, ...)
```

丢失了类型提示，IDE 不能补全。建议保留 `model: EchoMini` 或至少 `model: nn.Module`。

---

### 🟢 [低] `pyproject.toml` CRLF 警告

Git 显示 `warning: CRLF will be replaced by LF the next time Git touches it`。配置要求 LF，下次 commit 会自动转换，无实际问题。

---

## ✅ 正面评价

1. **推理代码去重** — chat.py / evaluate.py 原先各自复制了 70+ 行相同的 generate/load_model 逻辑，抽取到 `inference.py` 是正确做法
2. **设备选择统一** — 移除硬编码 `torch.device("cuda")` 改用 `get_device()`，符合项目铁律
3. **tokenize 阶段加 `<eos>`** — 让预训练数据有明确的文档边界信号，对模型学习终止生成有帮助
4. **`dropout` 移除** — echo-mini 模型体积小且未实际使用 dropout，删除减少配置噪音
5. **`inference.py` 的 `generate` 函数** — device 参数 None 时从 model 参数推断，API 设计合理

---

## 总结

| 级别 | 数量 | 说明 |
|------|------|------|
| 🔴 阻断 | 0 | — |
| 🟡 中等 | 3 | loss_count 半成品 / dropout 兼容 / sys.path 副作用 |
| 🟢 低 | 2 | 类型标注丢失 / CRLF |

**总体评价**：变更整体方向正确，不存在会导致运行崩溃的破坏性 Bug（前提是使用 git 跟踪的配置文件）。最值得关注的是 `loss_count` 引入不完整——虽然不会崩，但如果后续依赖它计算 avg_loss 而忘记同步修改除数，会产生统计错误。建议要么删掉 `loss_count`，要么把 `avg_loss = total_loss / log_interval` 改为 `total_loss / loss_count`。
