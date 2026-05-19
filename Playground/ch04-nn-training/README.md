# Playground/ch04-nn-training

ch04 课件配套练习。课件见 [`Doc/Courseware/ch04-nn-training/README.md`](../../Doc/Courseware/ch04-nn-training/README.md)。

| 脚本 | 内容 | 跑法 |
|---|---|---|
| `01_init_compare.py` | 朴素 / Xavier / He 初始化在 10 层 ReLU MLP 上的激活方差对比 | `uv run python Playground/ch04-nn-training/01_init_compare.py` |
| `02_optimizer_compare.py` | SGD / Momentum / Adam / AdamW 在 sin 拟合任务上的 loss 对比（加 `--plot` 出图） | 同上，加 `--plot` 生成走势图 |
| `03_lr_schedule.py` | 四种 lr schedule 的 ASCII 曲线（Constant / Step / Cosine / Warmup+Cosine），加 `--plot` 出图 | 同上，加 `--plot` 生成走势图 |
| `04_dropout_bn_ln.py` | Dropout 训/推差异；小 batch 下 BN 漂移 vs LN 稳定 | 同上 |

通过标准：每个脚本独立跑通，最后一行打印 `PASS`。

## 备注

- 全部脚本不依赖外部数据，3060 / Mac / 纯 CPU 都秒级跑完。
- `02_optimizer_compare.py` 用了 `get_device()`，其它脚本是纯 CPU 演示性质（无需 GPU）。
- 这些脚本是"现象演示"而非"训练任务"，目标是肉眼/打印观察差异，不是刷指标。
