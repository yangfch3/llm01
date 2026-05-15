"""ch04 练习 4：Dropout 训/推差异 + BN/LN 对小 batch 的鲁棒性。

两段独立 demo：
1. Dropout：同输入在 model.train() 多次结果不同、model.eval() 完全确定
2. BN vs LN：用极小 batch=2 的输入，比较 BN running stats 漂移和 LN 的稳定
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn


def demo_dropout() -> None:
    print("\n--- 1. Dropout 训/推差异 ---")
    torch.manual_seed(0)
    drop = nn.Dropout(p=0.5)
    x = torch.ones(8)  # 输入全 1，方便观察被丢的位置
    print(f"原始输入        : {x.tolist()}")

    drop.train()  # 切训练模式 → Dropout 生效
    out_train_1 = drop(x)
    out_train_2 = drop(x)
    print(f"train() 第 1 次 : {out_train_1.tolist()}")
    print(f"train() 第 2 次 : {out_train_2.tolist()}")
    # 训练时被保留的位置会被乘 1/(1-p)=2 补偿期望（inverted dropout）
    # 所以保留位置值 = 2.0 而不是 1.0
    nonzero = out_train_1[out_train_1 != 0]
    if len(nonzero) > 0:
        assert torch.allclose(nonzero, torch.tensor(2.0)), "训练时保留位置应被乘 2 补偿"

    drop.eval()  # 切推理模式 → Dropout 失活
    out_eval_1 = drop(x)
    out_eval_2 = drop(x)
    print(f"eval()  第 1 次 : {out_eval_1.tolist()}")
    print(f"eval()  第 2 次 : {out_eval_2.tolist()}")
    # 推理时 Dropout 等价于恒等映射
    assert torch.equal(out_eval_1, x) and torch.equal(out_eval_2, x)


def demo_bn_vs_ln_small_batch() -> None:
    print("\n--- 2. BN vs LN 在小 batch 下的差异 ---")
    torch.manual_seed(0)

    feature_dim = 4
    bn = nn.BatchNorm1d(feature_dim)
    ln = nn.LayerNorm(feature_dim)

    bn.train()
    ln.train()

    # 用 batch=2 喂 5 个不同 batch，观察 BN running stats 的漂移
    print(f"\n  feature_dim={feature_dim}, batch_size=2")
    print(f"  {'batch':<8}{'BN running_mean':<40}{'LN(无 running stats)':<25}")
    for i in range(5):
        # 每个 batch 分布不同（均值偏移），模拟真实场景
        x = torch.randn(2, feature_dim) + i * 0.5
        _ = bn(x)
        _ = ln(x)
        print(f"  {i:<8}{str([f'{v:.3f}' for v in bn.running_mean.tolist()]):<40}{'(每样本独立)':<25}")

    # 切到 eval，BN 用 running stats 而非当前 batch 统计
    bn.eval()
    ln.eval()
    test = torch.randn(2, feature_dim)  # 测试分布与训练第 0 个 batch 接近
    out_bn = bn(test)
    out_ln = ln(test)
    print(f"\n  推理输出 BN std: {out_bn.std().item():.4f}  (受 running stats 影响)")
    print(f"  推理输出 LN std: {out_ln.std().item():.4f}  (与 batch / 历史无关)")
    # LN 推理时输出方差应非常接近 1（每个样本各自归一化的结果）
    assert 0.5 < out_ln.std().item() < 2.0


def main() -> None:
    demo_dropout()
    demo_bn_vs_ln_small_batch()
    print("\n要点回顾:")
    print("- Dropout: 训练时随机置零并乘 1/(1-p)；推理必须 model.eval()")
    print("- 小 batch + 分布漂移 → BN running stats 不准；LN 完全无影响 → LLM 全用 LN")
    print("\nPASS")


if __name__ == "__main__":
    main()
