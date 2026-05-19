"""ch04 练习 3：学习率调度对照。

不实际训模型，只画 lr 随 step 变化的曲线（ASCII），看四种 schedule 形状：
- 固定 lr
- StepLR（每 N 步 ×0.1）
- CosineAnnealing（cos 衰减）
- Warmup + Cosine（前 K 步线性升 + 后续 cosine）
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def lr_constant(step: int, lr_max: float = 1e-3, **_) -> float:
    return lr_max


def lr_step(step: int, lr_max: float = 1e-3, step_size: int = 30, gamma: float = 0.5, **_) -> float:
    # 每 step_size 步乘 gamma；典型如 step_size=30, gamma=0.1
    return lr_max * (gamma ** (step // step_size))


def lr_cosine(step: int, lr_max: float = 1e-3, lr_min: float = 1e-4, total: int = 100, **_) -> float:
    # 标准 cosine annealing：从 lr_max 余弦衰减到 lr_min
    progress = min(step / total, 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def lr_warmup_cosine(
    step: int, lr_max: float = 1e-3, lr_min: float = 1e-4, warmup: int = 10, total: int = 100, **_
) -> float:
    # LLM 预训练事实标准：前 warmup 步线性升，再 cosine 衰减
    if step < warmup:
        return lr_max * step / warmup  # 线性 warmup
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(progress, 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def ascii_plot(values: list[float], lr_max: float, width: int = 50) -> str:
    # 把 lr 曲线压到 width 个柱形，每柱高度 ∝ lr / lr_max
    rows = 8
    lines = []
    for r in range(rows, 0, -1):
        threshold = lr_max * r / rows
        line = "".join("█" if v >= threshold else " " for v in values[:width])
        lines.append(f"{threshold:.1e} | {line}")
    lines.append(" " * 11 + "-" * width)
    return "\n".join(lines)


def main() -> None:
    total = 100
    lr_max = 1e-3
    schedules = {
        "Constant": lr_constant,
        "StepLR (size=30, gamma=0.5)": lr_step,
        "Cosine": lr_cosine,
        "Warmup(10) + Cosine": lr_warmup_cosine,
    }

    for name, fn in schedules.items():
        # 把 fn 需要的 kwargs 都喂全，避免每个 fn 单独写调用
        values = [fn(step=s, lr_max=lr_max, lr_min=1e-4, total=total) for s in range(total)]
        print(f"\n=== {name} ===")
        print(ascii_plot(values, lr_max))
        print(f"step 0 / 50 / 99 lr: {values[0]:.2e}  {values[50]:.2e}  {values[99]:.2e}")

    # 健全性断言
    assert lr_warmup_cosine(0, lr_max=lr_max, warmup=10, total=total) == 0.0  # warmup 起点 lr=0
    assert abs(lr_warmup_cosine(10, lr_max=lr_max, warmup=10, total=total) - lr_max) < 1e-9  # warmup 末端
    assert lr_warmup_cosine(99, lr_max=lr_max, lr_min=1e-4, warmup=10, total=total) < 2e-4  # 末端接近 lr_min

    # --plot：画 lr 走势对比图
    if "--plot" in sys.argv:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        for name, fn in schedules.items():
            values = [fn(step=s, lr_max=lr_max, lr_min=1e-4, total=total) for s in range(total)]
            plt.plot(values, label=name)
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.title("LR Schedule Comparison")
        plt.legend()
        plt.tight_layout()
        out_path = REPO_ROOT / "Doc" / "Courseware" / "ch04-nn-training" / "lr_schedule.png"
        plt.savefig(out_path, dpi=150)
        print(f"\n图已保存: {out_path.relative_to(REPO_ROOT)}")
        plt.show()

    print("\nPASS")


if __name__ == "__main__":
    main()
