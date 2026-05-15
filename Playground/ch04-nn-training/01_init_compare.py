"""ch04 练习 1：初始化方案对激活方差的影响。

10 层纯线性 + ReLU 网络，前向跑一遍，看每层激活的标准差怎么演化。
预期：
- 朴素 N(0, 1)：std 几层后爆炸或衰减到 0
- Xavier：浅层稳，深层略衰减（因为没补 ReLU 的 1/2）
- He：std 在 1 附近稳定（这就是它存在的理由）
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn


def build_net(init_kind: str, depth: int = 10, width: int = 256) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(depth):
        linear = nn.Linear(width, width, bias=False)  # 关 bias 隔离 weight 影响
        with torch.no_grad():
            if init_kind == "naive":
                # 朴素 N(0, 1)：std 太大，几层就炸
                linear.weight.normal_(mean=0.0, std=1.0)
            elif init_kind == "xavier":
                # Xavier (Glorot) normal: std = sqrt(1 / fan_in)
                # nonlinearity="linear" → gain=1，对应 Xavier 原意
                nn.init.xavier_normal_(linear.weight, gain=1.0)
            elif init_kind == "he":
                # Kaiming (He) normal: std = sqrt(2 / fan_in)，专为 ReLU 设计
                nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            else:
                raise ValueError(init_kind)
        layers.append(linear)
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


@torch.no_grad()
def forward_with_stats(net: nn.Sequential, x: torch.Tensor) -> list[float]:
    stds: list[float] = []
    h = x
    # 每两个 layer (Linear + ReLU) 视为"一层"
    for i, layer in enumerate(net):
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            stds.append(h.std().item())
    return stds


def main() -> None:
    torch.manual_seed(0)
    batch = 64
    width = 256
    x = torch.randn(batch, width)  # 输入本身 N(0, 1)

    print(f"输入: shape={tuple(x.shape)}  std={x.std().item():.4f}")
    print(f"\n{'层':<4}{'naive':>14}{'xavier':>14}{'he':>14}")
    print("-" * 46)

    results = {kind: forward_with_stats(build_net(kind, width=width), x) for kind in ("naive", "xavier", "he")}

    for layer_idx in range(len(results["he"])):
        # naive 后期会变成 inf 或 0，格式化时单独处理
        def fmt(v: float) -> str:
            if v != v or v == float("inf"):  # nan / inf
                return "  inf/nan"
            if v < 1e-6:
                return f"{v:.2e}"
            return f"{v:.4f}"

        n = results["naive"][layer_idx]
        x_ = results["xavier"][layer_idx]
        h_ = results["he"][layer_idx]
        print(f"{layer_idx + 1:<4}{fmt(n):>14}{fmt(x_):>14}{fmt(h_):>14}")

    he_final = results["he"][-1]
    # He 初始化在 ReLU 网络里激活 std 应稳在 1 量级（典型 0.5 ~ 2）
    assert 0.3 < he_final < 3.0, f"He 初始化应让最后层 std 在 1 附近，实际 {he_final}"
    print("\nPASS")


if __name__ == "__main__":
    main()
