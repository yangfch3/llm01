"""ch01 练习 3：VSCode 调试 demo。

本脚本故意埋一个 off-by-one 风格的 bug，目标：
1. 在 `BREAKPOINT 1` 处下断点，观察 `total` 与 `expected` 的差异
2. 在 `BREAKPOINT 2` 处下断点，单步进 `buggy_sum`，找出 bug
3. 修好后再次跑，应该看到 "PASS"

跑法：
    uv run python Playground/ch01-env-tools/vscode_debug_demo.py

调试入口：VSCode 打开本文件 → F5（用 README §5 的 launch.json）

提示：bug 在 `buggy_sum` 的 range 边界。
"""

from __future__ import annotations


def buggy_sum(n: int) -> int:
    """求 1 + 2 + ... + n。但实现里有一个边界 bug。"""
    total = 0
    # BUG: range(1, n) 漏掉了 n 本身
    for i in range(1, n):
        total += i
    return total


def main() -> None:
    n = 10
    expected = n * (n + 1) // 2  # 高斯求和公式

    # BREAKPOINT 1：在这里下断点，F10 单步过下面这行后看 total 的值
    total = buggy_sum(n)

    # BREAKPOINT 2：在这里下断点，F11 单步入 buggy_sum 找 bug
    print(f"n = {n}")
    print(f"expected = {expected}")
    print(f"got      = {total}")

    if total == expected:
        print("PASS")
    else:
        print(f"FAIL: 差 {expected - total}")


if __name__ == "__main__":
    main()
