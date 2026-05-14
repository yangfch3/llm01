# Playground/ch01-env-tools

配套课件：[`Doc/Courseware/ch01-env-tools/README.md`](../../Doc/Courseware/ch01-env-tools/README.md)

| 脚本 | 目标 | 跑法 |
|---|---|---|
| `hello_torch.py` | 用 `get_device()` 跑前向+反向 | `uv run python Playground/ch01-env-tools/hello_torch.py` |
| `hello_uv.py` | uv 常用命令速查（注释为主） | `uv run python Playground/ch01-env-tools/hello_uv.py` |
| `vscode_debug_demo.py` | VSCode 单步调试练习（含一个 bug） | `uv run python Playground/ch01-env-tools/vscode_debug_demo.py` |

## 通过标准

- `hello_torch.py` 打印的 `device` 在 Win 应为 `cuda`、Mac 应为 `mps`，且看到非空梯度
- `vscode_debug_demo.py` 默认输出 `FAIL`，找到 bug 修好后输出 `PASS`
