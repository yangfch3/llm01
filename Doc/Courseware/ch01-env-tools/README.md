# ch01 · 环境与工具

## 学习目标

1. 用 **uv** 管理 Python 项目（环境、依赖、运行）
2. 理解 PyTorch 在 **CUDA / MPS / CPU** 三档设备的选择策略，能让代码 "一份脚本两端跑"
3. 掌握 **VSCode 调试** 最小工作流，足以支撑后续章节练习

## 前置依赖

- 已读 [`README.md`](https://github.com/yangfch3/llm01) 完成 `uv sync`
- 已跑过 `python scripts/doctor.py`，至少看到 `[ OK ] 设备 smoke test 通过`

## 1. uv 是什么

`uv` 是 Astral 出品的 Python 项目管理器，一把替代 `pip + virtualenv + pip-tools + pipx`。本仓库**禁用 pip / conda**。

常用 4 条命令（覆盖 90% 日常）：

| 场景 | 命令 |
|---|---|
| 同步当前项目所有依赖 | `uv sync --extra dev --extra courseware --extra echo-mini --extra echo --extra train-cuda`（Win） |
| 运行脚本（自动用项目 venv） | `uv run python xxx.py` |
| 进入交互式 Python | `uv run python` |
| 临时跑一个工具（不污染项目） | `uvx ruff check .` |

`uv sync` 会读 `pyproject.toml` + `uv.lock`，确保锁文件一致。**改了依赖记得 commit `uv.lock`**。

## 2. 设备抽象

为什么不能写死为 `torch.device("cuda")` ?

Win 上 `torch.device("cuda")`，Mac 上要写 `torch.device("mps")`，CI 或纯 CPU 机器要写 `torch.device("cpu")`。三处都硬编码 = 一份代码三套分支。

仓库统一用 `Echo/shared/device.py::get_device()`：

```python
from Echo.shared.device import get_device

device = get_device()               # 自动 cuda → mps → cpu
device = get_device(prefer="cpu")   # 显式指定偏好（不可用时降级）
```

**铁律**：Playground / Echo 业务代码**禁止**出现字符串 `"cuda"` `"mps"` `"cpu"`。

### MPS 的坑（Mac 用户必读）

部分算子（如 `aten::_unique2`）MPS 后端尚未实现。两条解法：

1. **入口设环境变量**（推荐）：`enable_mps_fallback()` 或 `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")`
2. **dtype 降级**：MPS 不支持 `float64`，统一走 `mps_safe_to(tensor, device)`

## 3. PyTorch 安装（已由 uv 处理）

仓库 `pyproject.toml` 配置：

- Win/Linux：`torch` 走 `https://download.pytorch.org/whl/cu124`（CUDA 12.4 wheel）
- Mac：走默认 PyPI（自带 MPS 支持，无需额外驱动）

无需手动 `pip install torch`，`uv sync` 会按当前平台拉对的 wheel。

## 4. VSCode 调试最小工作流

`.vscode/launch.json` 一份最小配置（按需自建，不入仓库）：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: 当前文件",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": { "PYTHONPATH": "${workspaceFolder}" }
    }
  ]
}
```

要点：

- `PYTHONPATH` 加仓库根，否则 `from Echo.shared.device import ...` 会 ImportError
- 选解释器：`Ctrl+Shift+P` → `Python: Select Interpreter` → `.venv/Scripts/python.exe`（Win）/ `.venv/bin/python`（Mac）
- 断点：行号左侧点一下 → F5 启动 → F10 单步过、F11 单步入

### 可选：Jupyter Notebook

Jupyter 是交互式 Notebook 环境，可逐 cell 执行 Python 并即时看到结果/图表，适合数据探索与可视化调试。仓库已装 `jupyter` + `ipykernel`（`courseware` 组），VSCode 直接新建 `.ipynb`、右上角选 kernel = 项目 `.venv` 即用；浏览器版 `uv run jupyter lab`。

> 本仓库练习以 `.py` 脚本为主，notebook 仅做临时探索；不要把 `.ipynb` 提交进 `Playground/` 主路径，避免 diff 噪音。

## 5. 练习

跑 `Playground/ch01-env-tools/` 下三个脚本，逐个理解：

| 脚本 | 目标 |
|---|---|
| `01_hello_torch.py` | 用 `get_device()` 建张量，跑一次前向 + 反向，打印梯度 |
| `02_hello_uv.py` | 演示 uv 常用命令（注释形式，不执行 shell） |
| `03_vscode_debug_demo.py` | 故意留两个断点位置 + 一个等价 bug，练单步调试 |

跑通标准：每个脚本独立执行不报错，且看到设备名（Win 上应是 `cuda`，Mac 上 `mps`）。

## 自检

1. 为什么 `get_device()` 默认优先级是 `cuda > mps > cpu` 而不是反过来？反过来会怎样？
2. `uv run python xxx.py` 和 `python xxx.py` 在已激活 venv 后表现一致吗？不一致时差在哪？
3. `Playground/ch01-env-tools/01_hello_torch.py` 里写了 `sys.path.insert(0, str(REPO_ROOT))`，为什么需要这行？删掉会怎样？

<details markdown="1">
<summary>答案速查</summary>

1. 优先级反过来训练时永远只用 CPU，速度慢几十到几百倍；当前顺序确保有 GPU 就用 GPU。`prefer=` 参数留了手动降级的口子，调试时强制 CPU 复现 bug 用得上

2. 大体一致。差异：`uv run` 会先校验 `uv.lock` 与 `pyproject.toml` 同步、必要时自动补依赖，更稳。已激活 venv 直接 `python` 更快但不会自检环境漂移

3. `Echo` 是大写驼峰，**不是合法的 Python 包名**，import 机制默认找不到。把仓库根加到 `sys.path` 让 Python 把 `Echo/` 当作命名空间包扫描。删掉的话 `from Echo.shared.device import ...` 会 `ModuleNotFoundError`。VSCode 调试通过 `launch.json` 的 `PYTHONPATH` 解决同一问题

</details>

## 参考资料

- uv 官方文档：<https://docs.astral.sh/uv/>
- PyTorch MPS backend：<https://pytorch.org/docs/stable/notes/mps.html>
- VSCode Python 调试：<https://code.visualstudio.com/docs/python/debugging>
