# 踩坑记录

> 跨平台开发活文档，长期追加。
> 主索引文档（决策性内容）见 [`02-deps-compatibility.md`](02-deps-compatibility.md)。

## 格式约定

```
### YYYY-MM-DD · 平台 · 一句话标题

- **现象**：...
- **根因**：...
- **解决**：...
- （可选）**影响** / **后续**：...
```

新条目**置顶**（按日期倒序），便于看最近遇到的坑。

---

## 2026-05-25 · Win · trl 读取 jinja 模板报 GBK 编码错误

- **现象**：`from trl import SFTTrainer` 报 `UnicodeDecodeError: 'gbk' codec can't decode byte 0x9c`，出在 `trl/chat_template_utils.py` 读取 `deepseekv3.jinja`
- **根因**：trl 1.4.0 用 `Path.read_text()` 读模板文件，未指定编码，Windows 默认走 GBK，但文件是 UTF-8
- **解决**：设置环境变量 `PYTHONUTF8=1`（永久：`[System.Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")`）

## 2026-05-25 · Win · trl SFTTrainer import 触发 pyarrow segfault

- **现象**：`from trl import SFTTrainer`（在 `import torch` 之后）导致进程 segfault (0xC0000005)，无 Python traceback
- **根因**：与 `2026-05-14` 记录的 `torch + pyarrow` DLL 加载顺序冲突相同。trl 延迟 import sft_trainer 模块时链式触发 `from datasets import Dataset`
- **解决**：在脚本最顶部（`import torch` 之前）加 `import datasets`，确保 pyarrow DLL 先加载
- **影响**：echo SFT 脚本已加此 workaround；所有同时用 torch + datasets/trl 的新脚本都需遵循此顺序

## 2026-05-24 · Win · pretrain OOM: accelerate launch 忽略代码中 mixed_precision

- **现象**：pretrain.py 配置了 `mixed_precision="bf16"`，但 launch 日志显示 `'no'`，模型跑 fp32 导致 OOM
- **根因**：`accelerate launch` 命令行未传 `--mixed_precision`，其默认值 `no` 覆盖了 `Accelerator()` 构造函数的参数
- **解决**：launch 命令显式带 `--mixed_precision bf16`

## 2026-05-24 · Win · echo-mini scripts 报 `No module named 'echo_mini'`

- **现象**：`uv run python scripts/prepare_data.py ...` 报 ModuleNotFoundError
- **根因**：`src/echo_mini/` 不是 pip 安装的包，`sys.path` 中没有 `src/` 目录
- **解决**：scripts 顶部加 `sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))`

## 2026-05-14 · Win · `bitsandbytes` 自检子进程输出 GBK 编码

- **现象**：`import bitsandbytes` 时 doctor 主进程 reader thread 报 `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb2`
- **根因**：bnb 内部启动子进程查 CUDA 信息，输出走系统默认 GBK 编码，主进程默认 UTF-8 解码失败
- **解决**：doctor 中所有 `subprocess.run(..., text=True)` 调用统一加 `encoding="utf-8", errors="replace"`；bitsandbytes 检查也改走子进程隔离

## 2026-05-14 · Win · `torch` + `pyarrow/pandas` 同进程 ACCESS_VIOLATION

- **现象**：doctor 在 `import datasets` 时进程 native crash，exit code `0xC0000005`，stdout 截断
- **复现**：先 `import torch`，再 `import datasets`（其内部 `import pandas` 触发 `pyarrow.compat.pyarrow`）→ 崩；顺序反过来不崩
- **根因**：torch wheel 与 pyarrow wheel 各自携带不同版本的 native runtime DLL，先来后到加载时第二个不兼容旧地址，触发段错误。`KMP_DUPLICATE_LIB_OK=TRUE` 不够
- **解决**：`scripts/doctor.py` 把 ECHO_DEPS 的 import 检查改走**子进程隔离**（`subprocess.run([sys.executable, "-c", "import X"])`）。每个 import 独立进程，避免 DLL 串扰
- **训练代码影响**：训练脚本本身只在主进程 import torch，pyarrow/pandas 通过 `datasets` 走 dataloader worker 进程时是分开的，**实际训练不受影响**。仅自检脚本需要绕

## 2026-05-14 · Win · doctor.py 报 `No module named 'Echo'`

- **现象**：smoke test 时 `from Echo.shared.device import get_device` 失败
- **根因**：`Echo/` 是大写驼峰目录，不是标准 Python 包；`uv run` 跑脚本时 `sys.path` 没自动加仓库根
- **解决**：doctor 入口手动 `sys.path.insert(0, REPO_ROOT)`
- **后续**：业务代码也走仓库根 import 时同理处理；M4 之后若决定把 `Echo/shared` 重构为可发行包再统一调整

## 2026-05-14 · Win · uv 默认装到 CPU 版 torch

- **现象**：`uv sync --extra train-cuda` 后 `torch.__version__ == '2.12.0+cpu'`，`torch.cuda.is_available()` 为 False
- **根因**：PyPI 上的 `torch` Win wheel 是 CPU-only，CUDA 版要从 PyTorch 官方 index 拿
- **解决**：在 `pyproject.toml` 加 `[tool.uv.sources]` + `[[tool.uv.index]]` 配 `pytorch-cu124`，按 `sys_platform` marker 分流（Win/Linux 走 cu124，Mac 走默认 PyPI 的 MPS 版）
- **影响**：所有人首次 `uv sync` 后必须确认 `+cu124` 后缀；Mac 端不受影响

## 2026-05-14 · Win · `llama-cpp-python` 源码编译失败

- **现象**：M0 阶段 `uv sync --extra deploy` 触发本地编译，scikit-build-core 调 CMake → MSVC 编译 llama.cpp 失败
- **根因**：Win + Python 3.12 下官方未必发对应 wheel；本地编译要求 MSVC 版本、CMake、CUDA toolkit 多方匹配，对环境敏感
- **解决**：
  - 把 `llama-cpp-python` 从 `deploy` 拆到独立 extras `deploy-llamacpp`
  - M0–M5 全程不装，M6 部署阶段再按需安装
  - 真要装时优先走预编译 wheel：<https://github.com/abetlen/llama-cpp-python/releases>
- **影响**：M0 阶段 `uv sync` 命令更新，参见 README 与 `03-sync-strategy.md`
