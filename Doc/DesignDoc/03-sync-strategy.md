# 大文件同步策略 SOP

> 本文是 [`00-startup-proposal.md`](00-startup-proposal.md) §7.4 的操作层补充。策略已定，本文只补**怎么操作**。

## 1. 同步矩阵速查

| 类型 | 大小 | 去处 | 工具 |
|---|---|---|---|
| 代码、配置、课件、文档 | 小 | Git | `git push` / `git pull` |
| 训练日志（loss 曲线 json/tb 压缩后） | 中 | Git | 同上 |
| 数据集原始文件 | 大 | **不入 Git**，脚本下载 | `scripts/download_*.py` |
| 分词器产物 | 小（几 MB） | 可入 Git | `git push` |
| checkpoint / adapter 权重 | 大 | **HuggingFace Hub** | `huggingface-cli` |

**铁律**：仓库里只保留 HF repo id，不保留权重二进制。

## 2. HuggingFace Hub 仓库命名规则

占位用户名：`<your-hf-username>`（M5 启动前替换为真实账号）。

| 产物 | 仓库类型 | 命名 | 示例 |
|---|---|---|---|
| echo-mini Pretrain 权重 | model | `echo-mini-pretrain-vN` | `<your-hf-username>/echo-mini-pretrain-v1` |
| echo-mini SFT 权重 | model | `echo-mini-sft-vN` | `<your-hf-username>/echo-mini-sft-v1` |
| echo SFT LoRA adapter | model | `echo-sft-{base}-vN` | `<your-hf-username>/echo-sft-qwen25-0.5b-v1` |
| echo DPO LoRA adapter | model | `echo-dpo-{base}-vN` | `<your-hf-username>/echo-dpo-qwen25-0.5b-v1` |
| echo final 量化版 | model | `echo-{base}-gguf` | `<your-hf-username>/echo-qwen25-0.5b-gguf` |
| 自建评测集 | dataset | `echo-eval-vN` | `<your-hf-username>/echo-eval-v1` |
| 训练数据（如自建） | dataset | `echo-data-{purpose}-vN` | `<your-hf-username>/echo-data-sft-v1` |

**版本规则**：

- 主版本递增对应数据/配方重大变更（`v1` → `v2`）
- 同一版本内的小迭代用 git 分支或 commit hash 区分（HF model card 里写明）
- 公开/私有：教学产物默认 public，含商业风险数据时改 private

## 3. 常用命令

### 3.1 安装与登录

```bash
# huggingface-hub 已在 echo-mini extras 里
uv run huggingface-cli login
```

token 从 https://huggingface.co/settings/tokens 申请，权限选 `write`。

### 3.2 上传权重 / adapter

```bash
# 单文件
uv run huggingface-cli upload \
    <your-hf-username>/echo-mini-sft-v1 \
    Echo/echo-mini/checkpoints/sft-final.safetensors

# 整个目录（推荐，含 config / tokenizer / 权重）
uv run huggingface-cli upload \
    <your-hf-username>/echo-mini-sft-v1 \
    Echo/echo-mini/checkpoints/sft-final/ . \
    --repo-type model
```

参数说明：

- 第三个参数是**本地路径**，第四个 `.` 是**仓库内目标路径**
- `--repo-type` 可选 `model` / `dataset` / `space`，默认 `model`
- 大文件会自动走 LFS，断点续传由 hub 客户端处理

### 3.3 下载权重

```bash
# 整库下载到本地缓存（默认 ~/.cache/huggingface/）
uv run huggingface-cli download <your-hf-username>/echo-mini-sft-v1

# 下载到指定目录
uv run huggingface-cli download \
    <your-hf-username>/echo-mini-sft-v1 \
    --local-dir Echo/echo-mini/checkpoints/sft-final \
    --local-dir-use-symlinks False
```

**Win 注意**：默认软链接在 Win 上需开发者模式或管理员权限，建议加 `--local-dir-use-symlinks False`。

### 3.4 断点续传

`huggingface-cli` 默认支持。中断后再跑同一条命令即可，已下载分片不会重复传。

如遇网络不稳，可设环境变量：

```bash
# 启用 hf_transfer（更快、自动重试）
export HF_HUB_ENABLE_HF_TRANSFER=1
```

需先 `uv add hf_transfer`（按需，不放主依赖）。

### 3.5 在代码里加载

```python
from huggingface_hub import snapshot_download

ckpt_dir = snapshot_download(
    repo_id="<your-hf-username>/echo-mini-sft-v1",
    local_dir="Echo/echo-mini/checkpoints/sft-final",
    local_dir_use_symlinks=False,
)
```

或直接给 `transformers`：

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("<your-hf-username>/echo-mini-sft-v1")
```

## 4. 数据下载脚本骨架

每个数据集对应一个 `scripts/download_<dataset>.py`，脚本接口约定：

```python
"""Download <dataset name>.

Usage:
    uv run python scripts/download_<dataset>.py [--output DIR] [--force]
"""

from __future__ import annotations
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# 按目标产物切换：echo-mini 数据放 echo-mini/data，echo 数据放 echo/data
DEFAULT_OUTPUT = REPO_ROOT / "Echo" / "echo-mini" / "data" / "<dataset>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="覆盖已存在数据")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()) and not args.force:
        print(f"已存在数据，跳过：{args.output}（用 --force 覆盖）")
        return

    # 实际下载逻辑：huggingface-hub / requests / wget 任选
    ...

    print(f"完成：{args.output}")


if __name__ == "__main__":
    main()
```

约定：

- 数据落到 `Echo/{产物}/data/`，这两个路径都在 `.gitignore`
- 脚本入仓，数据不入仓
- 幂等：已下载完整数据应跳过；`--force` 覆盖
- 打印最终路径，方便后续脚本对接

## 5. 切机器 SOP

### 切出（离开当前机器前）

```bash
git status                           # 确认无遗漏
git add -A
git commit -m "..."
git push
# 权重若有更新：huggingface-cli upload ...
```

### 切入（到达新机器后）

```bash
git pull
uv sync --extra dev --extra courseware --extra echo-mini --extra train-{cuda|mps}
uv run python scripts/doctor.py      # 必跑
# 数据：按需 uv run python scripts/download_*.py
# 权重：按需 huggingface-cli download ...
# M6 部署阶段需要 llama-cpp-python 时再加 --extra deploy-llamacpp
```

## 6. 故障排查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `huggingface-cli login` 提示 token invalid | token 过期或权限不够 | 重新申请 `write` 权限 token |
| 上传速度极慢 | 默认 client 单线程 | 启用 `HF_HUB_ENABLE_HF_TRANSFER=1` |
| Win 下载报 symlink 权限错 | 开发者模式未开 | 命令加 `--local-dir-use-symlinks False` |
| `git push` 报 LFS 文件超限 | 误把权重 commit 进 Git | `git lfs untrack` + `.gitignore` 补漏 + `git filter-repo` 清理 |
| 两端 lock 不一致 | uv 解析器版本不同 | 升级 uv 到同 minor，重跑 `uv sync` |
