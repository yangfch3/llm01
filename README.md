# llm01

LLM 入门课件 + Echo 对话模型双线产出。

## 是什么

双线产出：

- **课件 + 练习**：从 PyTorch / 数学基础到 Transformer 全链路 LLM 知识
- **Echo 对话模型**：双产物
  - `echo-mini`：从零 Pretrain → SFT 全链路走通的迷你模型（教学价值）
  - `echo`：基于开源底座 SFT/DPO 出来的可对话模型（实用价值）

详细愿景与决策见 [`Doc/DesignDoc/00-startup-proposal.md`](Doc/DesignDoc/00-startup-proposal.md)。

## 快速开始

### 前置

- Python 3.12（minor 锁定，patch 由本地 pyenv/uv 选）
- [uv](https://docs.astral.sh/uv/)（依赖管理）
- Windows: NVIDIA RTX 3060 12GB（CUDA 12.x）/ Mac: Apple Silicon（MPS）

### 安装

```bash
# Windows (3060)
uv sync --extra dev --extra courseware --extra echo-mini --extra train-cuda

# Mac (Apple Silicon)
uv sync --extra dev --extra courseware --extra echo-mini --extra train-mps
```

> `llama-cpp-python` 拆到独立的 `deploy-llamacpp` extras，M6 部署阶段再按需装；
> Win 下若编译失败，参考 [`Doc/DesignDoc/02-deps-compatibility.md`](Doc/DesignDoc/02-deps-compatibility.md) §5。

### 自检

每次切换机器、上机第一件事：

```bash
uv run python scripts/doctor.py
```

### 文档站（MkDocs）

`Doc/` 下所有文档（课件、设计文档等）通过 MkDocs Material 主题渲染，支持 LaTeX 公式。

**线上**：<https://yangfch3.github.io/llm01/>（push main 后由 GitHub Actions 自动部署）

**本地预览**：

```bash
uv run python scripts/docs_serve.py
```

浏览器打开 `http://127.0.0.1:8000` 即可。

> 站点工具链与项目主依赖隔离，独立装在 `Misc/mkdocs/.venv-docs/`，依赖锁见 `Misc/mkdocs/requirements-docs.txt`。
> 首次本地预览前需执行：
> ```bash
> cd Misc/mkdocs && uv venv .venv-docs && uv pip install --python .venv-docs -r requirements-docs.txt
> ```

## 目录导览

```
Doc/
├─ DesignDoc/        根基文档（策划案 / 计划书 / 专题）
├─ Courseware/       课件（章节 markdown）
└─ UserDraft/        历史输入（只追加不改）

Playground/          配套练习代码（与课件 ch## 一一对应）

Echo/
├─ echo-mini/        从零教学产物
├─ echo/             开源底座微调产物
└─ shared/           两者共用工具（device.py 等）

scripts/             跨产物脚本（doctor.py 等）
```

详细约定见 [`Doc/DesignDoc/00-startup-proposal.md`](Doc/DesignDoc/00-startup-proposal.md) §5。

## 跨平台协作

- 训练**生产配置锁 Windows**，Mac 跑 tiny 配置做代码验证
- 学习/编码/推理/量化/部署 双端等价
- 切机器前 `git push`，上机 `git pull` + `scripts/doctor.py`
- 大文件不入 Git：数据走脚本下载，权重走 HuggingFace Hub
- 详见 [`Doc/DesignDoc/03-sync-strategy.md`](Doc/DesignDoc/03-sync-strategy.md)

## 学习路径

按里程碑顺序推进，详见 [`Doc/DesignDoc/01-project-plan.md`](Doc/DesignDoc/01-project-plan.md)：

| 阶段 | 内容 |
|---|---|
| M1 | PyTorch + 浅层数学 + 神经网络基础（ch01–ch04） |
| M2 | 注意力 + Transformer 架构 + 生成策略（ch05–ch07） |
| M3 | 分词器 / 预训练 / SFT / 对齐 / 评测 / 部署（ch08–ch13） |
| M4 | echo-mini 全链路落地 |
| M5 | echo 微调落地 |
| M6 | 对齐与量化部署 |

任务勾选清单见 [`Doc/DesignDoc/tasks.md`](Doc/DesignDoc/tasks.md)。

## License

待补（M7 阶段定，倾向 MIT / Apache-2.0）。
