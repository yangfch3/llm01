# LLM 学习仓库 Startup 策划案

> 本文是仓库根基文档，记录项目愿景、范围、技术决策与目录约定。后续所有文档以此为准。

## 1. 项目愿景

以 **学 + 做** 双线并行的方式，带领一名资深游戏工程师从零进入 LLM 领域，最终同时产出：

- 一套可复用的 LLM 入门课件（面向其他入门者开源）
- 一个代号 **Echo** 的可本地运行的对话模型

## 2. 目标与非目标

### 2.1 目标

1. 系统学习 LLM 制作与训练的全链路知识
2. 形成可复用的课件（大纲 + 章节 md）与配套练习代码
3. 产出 **Echo** 模型：
   - 中英双语基础对话能力
   - 智力达到初中生水平
   - 可在 RTX 3060 (12GB) 本地运行
4. 仓库可开源，能对其他入门者起到引导作用

### 2.2 非目标

- 不追求 SOTA 性能、不挑战大参数量模型
- 不构建生产级推理服务 / Web 前端
- 不做分布式多机训练（单卡为主）

## 3. 核心决策

| 决策项 | 选择 | 理由 |
|---|---|---|
| 技术路线 | 双产物并行：`echo-mini` 从零 Pretrain + `echo` 微调开源底座 | 前者保证全链路学习完整，后者保证最终产品可用 |
| 开源底座候选 | Qwen2.5-0.5B / 1.5B（待定 Pretrain 阶段再拍板） | 中英友好、尺寸合适、3060 可训可推 |
| 课程起点 | PyTorch + Transformer + 浅层数学 | 用户已有 Python/NumPy 基础 |
| 训练链路 | 数据清洗 → 分词器训练 → Pretrain → SFT → DPO/RLHF → 评测 → 量化部署 | 完整覆盖全链路 |
| 语言 | 中英双语 | 语料丰富、受众更广 |
| 环境管理 | **uv** + `pyproject.toml` + `uv.lock` | 跨平台、锁定可复现、Win/Mac 一致 |
| Python 版本 | 3.12（minor 锁定，Win 用 3.12.10 / Mac 用 3.12.13） | 用户指定，patch 由各平台决定 |
| 代码规范 | ruff（lint + format），类型注解建议不强制 | 教学代码轻量优先 |
| 硬件基线 | RTX 3060 12GB / Apple Silicon MPS | Win 主训练，Mac 用于推理 & 开发 |

## 4. 双产物设计（Echo）

### 4.1 echo-mini —— 教学产物

- 参数量：~10M–100M 级，可在 3060 上从零训练
- 用途：走完全链路，验证每一步管线
- 预期智力：极弱，能续写常见句式即可，不追求对话质量
- 价值：**学懂每个环节**

### 4.2 echo —— 实用产物

- 基于开源底座（如 Qwen2.5-0.5B）做 SFT + DPO
- 用途：作为最终"能与用户对话的 Echo"
- 预期智力：接近初中生水平的基础对话
- 价值：**交付可用成果**

### 4.3 两者关系

```
课件 & Playground 练习
        │
        ▼
echo-mini (全链路自建) ──学习经验──▶ echo (微调开源底座)
        │                                │
        ▼                                ▼
    弱模型 demo                     Echo 对话体验
```

### 4.4 经验流转机制

echo-mini 的训练过程中会沉淀大量"配方"（数据清洗参数、超参组合、踩坑解法）。流转规则：

- 通用基础设施（设备工具、数据加载、日志、评测）一律下沉到 `Echo/shared/`
- 训练配方差异写入各自 README（`echo-mini/README.md` / `echo/README.md`）
- 跨产物的踩坑统一追加到 `Doc/DesignDoc/troubleshooting.md`（活文档，按日期倒序）
- 每个产物里程碑结束（M4、M5、M6）时，强制做一次 retro，遵循 `Doc/DesignDoc/templates/retro-template.md`

### 4.5 Echo 验收标准

愿景里"初中生水平 + 基础对话"是定性目标，落地需有可操作的验收。下面只定 **echo（实用产物）** 的验收，echo-mini 不要求达成对话质量。

#### echo SFT 版（M5 完）

- **能力底线**：
  - 中英双语都能进行 ≥3 轮的连贯对话不跑题
  - 能正确回答常识题（人物、地理、基础科学）≥60%
  - 能进行简单算术（两位数加减）和字符级任务（数字符串里的 "a"）
- **量化方法**：
  - PPL：在留出 SFT 验证集上 ≤ 基线（底座 zero-shot）的 80%
  - 自建 50 题中英常识/算术评测集（一次性投入，复用到后续版本）
  - 人工对话 30 轮检查跑题率、人设一致性

#### echo final（M6 完，量化后）

- 上述指标量化后衰减不超过 5%
- 在 3060 上推理 ≥ 20 tokens/s（int4）；在 Mac M 系列 ≥ 15 tokens/s（GGUF Q4_K_M）
- Ollama `ollama run echo` 一行命令可启动对话

> 评测集落到 `Echo/echo/eval/` 下，详细方案在 M5 启动时落 `Doc/DesignDoc/04-eval-spec.md`（届时按需新建，不强求 M0 产出）。

## 5. 仓库目录结构

```
llm01/
├─ Doc/
│  ├─ UserDraft/               用户原始想法（只进不改）
│  ├─ DesignDoc/               根基文档（策划案 / 计划书 / 专题设计）
│  └─ Courseware/              课件
│     ├─ outline.md            课程大纲
│     └─ ch01-xxx/             各章节目录
│        ├─ README.md          章节课件
│        └─ assets/            图示
├─ Playground/                 配套练习代码
│  └─ ch01-xxx/                与课件章节一一对应
├─ Echo/                       Echo 项目
│  ├─ echo-mini/               教学产物
│  │  ├─ configs/
│  │  ├─ data/                 （.gitignore）
│  │  ├─ tokenizer/            （产物 .gitignore，脚本入仓）
│  │  ├─ src/echo_mini/
│  │  ├─ scripts/
│  │  ├─ checkpoints/          （.gitignore）
│  │  └─ README.md
│  ├─ echo/                    实用产物
│  │  ├─ configs/
│  │  ├─ data/                 （.gitignore）
│  │  ├─ src/echo/
│  │  ├─ scripts/
│  │  ├─ checkpoints/          （.gitignore）
│  │  └─ README.md
│  └─ shared/                  两者共用工具（数据加载、评测、日志）
├─ pyproject.toml              uv 统一管理依赖
├─ uv.lock
├─ .python-version             3.12（minor 锁定）
├─ .gitignore
└─ README.md                   面向开源访客的入口
```

### 目录约定要点

- `Doc/UserDraft` 下文档只追加、不修改，作为历史原始输入
- `Doc/DesignDoc` 下文档按 `NN-xxx.md` 两位数字排序
- `Doc/Courseware/ch##-xxx` 与 `Playground/ch##-xxx` 严格对应
- `Echo/shared` 只放两个产物都要用的代码（如日志、评测、通用 tokenizer 加载）
- 模型权重、数据集、缓存一律 `.gitignore`，脚本与配置入仓

## 6. 环境方案

### 6.1 依赖管理

- 单一 `pyproject.toml`，用 uv 的 **workspace** 或 **optional-dependencies** 管理子模块
- 分组原则：**按模块维度 + 按平台维度** 双轴分组
  - 模块维度：`courseware` / `echo-mini` / `echo` / `dev`
  - 平台维度：`train-cuda`（Win 专属，如 `bitsandbytes`、`flash-attn`）/ `train-mps`（Mac 专属）
- 跨平台通用依赖（torch、tokenizers、transformers 等）放模块分组，平台专属依赖放平台分组
- `uv.lock` 跨平台共享，靠 platform marker 保证同一 lock 在两端都能 resolve

### 6.2 首次 bootstrap

```bash
# Windows (3060)
uv sync --extra dev --extra echo-mini --extra train-cuda

# Mac (Apple Silicon)
uv sync --extra dev --extra echo-mini --extra train-mps

# 验证（两端都应通过 scripts/doctor.py）
uv run python scripts/doctor.py
```

## 7. 双平台协作方案

用户会在 Windows 和 Mac 之间**随时切换**进行学习、开发、训练、测试。以下规范保证切换零摩擦。

### 7.1 活动 × 平台矩阵

| 活动 | Windows (3060) | Mac (Apple Silicon) | 切换策略 |
|---|---|---|---|
| 学习 / 读课件 | ✅ | ✅ | 随时切 |
| Playground 练习 | ✅ | ✅ | 随时切，代码必须 device 无关 |
| echo-mini Pretrain（生产配置） | ✅ 主战场 | ❌ 时间成本不合理 | **锁 Win** |
| echo SFT / LoRA / DPO（生产配置） | ✅ 主战场 | ❌ 同上 | **锁 Win** |
| 训练代码的 **tiny 配置** 验证 | ✅ | ✅ | 随时切，用于代码正确性检查 |
| 推理 / 对话测试 | ✅ | ✅ | 随时切 |
| 量化（GGUF / int4） | ✅ | ✅ Metal 优化极好 | 随时切 |
| 本地 chat demo | ✅ | ✅ | 随时切 |

核心原则：**训练生产配置锁 Win，其余全部两端等价。**

### 7.2 代码设备无关性

- 禁止代码里硬编码 `cuda` / `mps` / `cpu`
- 统一走 `Echo/shared/device.py::get_device()`，按 `CUDA → MPS → CPU` 优先级自动选
- MPS 不支持的算子（部分）需检测后 fallback 到 CPU 或给出明确报错
- 所有路径用 `pathlib.Path`，禁止字符串拼接；文件编码统一 UTF-8，换行 LF

### 7.3 配置双份化（仅训练脚本）

所有训练任务（Pretrain / SFT / DPO）都要提供两套配置：

- `config-full.yaml`：生产配置，3060 上跑，大 batch、长步数
- `config-tiny.yaml`：验证配置，Mac/CPU 可跑，~100 步内完成，只为验证代码正确性

切换机器时，Mac 用 tiny 配置跑一遍确认不崩，然后推回 Win 跑 full 配置。

### 7.4 大文件同步策略

| 类型 | 大小 | 去处 |
|---|---|---|
| 代码、配置、课件、文档 | 小 | Git 仓库 |
| 训练日志（loss 曲线 json/tb） | 中 | Git 仓库（压缩后） |
| 数据集原始文件 | 大 | **脚本化下载**，入库 `scripts/download_*.py`，数据本身不入 Git |
| 分词器产物 | 小 | 可入 Git（几 MB） |
| checkpoint / adapter 权重 | 大 | **HuggingFace Hub 私有/公开仓库**，入库仅保留 HF repo id |

两台机切换前：
- 代码、配置：`git push` / `git pull`
- 权重：`huggingface-cli upload` / `download`
- 数据：两端各跑一次下载脚本

### 7.5 自检脚本 `scripts/doctor.py`

上机第一件事跑一遍，检查项：

- Python 版本满足 3.12（Win 3.12.10 / Mac 3.12.13）
- 平台与预期依赖组一致（Win 装了 train-cuda、Mac 装了 train-mps）
- `torch` 可用 + 设备可见（cuda/mps）
- 关键依赖 import 不报错
- Git 换行符配置为 LF（`core.autocrlf=false` + `.gitattributes`）
- 打印当前平台下推荐的训练配置入口

## 8. 质量与协作约定

- **文档风格**：简洁凝练，设计用伪代码/流程图替代完整代码
- **文件规范**：LF 换行、UTF-8 无 BOM、末尾留空行
- **AI 协作**：大的设计变更先在 `Doc/DesignDoc` 落文档，再动代码
- **开源就绪**：从第一天起 README、LICENSE、贡献指南逐步补齐

## 9. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 3060 显存不足以做 Pretrain | echo-mini 训练卡住 | 控制参数量在 10M–100M，启用 gradient checkpointing / 混合精度 |
| 中英双语稀释小模型效果 | echo-mini 效果差 | 允许 echo-mini 偏英文/单语，中英双语在 `echo` 上实现 |
| 开源底座许可证限制 | echo 无法开源权重 | 只开源训练脚本与配置，权重走 HF 链接引导下载 |
| 学习节奏拖长 | 项目烂尾 | 计划书按"里程碑 + 可交付物"驱动，见 `01-project-plan.md` |
| CUDA-only 库在 Mac 不可用（bitsandbytes / flash-attn 等） | Mac 端 import 失败 | 平台分组依赖 + 代码里 try-import fallback，必要时禁用对应特性 |
| checkpoint 跨机同步成本高 | 切换机器阻塞 | 统一走 HF Hub，只同步 repo id，不走 Git；训练中间态尽量不跨机 |
| MPS 算子覆盖不全 | 练习代码在 Mac 崩 | `get_device()` 内置 op-level fallback；课件练习在提交前两端都要跑过 |

## 10. 关联文档

- `01-project-plan.md`：项目计划书（任务拆分 & 里程碑）
- `02-deps-compatibility.md`：依赖兼容性与部署选型说明
- `tasks.md`：任务勾选清单（高频改动）
- `templates/retro-template.md`：里程碑复盘模板
- `Doc/Courseware/outline.md`：课程大纲（计划书推进后产出）
- `Doc/UserDraft/repo-target-idea.md`：原始需求来源
