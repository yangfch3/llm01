# LLM 0-1

欢迎来到 **LLM 0-1** 的文档站。

本项目是一份「LLM 入门课件 + 练习代码」与 **Echo 对话模型**的组合产出：

- **课件**：从数学基础起步，覆盖分词、Transformer、训练、推理、量化与部署等环节，按章节循序推进。
- **Echo 模型**：双产物路线 —— `echo-mini` 走从零全链路训练，`echo` 在开源底座上做微调，作为课件知识点的工程化落地。

## 本站导航

| 板块 | 说明 |
|------|------|
| [Courseware](Courseware/outline.md) | 课件正文，按章节推进 |
| DesignDoc | [0. 启动提案](DesignDoc/00-startup-proposal.md)（可首先阅读）<br>[1. 项目计划书](DesignDoc/01-project-plan.md)<br>[项目任务清单](DesignDoc/tasks.md)<br>[踩坑记录](DesignDoc/troubleshooting.md) |
| Reading | 文章或译文，延伸阅读 |

## 仓库与配套资源

- **GitHub 仓库**：<https://github.com/yangfch3/llm01>
- **课件配套练习代码**：存放于仓库的 [`Playground/`](https://github.com/yangfch3/llm01/tree/main/Playground) 目录。
- **Echo 模型代码**：见仓库的 [`Echo/`](https://github.com/yangfch3/llm01/tree/main/Echo) 目录，含 `echo-mini/`、`echo/` 与共享工具 `shared/`。
- [LLM 显存估算器](/llm01/tools/vram-estimator.html)
