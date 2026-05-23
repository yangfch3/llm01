# LLM 0-1

欢迎来到 **LLM 0-1** 的文档站。

本项目是一份「LLM 入门课件 + 练习代码」与 **Echo 对话模型**的组合产出：

- **课件**：从数学基础起步，覆盖分词、Transformer、训练、推理、量化与部署等环节，按章节循序推进。
- **Echo 模型**：双产物路线 —— `echo-mini` 走从零全链路训练，`echo` 在开源底座上做微调，作为课件知识点的工程化落地。

## 仓库与配套资源

- **GitHub 仓库**：<https://github.com/yangfch3/llm01>
- **课件配套练习代码**：与课件章节一一对应，存放于仓库的 [`Playground/`](https://github.com/yangfch3/llm01/tree/main/Playground) 目录下（`chNN-xxx/` 形式命名）。
- **工具**：
  - [LLM 显存计算器](/llm01/tools/vram-estimator.html)
- **Echo 模型代码**：见仓库的 [`Echo/`](https://github.com/yangfch3/llm01/tree/main/Echo) 目录，含 `echo-mini/`、`echo/` 与共享工具 `shared/`。

## 板块导航

| 板块 | 说明 |
|------|------|
| [Courseware](Courseware/outline.md) | 课件正文，按章节推进 |
| [DesignDoc](DesignDoc/00-startup-proposal.md) | 项目设计文档与技术方案，`00-startup-proposal.md` 为根基文档 |

> 阅读建议：第一次访问可以先翻 [启动提案](DesignDoc/00-startup-proposal.md) 了解项目定位与目录结构，再从 [课件大纲](Courseware/outline.md) 进入正文学习。
