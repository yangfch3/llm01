# echo

基于开源底座（Qwen2.5-0.5B / 1.5B 候选）SFT + DPO 微调的实用 Echo。

**当前状态**：占位。代码与配置在 M5/M6 里程碑落地，详见 [`../../Doc/DesignDoc/01-project-plan.md`](../../Doc/DesignDoc/01-project-plan.md)。

## 规划目录

```
echo/
├─ configs/           SFT / DPO 配置（full/tiny 双份）
├─ data/              数据（.gitignore）
├─ src/echo/          训练与推理代码
├─ scripts/           训练 / 推理 / 量化脚本
├─ checkpoints/       LoRA adapter / 合并权重（.gitignore，走 HF Hub）
├─ eval/              评测集与脚本（M5 启动时定）
└─ README.md          训练配方
```

## 价值定位

- 实用产物，目标"初中生水平"中英基础对话
- 验收标准见 [`../../Doc/DesignDoc/00-startup-proposal.md`](../../Doc/DesignDoc/00-startup-proposal.md) §4.5
- 部署走 GGUF + Ollama / llama-cpp-python，详见 [`../../Doc/DesignDoc/02-deps-compatibility.md`](../../Doc/DesignDoc/02-deps-compatibility.md) §2
