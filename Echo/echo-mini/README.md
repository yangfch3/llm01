# echo-mini

从零走完整链路的迷你 Echo（参数量 ~10M–100M）。

**当前状态**：占位。代码与配置在 M4 里程碑落地，详见 [`../../Doc/DesignDoc/01-project-plan.md`](../../Doc/DesignDoc/01-project-plan.md)。

## 规划目录

```
echo-mini/
├─ configs/           训练配置（full/tiny 双份）
├─ data/              数据（.gitignore，脚本下载）
├─ tokenizer/         分词器产物（.gitignore，脚本入仓）
├─ src/echo_mini/     模型与训练代码
├─ scripts/           训练 / 推理 / 评测脚本
├─ checkpoints/       权重（.gitignore，走 HF Hub）
└─ README.md          训练配方（M4 完后补）
```

## 价值定位

- 教学价值优先，效果不强求
- 走通"数据 → 分词器 → Pretrain → SFT → 评测 → 推理"全链路
- 与 `Echo/echo` 共用 `Echo/shared/` 下的工具
