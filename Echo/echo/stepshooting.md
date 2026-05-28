# echo 训练步骤速查

> cwd: `Echo/echo/`，所有命令均在此目录下执行。

echo 项目的锚点是**基于 base 模型从零跑通 SFT 全链路**——这是教学价值的来源，
也是默认的训练路线。Instruct 路线作为对照实验存在。

| 路线 | 文档 | 底座 | 默认？ | 用途 |
|---|---|---|---|---|
| **base（默认）** | [stepshooting-base.md](stepshooting-base.md) | `Qwen/Qwen2.5-1.5B` | ✅ | 主线。SFT 全链路 + 教模型学 ChatML / 停止行为 / 风格 |
| 对照 instruct | [stepshooting-instruct.md](stepshooting-instruct.md) | `Qwen/Qwen2.5-1.5B-Instruct` | 否 | 对照。验证已对齐底座做风格漂移的效果差异 |

所有脚本默认值（`--config` / `--adapter-dir` / `--val-file` / `--output-dir` 等）都
指向 base 路线。Instruct 路线必须显式传参，详见 stepshooting-instruct.md。

通用脚本说明见 `SPEC.md`，每个脚本的命令行参数自带 `--help`。
