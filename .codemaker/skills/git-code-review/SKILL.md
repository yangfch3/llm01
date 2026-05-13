---
name: git-code-review
description: >
  Git 代码审查（支持历史提交 & 本地未提交变更）。触发词："review 提交"、"Git review"、
  "review 本地改动"、"review 未提交的代码"、"review working tree"、"看看我改了什么"。
---

# Git Code Review Skill

对 Git 历史提交或本地未提交变更进行 Code Review，输出结构化报告。

## 前置检查

1. `git --version` — 失败则终止。
2. `git rev-parse --is-inside-work-tree` — 确认当前目录在 Git 仓库内，失败则终止。

## 模式判断

根据用户意图确定模式：

| 信号 | 模式 |
|------|------|
| 提到"提交"、"最近N条"、"git log"、指定 commit/分支/区间 | **历史提交** |
| 提到"本地"、"未提交"、"working tree"、"暂存区"、"改动" | **本地变更** |
| 无法判断 | 用 `ask_user_question` 让用户选择 |

---

## A. 历史提交模式

### A1. 获取 Git 作者信息

读取 `._proj/profile_rules.md` 查找 git author (name / email)。找到则确认，未找到则尝试 `git config user.name` / `git config user.email`；仍不明确则 `ask_user_question` 询问。

### A2. 获取提交记录

```
git log --author="{author}" -n {N} --pretty=format:"%h|%an|%ad|%s" --date=iso --name-status
```

默认 N=10，用户可指定。若用户指定了区间（如 `main..HEAD`、`HEAD~5..HEAD`），直接用区间替代 `-n {N}`。

### A3. 用户多选条目

以多选列表展示（短 hash + 时间 + 提交信息 + 关键文件摘要），用户选择要 review 的条目。

### A4. 筛选代码文件 & 获取 diff

对选中条目筛选代码文件（规则见下方），按时间从旧到新逐条执行：

```
git show {commit} -- {筛选后的文件...}
```

或只看 diff：`git diff {commit}^..{commit} -- {files...}`。
首个提交（无父）用 `git show --root {commit}`。
diff 超 500 行时按文件分别获取。

### A5. 报告

文件名：`.ai_review/YYYYMMDD_{short_hash_min}-{short_hash_max}_review.md`

---

## B. 本地变更模式

### B1. 获取变更文件列表

```
git status --short
```

覆盖已暂存（index）与未暂存（working tree）以及未跟踪（untracked）文件。

### B2. 筛选代码文件 & 获取 diff

从 status 输出中筛选代码文件（规则见下方），然后：

```
git diff HEAD -- {files...}
```

说明：
- `git diff HEAD` 同时涵盖已暂存 + 未暂存改动。
- 未跟踪（`??`）新文件 `git diff` 不显示，对其使用 `git diff --no-index /dev/null {file}`（Windows 用 `NUL`）或直接读取文件内容。
- diff 过长时按单文件分别获取。

如果用户要求只 review 部分文件，用 `ask_user_question` 让用户多选。

### B3. 报告

文件名：`.ai_review/YYYYMMDD_local_review.md`

报告模板中「审查范围」改为 `本地未提交变更`，「提交者」省略。

---

## 共享规则

### 代码文件筛选

- ✅：`.cs` `.lua` `.txt`(模板) `.py` `.js` `.ts` `.json`(配置) `.shader` `.cginc` `.hlsl` `.compute`
- ❌：`.prefab` `.meta` `.asset` `.unity` `.FBX` `.mat` `.anim` `.controller` `.png` `.jpg` `.tga` `.wav` `.mp3`

筛选后无代码文件变动 → 告知用户并结束。

### Review 守则

1. **语法与编译**：检查明显的语法错误、不可编译/运行的问题
2. **项目规范**：检查是否符合工作区 rules 中定义的编码规范（命名、文件组织、框架模式等）
3. **性能问题**：重点关注可能的性能问题（高频 GC、帧循环中的临时分配、不必要的重复计算等）
4. **架构设计**：如果涉及架构设计变更，检查设计是否合理，是否有未来隐患
5. **务实边界**：不苛求永远不会遇到的极端边界条件
6. **效率导向**：质量严格把关，同时抓重点、兼顾效率

特殊：Unity 版本只往新升级，无需考虑旧版本兼容。用户提供的自定义规则追加到上述守则之后。

### 报告模板

```markdown
# Git Code Review Report

**审查范围**：{commit 区间 或 本地未提交变更}
**审查日期**：YYYY-MM-DD
**提交者**：{git author，本地模式省略}

---

## {short_hash 或 文件名} — {提交信息或变更概述}

**变更文件**：{代码文件列表}

### ✅ 正面评价
{值得肯定的设计/实现}

### ⚠️ 问题

**[中等] 问题标题**
{描述 + 代码引用 + 建议}

---

## 总结

| 级别 | 数量 | 说明 |
|------|------|------|
| 🔴 阻断 | N | 必须修复 |
| 🟡 中等 | N | 建议修复 |
| 🟢 低 | N | 可选优化 |

**总体评价**：{一段话总结}
```

### 输出行为

1. 不在会话中展示 review 结果，只写入 `.ai_review/` 目录（不存在则创建）。
2. 文件使用 UTF-8 编码、LF 换行、末尾留一空行、无 BOM。

## 注意事项

- Windows 环境下 Git 命令不使用 `| cat` 管道；必要时加 `--no-pager`，例如 `git --no-pager log ...`、`git --no-pager diff ...`。
- 合并提交（multi-parent）默认 `git show` 不显示 diff，需加 `-m` 或 `--first-parent`；若无实质代码改动则跳过并在报告中说明。
- `git diff` 输出为空（纯重命名/权限变更）→ 跳过并在报告中说明。
