# shared

`echo-mini` 和 `echo` 共用的底层工具。

## 当前模块

- `device.py` · 统一设备选择（`cuda → mps → cpu` 优先级）+ MPS op fallback 工具

## 后续规划

随里程碑推进逐步补充：

- `data/` · 数据加载、packing、tokenize 通用接口
- `train/` · 训练循环、checkpoint 管理、混合精度、LR 调度
- `eval/` · PPL、自建评测集运行器
- `logging.py` · 统一日志（rich + 文件）

## 原则

- 只放**两个产物都要用**的代码，单产物专属代码留在各自 `src/`
- 不引入额外重型依赖，依赖与 `echo-mini` / `echo` extras 一致
