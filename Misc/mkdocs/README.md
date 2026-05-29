# mkdocs 本地环境说明

## 环境准备

站点工具链与项目主依赖隔离，独立环境预期位置：`Misc/mkdocs/.venv-docs/`，依赖锁见 `Misc/mkdocs/requirements-docs.txt`。

首次本地预览前需执行：
```bash
cd Misc/mkdocs

uv venv .venv-docs

uv pip install --python .venv-docs -r requirements-docs.txt
```

## 启动文档站

```bash
uv run python scripts/docs_serve.py
```

浏览器打开 `http://127.0.0.1:8000` 即可浏览。

