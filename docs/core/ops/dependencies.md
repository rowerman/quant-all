# 依赖安装与版本锁定

## 文件

- `requirements.txt`：项目运行时直接依赖，全部使用精确版本。
- `requirements.lock`：由 `uv pip compile --generate-hashes` 生成的完整运行时锁。
- `pyproject.toml`：测试和静态检查依赖位于 `dev` extra。

两份文件都必须同步更新。只升级一个包时，先改直接依赖版本，再在干净环境中重建
`requirements.lock`，不要手工删除锁文件里的传递依赖。

## 安装

```bash
cd /home/hanwen/quant
.venv/bin/pip install --require-hashes -r requirements.lock
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip check
```

测试会校验直接依赖与完整锁文件的版本一致：

```bash
.venv/bin/python -m pytest tests/test_dependencies.py -q
```

## 当前关键版本

| 组件 | 版本 |
|---|---|
| Python | 3.12.3 |
| NumPy / Pandas | 2.2.6 / 2.3.3 |
| DuckDB | 1.5.5 |
| pypdf | 6.21.2 |
| VectorBT / Numba | 0.28.5 / 0.61.2 |
| CMake / Ninja | 4.4.4 / 1.13.2 |
| pybind11 / scikit-build-core | 3.1.0 / 1.1.1 |
| scikit-learn / joblib | 1.9.1 / 1.6.0 |
| LightGBM | 4.7.0 |
| FastAPI / Starlette | 0.141.1 / 1.6.0 |
| pytest | 9.1.1 |
