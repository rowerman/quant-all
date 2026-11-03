# 快速开始

## 环境

- Python 3.12
- Linux 或可编译 C++20 扩展的环境
- CMake、Ninja 和 pybind11 由构建依赖自动安装

推荐使用 `uv`：

```bash
uv sync --extra dev
```

使用 pip 的最小流程：

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e .
```

## 合成演示

```bash
make demo-data
make demo
```

生成器默认写入 `data/demo`，包含 120 只股票、3 个自然年的确定性行情和 5 个内置动量
因子。演示会计算 MOM_20 评估和一次 Top-20 月频回测。

## 数据目录隔离

`QUANT_DATA_DIR` 同时覆盖行情库、因子库、模型和挖掘目录：

```bash
QUANT_DATA_DIR=/tmp/quant-demo make demo
```

默认值仍是项目内 `data/`。正式数据和演示数据不要共用目录。

## Web

```bash
QUANT_DATA_DIR="$PWD/data/demo" bash tools/monitor/serve.sh
```

页面默认监听 `127.0.0.1:8800`。该服务是本地研究工具，不应直接暴露到公网。

