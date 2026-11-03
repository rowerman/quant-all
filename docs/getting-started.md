# 快速开始

## 环境

- Python 3.12
- C++20 编译器、CMake 和 Ninja
- Linux 或可编译 PyBind11 扩展的环境

推荐使用 `uv`：

```bash
uv sync --extra dev
```

使用 pip：

```bash
python3.12 -m venv .venv
.venv/bin/pip install --require-hashes -r requirements.lock
.venv/bin/pip install -e ".[dev]"
```

## 合成演示

合成数据不需要 Tushare、模型 API、Docker 或私有数据库：

```bash
make demo-data
make demo
```

生成器默认写入 `data/demo`，包含 120 只股票、3 个自然年的确定性行情和 5 个动量因子。
演示会执行 MOM_20 因子评估和 Top-20 月频回测。

## 数据目录隔离

```bash
QUANT_DATA_DIR=/tmp/quant-demo make demo
```

`QUANT_DATA_DIR` 同时覆盖行情库、因子库、模型和挖掘目录。相对路径以当前项目根目录为
基准；未设置时仍使用 `data/`。

## 页面

```bash
QUANT_DATA_DIR="$PWD/data/demo" bash tools/monitor/serve.sh
```

默认监听 `127.0.0.1:8800`。该服务是单进程本地研究工具，不应直接暴露到公网。

