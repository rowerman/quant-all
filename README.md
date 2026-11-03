# ashare-quant-research

面向 A 股日频研究的开源工程，覆盖数据清洗、因子构建与评估、机器学习模型、
带交易规则的组合回测、受控 LLM 因子挖掘和本地 Web 审核。

项目重点是可追溯的研究工程，不是收益承诺或实盘交易系统。2024-2026 最终留出验证
未通过预注册门槛，项目不会把它包装成稳定 Alpha。

## 能力

- 使用 DuckDB 和 Parquet 管理行情、财务、指数、行业、公司行动和因子数据。
- 区分后复权研究价与原始成交价，处理停牌、涨跌停、ST、退市和板块交易单位。
- 提供 IC、分层、换手、衰减、风格暴露、增量 IC 和相关性等因子评估。
- 使用 purged walk-forward、embargo、冻结预测和封存留出验证模型。
- 使用自定义账本处理 T+1、成本、冲击、税费、分红送转和退市核销。
- 使用冻结 DSL、确定性门禁、人工审批和原子发布约束 LLM 因子挖掘。
- 通过 FastAPI、SSE 和原生 JavaScript 展示因子、挖掘和回测结果。
- 提供 C++20/pybind11 批量计算核和 Python 回退实现。

## 架构

```text
Tushare / gbbq
      |
      v
清洗与复权 -> DuckDB 主库 -> 因子 Parquet
                              |      |
                              v      v
                          因子评估   模型训练
                              \      /
                               v    v
                              组合回测
                                  |
                                  v
                             FastAPI 监测站
                                  ^
                                  |
                    受控 LLM 挖掘 -> 人工审批 -> 原子发布
```

当前推荐入口是 V3 自动因子挖掘。V1、V2、M5 和旧协议读取继续作为兼容能力保留。

## 五分钟演示

演示使用确定性合成数据，不需要 Tushare Token、DeepSeek Key、Docker 或私有数据库。
合成数据仅用于验证代码链路，不代表真实市场表现。

```bash
uv sync --extra dev
make demo-data
make demo
```

启动本地页面：

```bash
QUANT_DATA_DIR="$PWD/data/demo" bash tools/monitor/serve.sh
```

`QUANT_DATA_DIR` 会把数据、因子、模型和结果统一重定向到指定目录。相对路径以项目根
目录为基准，不设置时仍使用默认 `data/`。

## 真实数据

在项目根目录创建 `.env`：

```text
TUSHARE_TOKEN=<local token>
TUSHARE_HTTP_URL=http://jiaoch.site
FACTOR_DEEPSEEK_API=<worker deepseek key>
```

获取、清洗和建库：

```bash
.venv/bin/python -m scripts.sources.fetch_all reference
.venv/bin/python -m scripts.sources.fetch_all daily --start 20000101 --end 20261015
.venv/bin/python -m scripts.sources.fetch_all extra --end 20261015
.venv/bin/python -m scripts.sources.fetch_all index --end 20261015
.venv/bin/python -m scripts.sources.fetch_all core --end 20261015
.venv/bin/python -m scripts.sources.fetch_all gbbq
.venv/bin/python -m scripts.clean.pipeline
.venv/bin/python -m scripts.build_db.build
.venv/bin/python -m scripts.build_factors.build
.venv/bin/python -m scripts.build_factors.validate
```

全量取数需要数小时，主库重建会替换正式数据库。执行前应备份，并先核对命令中的日期。
更细的数据口径和表结构见 [数据层说明](docs/core/README.md)。

## 常用入口

```bash
# 因子评估
QUANT_DATA_DIR="$PWD/data/demo" .venv/bin/python -c \
  "from calculate.evaluate import evaluate; print(dict(evaluate('MOM_20', stages=['overview'])))"

# 回测 CLI
.venv/bin/python -m backtest --help

# 模型训练
.venv/bin/python -m scripts.build_alpha.train --help

# 页面
bash tools/monitor/serve.sh
bash tools/monitor/serve_v3.sh
```

## 研究结果

主模型为五种子 LightGBM rank 集成，使用周频、未来 5 日开盘到开盘收益标签和 purged
walk-forward 训练。

| 2024-2026 留出结果 | 年化净收益 | 净超额 | Sharpe | 最大回撤 |
|---|---:|---:|---:|---:|
| LightGBM 四周平滑 | 15.37% | 3.33% | 0.5263 | -45.54% |
| Ridge 原始分数 | 10.96% | -1.08% | 0.4242 | -40.02% |
| 全因子 rank 基线 | 19.43% | 7.39% | 0.7795 | -19.89% |

主模型只有年化净收益优于 Ridge，RankIC、ICIR、Sharpe 和最大回撤均没有稳定优势，
并且整体跑输简单基线。2024 年净超额为 -31.46%，收益集中在 2025 年。因此最终状态是
`fail_final_holdout`，不构成可交易模型。

## 测试

```bash
make test-unit          # 默认快速单元测试，不读正式数据
make test-integration   # 真实数据、原生扩展、Docker 等资源型测试
make test-e2e           # 完整链路或浏览器场景
make test-full          # 除付费和公网调用外的完整测试
make lint
make build-wheel
```

默认单元测试不依赖真实数据库、编译扩展、网络或付费模型。当前采集 1,800 余个用例，
快速层在开发机上约 1-2 分钟完成。

## 项目结构

| 路径 | 职责 |
|---|---|
| `core/data` | 取数、清洗、建库和 `quant` 读取层 |
| `core/factors` | 因子定义、算子、注册和发布 |
| `core/kernels` | C++ 批量核与 Python 回退 |
| `research/evaluation` | 因子评估指标和面板 |
| `research/modeling` | 数据集、模型、监控和生命周期 |
| `research/mining` | 受控 DSL、LLM Worker、审核和发布 |
| `trading/backtest` | A 股账本、交易规则、成本和结果存储 |
| `apps/monitor` | FastAPI、SSE 和前端页面 |
| `examples` | 合成数据与可运行演示 |
| `docs` | 架构、数据口径和研究说明 |

## 已知限制

- 不是实盘系统，没有券商接入、实时行情、盘中撮合、账户对账或风控熔断。
- Web 服务是单进程本地工具，没有多租户和身份认证。
- 最终留出窗口已消耗，不能用同一窗口重新调参后宣称独立成功。
- 自动挖掘测试段与历史模型留出存在区间重叠，不能宣称全项目独立样本发现。
- 因子评估的重叠标签标准误和部分统计检验仍有限制。
- 回测不模拟集合竞价队列、盘口深度、部分成交、撤单、融资融券和并购换股。

完整说明见 [系统架构](docs/architecture.md)、[研究方法](docs/research-methodology.md)
和 [已知限制](docs/limitations.md)。

## 许可证

代码使用 [MIT License](LICENSE)。Tushare、RD-Agent、Qlib、VectorBT 和其他第三方数据
或依赖继续遵守各自许可证，不包含在 MIT 授权范围内。
