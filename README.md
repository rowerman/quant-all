# A 股量化数据项目

从 Tushare 取全 A 股日线行情与辅助数据，清洗、入库，再用 `quant` 读取层或 DuckDB SQL 取用。

数据链路：`data/source/`（Tushare 与复权事件源数据） → `data/clean/`（清洗产物） → `data/databases/quant.duckdb`（入库结果，日常只读这一层）。

当前数据版本 `tushare_fullA_20000104_20261015`，2026-10-18 增量加入公司行动表（26 张表，约 7.3 GB）：
日线覆盖 2000-01-04 ~ 2026-10-15 共 6488 个交易日、5913 只股票，
另有 60 个指数与 31 个申万一级行业的日线行情，以及未来收益标签表。

在其上另建了因子库 `data/factors/` + `data/databases/factors.duckdb`：十一个内置因子族、
47 个已实现因子，
行集与日线库一致，另有 100+ 条待实现因子的公式清单（见 `docs/core/factors/catalog.md`）。

## 目录

| 路径 | 内容 |
|---|---|
| `core/` | 数据读取、取数/清洗/建库脚本、因子定义与公共路径 |
| `research/` | ML 模型与因子评估计算包 |
| `trading/` | 交易与回测模块 |
| `apps/monitor/` | 因子监测后端与前端 |
| `data/` | 原始源数据、清洗结果、因子值、模型产物与 DuckDB |
| `data/source/tushare/*.parquet` | 一次性参照表：交易日历、股票基础表、曾用名、停牌记录、指数权重 |
| `data/source/tushare/raw/<接口>/year=YYYY/month=MM.parquet` | 按交易日批量拉取的行情分片 |
| `data/clean/daily/year=YYYY/part.parquet` | 清洗后的日线长表（27 年） |
| `data/clean/{calendar,suspend_segments,anomalies}.parquet`、`data/clean/manifest.json` | 衍生表与产物清单 |
| `data/databases/quant.duckdb` | DuckDB 单文件数据库，日常取数入口 |
| `core/factors/` | 因子定义：算子库、注册表、十一个因子族、100+ 因子规划清单 |
| `data/factors/<族>/year=YYYY/part.parquet` | 因子值，一年一族一个文件 |
| `data/databases/factors.duckdb` | 因子目录库：定义、覆盖率、自检结论、构建记录与按族视图 |
| `core/data/scripts/build_factors/` | 因子构建、自检与文档生成脚本 |
| `demo.py` | 数据读写库用法速查：25 块覆盖全部读取接口与组合示例，可逐块复制进 ipynb |
| `core/data/quant/` | 读取层代码 |
| `core/data/scripts/` | 取数、清洗、建库脚本 |
| `tests/` | 单元测试 |
| `docs/` | 各模块文档 |
| `logs/` | 按模块分目录的运行日志 |
| `tools/v5/` | 取数/清洗/建库的串联脚本、看板与诊断脚本 |
| `research/evaluation/calculate/`、`apps/monitor/backend/`、`apps/monitor/frontend/` | 因子质量监测项目：计算包 / 后端接口 / 前端页面 |

## 快速上手

在项目根目录用自带的虚拟环境：

```bash
cd /home/hanwen/quant
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python
```

```python
import quant

# 长表：一次取一批字段。adjust 取 raw / hfq / qfq
df = quant.load_panel(
    start="2020-01-01", end="2020-12-31",
    fields=["open", "high", "low", "close", "ret", "limit_up", "is_limit_up", "status"],
    adjust="hfq",                # 做收益、因子研究用 hfq；看成交价用 raw
    stocks=["SH600000"],         # 不传就是全市场
    drop_suspended=True,         # 剔除停牌行
)

# 宽表：交易日 × 股票，方便做横截面运算
close = quant.load_panel_wide("close", start="2020-01-01", end="2020-12-31", adjust="hfq")

# 辅助表
quant.load_calendar(start="2026-01-01")
quant.load_stock_basic(status="listed")            # 或 "delisted"
quant.load_suspend_segments()
quant.load_index_weight("000300.SH", start="2026-01-01")   # 月末快照，带归一后的 stock_code
quant.load_index_daily("000300.SH", start="2026-01-01")    # 指数日线行情（60 个指数）
quant.load_sw_index_daily("801010.SI")                     # 申万一级行业日线（31 个行业）
quant.load_index_basic(market="CSI", category="规模指数")   # 指数目录

# 未来收益标签：N 取 1/3/5/10/15/20，price 取 close 或 open
lab = quant.load_label(start="2024-01-01", end="2024-12-31",
                       horizons=[1, 5, 20], price="close")
fwd5 = quant.load_label_wide(5, start="2024-01-01", end="2024-12-31", price="close")

# 需要写 SQL 时
from quant.db import query
query("select trade_date, close from md.daily_hfq where stock_code = 'SH600000' order by 1")
```

逐块探索用根目录 `demo.py`：25 块覆盖全部读取接口、元数据与常用组合用法，
每块以 `# %%` 开头，可单独粘进 ipynb，变量名互不覆盖。

## 数据库表清单

| schema.table | 行数 | 说明 |
|---|---|---|
| `md.daily_bar` / `md.daily_hfq` / `md.daily_qfq` | 各 17,999,882 | 三张同构日线表，键为 `stock_code` + `trade_date` |
| `md.fwd_return` | 17,999,882 | 未来 1/3/5/10/15/20 日收益标签，收盘与开盘两套成交通道 |
| `md.index_daily` | 291,110 | 60 个指数日线，含沪深300/中证500全收益指数 |
| `md.sw_index_daily` | 104,777 | 31 个申万一级行业指数日线，含行业估值与市值 |
| `md.calendar` | 6,480 | 交易日历 |
| `md.suspend_segment` | 71,319 | 停牌区间，含持续天数与是否长停牌 |
| `ref.stock_basic` | 5,907 | 股票基础表，含退市股的上市日与退市日 |
| `ref.namechange` | 14,215 | 证券简称变更历史，可推 ST 与退市整理期 |
| `ref.index_weight` | 984,140 | 12 个宽基的历史成分权重，月末快照；`con_code` 为 Tushare 原写法，join 请用 `stock_code` |
| `ref.index_basic` | 10,609 | 指数目录：来源、类别、发布日 |
| `ref.corporate_action` | 64,618 | 分红、送转、配股与可识别拆并的除权日账本 |
| `evt.suspend_daily` | 234,916 | 官方停牌记录明细 |
| `meta.manifest` | 1 | 数据版本与生成时间 |
| `meta.anomaly` | 57,157 | 清洗阶段标出的异常明细，`kind` 分四类 |
| `meta.data_dictionary` | 66 | 字段单位与口径说明 |

## 读取接口

| 接口 | 说明 |
|---|---|
| `load_panel(start, end, fields, adjust, stocks, drop_suspended)` | 日线长表，按 `trade_date`、`stock_code` 排序；`stock_code`/`trade_date` 始终返回 |
| `load_panel_wide(field, start, end, adjust)` | 把单个字段透视成「交易日 × 股票」宽表 |
| `load_calendar(start, end)` | 交易日历 |
| `load_stock_basic(status)` | `status` 支持 `"listed"` / `"delisted"`，也接受 Tushare 的 `"L"` / `"D"` / `"P"`，`None` 为全部 |
| `load_suspend_segments()` | 停牌区间表 |
| `load_index_weight(index_code, start, end)` | 指数历史成分权重，如 `"000300.SH"` |
| `load_index_daily(index_code, start, end)` | 指数日线行情（`md.index_daily`） |
| `load_sw_index_daily(index_code, start, end)` | 申万一级行业日线（`md.sw_index_daily`） |
| `load_index_basic(index_code, market, category)` | 指数目录（`ref.index_basic`） |
| `load_label(start, end, horizons, price, stocks)` | 未来收益标签长表；`price` 取 `close` / `open`，`horizons` 取 1/3/5/10/15/20 |
| `load_label_wide(horizon, start, end, price)` | 把某一档标签透视成「交易日 × 股票」宽表 |
| `load_members(table, name)` | 板块/概念成分，依赖的表本次未入库，调用会抛出带「可用表清单」的错误 |
| `load_financial(stock_code, table)` | 财务报表，依赖的 `fin` schema 本次未入库，同上 |
| `load_factor_panel(start, end, names, families, stocks)` | 因子长表，键 + 指定因子列 |
| `load_factor_wide(name, start, end)` | 单因子宽表（交易日 × 股票） |
| `load_factor_list(family, status)` | 因子清单：族、状态、方向、公式、白话步骤、实现表达式、覆盖率 |
| `built_families()` | 已落盘的因子族 |

## 因子库

因子层建在日线库之上：因子定义写在 `core/factors/` 包里，构建脚本算出因子值落成 Parquet，
目录库 `data/databases/factors.duckdb` 记下定义与覆盖率，读取走 `quant`。

**十一个因子族**：`momentum` 动量反转、`volatility` 波动风险、`liquidity` 流动性、
`volume_price` 量价关系、`technical` 技术形态、`valuation` 估值规模、`fundamental`
财务质量、`cn_special` A 股特色、`market` 市场相对强度、`alpha101` 与 `gtja191` 公式复现。

已实现 47 个因子（覆盖全部因子族），清单里另有待实现的公式，
照 `docs/core/factors/README.md` 的三步就能补：写 `define(...)` → 跑构建 → 跑自检与文档生成。

**每个因子都带公式和白话步骤**，想知道某个数是怎么算出来的：

```bash
# 公式 + 分步说明 + 一条可以直接跑的手工复现 SQL
.venv/bin/python -m scripts.build_factors.catalog --explain RSI_14
```

```python
import quant
quant.load_factor_list(family="momentum")[["name", "formula", "steps", "expr"]]
```

- `docs/core/factors/formulas.md`：47 个已实现因子的「公式 → 步骤 → 真实 SQL」；
- `docs/core/factors/cookbook.md`：入门教程，从一只股票用 pandas / SQL 手算并和库值对照
  （实测 MOM_20 最大差 3.7e-09，RSI_14 最大差 3.8e-06，都是 float32 的正常精度）。

```bash
# 构建（全历史、逐族逐年，可反复重跑；只建某族或某段年份都可以）
.venv/bin/python -m scripts.build_factors.build
.venv/bin/python -m scripts.build_factors.build --families momentum --start 2024-01-01 --end 2024-12-31

# 自检：行集、取值、前视偏差、方向（Rank IC / ICIR）
.venv/bin/python -m scripts.build_factors.validate

# 重新生成因子清单文档
.venv/bin/python -m scripts.build_factors.catalog --write-docs
```

```python
import quant

# 长表
df = quant.load_factor_panel(start="2024-01-01", end="2024-12-31",
                             names=["MOM_20", "EP_TTM", "PV_CORR_20"])
# 宽表
mom20 = quant.load_factor_wide("MOM_20", start="2024-01-01", end="2024-12-31")
# 清单
quant.load_factor_list(family="momentum")          # status="planned" 看还没实现的
```

口径要点：

1. 因子文件的行集与 `md.daily_hfq` **完全一致**，含停牌行与退市股；**停牌行的因子值为空**，
   与标签口径一致，下游按 `can_buy_close` 这些标记过滤；
2. 计算时先筛 `status='trading'`，窗口按有效交易日推进，算完再左连接回完整行集；
3. 价格类因子一律用后复权价；换手率、估值、市值直接用库内列；
4. 每年构建时往前多读该族最长回看窗口的交易日，所以 1 月初的长窗口因子也有值；
5. 因子值落 float32，落盘前把 `NaN` / `inf` 统一置空；
6. 只存原始值，截面标准化、排序、中性化在读取时现算。

## ML 模型层

`research/modeling/alpha/` 是模型层，已完成研究契约、周频数据集、purged walk-forward、模型训练、
通用评估、完整账本、监控和生命周期接入。当前可用后端包括 Ridge/ElasticNet、
LightGBM 和 Scikit-learn 直方图 GBDT。

新后端放入 `research/modeling/alpha/modeling/backends/` 后会自动发现并注册；训练命令完成后会自动
重建模型索引。模型索引同时汇总相关评估、账本、监控、生命周期和最终留出。

```bash
# P0 特征审计与基线
.venv/bin/python -m scripts.build_alpha.audit_features \
  --start 2012-01-01 --end 2023-12-31
.venv/bin/python -m scripts.build_alpha.run_baselines \
  --baseline B4_ALL_RANK --start 2012-01-01 --end 2023-12-31 --engine weekly

# P1 数据集与切分
.venv/bin/python -m scripts.build_alpha.build_dataset \
  --start 2012-01-01 --end 2023-12-31
.venv/bin/python -m scripts.build_alpha.build_folds

# 通用模型训练与索引查询
.venv/bin/python -m scripts.build_alpha.train \
  --dataset-id 6a59680fd71c50a8 \
  --spec configs/alpha/train_sklearn_hist_gbdt.json
.venv/bin/python -m scripts.build_alpha.registry --rebuild
```

研究契约见 `docs/research/modeling/alpha/research_contract.md`，基线结论见 `docs/research/modeling/alpha/feature_audit.md`。
2024-01-01～2026-10-15 已消耗并封存；下一留出窗口
2026-10-16～2027-10-15 已登记，需等数据和标签完整后执行。

## 字段口径

**代码格式**统一为 `SH600000` / `SZ000001` / `BJ920000`（交易所前缀 + 6 位数字）。
原始数据里的 `600000.SH`、`sh.600000`、`T600018.SH` 等写法已在入库时归一，各表 join 不用再转换。

**三张日线表结构完全一致，只有四价与 `vwap` 的口径不同**：

| `adjust` | 表 | 价格口径 |
|---|---|---|
| `raw` | `md.daily_bar` | 未复权原始价 |
| `hfq` | `md.daily_hfq` | 后复权价 = 原始价 × `factor` |
| `qfq` | `md.daily_qfq` | 前复权价 = 原始价 × `factor` / `factor_T`，`factor_T` 取该股最后一个有效交易日的因子 |

`factor`、`change`、`pre_close`、`ret`、`status`、涨跌停标记、可交易性标记这些列三张表内容一致。

**单位**（完整口径见 `meta.data_dictionary` 表）：

| 字段 | 单位 |
|---|---|
| 四价 / 涨跌停价 / `vwap` / `pre_close` / `change` | 元 |
| `volume_share` | 股 |
| `amount_cny` | 元 |
| `total_mv` / `circ_mv` | 万元 |
| `turnover_rate` / `dv_ttm` | % |
| `factor` | 无量纲 |

`status` 只有两个取值：`trading`、`suspended`。

## 取用时要注意

1. `qfq` 的基准是每只股票**最后一个有效交易日的因子**，所以同一只股票在不同时间取 `qfq`，数值会变；
   回测执行已改用原始价账本，公司行动单独入账；`qfq` 仍用于研究型因子和复权价格读取。
   回测运行会记录 `metadata.snapshot_id` 与分项摘要，复现时先比较快照而不是只看版本标签。
2. 只落 `[list_date, delist_date]` 区间内的行，上市前与退市后没有空行；要区分「停牌」和「未上市」看 `status` 与 `list_date`。
3. 涨跌停价优先用官方 `stk_limit`，缺失时才按「板块 + 日期 + ST + 退市整理期」推算，可用 `pre_close_source` 区分来源。
4. 读连接默认 `read_only=True`，可以和其它进程并存；写库要显式传 `read_only=False`。
5. 日线表没有索引，按日期区间取数时务必带 `start`/`end`，否则会把整表读进内存。
6. 股票代码是 `SH600000`，**指数代码保留 Tushare 写法** `000300.SH`，两套并存；
   `ref.index_weight` 里 join 日线表要用归一后的 `stock_code`。
7. `md.index_daily` 里 `399925.SZ` 基本面50（到 2019-01-25）与 `399984.SZ` 300等权
   （到 2015-04-29）**已经停编**，只有历史行情，别拿它们当基准。
8. 中证2000（`932000.CSI`）的成分权重覆盖率不足，未纳入 `ref.index_weight`；
   它的**日线是完整的**，可以直接用。
9. 申万行业日线从 2010-04-13 起，比中证系指数短，这是服务端限制。

## 重新生成数据

```bash
# 一次性参照数据（交易日历、股票基础表、曾用名）
.venv/bin/python -m scripts.sources.fetch_all reference

# 按交易日拉四个行情接口：daily / adj_factor / stk_limit / daily_basic
.venv/bin/python -m scripts.sources.fetch_all daily --start 20000101 --end 20261015

# 停牌记录 + 指数历史成分权重
.venv/bin/python -m scripts.sources.fetch_all extra

# 指数数据：含沪深300/中证500全收益指数
.venv/bin/python -m scripts.sources.fetch_all index --end 20261015

# 财务、事件、分红与申万历史行业成员原始数据（可断点续跑）
.venv/bin/python -m scripts.sources.fetch_all core --end 20261015
.venv/bin/python -m scripts.sources.fetch_all gbbq

# 清洗，按年处理
.venv/bin/python -m scripts.clean.pipeline

# 建库，每次删库重建
.venv/bin/python -m scripts.build_db.build

# 因子：构建 + 自检 + 生成清单文档
.venv/bin/python -m scripts.build_factors.build
.venv/bin/python -m scripts.build_factors.validate
.venv/bin/python -m scripts.build_factors.catalog --write-docs

# 全量测试
.venv/bin/python -m pytest -q
```

取数有硬约束：**全局并发上限 2 个请求**，只能「单进程 + 2 线程」，同一时刻不要跑第二个取数进程。
四个接口全量约 3~4 小时。整条链路幂等可续跑，中断后重跑同样命令即可，已完成的会跳过。

## 相关文档

| 文档 | 内容 |
|---|---|
| `demo.py` | 数据读写库用法速查：25 块覆盖全部读取接口、元数据与组合示例，可逐块粘进 ipynb |
| `docs/core/data/sources.md` | 取数模块 |
| `docs/core/data/cleaning.md` | 清洗模块 |
| `docs/core/data/database.md` | 数据库结构与访问方式 |
| `docs/core/data/label.md` | 未来收益标签的口径、缺失规则与校验方式 |
| `docs/core/factors/README.md` | 因子库：构建、读取、扩展因子的操作步骤 |
| `docs/core/factors/cookbook.md` | 入门教程：手算因子并与库值对照 |
| `docs/core/factors/formulas.md` | 已实现因子的公式、白话步骤与真实 SQL（自动生成） |
| `docs/core/factors/catalog.md` | 全部因子清单（自动生成）：状态、方向、公式、说明 |
| `docs/core/factors/conventions.md` | 因子口径、算子语义、已知限制与多层写法 |
| `docs/core/factors/missing_data.md` | 补哪些数据能做基本面与资金类因子 |
| `docs/core/ops/v5_runbook.md` | 串联脚本与运行手册 |
| `docs/trading/backtest/README.md` | VectorBT 回测模块：执行口径、股票池、资源上限与实测规模 |
| `docs/trading/backtest/api.md` | 回测接口与 SSE 流事件 |
| `docs/trading/backtest/todo.md` | 回测模块的待办清单、已知限制与建议顺序 |
| `AiRecord/version.md` | 需求与迭代记录 |

## 因子质量监测（V11）

```bash
bash tools/monitor/serve.sh
```

浏览器打开 http://127.0.0.1:8800，查看因子列表、单因子详情或最多4因子对比。
纯HTML/JS+本地ECharts，指标由calculate计算，行情/因子数据只读，结果不落库。
完整分析默认最近3年；全历史请选择轻量IC。完整分析保留2400交易日/800万行保护。
运行与口径见 [监测手册](docs/apps/monitor/README.md)，实测资源见
[性能记录](docs/apps/monitor/performance.md)，计算函数见 [calculate](research/evaluation/calculate/README.md)。
