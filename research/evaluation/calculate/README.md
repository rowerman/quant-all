# 因子质量监测项目：计算包 `calculate`

## 阅读顺序与运行方式

本文件是计算包的使用文档。先读“快速调用”，再按需要查指标、上下文与底层函数。
[面板细节](../../../docs/research/evaluation/panel.md) 和
[风格口径](../../../docs/research/evaluation/styles.md) 放在根 `docs/research/evaluation/`。

在项目根目录 `/home/hanwen/quant` 使用 `.venv/bin/python`，或在 notebook 的项目环境里
`import calculate`。不要在 research/evaluation/calculate/ 内直接执行 `python evaluate.py`，那会改变包导入路径。
使用 Python 3.12；依赖 numpy、pandas、scipy、duckdb（沿用项目 .venv）。
后端、前端与因子相关性阶段已接入；相关性 Spearman 默认经 `kernels` 的 `auto` 后端选择，
后端原生计算资源保护见 [原生计算资源保护](../../../docs/research/evaluation/native.md)。

## 快速调用

```python
from calculate import EvalParams, evaluate, build_panel, load_daily_ic, get_metric
from calculate.metrics import resolve_options
from calculate.stats import series_stats
from calculate.jsonutil import json_safe

params = EvalParams(factor="MOM_20", start="2024-01-02", end="2024-12-31",
                    universe="all_a", horizon=5, price="close", quantiles=10,
                    winsorize=True)

# 1. 流式评估：每一项是 (stage, payload)，payload 已转成标准 JSON 类型
for stage, payload in evaluate("MOM_20", params, stages=["overview", "decay"]):
    print(stage, payload.keys())

# 2. 单独计算一个指标：先建共享上下文，再调用注册表里的计算函数
ctx = build_panel(params)
metric = get_metric("bin_returns")
result = metric.compute(ctx, resolve_options(metric))
print(json_safe(result))  # 直接 compute 可能含 numpy 数组/NaN，展示前需要转换

# 3. 同一上下文复用中间结果，减少重复计算
ic = ctx.daily_ic()
bins = ctx.bins()
styles = ctx.styles()
exposures = ctx.exposures()  # 返回 ExposureResult；该方法带请求内缓存

# 4. 全历史 IC：不构建 1795 万行完整面板
history = load_daily_ic(EvalParams(factor="MOM_20", start="2000-01-04", end="2026-09-28"))
print(series_stats(history["rank_ic"].to_numpy()))
print(history.attrs["elapsed_ms"])

# 5. 从统一入口打开行业层；两个指标分别指定，避免意外改变另一项
options = {"exposure": {"industry": "stock_basic"},
           "incremental_ic": {"industry": "stock_basic"}}
for stage, payload in evaluate("MOM_20", params,
                               stages=["exposure", "incremental_ic"], metric_options=options):
    print(stage, payload.keys())
```

`stages=None` 跑全部已实现阶段；空序列只产出 meta/done。未知或未实现阶段报 EVAL-101。
`metric_options` 的指标名、参数名与枚举取值在读库前校验；未传的值取指标默认值。
`meta.stages` 是本次实际执行阶段，`meta.metric_options` 回显已校验参数。

默认 `industry="off"`。显式开启 `stock_basic` 使用静态行业，含历史前视风险，详见风格文档。
所有评估参数日期必须是合法 `YYYY-MM-DD`；`horizon`、`quantiles` 必须为整数，
`winsorize` 必须为布尔值。

## 参数、上下文与结果类型

| 类型 | 构造与用途 | 主要输出/方法 |
|---|---|---|
| `EvalParams(factor, start, end, universe="all_a", horizon=5, price="close", quantiles=10, winsorize=True)` | 不可变评估输入；周期 1/3/5/10/15/20，价格 close/open，组数 5/10/20 | `describe()`、标签列映射方法；完整字段见下面函数参考 |
| `EvalContext` | 通常由 `build_panel` 返回；手工构造需要 params/spec/universe/panel/summary/panel_elapsed_ms，db_file 可选 | `panel` 是 DataFrame；`daily_ic/bins/bin_matrices/styles/exposures` 复用请求内缓存 |
| `FactorView` | 描述正式或候选因子的展示元数据、目标 Parquet glob 和正式因子对照目录 | 可直接传给 `build_panel(..., factor=view)`、`load_daily_ic(..., factor=view)` |
| `ExposureResult` | `run_exposure` 或 `ctx.exposures()` 返回 | dates、coefficients、t_stats、intercept、r2、rows、skipped_days、residual、industry |
| `Metric(key, label, stage, compute, params=(), requires_quantiles=False)` | 指标声明；compute(context, options) → dict | `describe()`；新增指标调用 `register_metric` |
| `MetricParam(key, label, kind, default=None, choices=())` | 指标参数；kind 支持 int/bool/enum | 用 `resolve_options` 合并默认值与校验 |
| `UniverseSpec(key, label, detail, index_code=None)` | 股票池声明 | `is_index`；实际可选项来自 `list_universes()` |
| `StyleSpec(key, label, definition, window, min_obs, note="")` | 风格定义与展示说明 | `STYLE_FACTORS` 当前包含 7 个近似风格 |
| `IndustrySource` / `StockBasicIndustry()` | 行业读取协议与静态实现 | `load(conn)` 返回 stock_code/industry；自定义源登记到 `INDUSTRY_SOURCES` |

`EvalContext` 的 panel 至少满足所调用函数的列要求；指标不能向 panel 写新状态。
用手工 DataFrame 时，日期用 pandas datetime，键 stock_code/trade_date 唯一，因子与标签为数值。
低层函数无上下文缓存；`run_exposure(ctx)` 每次重新计算，`ctx.exposures()` 才带缓存。

## 指标清单与直接调用

所有计算函数使用 `(context, options)`，返回 dict；不带参数的指标传 `{}`。
行业参数在 exposure 与 incremental_ic 上独立生效。统一入口会自动校验；直接调用
时推荐 `get_metric(key).compute(ctx, resolve_options(get_metric(key), options))`。

| 指标 key | stage | 计算内容 | 直接导入函数 |
|---|---|---|---|
| coverage | overview | 全样本/池内/配对覆盖率、分布、每日样本数 | `calculate.metrics.coverage.compute` |
| ic_stats | overview | IC/RankIC 均值、ICIR、t、胜率、方向 | `calculate.metrics.ic.compute_stats` |
| ic_series | ic_series | 每日 IC/RankIC/n、累计 IC/RankIC 与 20 日滚动均值 | `calculate.metrics.ic.compute_series` |
| ic_dist | ic_dist | IC 直方图、分位数、偏度峰度 | `calculate.metrics.ic.compute_dist` |
| bin_returns | bins | 分位期收益、标准误、t、单调性；两套权重 | `calculate.metrics.bins.compute_bin_returns` |
| bin_curves | bin_curves | 同价格通道 1 日收益累乘；两套权重 | `calculate.metrics.bins.compute_bin_curves` |
| spread | spread | 最高组减最低组、净值、夏普、最大回撤 | `calculate.metrics.bins.compute_spread` |
| bin_dist | bin_dist | 每日分位均值箱线数据、个股收益分位点 | `calculate.metrics.bins.compute_bin_dist` |
| turnover | turnover | 连续池内分组变化率、自相关 lag 1/5/20 | `calculate.metrics.turnover.compute_turnover` |
| decay | decay | 五档独立配对 RankIC 与多头超额、多空价差 | `calculate.metrics.decay.compute_decay` |
| exposure | exposure | 7 个风格暴露、t、显著占比、R² | `calculate.metrics.exposure.compute_exposure` |
| incremental_ic | incremental_ic | 同样本 raw/residual RankIC 对照 | `calculate.metrics.exposure.compute_incremental_ic` |

## 底层计算示例

```python
from calculate.daily import daily_ic, daily_corr, daily_rank_ic_multi
from calculate.binning import assign_bins, bin_daily_returns, bin_matrix, rank_matrix, factor_autocorrelation
from calculate.stats import series_stats, compound, max_drawdown, annualised_sharpe
from calculate.normalize import mad_winsorize, zscore

# 自己构造的 panel 至少有 trade_date/factor/label/in_universe
per_day = daily_ic(ctx.panel, winsorize=False)
rank_only = daily_corr(ctx.panel.loc[ctx.panel.in_universe], "factor", "label", rank=True)
multi = daily_rank_ic_multi(ctx.panel, ["label", "label_h20"])

# 分箱表需要 stock_code/trade_date/factor/in_universe/total_mv 和指定标签列
binned = assign_bins(ctx.panel, 10, label_columns=ctx.label_columns)
equal = bin_daily_returns(binned, 10)
weighted = bin_daily_returns(binned, 10, weight_column="total_mv")
stability = factor_autocorrelation(rank_matrix(binned), (1, 5, 20))

# 统计工具；series_stats 的 t/IR 在零波动时为 None
values = [0.01, -0.02, 0.03]
print(series_stats(values), annualised_sharpe(values))
equity = compound(values)
print(max_drawdown(equity))  # 默认初始资本 1，包含首日亏损
print(zscore(mad_winsorize([1., 2., 3., 100.])))
```

`compound` 缺失收益按 0 处理，仅用于图形净值；这不代表末端标签被填充进 IC 或分位统计。
直接构造宽表时应补齐交易日索引，否则 shift 的 lag 是行数；`ctx.bin_matrices()` 会保留面板中的空池日期。

## 负责什么

把「因子值 + 未来收益标签 + 股票池标记 + 后续分析原料」对齐成一张评估面板，
在面板上算指标，并按固定顺序逐段产出结果，接口层直接转成流式事件。

本项目与数据项目（取数 / 清洗 / 建库 / 因子库）分开，目录约定：

```
research/evaluation/calculate/   计算包：只算，自带配置、错误码与日志     ← 本目录
apps/monitor/backend/    后端：接口、任务调度、流式输出
apps/monitor/frontend/   前端：纯 HTML/JS + ECharts
```

计算包对数据侧只有三个依赖：`data/databases/quant.duckdb`（行情、标签、指数与成分）、
`data/databases/factors.duckdb`（数据侧因子目录，当前评估不读取）、`data/factors/`（因子值）；另加数据项目的
`factors` 注册表用于取因子的定义与方向（`panel.py` 里的 `import factors`，
尚未彻底解耦：目前因子定义实际来自 `factors` 注册表，目录库不是评估必需读取项）。

三条边界：

1. **只算**：不展示、不落库、不产生新因子；
2. **只读**：数据一律经 DuckDB 只读连接（`data/databases/quant.duckdb` 的 `md` / `ref` 表 +
   `data/factors/` 的 parquet），不写数据产物；日志写集中日志目录，轻量 IC 的溢出临时文件自动清理；
3. **指标不缓存到磁盘**：结果只在单次请求的进程内存在，下一次请求重新算。

## 模块与依赖

| 模块 | 负责 | 被谁调用 | 调用了谁 | 改动时注意 |
|---|---|---|---|---|
| `config.py` | 数据路径与常量（库、因子产物、本包日志目录） | 全部模块 | — | 只认目录结构，不引用数据项目的 `scripts`；换数据位置改这里 |
| `errors.py` | `CalculateError` 基类与 `EVAL-1xx/2xx/3xx` 错误码 | 全部模块 | — | 后端 `MONITOR-1xx` 在 backend 里定义，别混进来 |
| `logging_conf.py` | 绑定本包日志目录，处理器复用 `common.logging_conf` | 全部模块 | `config`、`common` | 日志写 `logs/research/evaluation/` |
| `params.py` | `EvalParams` 定义与校验 | `panel`、指标、接口层 | `errors` | 加字段要同步 `describe()`，meta 事件靠它回显参数 |
| `universe.py` | 股票池定义与指数成分 CTE | `panel`、接口层 | `errors` | 全 A 池口径必须与数据项目的 `core/data/scripts/build_factors/validate.py` 的 `POOL_FILTER` 一字不差；指数池用**上一个已结束月份**的月末快照，别改成当月（会前视） |
| `panel.py` | 评估面板、连接、默认参数、`EvalContext` | `evaluate`、各指标、后续拆分阶段 | `factors` 注册表、`universe`、DuckDB、`config` | 面板列是公共契约，后续阶段只加不删；行数/交易日上限也在这里；**不要给面板加 ORDER BY**（几百万行排序要多花 2 秒以上，下游都按交易日分组） |
| `correlation.py` | 按月取数、成对Spearman、Top-N与矩阵 | metrics/correlation | DuckDB、factors、panel上下文、kernels | 完整矩阵与候选目标行经统一后端路由；成对缺失后重排名，不用标签；目录跟随context.factor_dir |
| `metrics/correlation.py` | 相关性指标声明 | 注册表/evaluate | correlation、metrics/base | Top-N只影响明细数量，不改变全矩阵 |
| `history.py` | 全历史 IC SQL 聚合与轻量读取 | 调用方 | panel/universe/DuckDB | 不构建完整面板；逐日平均并列排名，工作内存 1GB，临时文件自动清理 |
| `daily.py` | 逐日截面相关（IC / RankIC、通用 `daily_corr`、多档 `daily_rank_ic_multi` 与截面排名 `rank_pct_by_day`） | `metrics/ic.py`、`metrics/decay.py`，相关性模块独立读取 | `normalize` | RankIC 一律对**原始值**排名；去极值只作用于 Pearson IC；池内过滤默认开（`pool_only=True`）；`rank_pct_by_day` 必须与 pandas `groupby().rank(pct=True)` 同语义（并列取平均名次、按交易日内的位置算名次） |
| `normalize.py` | MAD 去极值、分位、zscore | `daily`、后续风格因子 | — | 都按交易日分组调用，NaN 原样保留 |
| `stages.py` | `STAGE_ORDER`（stage 顺序契约） | `evaluate`、指标注册表 | — | 单独成模块是为了让注册表能按它排序又不与评估入口循环依赖 |
| `stats.py` | 序列四件套、净值累乘、最大回撤、年化夏普 | IC 家族与 binplot 家族 | — | 加统计口径改这里，别在各指标里各写一遍 |
| `styles.py` | 7 个 Barra 近似风格因子的取数 SQL、`IndustrySource` 协议与静态行业源 | `exposure`、`EvalContext.styles()` | DuckDB、`config`、`errors` | 窗口按有效交易日推进（与数据项目因子构建一致）；只产出**原始值**，标准化在 `exposure` 里做（同一回归样本）；BETA_60 基准写死沪深300，与已实现的 `BETA_60` 因子一致 |
| `exposure.py` | 风格拆分核心：逐日截面 OLS、暴露系数与 t 值、残差 | `metrics/exposure.py` | `styles`、`normalize`、`daily` | 回归样本是「池内 + 因子非空」，**不看标签**；设计矩阵第 0 列是截距，风格系数从第 1 列起；风格缺失标准化后置 0 |
| `binning.py` | 分位分箱、分位收益、换手、因子自相关、宽表透视 | `metrics/bins.py`、`metrics/turnover.py`、`metrics/decay.py` | `daily`、`normalize` | 分箱只看因子值；宽表用 factorize + numpy 填充（pandas `pivot` 要慢一倍以上）；多档收益走 `bin_returns_multi`（bincount，比逐组 pandas 分组快 10 倍以上） |
| `metrics/base.py` | 指标注册表 | 各指标、接口层、前端 | `errors` | 注册顺序 = 前端卡片顺序；重名报 `EVAL-201` |
| `metrics/coverage.py` | 覆盖率与取值分布 | 经注册表调用 | `metrics/base` | 缺失（NaN）与非有限值（inf）分开统计，别混成一个数 |
| `metrics/ic.py` | IC / RankIC 汇总、时序、分布 | 经注册表调用 | `daily`、`metrics/base` | 有效交易日少于 20 天只给「样本不足」标记，不给数值 |
| `metrics/bins.py` | 分位平均收益与单调性、分位净值、多空价差、分位收益分布 | 经注册表调用 | `binning`、`stats` | 每张图都出两套权重；净值固定用 1 日收益，别改成期收益累乘 |
| `metrics/turnover.py` | 分位换手、全池换手、因子自相关（lag 1/5/20） | 经注册表调用 | `binning`、`stats` | 宽表从 `EvalContext.bin_matrices()` 取，别自己再透视一次 |
| `metrics/decay.py` | 多周期衰减：1/3/5/10/20 日标签的 RankIC 与多头超额、多空价差（各两套权重） | 经注册表调用 | `daily`、`binning`、`stats` | 各档独立取样本，末端缺标签只从该档剔除、不填充；主档取参数里的周期，参数用 15 档时回退到最接近的一档 |
| `metrics/exposure.py` | 风格暴露（系数/t 值/逐日序列）与剔除风格后的增量 IC | 经注册表调用 | `exposure`、`stats`、`daily` | 两个 stage 共用同一份回归结果（`EvalContext.exposures()` 带缓存）；行业开关是 `enum` 参数，默认 `off` |
| `evaluate.py` | stage 生成器与 meta 组装 | backend服务 | `panel`、`metrics` | `STAGE_ORDER` 是前后端的公共契约，后续阶段只往空位里填 |

正式入口 `evaluate()` 先解析注册因子并构建面板，再调用 `evaluate_context()`。需要评估
候选临时因子时，调用方构造 `FactorView(formal=False, parquet_glob=...)`，通过
`build_panel(..., factor=view)` 构建同一面板后交给 `evaluate_context()`；指标注册、
stage 顺序和输出结构完全一致。

## 怎么用

```python
from calculate import EvalParams, evaluate

# 默认口径：最近 3 年 + 全A可交易池 + 未来 5 日收盘收益
for stage, payload in evaluate("MOM_20"):
    print(stage, payload)

# 自定义：指数成分池、20 日标签、开盘通道
params = EvalParams(factor="MOM_20", start="2024-01-02", end="2024-12-31",
                    universe="index:000300.SH", horizon=20, price="open")
for stage, payload in evaluate("MOM_20", params, stages=["overview", "ic_series"]):
    ...
```

`stage` 与 payload 结构：

| stage | payload |
|---|---|
| `meta` | 因子定义、参数回显、股票池与面板概览、指标清单（前端据此先摆骨架） |
| `overview` | `metrics.coverage`（覆盖率与分布）、`metrics.ic_stats`（IC/RankIC 汇总） |
| `ic_series` | 逐日 IC / RankIC / 样本数、累计 IC / RankIC 与 20 日滚动均值 |
| `ic_dist` | 40 桶直方图、分位数、偏度峰度 |
| `bins` | `metrics.bin_returns`：分位平均期收益、t 值、标准误、胜率、单调性（两种权重） |
| `bin_curves` | `metrics.bin_curves`：分位累计净值曲线（1 日收益累乘，两种权重） |
| `spread` | `metrics.spread`：多空价差逐日序列、净值、t 值、年化夏普、最大回撤 |
| `bin_dist` | `metrics.bin_dist`：每日分位均值的箱线数据 + 分位内个股收益分位点 |
| `turnover` | `metrics.turnover`：分位换手、全池逐日换手、因子自相关（lag 1/5/20） |
| `decay` | `metrics.decay`：1/3/5/10/20 日标签的 RankIC 均值/ICIR/t/胜率/天数（衰减曲线），以及「最高分位 − 池内均值」多头超额与「最高 − 最低」多空价差（各两套权重） |
| `exposure` | `metrics.exposure`：7 个风格上的暴露系数（时序均值 + Fama-MacBeth t 值 + 显著占比）、逐日系数序列、平均 R²、行业层开关与说明 |
| `incremental_ic` | `metrics.incremental_ic`：剔除风格后残差的 RankIC（`residual`）与原始 RankIC（`raw`）对照、`delta`、`survivor_share`、逐日两条序列 |
| `done` | 各 stage 耗时、总耗时、面板概览 |

`correlation` 阶段已实现：与全部已注册且落盘因子的逐日成对 Spearman 均值、Top-N和全矩阵。
前后端已接入，启动见 `apps/monitor/backend/README.md`。

## 怎么加一个指标

1. 在 `research/evaluation/calculate/metrics/` 下新增一个模块，写 `compute(context, options) -> dict`；
2. 调 `register_metric(Metric(key=..., label=..., stage=..., compute=compute))` 注册；
3. 重启调用进程，包会自动发现并导入该模块，不用编辑导入清单。
   已运行的进程可调用 `calculate.metrics.discover_metrics()` 发现新增文件（不会重载已导入模块）。

所用 `stage` 已经在 `STAGE_ORDER` 里的话，接口层与前端**自动生效**；
若是全新 stage，先在 `STAGE_ORDER` 里补一个位置；非法 stage 注册时直接报错。
同 stage 内保持注册顺序，跨 stage 按 `STAGE_ORDER` 排序。
前后端尚未实现，因此“自动生效”目前指计算阶段与 meta 中的指标清单。

## 口径速查

- 股票池：全 A 可交易 = `can_buy_close` + 非 ST + 上市满 60 个自然日；
  指数池 = 上一个已结束月份的月末权重快照 ∩ 同一套可交易性过滤。
- IC：逐日截面 Pearson，默认对因子与标签各做 3 倍 MAD 去极值（`winsorize=False` 可关）。
- RankIC：先取因子与标签的有效配对，再逐日按平均并列名次算 Spearman。
  与 `validate.py` 已统一，按原定绝对误差 1e-6 验收，由 `tests/test_eval_consistency.py` 守住。
  多周期与自相关也必须先取各自有效配对，不能复用删除样本前的完整截面排名。
- ICIR = 均值 / 样本标准差；t 值 = ICIR × √有效交易日；胜率 = IC 同号占比。
- 单日有效配对少于 5 只的截面不参与统计；有效交易日少于 20 天只报「样本不足」。
- **分箱**：只看因子值（不依赖标签），每个交易日把池内非空因子值按平均名次均分成 Q 组，
  组号 1 最小、Q 最大；并列值用平均名次，落在同一组（可复现，不依赖面板行序）。
- **两种权重**：等权 = 组内对标签取均值；市值加权 = 按 `total_mv` 加权，
  缺市值或市值不大于 0 的样本不进加权口径。
- **净值与多空曲线固定用 1 日前瞻收益**累乘。期收益（如 5 日）逐日重叠，
  直接累乘会把同一个区间重复计入，所以曲线与期收益分开报；页面标注这一点。
- **换手** = 今天在该组、上一交易日在池内但不在同组的股票占「今天在组里且昨天也有值」的比例；
  **因子自相关** = 当日与 lag 个交易日前在同一批股票上的截面秩相关。
- **多周期衰减**：同一批交易日上按 1/3/5/10/20 档标签各算一次 RankIC（口径与 IC 家族一致），
  每档**只用自己的非空标签样本**，末端不足 N 个交易日与停牌的行只从该档剔除、不做任何填充；
  多头超额 = 最高分位 − 池内均值，多空价差 = 最高分位 − 最低分位，各出等权与分位内市值加权。
  期收益逐日重叠（1 日与 20 日的持有区间互相套嵌），跨档的 t 值不可直接比较。
- **风格拆分**：7 个 Barra 近似风格因子的定义、窗口与最小观测见
  [风格口径](../../../docs/research/evaluation/styles.md)；因子与风格都在回归样本上逐日做 3 倍 MAD 去极值 + z-score，
  回归系数统一为「因子每 1 个标准差的风格暴露」，逐日截面回归（Fama-MacBeth），
  报系数时序均值与其 t 值、|日度 t| > 1.96 的占比；单日样本 < 30 只或因子常数时跳过，
  有效回归日 < 20 天只报「样本不足」。
- **增量 IC** = 回归残差与标签的逐日 RankIC；原始 RankIC 与它同样本同口径，
  `survivor_share` = 残差 IC 均值 / 对照 raw IC 均值；不是风格解释率，可超出 [0,1]，
  raw 接近 0 时比例不稳定。raw 在残差的同一有效样本与日期上重算，不保证等于 overview。
  `delta` 为 raw 减 residual 的对照差值，不是严格可加的风格贡献。
  完全被风格解释时残差置零，RankIC 无定义；常数因子回归日跳过。
- **行业层**默认关闭（库里只有静态行业分类，非时点、退市股缺失），
  指标参数 `industry` 可选 `off` 或已注册的行业源，开启后设计矩阵加行业哑变量（丢第一类）。
- 完整评估上限：区间 ≤ 2400 个交易日；面板 ≤ 800 万行（parquet 文件级元数据的保守上界）。
  全历史 IC 使用 `load_daily_ic`，不受完整面板行数和天数上限限制。
- t 值与误差棒使用普通样本标准误，假设日度观测独立；未作 HAC/Newey-West 修正。
  多日标签重叠、风格系数自相关时，显著性只能作参考。
- 分位换手只统计连续池内股票的分组变化，不包含新进/退出股票，也不等于实际交易换手。

## 历史耗时参考（修复前，2026-10-10～11）

| 区间 | 面板行数 | 建面板 | overview | 合计 |
|---|---|---|---|---|
| 默认最近 3 年 | 3,925,208 | 4.4 s | 4.0 s | 14.4 s（V11-2 口径，9 个指标） |
| 2021-01-01 起（5.7 年） | 7,113,634 | 13.0 s | 10.3 s | 36.1 s（V11-2 口径，9 个指标） |
| 默认最近 3 年（含 V11-3 衰减） | 3,925,208 | 5.7 s | 4.1 s | 26.1 s（10 个指标） |
| 默认最近 3 年（含 V11-3 + V11-4） | 3,925,208 | 9.7 s | 5.4 s | 53.9 s（12 个指标） |

各阶段的耗时（默认区间）：`overview` 4.0 s、`bins` 2.0 s、`bin_curves` 0.6 s、
`spread` 0.6 s、`bin_dist` 0.9 s、`turnover` 2.0 s；5.7 年区间上分别是
10.3 / 4.2 / 1.4 / 1.3 / 1.8 / 4.0 s。
加入衰减后默认区间的 `decay` 单 stage 约 10.0 s，主要花在 5 档标签的截面排名上
（`rank_pct_by_day` 一次排序约 1 秒/档）；两套权重的各档分位收益已经压到 0.4 秒。
加入风格拆分后 `exposure` 单 stage 约 18.0 s（风格取数 SQL 6.4 s、与面板合并 2.1 s、
8 列原料与因子截面标准化 3.6 s、逐日回归 3 s），`incremental_ic` 2.5 s。
流式推送下用户先看到 IC，再看到各项分层结果；进一步的压测与优化见 V11-9。

日志按模块分目录写在 `logs/research/evaluation/calculate_panel/`、`logs/research/evaluation/calculate_evaluate/`
（模块名里的点换成下划线，与数据项目的日志命名规则一致）。

目录约定与后续安排见 `AiRecord/plans/20261011-1.md`（项目拆分）；
前后端代码落在 `apps/monitor/backend/` 与 `apps/monitor/frontend/`，
本包只提供纯函数式的计算与流式 stage。

## 完整可调用函数参考

下面按模块列出所有不以下划线开头的模块函数，以及公开类的方法/属性。
计算入口通常从包根导入；未导出的底层函数按模块路径导入。
以下划线开头的内部辅助函数不属于公开调用契约。`clear_metrics` 仅供隔离测试。

### calculate.binning

#### `assign_bins`

```python
calculate.binning.assign_bins(panel: pd.DataFrame, quantiles: int, *, label_columns=('label', 'label_1d')) -> pd.DataFrame
```

给池内非空因子值分配分位组号。

参数:
    panel: 评估面板。
    quantiles: 分组数。
    label_columns: 一并带进分位表的标签列，默认所选周期与 1 日档；
        多周期衰减要把几档标签都带出来（列名见 ``EvalParams.horizon_label_column``）。

返回:
    列含 ``trade_date``、``stock_code``、``factor``、``label_columns``、
    ``total_mv``、``bin`` 的数据表；池外或因子为空的行不出现。

#### `bin_daily_returns`

```python
calculate.binning.bin_daily_returns(frame: pd.DataFrame, quantiles: int, *, label_column: str='label', weight_column: str | None=None) -> pd.DataFrame
```

算每个交易日的分位组合收益。

参数:
    frame: ``assign_bins`` 的输出。
    quantiles: 分组数。
    label_column: 用哪一列算收益，默认所选周期的标签。
    weight_column: 加权列；为空表示等权（组内取均值）。

返回:
    以交易日为索引、列为 1~Q 的收益表，当天没有有效样本的组为空值。

#### `bin_returns_multi`

```python
calculate.binning.bin_returns_multi(frame: pd.DataFrame, quantiles: int, columns, weight_column: str | None=None) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]
```

一次算多档标签的分位组合收益与池内均值。

多周期衰减要按 5 档标签 × 2 套权重算 10 组收益，逐组走 pandas 分组太慢
（实测默认区间约 7 秒），这里改成「把 (交易日, 分位) 压成一个格子号，
用 bincount 一次性求和」，实测降到 1 秒以内。口径与 ``bin_daily_returns``
完全一致：等权取组内均值，市值加权只在 ``total_mv`` 非空且大于 0 的样本上算。

参数:
    frame: ``assign_bins`` 的输出。
    quantiles: 分组数。
    columns: 标签列名序列。
    weight_column: 加权列；为空表示等权。

返回:
    ``(每档的逐日分位收益表, 每档的池内逐日均值)``；
    分位收益表的行是交易日、列是 1~Q，当天没有有效样本的组为空值。

#### `wide_matrix`

```python
calculate.binning.wide_matrix(frame: pd.DataFrame, values) -> pd.DataFrame
```

把长表摊成「交易日 × 股票」宽表，缺值表示当天该股票不在池内。

用 factorize 加 numpy 直接填充，而不是 pandas 的 pivot：几百万行的透视
实测要 1.5 秒一次，这样写只要零点几秒。

参数:
    frame: ``assign_bins`` 的输出。
    values: 与 frame 等长的取值序列。

#### `bin_matrix`

```python
calculate.binning.bin_matrix(frame: pd.DataFrame) -> pd.DataFrame
```

把分位表摊成「交易日 × 股票」的组号宽表。

参数:
    frame: ``assign_bins`` 的输出。

#### `rank_matrix`

```python
calculate.binning.rank_matrix(frame: pd.DataFrame) -> pd.DataFrame
```

把分位表摊成「交易日 × 股票」的截面秩宽表。

参数:
    frame: ``assign_bins`` 的输出。

#### `bin_turnover`

```python
calculate.binning.bin_turnover(matrix: pd.DataFrame, quantiles: int) -> tuple[pd.DataFrame, pd.Series]
```

算每个分位、每个交易日的换手。

参数:
    matrix: ``bin_matrix`` 产出的组号宽表。
    quantiles: 分组数。

返回:
    ``(每个分位的逐日换手表, 全池逐日换手序列)``；
    分母是「今天在池内且上一交易日也有值」的股票数，首日为空。

#### `row_correlation`

```python
calculate.binning.row_correlation(current: pd.DataFrame, previous: pd.DataFrame, *, rank: bool=False) -> pd.Series
```

逐行（逐交易日）算两个宽表的 Pearson 相关，缺失按配对剔除。

参数:
    current: 当期取值（交易日 × 股票）。
    previous: 上一期取值，与 current 的行位置和股票列对应。
    rank: 真则对逐行有效配对重新排名，返回 Spearman。

#### `factor_autocorrelation`

```python
calculate.binning.factor_autocorrelation(ranks: pd.DataFrame, lags: tuple[int, ...]=AUTOCORR_LAGS) -> pd.DataFrame
```

算因子自相关：当日与 lag 个交易日前在同一批股票上的截面秩相关。

参数:
    ranks: ``rank_matrix`` 产出的秩宽表。
    lags: 滞后阶数（交易日）。

返回:
    以交易日为索引、每阶一列的相关系数表。

### calculate.daily

#### `rank_pct_by_day`

```python
calculate.daily.rank_pct_by_day(codes: np.ndarray, values: np.ndarray) -> np.ndarray
```

按交易日分组算截面分位（0~1），与 ``groupby(...).rank(pct=True)`` 同语义。

参数:
    codes: 每个样本所属交易日的整数编码，取值为 ``0..n-1``。
    values: 一维取值数组，空值（NaN）原样保留为空。

返回:
    分位数组；并列值取平均名次，分母是该交易日该列的非空样本数。

说明：pandas 的 ``groupby().rank(pct=True)`` 在多标签上要 8 秒以上，
这里改成「整体按 (交易日, 取值) 排一次序，再对并列块取平均名次」，
实测快 3 倍以上，取值与 pandas 完全一致。

#### `daily_corr`

```python
calculate.daily.daily_corr(frame: pd.DataFrame, x: str, y: str, *, min_n: int=MIN_CROSS_SECTION, rank: bool=False) -> pd.DataFrame
```

按交易日算两列的截面相关系数。

参数:
    frame: 至少含 ``trade_date``、``x``、``y`` 三列的数据表。
    x: 左变量列名。
    y: 右变量列名。
    min_n: 单日最少有效配对数。
    rank: 真则先做截面分位再算相关，等价于 Spearman。

返回:
    以交易日为索引、含 ``corr`` 与 ``n`` 两列的表；``n`` 为当日有效配对数。

#### `daily_ic`

```python
calculate.daily.daily_ic(panel: pd.DataFrame, *, winsorize: bool=True, min_n: int=MIN_CROSS_SECTION, pool_only: bool=True) -> pd.DataFrame
```

算因子与标签的逐日 IC 与 RankIC。

参数:
    panel: 评估面板，需要 ``trade_date``、``factor``、``label`` 三列。
    winsorize: 是否对因子与标签做截面 3 倍 MAD 去极值（只影响 Pearson IC）。
    min_n: 单日最少有效配对数。
    pool_only: 是否只用池内样本；默认真，与股票池口径保持一致。

返回:
    以交易日为索引、含 ``ic``、``rank_ic``、``n`` 三列的表。

#### `daily_rank_ic_multi`

```python
calculate.daily.daily_rank_ic_multi(panel: pd.DataFrame, columns, *, min_n: int=MIN_CROSS_SECTION, pool_only: bool=True) -> dict[str, pd.DataFrame]
```

对多档标签列各算一条逐日 RankIC 序列（多周期衰减用）。

参数:
    panel: 评估面板，需要 ``trade_date``、``factor`` 与待算的标签列。
    columns: 标签列名序列，例如 ``("label", "label_h20")``。
    min_n: 单日最少有效配对数。
    pool_only: 是否只用池内样本；默认真，与股票池口径保持一致。

返回:
    ``{列名: DataFrame(rank_ic, n)}``，行是交易日，索引与 ``daily_ic`` 一致。

口径与 ``daily_ic`` 的 RankIC 相同（截面平均名次的 Spearman）：
各档的因子秩与标签秩都在同一批有效配对上计算。
各档必须先取因子与标签的有限配对，再在配对样本内排名。
删除截面的一部分股票会改变名次间距，不能复用完整截面的因子秩；
末端不足 N 个交易日的行只从该档剔除，不做任何填充。

### calculate.evaluate

#### `planned_stages`

```python
calculate.evaluate.planned_stages() -> tuple[str, ...]
```

列出当前已注册指标会真正产出的 stage（不含 meta 与 done）。

#### `meta_payload`

```python
calculate.evaluate.meta_payload(context, stages=None, metric_options=None) -> dict
```

组装 meta 事件：参数回显、因子定义、面板概览与指标清单。

参数:
    context: 评估上下文。
    stages: 本次实际执行的阶段；None 表示全部已实现阶段。
    metric_options: 本次已校验的指标参数，用于回显。

#### `evaluate`

```python
calculate.evaluate.evaluate(factor: str, params: EvalParams | None=None, *, db_file: Path | str | None=None, factor_dir: Path | str | None=None, stages=None, metric_options: Mapping | None=None) -> Iterator[tuple[str, dict]]
```

跑一次因子评估，按 stage 逐段产出结果。

参数:
    factor: 因子名。
    params: 评估参数；留空按「最近 3 年 + 默认口径」构造。
    db_file: 日线主库路径，默认取项目配置。
    factor_dir: 因子数据根目录，默认取项目配置。
    stages: 只跑这些已实现 stage；留空表示全部。
    metric_options: 指标 key 到参数映射，例如
        {"exposure": {"industry": "stock_basic"}}。不改变其他指标的参数。

产出:
    ``(stage, payload)`` 序列，payload 保证是 JSON 可序列化的 dict。

### calculate.exposure

#### `run_exposure`

```python
calculate.exposure.run_exposure(context, industry_key: str=INDUSTRY_OFF) -> ExposureResult
```

跑一次风格拆分：逐日截面回归 + 残差。

参数:
    context: 评估上下文。
    industry_key: 行业数据源标识，默认 ``off``。

#### `residual_rank_ic`

```python
calculate.exposure.residual_rank_ic(result: ExposureResult, *, min_n: int=MIN_CROSS_SECTION) -> pd.DataFrame
```

残差与标签的逐日 RankIC。

参数:
    result: 风格拆分结果。
    min_n: 单日最少有效配对数。

#### `describe_styles`

```python
calculate.exposure.describe_styles() -> list[dict]
```

风格因子的展示信息（名称、口径、窗口与最小观测）。

### calculate.history

#### `ic_sql`

```python
calculate.history.ic_sql(params: EvalParams, *, factor_dir: Path | str=FACTOR_DIR) -> str
```

生成逐日 IC 的 SQL：有效配对内平均并列排名，Pearson 可选 MAD 去极值。

参数:
    params: 因子、股票池、区间、标签与去极值参数。
    factor_dir: 因子 parquet 根目录。

#### `load_daily_ic`

```python
calculate.history.load_daily_ic(params: EvalParams, *, db_file: Path | str=DB_FILE, factor_dir: Path | str=FACTOR_DIR) -> pd.DataFrame
```

读取任意区间（含全历史）的日度 IC、RankIC 与配对数。

参数:
    params: 同 build_panel 的评估参数；仅使用因子、日期、股票池、标签与 winsorize。
    db_file: 行情主库路径。
    factor_dir: 因子 parquet 根目录。

返回:
    交易日索引的 DataFrame，列 ic、rank_ic、n。无配对的日期不出现；
    不设完整面板的行数/天数上限。DuckDB 工作内存限制 1GB，溢出临时文件自动清理。
    attrs['elapsed_ms'] 为本次含校验和取数的耗时，结果不持久化。

### calculate.jsonutil

#### `json_safe`

```python
calculate.jsonutil.json_safe(value: Any) -> Any
```

递归地把值转成 JSON 可序列化形式。

参数:
    value: 任意计算结果，支持 dict / list / ndarray / Series / 标量 / 时间戳。

返回:
    只含 dict、list、str、int、float、bool、None 的结构；
    非有限浮点数一律转成 None。

### calculate.normalize

#### `mad_winsorize`

```python
calculate.normalize.mad_winsorize(values, n_mad: float=3.0) -> np.ndarray
```

按中位数 ± n 倍 MAD 做裁剪。

参数:
    values: 一维取值数组，允许含 NaN。
    n_mad: 倍数，默认 3。

返回:
    裁剪后的数组；有效样本不足 3 个或 MAD 为 0 时原样返回。

#### `winsorize_by_day`

```python
calculate.normalize.winsorize_by_day(dates, values, n_mad: float=3.0) -> np.ndarray
```

按交易日分别做 MAD 裁剪。

参数:
    dates: 与 values 等长、表示交易日的序列。
    values: 一维取值序列。
    n_mad: 倍数，默认 3。

返回:
    裁剪后的数组，顺序与输入一致。

#### `rank_pct`

```python
calculate.normalize.rank_pct(values) -> np.ndarray
```

截面分位（0~1），并列取平均名次，NaN 保持 NaN。

#### `zscore`

```python
calculate.normalize.zscore(values) -> np.ndarray
```

截面标准化；有效样本不足 2 个或标准差为 0 时返回全 NaN。

#### `zscore_by_day`

```python
calculate.normalize.zscore_by_day(dates, values) -> np.ndarray
```

按交易日分别做截面标准化。

参数:
    dates: 与 values 等长、表示交易日的序列。
    values: 一维取值序列，允许含 NaN。

返回:
    标准化后的数组，顺序与输入一致；有效样本不足 2 个或标准差为 0 的
    交易日返回空值。

### calculate.panel

#### `panel_columns`

```python
calculate.panel.panel_columns(params: EvalParams) -> tuple[str, ...]
```

面板的完整列清单：固定列 + 多周期标签列（V11-3 加入）。

参数:
    params: 评估参数，决定除 ``label`` / ``label_1d`` 之外还要带哪几档标签。

#### `label_select`

```python
calculate.panel.label_select(params: EvalParams) -> list[str]
```

标签 CTE 要产出的列（含多周期标签）。

参数:
    params: 评估参数。

#### `open_source`

```python
calculate.panel.open_source(db_file: Path | str=DB_FILE) -> duckdb.DuckDBPyConnection
```

打开内存连接并以只读方式挂载日线主库。

参数:
    db_file: 主库路径。

返回:
    连接对象，主库别名固定为 ``src``。

#### `factor_glob`

```python
calculate.panel.factor_glob(family: str, factor_dir: Path | str=FACTOR_DIR) -> str
```

某个族全部年份的 parquet 通配路径。

参数:
    family: 因子族。
    factor_dir: 因子数据根目录。

#### `resolve_factor`

```python
calculate.panel.resolve_factor(name: str, factor_dir: Path | str=FACTOR_DIR) -> FactorSpec
```

取得因子定义，并确认它已经有落盘的数值列。

参数:
    name: 因子名。
    factor_dir: 因子数据根目录。

#### `count_trade_days`

```python
calculate.panel.count_trade_days(conn: duckdb.DuckDBPyConnection, start: str, end: str) -> int
```

数区间内的交易日数。

参数:
    conn: 已挂载主库的连接。
    start: 起始日。
    end: 截止日。

#### `check_label_columns`

```python
calculate.panel.check_label_columns(conn: duckdb.DuckDBPyConnection, label_columns) -> None
```

确认标签列存在，避免拼出错误列名后报一堆无关的 SQL 错误。

参数:
    conn: 已挂载主库的连接。
    label_columns: 期望存在的标签列名序列。

#### `estimate_panel_rows`

```python
calculate.panel.estimate_panel_rows(conn: duckdb.DuckDBPyConnection, family: str, factor_dir: Path | str, start: str, end: str) -> int
```

用 parquet 元数据估算区间内的面板行数上界（不扫数据本身）。

参数:
    conn: 已挂载主库的连接。
    family: 因子族。
    factor_dir: 因子数据根目录。
    start: 起始日。
    end: 截止日。

#### `panel_sql`

```python
calculate.panel.panel_sql(params: EvalParams, spec: FactorSpec, universe: UniverseSpec, factor_dir: Path | str=FACTOR_DIR) -> str
```

生成评估面板的 SQL。

参数:
    params: 评估参数。
    spec: 因子定义。
    universe: 股票池。
    factor_dir: 因子数据根目录。

#### `optimize_panel`

```python
calculate.panel.optimize_panel(frame: pd.DataFrame, params: EvalParams) -> pd.DataFrame
```

把面板列压成省内存的类型，并固定列顺序。

参数:
    frame: 查询结果。
    params: 评估参数，决定多周期标签列有哪些。

#### `summarize_panel`

```python
calculate.panel.summarize_panel(frame: pd.DataFrame, universe: UniverseSpec) -> dict
```

算一份面板概览，供 meta 事件与覆盖率口径对照。

参数:
    frame: 评估面板。
    universe: 股票池。

#### `EvalContext.label_columns`

```python
calculate.panel.EvalContext.label_columns(self) -> tuple[str, ...]
```

面板上全部标签列，顺序即档位顺序。

#### `EvalContext.horizon_label`

```python
calculate.panel.EvalContext.horizon_label(self, horizon: int) -> str
```

取某个标签档位对应的面板列名。

参数:
    horizon: 标签周期，取值见 ``params.DECAY_HORIZONS``。

#### `EvalContext.styles`

```python
calculate.panel.EvalContext.styles(self) -> pd.DataFrame
```

风格因子原始值（带缓存）：键 stock_code + trade_date，7 个风格列。

风格因子需要 260 个交易日的回看，塞进面板要多占几百万行内存，
所以单独取一次，只有用到拆分阶段才触发；耗时以包 README 的实测为准。

#### `EvalContext.exposures`

```python
calculate.panel.EvalContext.exposures(self, industry_key: str='off')
```

风格拆分结果（带缓存）：逐日截面回归的系数、t 值与残差。

参数:
    industry_key: 行业数据源标识，``off`` 表示不加行业层。

#### `EvalContext.daily_ic`

```python
calculate.panel.EvalContext.daily_ic(self, winsorize: bool | None=None) -> pd.DataFrame
```

逐日 IC 序列（带缓存），列：ic、rank_ic、n。

参数:
    winsorize: 是否去极值；留空用参数里的设置。

#### `EvalContext.bins`

```python
calculate.panel.EvalContext.bins(self, quantiles: int | None=None) -> pd.DataFrame
```

池内样本的分位分组表（带缓存），列见 ``binning.assign_bins``。

参数:
    quantiles: 分组数；留空用参数里的设置。

#### `EvalContext.bin_matrices`

```python
calculate.panel.EvalContext.bin_matrices(self, quantiles: int | None=None) -> tuple[pd.DataFrame, pd.DataFrame]
```

（组号宽表, 截面秩宽表）二元组，带缓存：换手与自相关共用同一次透视。

参数:
    quantiles: 分组数；留空用参数里的设置。

#### `build_panel`

```python
calculate.panel.build_panel(params: EvalParams, *, db_file: Path | str=DB_FILE, factor_dir: Path | str=FACTOR_DIR, conn: duckdb.DuckDBPyConnection | None=None) -> EvalContext
```

构建评估面板。

参数:
    params: 评估参数。
    db_file: 日线主库路径。
    factor_dir: 因子数据根目录。
    conn: 已有的主库连接；留空则自行打开并在结束时关闭。

#### `default_params`

```python
calculate.panel.default_params(factor: str, *, db_file: Path | str=DB_FILE, years: int=3, **overrides) -> EvalParams
```

按「最近 N 年」构造默认参数。

参数:
    factor: 因子名。
    db_file: 日线主库路径。
    years: 回看的年数，默认 3。
    overrides: 需要覆盖的其它参数。

### calculate.params

#### `EvalParams.label_column`

```python
calculate.params.EvalParams.label_column(self) -> str
```

面板标签列对应的 ``md.fwd_return`` 列名。

#### `EvalParams.label_column_for`

```python
calculate.params.EvalParams.label_column_for(self, horizon: int) -> str
```

取指定周期的标签列名，净值曲线等需要换周期时用。

参数:
    horizon: 标签周期。

#### `EvalParams.horizon_label_column`

```python
calculate.params.EvalParams.horizon_label_column(self, horizon: int) -> str
```

把衰减分析用到的标签档位映射到评估面板的列名。

参数:
    horizon: 标签周期，必须落在 ``DECAY_HORIZONS`` 里。

#### `EvalParams.extra_label_specs`

```python
calculate.params.EvalParams.extra_label_specs(self) -> tuple[tuple[int, str], ...]
```

除 ``label`` 与 ``label_1d`` 之外，面板还要带的标签档位与列名。

#### `EvalParams.extra_label_columns`

```python
calculate.params.EvalParams.extra_label_columns(self) -> tuple[str, ...]
```

除 ``label`` 与 ``label_1d`` 之外，面板还要带的多周期标签列。

#### `EvalParams.describe`

```python
calculate.params.EvalParams.describe(self) -> dict
```

把参数摊平成可序列化的字典，供 meta 事件回显。

### calculate.stats

#### `series_stats`

```python
calculate.stats.series_stats(values) -> dict
```

把一条收益或相关序列压成均值、标准差、标准误、IR、t 值与胜率。

参数:
    values: 逐日序列，允许含 NaN。

#### `compound`

```python
calculate.stats.compound(returns) -> np.ndarray
```

把逐日收益累乘成净值曲线。

参数:
    returns: 逐日收益序列，缺失值按 0 处理。

#### `max_drawdown`

```python
calculate.stats.max_drawdown(equity) -> float
```

最大回撤，返回负数（例如 -0.12 表示最深回撤 12%）。

参数:
    equity: 净值曲线。

#### `annualised_sharpe`

```python
calculate.stats.annualised_sharpe(returns, periods: int=TRADING_DAYS) -> float | None
```

年化夏普（不扣费用、不做无风险利率调整），样本不足或波动为 0 时返回空。

参数:
    returns: 逐日收益序列。
    periods: 一年的交易日数。

### calculate.styles

#### `lookback_start`

```python
calculate.styles.lookback_start(conn: duckdb.DuckDBPyConnection, start: str, days: int=STYLE_LOOKBACK_DAYS) -> str
```

取区间起始日往前第 ``days`` 个交易日的日期。

参数:
    conn: 已挂载主库 ``src`` 的连接。
    start: 评估区间起始日。
    days: 需要回溯的交易日数。

#### `style_sql`

```python
calculate.styles.style_sql(params: EvalParams, window_start: str) -> str
```

生成风格因子的取数 SQL。

参数:
    params: 评估参数，取它的 ``start`` / ``end``。
    window_start: 市场回看起点（含）；个股有效历史不足时向前补足 250 次交易。

#### `build_styles`

```python
calculate.styles.build_styles(params: EvalParams, *, db_file: Path | str=DB_FILE, conn: duckdb.DuckDBPyConnection | None=None) -> pd.DataFrame
```

取评估区间内全部股票的风格因子原始值。

参数:
    params: 评估参数。
    db_file: 日线主库路径。
    conn: 已有的主库连接；留空则自行打开并在结束时关闭。

#### `style_specs`

```python
calculate.styles.style_specs(keys=None) -> tuple[StyleSpec, ...]
```

按名字取风格定义，留空返回全部。

参数:
    keys: 风格名序列。

#### `IndustrySource.load`

```python
calculate.styles.IndustrySource.load(self, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame
```

返回含 ``stock_code`` 与 ``industry`` 两列的表。

参数:
    conn: 已挂载主库 ``src`` 的连接。

#### `StockBasicIndustry.load`

```python
calculate.styles.StockBasicIndustry.load(self, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame
```

读静态行业映射。

参数:
    conn: 已挂载主库 ``src`` 的连接。

#### `list_industry_sources`

```python
calculate.styles.list_industry_sources() -> tuple[IndustrySource, ...]
```

返回可用的行业数据源，供指标参数枚举。

#### `resolve_industry_source`

```python
calculate.styles.resolve_industry_source(key: str) -> IndustrySource | None
```

按标识取行业数据源；``off`` 返回 None。

参数:
    key: 数据源标识，或 ``off`` 表示关闭。

#### `industry_choices`

```python
calculate.styles.industry_choices() -> tuple[str, ...]
```

指标参数里行业开关的可选值。

### calculate.universe

#### `UniverseSpec.is_index`

```python
calculate.universe.UniverseSpec.is_index(self) -> bool
```

是否是指数成分池。

#### `list_universes`

```python
calculate.universe.list_universes() -> tuple[UniverseSpec, ...]
```

返回全部可选股票池，全 A 池排第一。

#### `resolve_universe`

```python
calculate.universe.resolve_universe(key: str) -> UniverseSpec
```

按标识取股票池，非法标识直接报错并列出可选项。

参数:
    key: 股票池标识。

#### `membership_cte`

```python
calculate.universe.membership_cte(spec: UniverseSpec, start: str, end: str) -> str
```

生成指数成分的 CTE（逐交易日的成分集合）。

参数:
    spec: 股票池定义，必须是指数池。
    start: 评估区间起始日。
    end: 评估区间截止日。

返回:
    ``member AS (...)`` 的 CTE 文本，产出 ``trade_date`` 与 ``stock_code`` 两列。

### calculate.metrics.base

#### `Metric.describe`

```python
calculate.metrics.base.Metric.describe(self) -> dict
```

摊平成可序列化字典，供接口层与 meta 事件使用。

#### `register_metric`

```python
calculate.metrics.base.register_metric(metric: Metric, overwrite: bool=False) -> Metric
```

登记一个指标。

参数:
    metric: 指标定义。
    overwrite: 同名时是否覆盖，默认报错。

#### `get_metric`

```python
calculate.metrics.base.get_metric(key: str) -> Metric
```

按 key 取指标。

参数:
    key: 指标标识。

#### `unregister_metric`

```python
calculate.metrics.base.unregister_metric(key: str) -> None
```

注销一个指标，找不到就静默返回。

参数:
    key: 指标标识。

#### `list_metrics`

```python
calculate.metrics.base.list_metrics() -> tuple[Metric, ...]
```

返回全部指标，按 stage 顺序排列（即前端卡片顺序）。

#### `metrics_by_stage`

```python
calculate.metrics.base.metrics_by_stage(stage: str) -> tuple[Metric, ...]
```

返回某个阶段下的指标，保持注册顺序。

参数:
    stage: 阶段名。

#### `default_options`

```python
calculate.metrics.base.default_options(metric: Metric) -> dict
```

取出某个指标的默认参数取值。

参数:
    metric: 指标定义。

#### `resolve_options`

```python
calculate.metrics.base.resolve_options(metric: Metric, options: Mapping[str, Any] | None=None) -> dict
```

合并并校验指标参数，错误在读库前返回。

参数:
    metric: 指标定义。
    options: 参数名到取值的映射，未提供的参数取默认值。

#### `clear_metrics`

```python
calculate.metrics.base.clear_metrics() -> None
```

清空注册表，仅供测试使用。

### calculate.metrics.bins

#### `compute_bin_returns`

```python
calculate.metrics.bins.compute_bin_returns(context, options) -> dict
```

算分位平均期收益、t 值与单调性（两种权重）。

参数:
    context: 评估上下文。
    options: 指标参数（本指标无参数）。

#### `compute_bin_curves`

```python
calculate.metrics.bins.compute_bin_curves(context, options) -> dict
```

算分位累计净值曲线（两种权重，1 日前瞻收益累乘）。

参数:
    context: 评估上下文。
    options: 指标参数（本指标无参数）。

#### `compute_spread`

```python
calculate.metrics.bins.compute_spread(context, options) -> dict
```

算多空价差：最高分位减最低分位的逐日收益、净值与统计量。

参数:
    context: 评估上下文。
    options: 指标参数（本指标无参数）。

#### `compute_bin_dist`

```python
calculate.metrics.bins.compute_bin_dist(context, options) -> dict
```

算分位收益分布：每日分位均值的箱线数据 + 分位内个股收益分位点。

参数:
    context: 评估上下文。
    options: 指标参数（本指标无参数）。

### calculate.metrics.coverage

#### `compute`

```python
calculate.metrics.coverage.compute(context, options) -> dict
```

算覆盖率、池内分布与每日有效样本数。

参数:
    context: 评估上下文。
    options: 指标参数（本指标无参数）。

### calculate.metrics.decay

#### `compute_decay`

```python
calculate.metrics.decay.compute_decay(context, options) -> dict
```

算多周期衰减：各档 RankIC 与分位收益。

参数:
    context: 评估上下文。
    options: 指标参数（本指标无参数）。

### calculate.metrics.exposure

#### `compute_exposure`

```python
calculate.metrics.exposure.compute_exposure(context, options) -> dict
```

算风格暴露：逐日截面回归的系数、t 值与逐日序列。

参数:
    context: 评估上下文。
    options: 指标参数（``industry``）。

#### `compute_incremental_ic`

```python
calculate.metrics.exposure.compute_incremental_ic(context, options) -> dict
```

算增量 IC：剔除风格后残差的 RankIC，与原始 RankIC 对照。

参数:
    context: 评估上下文。
    options: 指标参数（``industry``）。

### calculate.metrics.ic

#### `compute_stats`

```python
calculate.metrics.ic.compute_stats(context, options) -> dict
```

算 IC / RankIC 汇总指标与方向一致性。

参数:
    context: 评估上下文。
    options: 指标参数（本指标无参数）。

#### `compute_series`

```python
calculate.metrics.ic.compute_series(context, options) -> dict
```

输出逐日 IC 序列与 20 日滚动均值，供前端画柱状图与曲线。

参数:
    context: 评估上下文。
    options: 指标参数（本指标无参数）。

#### `compute_dist`

```python
calculate.metrics.ic.compute_dist(context, options) -> dict
```

输出 IC 与 RankIC 的分布，供前端画直方图。

参数:
    context: 评估上下文。
    options: 指标参数（本指标无参数）。

### calculate.metrics.turnover

#### `compute_turnover`

```python
calculate.metrics.turnover.compute_turnover(context, options) -> dict
```

算分位换手、全池换手与因子自相关。

参数:
    context: 评估上下文。
    options: 指标参数（本指标无参数）。

### 配置、常量与错误处理

根包 `calculate.__all__` 中的常量包括股票池、风格、标签周期、分箱组数、stage 顺序和面板上限。
数据路径定义在 `calculate.config`：DB_FILE 是行情库，FACTOR_DIR 是因子 parquet 根目录，
FACTOR_DB_FILE 是数据侧目录库；LOG_DIR 是本包日志目录。入口提供 db_file/factor_dir 参数，
替换数据位置优先传参数。

预期错误从 `calculate.errors` 导入：CalculateError 是基类，异常有 code/message。
EVAL-101 参数/日期/阶段，102 面板，103 股票池，104 因子，201 指标参数/注册，
301 拆分，302 风格，303 行业。异常不会当作正常结果吞掉；调用方负责显示或记录错误。

`calculate.logging_conf.get_logger(module_name)` 返回按模块分目录的 logger；处理器由根目录
`common.logging_conf` 统一实现。日志用于阶段耗时
与样本规模核查，不代替数值正确性验证。不要在日志里记录完整原始面板。

### 验证命令

```bash
.venv/bin/python -m pytest research/evaluation/calculate/tests tests -q
```

真实库相关测试需要 data/databases/quant.duckdb 与 data/factors/；其余合成数据回归不依赖网络。

## 修复后性能核验（2026-10-12）

全历史 MOM_20，2000-01-04～2026-09-28、all_a、close 5 日标签：

| 路径 | 配对数 | 日期数 | 含校验取数耗时 | 峰值进程 RSS |
|---|---|---|---|---|
| load_daily_ic，默认 MAD Pearson IC + 原始秩 RankIC | 16,248,231 | 6,454 | 12.4 秒 | 约 1,305 MB |
| evaluate，最近三年完整 12 指标（共享面板） | 全面板约 392 万行 | 最近三年 | 52.1 秒 | 本轮未单独测峰值 |

这是本机一次独立进程的实测，非跨机器/并发请求的性能保证。
原定全历史 IC ≤15 秒通过此轻量路径验收；完整 12 指标的耗时另算，
不能把这条记录当作完整评估 ≤15 秒。完整路径仍保留面板上限，
V11-5～V11-9 的相关性、前后端与多请求压测仍按原计划实施。

## V11-5 相关性接口（2026-10-12）

模块 `calculate.correlation` 只读取产物，不使用数据项目内部脚本。
`EvalContext.factor_dir` 保存此次评估所用的因子目录（默认 `data/factors/`）；
`build_panel` 已传入此字段，相关性不会绕过调用者指定的自定义目录。
`open_source` 默认每连接工作内存1GB、2线程；全历史轻量IC仍独立设4线程。

| 公开函数 | 输入与输出 |
|---|---|
| `available_factors(factor_dir=FACTOR_DIR)` | parquet根目录 → `{族: 因子名元组}`；仅保留注册且实际存在的列，空目录返回空字典 |
| `daily_spearman_matrix(values, *, target_index=None, backend=None)` | 股票×因子float矩阵 → 成对秩相关方阵或目标行；默认按环境选择后端，目标模式其他位置为NaN |
| `accumulate_correlations(frame,names,sums,days,backend=None)` | 含trade_date的月长表、因子列名、和/计数方阵 → 原地累计；可复用同一后端 |
| `correlation_result(names,sums,days,factor,top_n=10)` | 累计矩阵、目标因子 → names/matrix/days/top等字典；不足20有效日返回NaN，Top-N排除自身，范围1～100 |
| `compute_correlation(context,options)` | 评估上下文与`{'top_n':10}` → 按月计算的完整结果；由注册指标调用，无长期缓存 |
| `compute_target_correlation(context,options,backend=None)` | 非正式候选上下文与`{'top_n':10}` → 目标对正式因子的目标行/列；公共矩阵仍保持原形状 |

```python
import calculate
from calculate.jsonutil import json_safe
params = calculate.EvalParams('MOM_20', '2024-01-01', '2024-12-31')
result = dict(calculate.evaluate('MOM_20', params, stages=['correlation'],
    metric_options={'correlation': {'top_n': 10}}))
result['correlation']['metrics']['correlation']['top']
```

新增指标 `correlation`（stage同名、中文名“与已有因子的相关性”），参数top_n默认10。
结果：names列名、matrix时间均值方阵、days每对有效日方阵、top记录数组
（factor/correlation/days），另有min_days/min_stocks/basis/note/insufficient。
原始函数可含ndarray/NaN；evaluate统一用json_safe转为数组/null。
insufficient表示没有足够样本的可比较同伴，矩阵仍可有有效对角线。

相同有限掩码分组复用排序；成对删除后按保留的并列块重新排名，不能先对全截面
排名再直接删股票。相关性不筛收益标签，不做去极值。不同因子对可以有不同样本，
因此均值矩阵不保证半正定；Top-N按绝对相关性排序并保留正负号。

逐日计算由 `_DailySpearmanWorkspace` 管理。每个因子每天只生成一次稳定排序索引；
相同缺失掩码和组对交集用打包字节键复用解包掩码、样本位置与有效数。排名矩阵直接预分配，
中心和范数由并列块大小直接计算；排名缓存只存在于当前交易日的调用内，不跨日或跨请求。

正式因子页面使用完整相关热力图，因此 `compute_correlation` 继续计算完整矩阵。
非正式候选只展示目标因子与正式因子的关系，`compute_target_correlation` 只计算目标行、
目标列和对角线；返回的 `matrix`、`days`、`names` 数组仍保持原字段与形状，其他矩阵位置
分别为 `NaN` 和 `0`。两条路径共用缺失掩码分组、稳定排序、并列平均名次和相关计算。

2026-10-27 三年固定案例中，47 个相关性列的候选阶段由 233.5 秒降至 132.9 秒，峰值 RSS
由 2788.2MB 降至 2682.3MB；正式完整矩阵与 P0-01 金标准一致。该优化仍为 Python 算法
调整，不代表已经采用 C++。

P0-03 进一步把正式三年相关性 stage 从 228.6 秒降至 209.1 秒，峰值 RSS 约 2680.0MB；
金标准完全一致。该轮仍是 Python 算法优化，未引入 C++。

P1-03 让 `compute_correlation` 每次请求只选择一次 `auto`/`python`/`native` 后端。默认
`auto` 在 C++ 扩展可用时使用 `quant_native.pairwise_spearman`，扩展不可导入时回退 Python；
已加载原生核的执行异常不静默回退。正式完整矩阵调用完整模式，候选目标行传
`target_index`，C++ 只计算目标行、目标列和对角线，避免重新引入全矩阵浪费。

2026-10-27 固定三年案例中，正式 stage 为 Python 217.770 秒、native 140.460 秒、auto
135.788 秒；候选目标行为 Python 137.599 秒、native 118.680 秒、auto 120.995 秒。
相关系数在 `1e-10` 内一致，有效计数和 `NaN` 位置完全一致。该轮实际登记 1 个 C++ 替换
模块，证据位于 `AiRecord/benchmarks/V114/p1-03-20261027/`。

完整分析性能请看 `docs/apps/monitor/performance.md`，旧12指标耗时不包含本阶段。
