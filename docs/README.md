# 因子质量监测项目：计算包 `calculate`

## 负责什么

把「因子值 + 未来收益标签 + 股票池标记 + 后续分析原料」对齐成一张评估面板，
在面板上算指标，并按固定顺序逐段产出结果，接口层直接转成流式事件。

本项目与数据项目（取数 / 清洗 / 建库 / 因子库）分开，目录约定：

```
calculate/   计算包：只算，自带配置、错误码与日志     ← 本目录
backend/     后端：接口、任务调度、流式输出（待实现）
frontend/    前端：纯 HTML/JS + ECharts（待实现）
```

计算包对数据侧只有三个依赖：`db/quant.duckdb`（行情、标签、指数与成分）、
`db/factors.duckdb`（因子目录）、`factor_data/`（因子值）；另加数据项目的
`factors` 注册表用于取因子的定义与方向（`panel.py` 里的 `import factors`，
如果以后要彻底解耦，可以改成只读 `db/factors.duckdb` 的 `meta.factor_definition`）。

三条边界：

1. **只算**：不展示、不落库、不产生新因子；
2. **只读**：数据一律经 DuckDB 只读连接（`db/quant.duckdb` 的 `md` / `ref` 表 +
   `factor_data/` 的 parquet），不写任何文件；
3. **指标不缓存到磁盘**：结果只在单次请求的进程内存在，下一次请求重新算。

## 模块与依赖

| 模块 | 负责 | 被谁调用 | 调用了谁 | 改动时注意 |
|---|---|---|---|---|
| `config.py` | 数据路径与常量（库、因子产物、本包日志目录） | 全部模块 | — | 只认目录结构，不引用数据项目的 `scripts`；换数据位置改这里 |
| `errors.py` | `CalculateError` 基类与 `EVAL-1xx/2xx/3xx` 错误码 | 全部模块 | — | 后端 `MONITOR-1xx` 在 backend 里定义，别混进来 |
| `logging_conf.py` | 按模块分目录、128MB 轮转、起始-结束命名的日志 | 全部模块 | `config` | 日志写 `calculate/logs/`，与数据项目的 `logs/` 分开 |
| `params.py` | `EvalParams` 定义与校验 | `panel`、指标、接口层 | `errors` | 加字段要同步 `describe()`，meta 事件靠它回显参数 |
| `universe.py` | 股票池定义与指数成分 CTE | `panel`、接口层 | `errors` | 全 A 池口径必须与数据项目的 `scripts/build_factors/validate.py` 的 `POOL_FILTER` 一字不差；指数池用**上一个已结束月份**的月末快照，别改成当月（会前视） |
| `panel.py` | 评估面板、连接、默认参数、`EvalContext` | `evaluate`、各指标、后续拆分阶段 | `factors` 注册表、`universe`、DuckDB、`config` | 面板列是公共契约，后续阶段只加不删；行数/交易日上限也在这里；**不要给面板加 ORDER BY**（几百万行排序要多花 2 秒以上，下游都按交易日分组） |
| `daily.py` | 逐日截面相关（IC / RankIC 与通用 `daily_corr`） | `metrics/ic.py`，后续 binplot 与相关性阶段 | `normalize` | RankIC 一律对**原始值**排名；去极值只作用于 Pearson IC；池内过滤默认开（`pool_only=True`） |
| `normalize.py` | MAD 去极值、分位、zscore | `daily`、后续风格因子 | — | 都按交易日分组调用，NaN 原样保留 |
| `stages.py` | `STAGE_ORDER`（stage 顺序契约） | `evaluate`、指标注册表 | — | 单独成模块是为了让注册表能按它排序又不与评估入口循环依赖 |
| `stats.py` | 序列四件套、净值累乘、最大回撤、年化夏普 | IC 家族与 binplot 家族 | — | 加统计口径改这里，别在各指标里各写一遍 |
| `binning.py` | 分位分箱、分位收益、换手、因子自相关、宽表透视 | `metrics/bins.py`、`metrics/turnover.py` | `daily`、`normalize` | 分箱只看因子值；宽表用 factorize + numpy 填充（pandas `pivot` 要慢一倍以上） |
| `metrics/base.py` | 指标注册表 | 各指标、接口层、前端 | `errors` | 注册顺序 = 前端卡片顺序；重名报 `EVAL-201` |
| `metrics/coverage.py` | 覆盖率与取值分布 | 经注册表调用 | `metrics/base` | 缺失（NaN）与非有限值（inf）分开统计，别混成一个数 |
| `metrics/ic.py` | IC / RankIC 汇总、时序、分布 | 经注册表调用 | `daily`、`metrics/base` | 有效交易日少于 20 天只给「样本不足」标记，不给数值 |
| `metrics/bins.py` | 分位平均收益与单调性、分位净值、多空价差、分位收益分布 | 经注册表调用 | `binning`、`stats` | 每张图都出两套权重；净值固定用 1 日收益，别改成期收益累乘 |
| `metrics/turnover.py` | 分位换手、全池换手、因子自相关（lag 1/5/20） | 经注册表调用 | `binning`、`stats` | 宽表从 `EvalContext.bin_matrices()` 取，别自己再透视一次 |
| `evaluate.py` | stage 生成器与 meta 组装 | 接口层（后续 FastAPI） | `panel`、`metrics` | `STAGE_ORDER` 是前后端的公共契约，后续阶段只往空位里填 |

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
| `ic_series` | 逐日 IC / RankIC / 样本数与 20 日滚动均值 |
| `ic_dist` | 40 桶直方图、分位数、偏度峰度 |
| `bins` | `metrics.bin_returns`：分位平均期收益、t 值、标准误、胜率、单调性（两种权重） |
| `bin_curves` | `metrics.bin_curves`：分位累计净值曲线（1 日收益累乘，两种权重） |
| `spread` | `metrics.spread`：多空价差逐日序列、净值、t 值、年化夏普、最大回撤 |
| `bin_dist` | `metrics.bin_dist`：每日分位均值的箱线数据 + 分位内个股收益分位点 |
| `turnover` | `metrics.turnover`：分位换手、全池逐日换手、因子自相关（lag 1/5/20） |
| `done` | 各 stage 耗时、总耗时、面板概览 |

后续阶段会填入 `decay` / `exposure` / `incremental_ic` / `correlation`，接口与前端不用改。

## 怎么加一个指标

1. 在 `calculate/metrics/` 下新增一个模块，写 `compute(context, options) -> dict`；
2. 调 `register_metric(Metric(key=..., label=..., stage=..., compute=compute))` 注册；
3. 在 `calculate/metrics/__init__.py` 里 import 这个模块（导入即注册）。

所用 `stage` 已经在 `STAGE_ORDER` 里的话，接口层与前端**自动生效**；
若是全新 stage，在 `STAGE_ORDER` 里补一个位置即可。

## 口径速查

- 股票池：全 A 可交易 = `can_buy_close` + 非 ST + 上市满 60 个自然日；
  指数池 = 上一个已结束月份的月末权重快照 ∩ 同一套可交易性过滤。
- IC：逐日截面 Pearson，默认对因子与标签各做 3 倍 MAD 去极值（`winsorize=False` 可关）。
- RankIC：逐日截面 Spearman；与 `validate.py` 的差异只来自并列名次语义（1e-6 量级），
  由 `tests/test_eval_consistency.py` 守住。
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
- 上限：区间 ≤ 2400 个交易日；面板 ≤ 800 万行（按 parquet 元数据估的上界，
  因此实际可用的最长区间约 6 年）。

## 耗时参考（实测，2026-10-10）

| 区间 | 面板行数 | 建面板 | overview | 合计（9 个指标） |
|---|---|---|---|---|
| 默认最近 3 年 | 3,925,208 | 4.4 s | 4.0 s | 14.4 s |
| 2021-01-01 起（5.7 年） | 7,113,634 | 13.0 s | 10.3 s | 36.1 s |

各阶段的耗时（默认区间）：`overview` 4.0 s、`bins` 2.0 s、`bin_curves` 0.6 s、
`spread` 0.6 s、`bin_dist` 0.9 s、`turnover` 2.0 s；5.7 年区间上分别是
10.3 / 4.2 / 1.4 / 1.3 / 1.8 / 4.0 s。
流式推送下用户先看到 IC，再看到各项分层结果；进一步的压测与优化见 V11-9。

日志按模块分目录写在 `calculate/logs/calculate_panel/`、`calculate/logs/calculate_evaluate/`
（模块名里的点换成下划线，与数据项目的日志命名规则一致）。

目录约定与后续安排见 `AiRecord/plans/20261011-1.md`（项目拆分）；
前后端代码会分别落在 `backend/` 与 `frontend/`，本包只提供纯函数式的计算与流式 stage。
