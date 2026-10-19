# 核心模块深挖

## 核心模块判定

| 模块 | 判断依据 | 主要贡献 |
|---|---|---|
| `data.py` | 定义输入数据的统一索引/列契约 | 让所有后续矩阵可以安全对齐 |
| `signals.py` | 由因子产生选股和目标权重 | 决定策略持仓 |
| `orders.py` | 处理状态机、拒单、费用和退市强平 | 决定哪些目标真正能提交 |
| `engine.py` | 调用 VectorBT 并标准化结果 | 产生权威组合价值和订单 |
| `pipeline.py` | 管理端到端生命周期 | 连接入口、取数、执行和存储 |
| `store.py` | 保存版本和运行产物 | 保证可追溯与复现 |

## `data.py`

### 职责边界

负责从 `quant` 与因子目录读取并对齐输入；不负责计算信号、不负责订单规则、不负责现金账本。

### 关键数据结构

| 名称 | 字段 | 含义 | 谁写入 | 谁读取 |
|---|---|---|---|---|
| `MarketData` | 价格、标记、市值、因子、上市/退市日、成分、基准、版本 | VectorBT 和信号层的统一宽表 | `load_market_data` | `signals`、`orders`、`engine` |
| `PANEL_FIELDS` | 价格、标记、市值字段名 | 控制逐字段取数和统一校验 | 模块常量 | `MarketData`、`load_market_data` |

### 调用链

```text
pipeline.iter_run
  -> load_market_data
     -> resolve_universe / universe_stocks
     -> check_limits
     -> _load_field × PANEL_FIELDS
     -> _load_factor × factor_names
     -> membership_frame
     -> _load_benchmark
     -> MarketData.__post_init__
```

### 关键函数

`MarketData.__post_init__(self) -> None`（`data.py:85-104`）检查所有矩阵的形状、索引和列，并验证 `list_date`/`delist_date` 的索引与股票列一致。任何错位都立刻转为 `DataAlignmentError`。

`MarketData.crop(start, end) -> MarketData`（`data.py:116-145`）按日期裁剪所有字段。它特别处理基准元数据，让直接调用引擎传入更宽区间的数据时，最终 `benchmark_info` 的覆盖范围仍与实际运行区间一致。

`_load_field(field, start, end, stocks, db_path) -> DataFrame`（`data.py:148-177`）根据字段选择 `qfq` 或 `raw`，用 `quant.load_panel` 读取长表，再 pivot 为宽表。价格/市值使用 `float32`；交易状态缺失填 False。

`_load_benchmark(...) -> tuple[Series | None, dict | None]`（`data.py:297-327`）严格按回测交易日重建基准序列。基准为空或缺日都报错，并返回收益口径元数据。

### 关键边界

| 条件 | 处理 |
|---|---|
| 面板形状或列不一致 | `BT-201` |
| 区间没有交易日 | `MarketData.crop` 抛 `BT-201` |
| 基准缺日 | `_load_benchmark` 抛 `BT-201` |
| 面板太大 | `check_limits` 抛 `BT-401` |
| 缺少退市日期列 | 记录 warning，不做退市强平 |

## `signals.py`

### 职责边界

负责信号和目标权重，不负责实际成交、现金和成本。

### 调用链

```text
run_backtest
  -> build_signal_targets
     ├─ 无 provider：build_target_weights
     │  -> build_scores / eligible_mask / rebalance_mask / _weights_for
     └─ 有 provider：get_provider -> provider.build(data, eligible, params)
```

### 关键函数

`rebalance_mask(index, frequency) -> Series`（`signals.py:17-28`）把交易日映射到日/周/月周期，只标出每个周期的最后交易日。

`build_scores(data, spec) -> DataFrame`（`signals.py:39-56`）验证因子存在，按配置 winsorize 和标准化，再按权重线性合成。标准差为 0 的截面变为 NaN，避免平因子形成无意义排名。

`eligible_mask(data) -> DataFrame`（`signals.py:59-78`）综合可买入收盘、非停牌、非 ST、上市 60 日和指数成分矩阵。上市日期以自然日加 60 天比较。

`build_target_weights(...) -> tuple[DataFrame, DataFrame]`（`signals.py:101-137`）只在调仓日生成目标。先把整行置为 0，再给 Top-N 股票写入正权重；非调仓日保持 NaN。

`build_signal_targets(data, strategy, rebalance_days=None) -> tuple[DataFrame, DataFrame]`
是因子策略和提供者策略的共同入口。提供者路径拒绝同时传 `rebalance_days`，由提供者返回的
矩阵决定调仓日；返回矩阵会经过形状、部分 `NaN`、有限性、非负性、单行合计和 eligible
正权重校验，再按正权重派生排名。

### 提供者假设

- `data` 是当前面板的 `MarketData`，`eligible` 是与它同形的布尔矩阵；提供者不获得真实现金、
  持仓和订单状态。
- 整行 `NaN` 表示不调仓，全 0 行表示清仓；非空行合计可以小于 1，剩余部分留现金。
- 提供者必须确定性、因果、只读输入，不能修改 `data` 或 `eligible`。
- 跨资源块的滚动计算必须使用因子库中已经算好的因子；提供者不会拿到块边界之前的完整行情。
- `signal.factors` 是输入因子声明；`required_factors` 必须是它的子集，`signal.weights` 必须为空。
- 提供者名称、版本和参数必须匹配；代码改动要提升版本，源码哈希只用于审计，不是 P1-3 的
  不可变快照。

### 设计取舍

用矩阵表达 `NaN/0/正数` 三种状态，能够直接映射到 VectorBT 的“今天不下单/清仓/目标仓位”。如果把这三种状态压成一个布尔信号，清仓和不动作会无法区分。

## `providers.py` 与 `strategies/`

### 职责边界

负责提供者的注册、自动发现、版本/参数/必需因子校验和源码哈希；不负责信号计算之后的
风险上限、订单状态机、资金分配和账本。`strategies/` 下的模块在导入时注册，文件名以下划线
开头的不自动导入。

### 调用链

```text
build_signal_targets / store.create_strategy / backtest_service.prepare / options
  -> get_provider
     -> discover_providers
        -> import backtest.strategies.*
  -> validate_strategy_provider
     -> 版本、必需因子、validate_params
```

### 关键函数

- `register_provider(provider)`：注册实例或类，检查名称、版本、描述、`required_factors`
  和 `build` 是否存在，重复名称直接报 `BT-101`。
- `discover_providers()`：导入 `backtest.strategies` 下的模块，触发模块内注册。
- `get_provider(spec)`：按名称取提供者并校验版本，未知名称列出可用提供者。
- `validate_strategy_provider(spec)`：校验提供者版本、必需因子和 `validate_params`。
- `provider_metadata(spec)` / `signal_metadata(signal)`：生成运行元数据，含模块和源码哈希。

### 设计取舍与注意事项

- 固定目录自动发现比策略 JSON 保存任意 `module:function` 更容易审计，也避免 HTTP 创建策略时
  引入任意代码导入；新增提供者必须把模块放进 `backtest/strategies/`。
- `params` 必须是可 JSON 序列化对象，不能含 `NaN`；版本和参数进入策略配置哈希，
  源码哈希只进入运行元数据。代码改动必须提升提供者版本，不能只改源码。
- 普通提供者只能依赖行情、因子和 eligible 矩阵；需要止损或动态加减仓时声明
  `uses_position_state=True`，读取执行层状态，不应在提供者里模拟持仓。
- `providers` 不保证跨资源块的历史行情；提供者应使用已落盘因子做滚动计算。

## `orders.py`

### 职责边界

负责交易规则和订单计划，不维护现金余额，也不替代 VectorBT 执行。

### 关键数据结构

| 名称 | 字段 | 含义 |
|---|---|---|
| `OrderPlan` | `target_percent` | 最终目标权重矩阵 |
| `OrderPlan` | `fees` / `fixed_fees` | VectorBT 费用输入 |
| `OrderPlan` | `rejection` | 规则拒单明细 |
| `OrderPlan` | `direction` | 状态机判定方向 |
| `OrderPlan` | `liquidations` | 实际提交的退市强平 |
| `OrderPlan` | `plan_stats` | 计划校准统计 |

### 关键函数

`execution_targets(target_weights) -> DataFrame`（`orders.py:58-64`）整体向后移动一天，保证收盘信号次日开盘执行。

`apply_rules(data, targets, execution, writeoffs) -> tuple[...]` 维护 `held` 和 `shadow` 状态。它把当前目标和上一成功目标比较，判定买、卖、再平衡；被拦下的格子不更新状态，下一次调仓继续尝试。退市整理期持仓从首个可卖日起强制目标为 0。

`writeoff_cells(data) -> DataFrame` 处理区间内正式退市的股票，返回核销行、退市日和参考收盘价；`liquidation_cells()` 保留为兼容入口。

成本不再由 `orders.py` 生成费率矩阵。`costs.py` 的 `CostSpec.charge()` 与
`charge_arrays()` 统一处理佣金、最低佣金、历史印花税、历史过户费和历史红利税，
账本成交与资金分配共用同一套规则。

### 关键分支

| 条件 | 结果 |
|---|---|
| 停牌且非退市强制退出 | 目标置 NaN，记录 `suspended` |
| 涨停买入 | 目标置 NaN，记录 `limit_up_or_not_buyable` |
| 跌停卖出 | 目标置 NaN，记录 `limit_down_or_not_sellable` |
| 退市整理期持仓 | 强制目标为 0，遵守停牌和跌停并每日重试 |
| 退市日仍有残仓 | 账本按零价值核销，不计费用和红利税 |
| 目标为 0 且空仓 | 不提交无效订单 |
| 目标与上一目标相同 | 作为方向未知的再平衡，任一侧被封则不提交 |

## `engine.py`

### 职责边界

负责调用 VectorBT、提取标准结果、计算指标和保存运行元数据；不重新实现现金账本。

### 调用链

```text
run_backtest
  -> data.crop
  -> build_signal_targets（或使用分块准备阶段传入的 target_override）
  -> execution_targets
  -> _run_portfolio (探针，无费用)
  -> _probe_frame
  -> build_order_plan
  -> _run_portfolio (最终，带费用)
  -> _daily_frame / _holdings_frame / _liquidation_records
  -> summarize
  -> BacktestResult
```

`_run_portfolio(...)` 现在调用原始价账本：探针与基准遍使用目标百分比，最终遍使用按板块规则分配出的目标股数。主板/创业板按 100 股整数倍，科创板最低 200 股并按 1 股递增，北交所最低 100 股并按 1 股递增；成交价和估值价都使用原始口径。

`_daily_frame(...)`（`engine.py:133-166`）生成组合价值、现金、持仓市值、收益、净值、回撤、成本、换手和基准超额列。

`run_backtest(...)`（`engine.py:282-370`）是执行层总入口。它还负责发出 `signals`、`probe`、`orders`、`execute` 进度事件，并在元数据中保存策略哈希、价格口径、数据版本、VectorBT 版本和校准信息。

分块回测时，`chunked._prepare_chunks` 先按资源块调用一次提供者，`run_windows` 把目标权重和
排名按调仓段传给 `run_backtest(target_override=...)`。提供者路径的风险上限只在这个执行入口
统一施加，避免准备阶段先封顶导致风险统计被清零；因子路径仍沿用原有的准备和执行口径。

## `pipeline.py` 与 `store.py`

### `pipeline.py` 调用链

```text
CLI
  -> BacktestStore.get_version
  -> effective_spec
  -> BacktestStore.start_run
  -> load_market_data
  -> Thread(_run_in_thread)
  -> Queue 阶段事件
  -> finish_run / fail_run / cancel_run
```

`effective_spec()`（`pipeline.py:46-61`）用 `replace()` 生成运行级策略，不污染数据库中的原版本。`iter_run()`（`pipeline.py:79-175`）将长任务拆成可观察阶段，并保证异常和取消写入运行状态。

### `store.py` 表和生命周期

策略表：

- `strategy.definition`：策略 ID、名称、当前版本；
- `strategy.version`：每个版本的 JSON、哈希和时间。

运行表：

- `bt.run`：运行 ID、策略版本、区间、资金、状态、元数据、错误和时间。

结果表：

- `bt.daily`、`bt.position`、`bt.order`、`bt.rejection`、`bt.target_weight`、`bt.holding`、`bt.liquidation`。

`finish_run()`（`store.py:280-314`）把 pandas 结果转成长表并按 `run_id` 追加；`get_metrics()` 从 `bt.daily` 重算指标，使指标读取不依赖当时是否完整保存了 metadata。

## 6. 测试与剩余风险

测试已经覆盖主要模块和跨模块联动，但本文没有执行测试。静态可见的关键保护包括：

- 数据契约错位保护：`test_data.py`；
- 规则与 VectorBT 联动：`test_engine.py`、`test_orders.py`；
- 信号到目标权重：`test_signals.py`；
- 配置序列化到版本读取：`test_specs.py`、`test_store.py`；
- 指数快照与资源限制：`test_universe.py`。

运行前仍应确认实际环境中的依赖版本、主行情库和因子产物存在，并按项目已有测试命令执行完整测试。本文不把这些未执行事项写成已验证结论。

