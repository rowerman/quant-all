# 回测数据契约

## MarketData

行情与规则面板共用同一个交易日索引和同一套股票列，形状不一致直接报 `BT-201`。

| 字段 | 类型 | 说明 |
|---|---|---|
| `open`、`close` | 交易日 × 股票 `float32` | **原始价**成交/估值价；停牌日为 NaN |
| `can_buy_close` | 布尔宽表 | 收盘可买入（股票池过滤用） |
| `can_buy_open_next`、`can_sell_open_next` | 布尔宽表 | **第 t 行描述的是 t+1 日**开盘可买/可卖 |
| `is_suspended`、`is_st`、`is_delisting`、`is_limit_up`、`is_limit_down` | 布尔宽表 | 当日状态标记；`is_delisting` 表示退市整理期 |
| `total_mv` | `float32` 宽表 | 当日总市值，市值权重用 |
| `factors` | `因子名 → 宽表` | 因子值只能是信号日及之前的信息 |
| `list_date` | 每股上市日的 `Series`（索引为股票列） | 上市满 60 个自然日才可入选 |
| `board` | 每股板块的 `Series`，取值主板/创业板/科创板/北交所 | 来源 `ref.stock_basic.market`；交易单位按它解析，缺失或未知板块直接报错 |
| `delist_date` | 每股退市日的 `Series` 或 None | 回测区间内退市的持仓按此退出并在退市日核销残仓；None 表示不核销 |
| `membership` | 布尔宽表或 None | 指数成分池下为 True 表示当日属于成分 |
| `industry_l1`、`industry_labels` | `int16` 宽表 + 字典或 None | 时点申万一级行业编码与标签；启用行业约束或行业报告时加载 |
| `impact_volatility` | `float32` 宽表或 None | 执行日前 20 个交易日复权收益波动率；平方根冲击开启时加载 |
| `benchmark` | 收盘价 `Series` 或 None | 配置后必须覆盖全部回测交易日，缺失直接报错 |
| `benchmark_info` | 字典或 None | 基准代码、名称、价格/全收益口径与覆盖区间 |
| `versions` | 字典或 None | 行情库与因子库的版本摘要，只进运行元数据 |
| `data_snapshot` | 字典或 None | 本次取数范围的数据分项摘要；运行时会与规则摘要合成完整输入快照 |
| `corporate_actions` | 长表或 None | 除权日公司行动账本，含送转比例、税前每股分红、配股比例/价格与到账日 |

注意两处容易踩错的时点：

1. `can_buy_open_next` / `can_sell_open_next` 是「下一交易日」口径。目标权重已经平移到执行日，
   规则层必须把这两个标记同样平移一格再用（`orders._shifted_flags`）。
2. `close` 在停牌日是 NaN。执行价用 `open`（缺失时不成交），估值价用 `close.ffill()`，
   避免整块净值被停牌打断。

## 目标权重提供者输入契约

普通提供者实现 `build(data, eligible, params)`；需要真实持仓的提供者声明
`uses_position_state=True`，实现 `build(data, eligible, params, state)`，其中：

- `data` 是当前回测面板的 `MarketData`，`data.factors` 只包含
  `signal.factors` 声明的输入因子；提供者需要的因子必须在策略配置中列出，并由
  `required_factors` 校验。
- `eligible` 是与 `data.index`、`data.columns` 完全一致的布尔宽表，已经统一包含可买入收盘、
  非停牌、非 ST、非退市整理期、上市满 60 个自然日和指数成分过滤。提供者不能自己放宽。
- `state` 是上一遍账本执行结果的逐日 `ProviderState`：现金、组合价值、持仓股数、
  平均成本、估值价、权重、浮动收益、建仓以来最高收盘价、建仓日和最近清仓。
  提供者只能读取状态，不能自行模拟成交或持仓。
- 提供者不获得资源块边界之前的完整行情。需要滚动历史的策略必须使用因子库中已算好的因子，
  不能用块内 `close` 自行滚动，否则单块和分块可能不一致。

返回宽表必须与输入面板同形：

- 整行 `NaN` 表示当天不调仓，保持原持仓；
- 全 0 行表示清仓；
- 非空行必须是有限、非负权重，单行合计不超过 1，剩余部分保留为现金；
- 正权重只能出现在 `eligible=True` 的股票上，当前仍只支持多头。

违反形状、部分 `NaN`、负权重、无穷值、合计超过 1 或 eligible 限制时返回 `BT-201`。
提供者名称、版本、参数和必需因子在创建/运行前校验；`params` 必须可 JSON 序列化。
新建策略页使用 `trading/backtest/strategies/_param_schema.py` 中的扁平字段说明生成输入控件，
由 `trading/backtest/provider_schema.py` 校验；该说明只影响页面展示，不改变提供者源码哈希。
创建提供者策略时，策略版本与 `bt.provider_revision` 同事务提交，源码冲突不会留下半成品策略。

读取外部模型或数据产物的提供者可以实现可选的
`artifact_snapshot(params) -> dict`。该结果只进入回测输入快照，不改变策略参数结构。
`ml_rank` 用它记录模型文件、训练数据集、特征、折、种子和预测的实际哈希。

## 输入快照

每次运行都会生成内容寻址的 `snapshot_id`，并写入 `bt.input_snapshot`：

- `scope` 固定起止日、股票池、股票并集摘要、因子清单、基准、行业开关、
  `price_adjust=raw` 与 `state_adjust=raw`。
- `market`、`calendar`、`universe`、`factors`、`corporate_actions`、`industry`、
  `impact_volatility`、`benchmark` 分别记录精确行集的 SHA-256；未加载的行业、冲击波动率
  或基准记为关闭。
- `rules` 记录策略完整配置与 `strategy_hash`、板块规则版本与规则表、成本模型版本与
  全部历史费率表、退市处置版本、`corporate_action_model=cn_a_corporate_action_v2`、
  `rights_issue_policy=cash_limited`，以及 `execution_model`。
- `execution_model` 包含执行模型版本、`backend=ledger`、`mode=causal_segmented`、
  核心执行模块逐文件 SHA-256 和源码集合摘要。引擎、账本、分配、风控、成本、信号等
  代码变化会自动改变 `snapshot_id`。
- 提供者策略在 `rules.provider` 中额外保存名称、版本、参数、必需因子、模块名、
  完整源码和源码 SHA-256。同一名称与版本出现不同源码时抛 `BT-701`。
- 实现 `artifact_snapshot()` 的提供者还会在 `rules.provider.artifacts` 保存外部产物摘要；
  没有实现该方法的提供者保持原有结构。`ml_rank` 的产物摘要包含实际模型文件、
  训练数据集和预测内容哈希，任一内容变化都会改变最终 `snapshot_id`。

`snapshot_id` 只对 `schema_version/scope/components` 做规范化 JSON 的 SHA-256；
`created_at`、构建清单时间和 `provenance` 不参与。因此相同内容的数据库重建会保持同一
快照，运行区间之外的数据刷新也不会改变快照。内容摘要不复制原始数据：区间内数据被修订时
会生成新快照，旧运行保留旧摘要但不能用旧摘要重放新数据。

当前快照 schema 为 `backtest_input_snapshot_v5`；旧 `v1/v2/v3/v4` 快照仍按原 JSON 读取，
不重算、不回填。

## 价格口径

价格字段使用 `md.daily_bar` 的**原始价**，交易标记与 `total_mv` 同样使用原始口径；
`metadata.price_adjust` 记录本次运行的口径，历史运行按它区分。

- 公司行动不再通过前复权价格隐式处理：除权日按 `ref.corporate_action` 调整持股、
  确认税前现金分红并在到账日进入现金；配股按现有现金自动认购，不足部分不融资。
- 默认基准使用全收益指数 `H00300.CSI`，组合分红与到账现金由账本单独记录；
  价格指数仍可显式选择，用于复核股息收益差异。
- 读取面板前先检查交易日历、日线表和基准的实际覆盖；可预期的缺日起止问题在创建
  运行记录前报 `BT-201`，不会先建一条失败运行再报错。
- 旧运行可能是 `qfq` 口径；新旧结果不能直接比较，需结合 `metadata.price_adjust`
  和公司行动摘要判断。

## 取数

```python
from backtest import load_market_data

data = load_market_data("2024-01-01", "2026-10-15", ["MOM_20", "EP_TTM"],
                        universe="index:000905.SH", benchmark_code="H00905.CSI")
```

- 逐字段读取并转 `float32` 宽表：全 A 三年实测峰值 3.2GB，一次性读全字段长表会明显更高；
- 指数成分池先把**成分并集**传给读取层，只取可能用到的股票；
- 取数前按交易日数与股票数做规模预检（`BT-401`），不要绕过；
- 因子按需加载：`spec.signal.factors` 里有哪些就只读哪些。

## 落库口径

`data/databases/backtest.duckdb` 里的 `bt.daily` / `bt.holding` / `bt.order` / `bt.rejection` /
`bt.target_weight` / `bt.position` / `bt.allocation` / `bt.liquidation` /
`bt.corporate_action` / `bt.industry_exposure` 都按 `run_id` 分区存储：

- `daily`：组合价值、现金、持仓市值、收益、净值、回撤、成本分项、换手与基准列，索引名为 `trade_date`；
  `cost = commission + stamp_tax + transfer_fee + dividend_tax` 是现金成本，
  `total_cost = cost + slippage_cost + impact_cost` 是含滑点和冲击的经济成本；
  `buy_value` / `sell_value` / `turnover` / `turnover_one_way` 分别是买卖成交额、
  双边成交额换手和单边换手；
  `writeoff_size` / `writeoff_value` 是当日零价值核销股数与核销前参考市值；
  `rights_subscription_cash` / `rights_subscribed_shares` / `rights_unsubscribed_shares`
  分别是配股认购现金、实际认购股数和现金不足未认购股数；
- `holding`：逐日逐股持仓市值与权重，只存有仓位的格子；
- `order`：账本成交明细（`stock_code`、`trade_date`、`Size`、`Price`、`Fees`、`Side`、
  `Board`、`Order Valid`、`reference_price`、`gross_amount`、`commission`、`stamp_tax`、
  `transfer_fee`、`slippage_rate`、`slippage_cost`、`impact_rate`、`impact_cost`、
  `total_cost`、`forced_exit`）；
  `forced_exit=True` 表示退市整理期或退市日的强制退出成交；
  `Fees = commission + stamp_tax + transfer_fee`，
  `total_cost = Fees + slippage_cost + impact_cost`；
- `rejection`：规则拒单（`trade_date`、`stock_code`、`reason`），一事一条——
  被拦下的格子之后每个交易日重试，同一事件的后续失败不再重复写行，次数记在校准的重试统计里；
  `reason` 取 `suspended`（停牌）、`no_price`（缺执行日开盘价）、
  `limit_up_or_not_buyable`、`limit_down_or_not_sellable`。停牌与缺价都不会成交；
  `reject_suspended=False` 只是不写停牌拒单，订单仍保留为待重试；
- `liquidation`：退市残仓零价值核销明细（`trade_date`、`stock_code`、`delist_date`、
  `size`、`price`、`value`、`reference_price`、`reason`）；`price` 和 `value` 固定为 0；
- `target_weight`：实际提交的目标权重（只存下单的格子）；
- `allocation`：资金分配明细（`trade_date`、`stock_code`、`target_weight`、`price`、
  `baseline_shares`、`allocated_shares`、`board`、`order_shares`、`order_valid`、`status`），
  `status` 取
  `allocated`（正常分配）/ `bumped`（用取整余量补到最小申报量）/
  `unaffordable`（最小申报量都装不下）/
  `below_min_order`（持仓差额不足最小申报量）/ `no_price`（缺执行价）/ `flat`（目标 0）；
- `position`：账本按买入批次形成的往返持仓记录（含建仓/平仓时间、分红和费用），
  **不是**逐日持仓，逐日持仓看 `holding`。
- `corporate_action`：逐日公司行动明细（送转、分红确认、到账、红利税、零头折现和配股认购），
  红利税行同时记录 `tax_rate`；配股行记录配股价、可配股数、实际认购和未认购股数。
- `industry_exposure`：逐日行业暴露（`trade_date`、`industry`、`market_value`、`weight`、
  `is_cash`），包含现金与未分类行，用于报告页面和指标重算。

## 执行口径

最终一遍按**目标股数**下单，目标股数由 `trading/backtest/allocation.py` 按
「执行日开盘时刻的真实现金与持仓」算出，并预留 `execution.cash_buffer`（默认 2%）。
买入按所属板块校验：主板/创业板 100 股整数倍，科创板不少于 200 股且可按 1 股递增，
北交所不少于 100 股且可按 1 股递增；卖出遵守最小申报量，清仓可一次卖出全部零股。
`MarketData.amount_cny` 是执行日原始成交额，用于默认启用的
`execution.max_participation_rate=0.1` 容量约束；退市强制退出同样受此约束。
它必须与其他行情面板共享日期和股票列。
容量约束先限制每格持仓变化；卖单缩量后，再用扣除 FIFO 红利税的实际卖出净所得计算同日买单预算。
运行元数据的 `cost_model` 字段记录成本模型版本、历史费率表、取整方式和旧固定税率覆盖；
`execution` 字段记录 `sizing`、`cash_buffer`、成交额参与率、旧字段 `lot_size`、
`lot_rule_version`、板块规则摘要、退市处置规则与分配迭代次数。`lot_size` 仅为旧配置兼容，
不再控制交易单位。
用来区分 2026-10-15 之前的「目标百分比、无缓冲」口径。细节与实测对照见
`vectorbt_adapter.md`。

