# 策略配置

`backtest.specs.StrategySpec` 序列化成稳定 JSON，`config_hash` 是它的 SHA-256 前 16 位。
策略版本只增不改：同一 `(strategy_id, version)` 的配置永不变，回测运行必须绑定具体版本。

## 字段

| 段 | 支持值 | 说明 |
|---|---|---|
| `name`、`version` | 非空字符串 / 正整数 | 策略名与版本 |
| `universe` | `all_a` 或 `index:<指数代码>` | 全 A 可交易池，或任意已收录指数成分池 |
| `signal.factors`、`signal.weights` | 因子名序列 + 同长度权重 | 多因子线性组合，权重可为负；使用提供者时 factors 是输入因子，weights 必须为空 |
| `signal.provider` | `ProviderSpec` 或省略 | 外部目标权重提供者；`name`、`version` 必须匹配 `trading/backtest/strategies/` 中注册的对象，`params` 是 JSON 对象 |
| `signal.winsorize`、`signal.standardize` | 布尔 | 3×MAD 去极值、截面标准化；仅因子策略使用，提供者策略忽略 |
| `portfolio.selection`、`top_n` | `top_n` 或 `provider` + 正整数 | 因子策略取分数最高的 N 只；提供者策略由提供者返回权重，top_n 不参与计算 |
| `portfolio.weighting` | `equal`、`market_cap` | 等权，或选中股票内按当日总市值归一；提供者策略忽略 |
| `portfolio.long_only` | 必须为 True | 只做多 |
| `rebalance.frequency` | `daily`、`weekly`、`monthly`、`custom` | 因子策略使用前三种；提供者策略使用 `custom`，由提供者返回的矩阵决定调仓日 |
| `rebalance.signal_at`、`execute_at` | `close` / `next_open` | 收盘生成信号、次一交易日开盘执行 |
| `execution.lot_size` | 固定 100 | 旧配置兼容字段；实际交易单位按股票板块解析，该字段不再控制下单 |
| `execution.cash_buffer` | `[0, 0.5)` 的小数，默认 0.02 | 执行层预留的现金比例；分配目标股数时按 `1 - cash_buffer` 计算可用预算 |
| `execution.max_participation_rate` | `None` 或 `(0, 1]` 的小数，默认 `0.1` | 单票单日未复权成交额参与率；容量不足时按可成交金额缩小并在后续交易日重试 |
| `execution.reject_*` | 布尔 | 停牌、涨停买入、跌停卖出的拒单记录开关；关闭停牌开关后订单仍静默顺延到复牌，不伪造成交 |
| `execution.enforce_t_plus_1` | 布尔 | 订单层限制卖出不得超过上一交易日收盘持股；当前日频路径结构上不会同日双向 |
| `risk.max_single_weight` | `None` 或 `(0, 1]` | 单股目标权重上限；超出部分在同一入选池内按原权重比例再分配，不能分配的部分保留为现金 |
| `risk.max_industry_weight` | `None` 或 `(0, 1]` | 时点申万一级行业目标权重上限；`2021-12-13` 前用 SW2014、之后用 SW2021，与单股上限联合使用剩余容量，异常行业数据直接报错 |
| `risk.max_one_way_turnover` | `None` 或 `[0, 1]` | 含现金的单边计划换手上限；超限时沿当前组合到目标组合整体缩放 |
| `costs` | 佣金、最低佣金、滑点、冲击、历史税费 | 印花税和过户费按成交日自动分段；`dividend_tax_mode` 取 `fifo_deferred`（默认）或 `disabled`；新策略默认 `impact_model=sqrt`，旧 JSON 缺少冲击字段时解释为关闭；`stamp_tax_rate` 仅保留为旧策略 JSON 的固定卖出税率覆盖 |
| `benchmark.index_code` | 指数代码或 None | 默认 `H00300.CSI`；可选价格指数或全收益指数 |

运行级覆盖：不传 `benchmark` 时沿用策略版本；传 `null` 或空字符串表示本次关闭基准，
CLI 使用 `--no-benchmark`。关闭后不生成基准和超额列，策略版本本身不变。
运行级参数 `risk_free_rate` 是年化小数，默认 `0.02`，只影响指标，不进入策略版本哈希。

## 内置策略

内置策略就是预先写进策略表的普通版本，用固定标识，与后续前端新建的策略走同一条运行路径：

| 标识 | 配置 |
|---|---|
| `builtin-mom20-top30` | `MOM_20` 全 A、Top-30、等权、月频、基准 H00300.CSI |
| `builtin-multifactor-top30` | `MOM_20 + EP_TTM - VOL_20` 全 A、Top-30、等权、月频、基准 H00300.CSI |

`BacktestStore.ensure_builtins()` 幂等：标识已存在且定义一致就不动；定义变了就追加新版本并推进
`current_version`，历史版本仍可复现。

## 前端新建策略

`#/backtest/new` 使用现有 `POST /api/strategies` 创建第一版，不存在另一套前端专用配置。
页面可在“因子 Top-N”和“目标权重提供者”之间切换。因子模式开放名称、说明、股票池、基准、
因子与权重、标准化/去极值、Top-N、权重方式、调仓频率；提供者模式按参数结构生成字段，
自动带入必需因子、`selection=provider` 和 `frequency=custom`，其他编辑项自动隐藏。
两种模式共用现金缓冲、成交额参与率、三类规则拦截、T+1 与佣金/最低佣金/滑点配置；
印花税和过户费由系统按历史分段。
新建策略页同时可配置平方根冲击模型与冲击系数，其他冲击参数使用默认值并写入运行元数据。
固定能力保持为只做多、按板块申报单位、收盘生成信号、下一交易日开盘执行。

提供者模式先调用 `POST /api/providers/validate` 校验并取得规范化参数，再创建策略。
创建时策略版本与提供者源码修订在同一事务中写入；两者都成功后才返回回测页、刷新选项并选中新策略。
前端做即时范围检查，`StrategySpec.from_dict` 仍是最终校验来源；策略版本只增不改，
页面不允许覆盖内置策略或历史版本。

## 通用目标权重提供者

P1-2 起可以在 `trading/backtest/strategies/` 新增 `*.py` 模块，模块导入时调用
`backtest.providers.register_provider`。回测启动时自动发现该目录；文件名以下划线开头的模块
不会被自动导入。提供者适合实现择时、配对、动态目标权重等“只依赖行情/因子”的日频策略。

提供者接口：

```python
class MyProvider:
    name = "my_provider"
    version = "1"
    description = "说明"
    required_factors = ("MOM_20",)

    def build(self, data, eligible, params):
        # data: MarketData；eligible: 交易日 × 股票布尔矩阵；params: 策略参数
        # 返回与 data.index、data.columns 完全一致的 DataFrame
        ...
```

返回矩阵口径：

- 整行 `NaN` 表示当天不调仓；全 0 行表示清仓；
- 非空行必须是有限、非负权重，单行合计不超过 1，剩余部分保留为现金；
- 正权重只能落在 `eligible=True` 的股票上；当前仍只支持多头；
- 排名由引擎按正权重降序派生，用于 `rank_rows` 审计；
- 提供者必须确定性、因果、只读输入；跨资源块的滚动历史应使用已落盘因子，
  不能依赖资源块边界之前的原始价格。

### 提供者实现注意事项

- 新建策略页的结构化字段定义在 `trading/backtest/strategies/_param_schema.py`，按
  `(name, version)` 索引；`trading/backtest/provider_schema.py` 负责字段校验和默认值检查。
  字段支持整数、普通数值、百分比、布尔、字符串和枚举；百分比字段在后端仍保存为小数。
  未声明结构的提供者仍可通过 HTTP API 创建，页面显示无参数可编辑。
- 参数结构属于页面元数据，不写入提供者模块源码，因此给既有 v1 提供者补表单不会改变
  `code_sha256` 或既有策略的源码绑定。
- `params` 必须是可以稳定 JSON 序列化的对象，不能包含 `NaN` 或无穷值；`name`、`version`
  必须与注册对象一致，未知名称、版本不匹配和缺少 `required_factors` 都会在创建或运行前报
  `BT-101`。
- provider 注册表允许同一 `name` 的不同 `version` 并存；显式 `ProviderSpec.version`
  取对应版本，前端列表默认展示最新版本。历史策略仍可按原版本取得原实现。
- 研究基线还提供一个不进入常规发现列表的 `factor_rank` 提供者，源码位于
  `trading/backtest/strategies/_factor_rank.py`，参数为 `top_n` 和 `factor_names`。它只在周最后
  交易日输出 Top-N 等权目标，用于 P0 的全因子 rank 基线；文件名以下划线开头，不会出现在
  前端常规提供者列表中。
- `ml_rank` 读取 `alpha` 已落盘模型预测，参数为必需 `model_id` 和默认 30 的 `top_n`。
  它在周最后交易日对预测做当前及此前 3 个信号日的 rank 平滑，再在当前可交易池内
  选 Top-N 等权。实现位于 `trading/backtest/strategies/ml_rank.py`，详细口径见
  `docs/trading/backtest/ml_rank.md`。
- `ml_rank` v2 另接受可选 `rebalance_dates`。提供时只在显式日期调仓，用于对齐
  walk-forward 各年测试窗口实际存在的预测日期；v1 保持“缺失自然周预测直接报错”。
- `ml_rank` v3 读取 P5-3 生命周期包，在 `normal`、`reduced`、`rollback` 和
  `paused` 状态间切换主模型、半仓、Ridge 或清仓；v1/v2 保持原行为。
- 隐藏的 `factor_rank` v2 同样接受可选 `rebalance_dates`，供 P4-4A 让 B4 与 ML
  候选使用完全相同的调仓日。
- 读取外部产物的 provider 可实现可选 `artifact_snapshot(params)`；返回内容写入
  输入快照的 `rules.provider.artifacts`，不改变策略参数结构或 provider 版本。
- 提供者代码改动必须提升 `version`。创建策略时即登记模块源码、源码 SHA-256、名称、版本和
  必需因子；输入快照继续保存这些信息及参数。同一名称与版本出现不同源码时，创建或运行直接
  返回 `BT-701`。
- 提供者只能看到行情、因子和 `eligible` 矩阵，看不到真实现金、持仓和订单；风险上限、
  订单状态机和账本都在执行层统一处理，提供者不要提前按风险上限改写目标。
- 提供者路径在分块准备阶段只调用一次信号，之后各调仓段复用同一份目标；跨资源块的滚动
  计算必须使用已落盘因子，否则单块与分块可能不一致。
- 示例 `momentum_timing` 只是可用提供者，不是内置策略；前端新建策略页与
  `POST /api/strategies` 共用同一条创建路径。

策略配置示例：

```json
{
  "name": "广度择时",
  "signal": {
    "provider": {
      "name": "momentum_timing",
      "version": "1",
      "params": {"top_n": 30, "min_breadth": 0.5}
    },
    "factors": ["MOM_20"],
    "weights": []
  },
  "portfolio": {"selection": "provider", "top_n": 30},
  "rebalance": {"frequency": "custom"}
}
```

仓库自带 `momentum_timing` 示例：广度达标时持有 `MOM_20` 分数最高的 Top-N 等权，
否则清仓。另有 `momentum_trailing_stop`：每周调仓，持仓触发固定止损或移动止损时，
在信号日收盘把目标权重降到 0，次一交易日开盘卖出。两者只注册为可用提供者，不自动成为
内置策略。`momentum_trailing_stop` 的 `cooldown_days` 按自然日计，用于避免清仓后
立即买回同一只股票。

需要真实持仓的提供者声明 `uses_position_state = True`，`build` 签名为
`build(data, eligible, params, state=None)`。`state` 是上一遍账本执行结果的逐日快照：
现金、组合价值、持仓股数、平均成本、估值价、权重、浮动收益、建仓以来最高收盘价和最近
清仓。提供者只能读取状态，不能自行模拟持仓。第一次运行 `state=None`，提供者必须能只凭
行情、因子和股票池给出基准目标。

状态反馈在流水线层按固定点执行：用上一遍最终账本状态重算目标，直到目标权重逐位重复；
最多 20 遍，未收敛直接报错，不输出半自洽结果。止损和再入场会把状态变化逐周传下去，
实测短区间需要 11 轮，运行耗时明显高于普通提供者。前端新建策略页可直接编辑
`momentum_trailing_stop` 的止损、移动止损和冷却天数，并显示其会读取真实持仓状态。

## 运行级覆盖

一次运行可以覆盖 `universe` 与 `benchmark`，用来「同一个策略换池子跑一遍」。
策略版本本身不变，实际执行的配置快照、数据分项摘要与 `snapshot_id` 写进运行元数据，
保证可复现。旧运行没有 `snapshot_id` 时仍可查询，但不会补算当时的数据摘要。

## 股票池口径

两套池子共用同一套可交易性过滤（可买入收盘、非 ST、上市满 60 个自然日），
与 `calculate.universe` 一致，不要在回测侧重写：

- 全 A：全部可交易股票；
- 全 A 在 `2021-11-15` 前排除北交所代码，与 P0/P3 周频研究模拟器一致；
- 指数池：用**上一个已结束月份**的月末权重快照，避免用到当月才披露的权重（前视）。

## 已知口径

- 目标权重合计 1.0，执行层另预留 `cash_buffer`（默认 2%）：最终一遍按分配出的目标股数下单，
  现金缓冲覆盖费用、滑点与取整余量，不再出现「队尾格子被整笔裁掉」；
- 提供者策略的目标权重可以留现金（单行合计小于 1）；`NaN` 行保持原持仓，`0` 行清仓；
  风险上限仍由执行层统一施加；
- 新策略默认 `max_participation_rate=0.1`，执行日每只股票的订单金额不超过
  `amount_cny × max_participation_rate`；`amount_cny` 使用未复权原始成交额，缺失或为零时当日拒绝并重试。
  旧策略 JSON 没有该字段时仍按 `None` 解释，历史版本哈希不变；
- 退市整理期股票不进入信号股票池、不允许买入；已持仓股票从首个可卖日开始按停牌/跌停规则
  每日尝试退出，退市日仍未卖出的残仓按零价值核销，不产生现金、费用或红利税；
- 分配明细落 `bt.allocation`，逐格给出目标股数、板块、订单股数、合法性、补最小申报量与未建仓记录；
- 交易成本统一由 `trading/backtest/costs.py` 计算：佣金按笔最低 5 元；印花税按历史生效日期区分双边/卖出单边；
  过户费按沪深北市场和日期分段，2015-08-01 前沪市按成交面额、深市按成交金额；
  红利税按卖出日政策和 FIFO 持有期结算；佣金、印花税、过户费逐笔按分四舍五入；
- 新策略默认启用平方根冲击：`impact = min(0.5 × 20日复权波动 × sqrt(成交额 / 当日成交额), 5%)`；
  成交额或波动率缺失时冲击为 0。冲击与固定滑点直接体现在成交价里；
- `bt.daily.cost` 是现金成本（佣金、印花税、过户费、红利税），
  `bt.daily.total_cost = cost + slippage_cost + impact_cost`；滑点和冲击不能再从现金扣一次；
- 主板、创业板买入按 100 股整数倍；科创板买入不少于 200 股并按 1 股递增；北交所买入
  不少于 100 股并按 1 股递增。卖出不足最小申报量时保持原仓位，清仓允许卖出全部零股。
- 同一目标连续两次不变时按「再平衡」提交（方向未知），买卖任一侧受限就跳过该格；
  因此停牌/涨跌停日的漂移再平衡会被跳过，这是有意的保守选择。
- 停牌和缺少执行日开盘价时现金、持股和规则状态都不变，待成交目标逐日重试；
  `no_price` 属于数据问题，始终记录拒单并重试，不受 `reject_suspended` 开关影响。
- 风险约束默认关闭；行业归属取信号日的时点申万一级行业，闭区间按
  `in_date <= signal_date <= out_date` 生效。`2021-12-13` 前使用 SW2014，之后使用 SW2021；
  SW2021 完全没有记录时沿用最近已知行业，并在运行元数据中标记 carry-forward；最早一条记录
  晚于上市日时按该记录向前补到上市日，标记 `backfill_earliest_*`。
- 行业来源重叠已在建库时解析成唯一时点区间；启用行业上限后，目标股票仍缺失行业或同日冲突时
  回测直接报错，不使用 `stock_basic.industry` 静态分类兜底。

## 口径变更与旧运行

2026-10-15 起最终一遍从「目标百分比」改为「目标股数 + 现金缓冲」，
运行元数据里新增 `execution`（`sizing`、`cash_buffer`、`lot_rule_version`、板块规则摘要、
迭代次数与是否收敛）用于区分。`lot_size` 仅用于识别旧配置。
旧运行是「目标百分比、无缓冲」口径，两者不可直接比较：同一份数据上，
全 A 三年累计收益 −42.62% → −54.08%、中证500 十年 +39.89% → +45.53%，
差异主要来自旧口径被动留了 10% 以上的现金（见 `docs/trading/backtest/vectorbt_adapter.md` 的实测表）。

