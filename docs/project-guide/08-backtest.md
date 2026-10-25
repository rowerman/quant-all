# 回测

回测模块把策略版本、行情、因子和股票池转换成逐日目标权重，再通过资金分配和原始价账本
模拟 A 股成交、公司行动、费用、持仓和绩效。正式执行路径已经不再用 VectorBT 记账。

核心源码位于 [trading/backtest](../../trading/backtest)。

## 1. 技术栈

| 技术 | 用途 |
|---|---|
| pandas / NumPy | 日期 × 股票面板、订单、持仓和资金迭代 |
| DuckDB | 行情读取、回测结果和策略版本存储 |
| Parquet | 因子值读取 |
| VectorBT 0.28.5 | 历史基础库和接口兼容参考 |
| 自定义 `LedgerPortfolio` | 当前原始价成交、现金、股数和税务批次账本 |
| FastAPI / SSE | 页面提交回测并流式报告阶段 |

## 2. VectorBT 基础库

### 2.1 上游定位

VectorBT 是面向数组化回测和策略分析的 Python 库。当前锁定版本 0.28.5，包摘要为
“Python library for backtesting and analyzing trading strategies at scale”，
上游地址：

```text
https://github.com/polakowo/vectorbt
```

它的主要设计特点是：

- 输入通常是交易日 × 标的的 pandas 数组。
- 用向量化订单和组合状态批量处理数百到数千个标的。
- `Portfolio.from_orders()` 接受价格、订单大小、订单类型、现金共享、手续费和成交序列。
- 支持 `targetpercent`、`targetamount`、`amount` 等订单口径。
- 自带组合价值、现金、持仓和订单记录接口，便于计算绩效。

### 2.2 项目早期为什么使用它

最初适配层用 VectorBT 组合对象承载目标权重、成交和每日估值。项目在其上层实现：

- A 股股票池和执行日信号。
- 涨跌停、停牌和申报单位。
- 资金分配和现金缓冲。
- 基准和绩效报告。

### 2.3 为什么改成自定义账本

普通 VectorBT 组合不能完整表达 A 股公司行动：

- 现金分红应收与到账。
- 送转、拆并和零头折现。
- 配股认购和未认购股数。
- FIFO 红利税批次。
- 退市整理期退出和零价值核销。

历史实现还遇到两个具体问题：

1. `targetpercent` 遇到最小申报单位时可能整只股票买不进。
2. 目标百分比满仓加费用后，后处理订单可能因现金不足被整笔裁剪。

因此项目保留了 VectorBT 风格的 `from_orders()` 和结果接口，但实现
[trading/backtest/ledger.py](../../trading/backtest/ledger.py) 的 `LedgerPortfolio`。
当前运行时 `engine.py` 不调用 VectorBT 做交易记账；依赖仍锁定，回测 metadata 仍记录
`vectorbt_version` 作为环境追踪信息。

### 2.4 “二次开发”的实际含义

本项目不是给 VectorBT 写一个简单参数包装器，而是经历了两个阶段：

| 阶段 | 关系 |
|---|---|
| 历史阶段 | 使用 VectorBT 组合执行，外层增加 A 股规则、成本和结果标准化 |
| 当前阶段 | 保留部分接口形状，使用自定义账本取代正式现金与持股记账 |

因此阅读代码时要区分：

- `LedgerPortfolio.from_orders()` 是现役实现。
- `vectorbt` 包是历史基础和锁定依赖。
- `BacktestResult.portfolio` 可以是兼容接口的组合对象，不代表当前由 VectorBT 计算。

## 3. 提供功能

### 3.1 策略版本

`StrategySpec` 包含：

- 股票池。
- 因子线性组合或目标权重 provider。
- Top-N 或 provider 选择规则。
- 等权或市值权重。
- 日/周/月/custom 调仓。
- 收盘信号、次日开盘执行。
- 现金缓冲、申报名义和成交额参与率。
- 佣金、最低佣金、滑点、冲击、印花税、过户费和红利税。
- 单股、行业和换手风险约束。
- 基准。

策略版本写入 `data/databases/backtest.duckdb`，只增不改。内置策略定义变化时追加新版本。

### 3.2 股票池

- `all_a`：全 A 可交易。
- `index:<代码>`：上一个已结束月份的月末权重快照 ∩ 可交易过滤。

回测读取层只取股票池的成分并集，并在信号日再次应用可交易规则。

### 3.3 信号

因子模式：

1. 读取因子宽表。
2. 可选 3 倍 MAD 去极值和截面标准化。
3. 按方向和权重线性组合。
4. 在合格股票中选 Top-N。
5. 等权或按总市值归一。

Provider 模式：

```python
class MyProvider:
    name = "my_provider"
    version = "1"
    required_factors = ("MOM_20",)

    def build(self, data, eligible, params):
        ...
```

返回矩阵与 `MarketData.index/columns` 完全一致：

- 整行 `NaN`：当天不调仓。
- 全 0 行：清仓。
- 正值：目标权重。
- 单行合计不超过 1，剩余为现金。
- 只允许多头，只允许在 `eligible=True` 的股票上持有。

需要真实持仓的 provider 设置 `uses_position_state=True`，可读取上一遍账本的现金、持股、
成本、权重、浮动收益、历史最高价和最近清仓。

### 3.4 风控

- 单股权重上限。
- 时点申万一级行业权重上限。
- 含现金的单边换手上限。

单股或行业超限后在同一入选池重新分配；换手约束按当前真实持仓向目标组合整体缩放。

### 3.5 交易规则

- 主板买入 100 股整数倍。
- 科创板买入至少 200 股、按 1 股递增。
- 北交所买入至少 100 股、按 1 股递增。
- 停牌、缺执行价、涨停买入、跌停卖出会被拒绝。
- 被拒绝的订单在后续交易日重试，直到成交或被新目标覆盖。
- T+1 卖出量不超过上一交易日收盘持仓。
- 退市整理期不买；持仓从可卖日起每日尝试退出；退市日残仓按零价值核销。

### 3.6 成本

`CostSpec` 统一计算：

- 佣金和最低佣金。
- 固定滑点。
- 平方根冲击。
- 历史印花税。
- 历史过户费。
- FIFO 递延红利税。

费用由资金分配、账本和归因共用同一模型，不能在不同模块另写费率。

### 3.7 公司行动

- 送转/拆并按倍数调整持股，零头按除权日开盘价折现。
- 现金分红在除权日确认应收，到账日转现金。
- 配股只在现有现金范围内认购，不融资。
- 买入形成税务批次；卖出按 FIFO 和持有期结算红利税。

### 3.8 结果

`BacktestResult` 包含：

- `daily`：净值、收益、回撤、现金、成本和换手。
- `orders`：成交和费用分项。
- `positions`：已平仓批次。
- `holdings`：逐日逐股持仓市值和权重。
- `target_weights`：提交的目标。
- `rejections`：规则拒单。
- `liquidations`：退市零价值核销。
- `allocations`：目标股数、容量和补回明细。
- `corporate_actions`：公司行动明细。
- `industry_exposure`：逐日行业暴露。
- `metrics` 和 `metadata`。

## 4. 一次回测从头到尾如何运作

### 4.1 运行入口

```text
CLI 或 Web
  -> BacktestStore 读取不可变策略版本
  -> 应用运行级 universe/benchmark 覆盖
  -> validate_window 检查数据覆盖
  -> build_data_snapshot 摘要区间内数据
  -> build_input_snapshot 合并策略、规则和外部产物哈希
  -> start_run 写 pending/running 状态
  -> estimate_panel 决定单块或分块
  -> load_market_data
```

### 4.2 输入数据

`MarketData` 固定所有面板共用交易日索引和股票列：

- 原始 `open`、`close`。
- `can_buy_close`、`can_buy_open_next`、`can_sell_open_next`。
- 停牌、ST、涨跌停、退市整理期。
- 市值和成交额。
- 因子字典。
- 上市日、退市日和板块。
- 指数成分矩阵。
- 行业、冲击波动率和基准。
- 公司行动和输入快照。

价格为原始价，避免把前复权形成的“合成股数”当作真实交易股数。

### 4.3 信号

收盘数据计算目标权重。`execution_targets()` 把目标权重平移到下一交易日执行。

`NaN` 表示不提交订单，0 表示清仓，正数表示目标权重。提供者只能输出目标，不自行模拟
成交。

### 4.4 三遍账本

#### 探针遍

- 未过滤目标、零费用。
- 只用于得到每格预期买卖方向和成交估计。

#### 基准遍

- 应用停牌、涨跌停和申报规则。
- 按 `1 - cash_buffer` 的目标百分比下单。
- 带完整费用。
- 提供逐日真实现金和持股，作为资金分配预算。

#### 最终遍

- 按分配出的目标股数下单。
- 带完整费用、公司行动和退市核销。
- 是权威成交、现金、持仓和净值的来源。

### 4.5 资金分配

对每个执行日计算：

```text
预算 = 上一日现金 + 上一日持股 × 执行价
目标市值 = 预算 × (1 - cash_buffer) × 目标权重
目标股数 = 按板块申报单位向下取整
```

执行价优先用开盘价；停牌时用上一有效收盘价估值。

随后应用：

- 成交额参与率限制。
- 卖出实际净所得限制同日买入。
- 最小申报单位。
- 取整余量补最小订单。
- 现金不足时等比缩小。
- 对实际缺口创建下一交易日补回事件。

### 4.6 固定点迭代

目标股数依赖上一轮最终账本的真实现金和持股，账本结果又依赖目标股数。系统因此反复：

```text
分配目标股数
  -> 运行最终账本
  -> 读取真实成交状态
  -> 重新分配
```

直到目标股数、补回事件和最终结果不再变化，或达到迭代上限。未收敛不会被伪装成成功，
metadata 保留 `allocation_converged=false`。

### 4.7 每个交易日账本

按以下顺序：

1. 收到期现金分红。
2. 应用送转/拆并和零头折现。
3. 按现有现金认购配股。
4. 先卖后买。
5. 按板块和交易规则校验订单。
6. 按原始开盘价、滑点、冲击、佣金、印花税和过户费成交。
7. 更新 FIFO 税务批次并结算卖出红利税。
8. 处理退市残仓核销。
9. 按收盘价估值，停牌沿用上一有效价。

### 4.8 标准化和归因

账本输出转换为逐日日结果：

```text
组合价值 = 现金 + 应收分红 + 持仓市值
return = 组合价值日变化率
nav = 组合价值 / 初始组合价值
drawdown = nav / 累计最高 nav - 1
```

绩效模块从日结果、成交和持仓推导：

- 总收益、年化收益、波动、Sharpe、Sortino、Calmar、最大回撤。
- Beta、Alpha、跟踪误差、信息比率和年化超额。
- 单边/双边换手。
- 成本分项及其占净值/成交额 bps。
- 行业平均、最大、期末权重、现金比例和 HHI。

### 4.9 落库

`BacktestStore.finish_run()` 把日结果、订单、持仓、拒单、目标权重、分配明细、公司行动、
行业暴露和指标在同一事务写入 `backtest.duckdb`，最后把运行标记为成功。任何失败都回滚
本次运行结果并写失败状态。

## 5. 分块回测

当估计面板超过上限时，按时间自动分块：

```text
全段数据摘要和信号口径
  -> 按约 150 万格准备资源块
  -> 按真实调仓执行日划分逻辑段
  -> 每段继承上一段现金、持股、税批次、应收分红和规则状态
  -> 拼接日结果和明细
```

资源块只控制内存。不可跨块的滚动因子应使用已落盘因子，不能依赖块内临时价格历史。

## 6. 内部模块结构

| 模块 | 责任 |
|---|---|
| `specs.py` | 策略、执行、成本、风险和参数契约 |
| `data.py` | `MarketData` 读取和字段对齐 |
| `universe.py` | 全 A、指数成分并集和资源上限 |
| `signals.py` | 因子打分、合格池、调仓日和 provider 路由 |
| `providers.py` | provider 注册、发现、版本和源码哈希 |
| `risk.py` | 单股、行业和换手约束 |
| `allocation.py` | 目标股数、容量、现金和补回 |
| `orders.py` | 规则过滤、订单状态和退市候选 |
| `trading_rules.py` | 板块申报单位 |
| `costs.py` | 历史费率、滑点、冲击和红利税 |
| `ledger.py` | 原始价成交和公司行动账本 |
| `engine.py` | 三遍执行、固定点和结果标准化 |
| `chunked.py` | 分块、状态交接和拼接 |
| `metrics.py` | 绩效指标 |
| `attribution.py` | 换手、成本和行业归因 |
| `snapshot.py` | 数据和规则内容摘要 |
| `store.py` | 策略版本、运行和结果事务 |
| `pipeline.py` | 取数、执行、进度和落库编排 |

## 7. 如何使用

### 7.1 Python

```python
from backtest import BacktestParams, StrategySpec, load_market_data, run_backtest

data = load_market_data(
    "2024-01-01",
    "2024-12-31",
    ["MOM_20"],
    universe="all_a",
    benchmark_code="H00300.CSI",
)
result = run_backtest(
    data,
    StrategySpec(),
    BacktestParams("2024-01-01", "2024-12-31"),
)
print(result.metrics["annual_return"])
```

### 7.2 CLI

```bash
.venv/bin/python -m backtest \
  --strategy mom20 --start 2024-01-01 --end 2024-09-30
.venv/bin/python -m backtest \
  --strategy multifactor --start 2024-01-01 --end 2024-09-30 \
  --universe index:000905.SH --benchmark H00905.CSI
```

### 7.3 Web

```text
#/backtest
#/backtest/new
#/backtest/{run_id}
#/backtest/compare?run_ids=id1,id2
```

页面可以创建策略版本、运行、取消、查看日结果、成交、持仓和归因，并比较 2～4 次成功运行。

## 8. 复现与输入快照

输入快照包含：

- 行情、交易日历、股票池和因子摘要。
- 公司行动、行业和基准摘要。
- 策略配置哈希。
- 规则和执行模型版本。
- 各核心执行文件的 SHA256。
- provider 源码、参数和外部模型产物摘要。

任一区间内实际输入或执行代码变化都会改变 `snapshot_id`。区间外数据刷新不影响本次快照。

## 9. 日志与错误

日志写入 `logs/trading/backtest/<module>/`。常见错误：

| 错误码 | 含义 |
|---|---|
| `BT-1xx` | 策略配置或数据契约 |
| `BT-2xx` | provider 或模型输入 |
| `BT-4xx` | 资源限制、取消或数据对齐 |
| `BT-5xx` | 执行和资金分配 |
| `BT-7xx` | 策略 provider 版本冲突 |

## 10. 边界与修改注意事项

1. 不使用未来数据。收盘信号只能下一交易日开盘执行。
2. 原始价执行，不能把后复权价格当作成交价。
3. 任何费用、滑点或冲击都不能在成交价和现金费用中重复扣减。
4. 被规则拦下的订单必须保留状态并重试，不能静默丢弃。
5. 资金分配必须使用真实现金与持股；不能用计划市值假装已成交。
6. 修改板块规则、成本、公司行动或 provider 版本时，应触发新快照和相应测试。
7. VectorBT 文档中的默认行为不能直接推演本项目结果，现役权威实现是
   `LedgerPortfolio` 和固定点分配。
8. 回测经济结果与研究模拟器结果不能混表比较，因为费用、成交和资金口径不同。

