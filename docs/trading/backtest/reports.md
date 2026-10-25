# 绩效指标与归因

`trading/backtest/attribution.py` 只从最终日结果、成交和逐日持仓推导指标，前端不重算。
默认无风险利率是运行级年化 `risk_free_rate=0.02`，不进入策略配置哈希。

## 收益与风险

- 日无风险利率：`(1 + risk_free_rate) ** (1 / 252) - 1`
- 夏普：日超额收益均值 ÷ 日超额收益标准差 × `sqrt(252)`
- Sortino：日超额收益均值 × `sqrt(252)` ÷ 下行波动，下行波动按
  `sqrt(mean(min(excess, 0) ** 2)) * sqrt(252)` 计算
- Beta：组合日收益与基准日收益的协方差 ÷ 基准日收益方差
- Alpha：`(组合日均超额 - Beta × 基准日均超额) × 252`

## 换手

- 双边成交额换手：`(买入成交额 + 卖出成交额) / 前一日组合价值`
- 单边换手：`max(买入成交额, 卖出成交额) / 前一日组合价值`
- 首日分母取首日组合价值；年化值为逐日换手求和后乘 `252 / 交易日数`
- `metrics.turnover_annual` 保留为双边口径兼容字段，
  新字段为 `turnover_two_way_annual` 与 `turnover_one_way_annual`

## 成本

`daily.cost` 是现金费用（佣金、印花税、过户费、红利税）；
`daily.total_cost = cost + slippage_cost + impact_cost`。滑点和冲击已进入成交价，
不能再作为现金费用重复扣除。成本报告同时给出金额、占平均组合净值 bps 和占成交额 bps，
并按买入、卖出和退市强制退出拆分。

平方根冲击默认参数：

```text
impact = min(0.5 * sigma_20 * sqrt(成交额 / 当日成交额), 0.05)
```

`sigma_20` 取执行日前 20 个交易日 `md.daily_hfq.ret` 的样本标准差；成交额或波动率缺失时
冲击为 0。买入成交价上浮、卖出成交价下调。

## 行业暴露

行业暴露从实际逐日持仓和申万一级时点行业计算，包含现金与未分类行。报告给出各行业
平均权重、最大权重、期末权重、现金占比、未分类占比和 HHI 均值；不假设目标权重合计为 1，
也不假设每天都有调仓。行业数据缺失时 `metrics.industry_exposure.available=false`。

