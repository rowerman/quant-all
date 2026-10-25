# factor_linear 显式日期线性基线提供者

## 职责

`factor_linear` v1 是 P4-4B1 使用的隐藏目标权重 provider。它把已有因子按给定权重
组合，在显式调仓日选择可交易股票中的 Top-N 等权持有。它只服务研究验收的相同日期
比较，不进入通用新建策略页面。

## 输入

- `factor_names`：非空且不重复的因子名序列。
- `factor_weights`：与因子一一对应的有限数值权重。
- `standardize`：是否按日做截面 z-score，默认 `True`。
- `winsorize`：是否做截面 1%/99% 缩尾，默认 `False`。
- `top_n`：正整数持仓数。
- `rebalance_dates`：必填、排序去重的 `YYYY-MM-DD` 日期序列。

## 行为

1. 复用 `backtest.signals.build_scores()` 的因子截面标准化和线性组合。
2. 只在给定调仓日与行情面板的交集中生成目标，其他日期返回整行 `NaN`。
3. 在 eligible 股票中按分数降序选 `top_n` 只，等权持有。
4. provider 返回矩阵仍由信号层统一校验正权重、有限值和股票池约束。

## 调用关系

- 上游：P4-4B1 验收脚本显式调用 `ensure_registered()`。
- 下游：`build_signal_targets()` → 完整回测账本。
- 数据：使用 `MarketData.factors` 中已有因子，不写回 `data/factors/`。

## 注意

- 模块名以下划线开头，避免自动发现；类声明 `hidden=True`，即使显式注册也不会
  出现在 `list_providers()` 的通用页面列表。
- 显式日期只改变信号产生日，不改变因子计算和信号时点，不允许未来数据。
- B1-B3 必须保留原基线因子和权重，不能用 `factor_rank` 替代 z-score 语义。
