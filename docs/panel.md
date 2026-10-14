# 评估面板与股票池口径

## 面板是什么

一次评估请求只扫一遍库：`calculate/panel.py` 用一条 SQL 把因子值、标签、
股票池标记与后续分析要用的原料对齐成一张表，之后所有指标都在这张表上算。

```
factor_data/<族>/year=*/part.parquet   因子值
        ├── md.fwd_return              fwd_ret_{close|open}_{1|3|5|10|15|20}
        └── md.daily_hfq               池子标记 + 市值 / 换手 / 估值 / 成交额 / 收益
```

**面板列（顺序固定，后续阶段只加不删）**：

`stock_code`、`trade_date`、`factor`、`label`、`label_1d`、`in_universe`、`is_st`、
`listed_days`、`can_buy_close`、`can_sell_close`、`total_mv`、`circ_mv`、
`turnover_rate`、`pe_ttm`、`pb`、`amount_cny`、`close`、`ret`，
再往后跟**多周期标签列**（V11-3 加入，只有衰减阶段用）

- 主键是 `stock_code + trade_date`，唯一性在 `build_panel` 里断言；
- `factor` / `label` 用 float64 参与计算，原料列用 float32 省内存；
- `label` 允许为空：停牌行没有未来价，区间末端不足 N 个交易日的也没有；
- `label_1d` 是**同价格通道的 1 日前瞻收益**，专门给分位净值与多空曲线用：
  期收益逐日重叠，累乘会把同一区间重复计入，曲线必须用 1 日口径（V11-2 加入）；
- 多周期标签列＝衰减档位里除 `label`（= `params.horizon`）与 `label_1d`（= 1 日档）
  之外的那几档，命名为 `label_h{h}`，同价格通道、同为 float64。
  默认（5 日档 + close）面板多出 `label_h3` / `label_h10` / `label_h20` 三列；
  列名一律通过 `EvalParams.horizon_label_column(h)` 或 `EvalContext.horizon_label(h)`
  取，别在指标里手写列名。多出来的列不复制 `label` / `label_1d`，避免白占几百万行内存；
- 面板**不做排序**：几百万行的 `ORDER BY` 要多花 2 秒以上，下游都按交易日分组。
  风格因子（V11-4）需要个股至少250次有效交易的回看，**不放进面板**，由 `EvalContext.styles()`
  单独取一次（旧实现实测约 6.4 秒，新实现耗时见包 README），只有进入拆分阶段才触发。

## 股票池

| 池子 | 口径 |
|---|---|
| `all_a` 全A可交易 | `can_buy_close` + 非 ST + 从上市日起满 60 个自然日 |
| `index:<代码>` | 上一个已结束月份的月末权重快照（`ref.index_weight`，`weight > 0`）∩ 上表同一套可交易过滤 |

覆盖的指数（12 个）：沪深300、中证500、中证1000、上证50、中证800、中证100、
中证A500、科创50、深证成指、创业板指、国证2000、北证50。

两个要点：

1. **全 A 池与 `scripts/build_factors/validate.py` 的 `POOL_FILTER` 完全相同**，
   否则评估结论会与既有自检对不上；`tests/test_eval_panel.py` 会拿 SQL 直接核对样本数。
2. **指数成分用上一个已结束月份的快照**，不用当月：月末权重要到当月收盘后才知道，
   拿它算当月 IC 就是前视。2005-04 之前没有沪深300 权重快照，区间落在那里时池子为空。

## 保护与报错

| 情况 | 错误码 | 提示 |
|---|---|---|
| 区间内没有交易日 | `EVAL-101` | 检查起止日期 |
| 交易日数 > 2400 | `EVAL-101` | 缩小起止范围 |
| 面板行数 > 800 万 | `EVAL-102` | 缩小起止范围 |
| 主键重复 | `EVAL-102` | 面板构建有 bug |
| 股票池名非法 | `EVAL-103` | 列出可选池子 |
| 因子未注册 / 未落盘 | `EVAL-104` | 提示跑构建脚本或查已实现因子 |

行数上限用 parquet 的文件级元数据估（不扫数据），是个**上界**：区间跨越的整年都算进去，
所以临界情况下会把实际没超的区间也拦下来。实测最长可用区间约 6 年（≈710 万行）。

## 全历史 IC 的独立路径

`load_daily_ic(params, db_file=..., factor_dir=...)` 用 DuckDB 内聚合返回日度
ic/rank_ic/n，不构建完整面板，所以不受本页完整面板的 2400 日/800 万行限制。
两条路径都在有效配对上按平均并列名次计算 RankIC；只有 Pearson IC 受 winsorize 影响。
独立连接设置工作内存 1GB、4 个数据库线程，溢出文件位于自动清理的临时目录。
该限制是 DuckDB 工作内存限制，不是整个 Python 进程 RSS 上限。
覆盖率、分位和风格分析仍走完整面板；不可把全历史 IC 的耗时当作完整分析的耗时。

函数参考与示例见 [../README.md](../README.md)。

## 相关性与连接资源

上下文保留 `factor_dir`，相关性按月从该目录读取各族；只用面板的池内键对齐，
不要求目标因子或收益标签非空。`open_source`默认工作内存1GB、2线程，
每次调用拥有独立只读连接；这不是Python进程RSS的硬上限。
