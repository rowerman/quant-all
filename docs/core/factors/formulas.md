# 因子公式与计算步骤

本文件由 `python -m scripts.build_factors.catalog --write-docs` 生成，不要手工编辑；
因子的唯一来源是 `factors/` 包里的注册表。

已实现并已算出全历史数据的因子 47 个（每个都给 公式 → 白话步骤 → 真实 SQL），
另有规划因子 139 条，公式列在各族末尾的表里。

## 怎么读这份文档

| 写法 | 含义 |
|---|---|
| `close_t` | 第 t 个交易日的收盘价（后复权） |
| `close_(t-20)` | 20 个交易日之前的收盘价 |
| `lag(close, 20)` | 同上，SQL 里的写法 |
| `mean(x, 20)` | 最近 20 个有效交易日的平均 |
| `std(x, 20)` | 最近 20 个有效交易日的样本标准差 |
| `corr(x, y, 20)` | 最近 20 日 x 与 y 的相关系数 |
| `rank(x)` | 同一天全部股票里 x 的排名分位（0~1） |
| `w` | 窗口定义：按 `stock_code` 分组、按 `trade_date` 排序 |

三条必须知道的口径：

1. 价格一律是**后复权价**，除权日不会出现假跳空；
2. 停牌日**不占窗口**，「20 个交易日」指 20 个真正有交易的日子，
   停牌行的因子值直接为空；
3. 历史不够时结果为空，不做填充。所以次新股的前 60 个交易日没有 60 日动量。

想自己动手算一遍、和库里的数对照，看 `cookbook.md`。

## 动量与反转（`momentum`）

### `MOM_20`

**说明**：20 交易日累计收益，短中期动量；A 股短端更偏反转，方向以 IC 实测为准。

**方向**：值越大越看多　**依赖字段**：`close`　**回看窗口**：20 个交易日　**层数**：1

**公式**：`close_t / close_(t-20) - 1`

**计算步骤**

1. 取当日后复权收盘价 close
2. 取 20 个交易日之前的收盘价 lag(close, 20)
3. 相除再减 1，得到这 20 个交易日的累计收益率

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
close / lag(close, 20) over w - 1
```

### `MOM_60`

**说明**：60 交易日累计收益，季度动量。

**方向**：值越大越看多　**依赖字段**：`close`　**回看窗口**：60 个交易日　**层数**：1

**公式**：`close_t / close_(t-60) - 1`

**计算步骤**

1. 取当日后复权收盘价 close
2. 取 60 个交易日之前的收盘价 lag(close, 60)
3. 相除再减 1，得到这 60 个交易日的累计收益率

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
close / lag(close, 60) over w - 1
```

### `MOM_60_20`

**说明**：跳过最近 20 日的中期动量，剔除短期反转对动量因子的稀释。

**方向**：值越大越看多　**依赖字段**：`close`　**回看窗口**：60 个交易日　**层数**：1

**公式**：`close_(t-20) / close_(t-60) - 1`

**计算步骤**

1. 取 20 个交易日前的收盘价
2. 取 60 个交易日前的收盘价
3. 相除再减 1，得到「跳过最近 20 日」那 40 个交易日的收益率

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
lag(close, 20) over w / lag(close, 60) over w - 1
```

### `REV_20`

**说明**：20 日反转：过去一个月跌幅越大，因子值越大。

**方向**：值越大越看多　**依赖字段**：`close`　**回看窗口**：20 个交易日　**层数**：1

**公式**：`-(close_t / close_(t-20) - 1)`

**计算步骤**

1. 先算 20 日累计收益率：close / lag(close, 20) - 1
2. 再取负号，这样「跌得多」的股票因子值更大

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
-(close / lag(close, 20) over w - 1)
```

### `REV_5`

**说明**：5 日反转：过去一周跌得多的股票，因子值更大。

**方向**：值越大越看多　**依赖字段**：`close`　**回看窗口**：5 个交易日　**层数**：1

**公式**：`-(close_t / close_(t-5) - 1)`

**计算步骤**

1. 先算 5 日累计收益率：close / lag(close, 5) - 1
2. 再取负号，这样「跌得多」的股票因子值更大

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
-(close / lag(close, 5) over w - 1)
```

### 待实现（13 条）

| 因子 | 公式 | 实现要点 |
|---|---|---|
| `EMA_MOM` | `ema(close,12) / ema(close,26) - 1` | EMA 无内置算子，用多层或近似 |
| `HIGH_52W_DIST` | `close_t / max(high, 250) - 1` | 距 52 周高点的距离，越接近 0 越强 |
| `LOW_52W_DIST` | `close_t / min(low, 250) - 1` | 距 52 周低点的距离 |
| `MOM_10` | `close_t / close_(t-10) - 1` |  |
| `MOM_120` | `close_t / close_(t-120) - 1` | 半年动量 |
| `MOM_250` | `close_t / close_(t-250) - 1` | 一年动量 |
| `MOM_250_20` | `close_(t-20) / close_(t-250) - 1` | 12-1 动量，国际通用口径 |
| `MOM_5` | `close_t / close_(t-5) - 1` | 一周动量，实际更常作为反转用 |
| `MOM_ACC` | `MOM_20 - delay(MOM_20, 20)` | 动量加速度，需多层写法 |
| `REV_10` | `-(close_t / close_(t-10) - 1)` |  |
| `REV_60` | `-(close_t / close_(t-60) - 1)` | 长端反转，A 股通常不显著 |
| `TREND_SLOPE_T` | `close 对时间的滚动回归 t 值，窗口 60` | 趋势强度，用 regr_slope 与 regr_r2 组合 |
| `WGT_MOM_60` | `sum(w_i * ret_i, 60)，w_i 线性递增` | 线性加权动量，需多层写法 |

## 波动与风险（`volatility`）

### `DOWNSIDE_VOL_20`

**说明**：20 日下行波动，只统计亏损方向的波动，比总波动更贴近风险厌恶。

**方向**：值越大越看空　**依赖字段**：`ret`　**回看窗口**：20 个交易日　**层数**：1

**公式**：`sqrt(mean(min(ret, 0)^2, 20))`

**计算步骤**

1. 取最近 20 个交易日的日收益 ret
2. 把正收益截成 0，只保留亏损
3. 平方后取平均再开方，得到只反映下跌的下行波动

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
sqrt(avg(power(least(ret, 0), 2)) over (partition by stock_code order by trade_date rows between 19 preceding and current row))
```

### `MAX_RET_20`

**说明**：20 日最大单日收益，彩票效应：极值收益高的股票未来偏弱。

**方向**：值越大越看空　**依赖字段**：`ret`　**回看窗口**：20 个交易日　**层数**：1

**公式**：`max(ret, 20)`

**计算步骤**

1. 取最近 20 个交易日的日收益 ret
2. 取其中最大的一个，代表近期单日涨幅最猛的程度

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
max(ret) over (partition by stock_code order by trade_date rows between 19 preceding and current row)
```

### `SKEW_20`

**说明**：20 日收益偏度，右偏（博彩型）股票预期收益偏低。

**方向**：值越大越看空　**依赖字段**：`ret`　**回看窗口**：20 个交易日　**层数**：1

**公式**：`skewness(ret, 20)`

**计算步骤**

1. 取最近 20 个交易日的日收益 ret
2. 算偏度：右偏为正、左偏为负

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
skewness(ret) over (partition by stock_code order by trade_date rows between 19 preceding and current row)
```

### `VOL_20`

**说明**：20 日收益波动率，低波动异象：波动越低未来收益越好。

**方向**：值越大越看空　**依赖字段**：`ret`　**回看窗口**：20 个交易日　**层数**：1

**公式**：`std(ret, 20)`

**计算步骤**

1. 取最近 20 个有效交易日的日收益 ret
2. 算这 20 个数的样本标准差
3. 标准差越大说明波动越高

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
stddev_samp(ret) over (partition by stock_code order by trade_date rows between 19 preceding and current row)
```

### 待实现（16 条）

| 因子 | 公式 | 实现要点 |
|---|---|---|
| `ATR_14` | `mean(true_range, 14) / close` | 真实波幅占价格比例 |
| `DOWNSIDE_VOL_60` | `sqrt(mean(min(ret,0)^2, 60))` |  |
| `GK_VOL_20` | `Garman-Klass 高低开收估计量，20 日` |  |
| `KURT_20` | `kurtosis(ret, 20)` |  |
| `MAXDD_20` | `窗口内最大回撤，20 日` | 需多层：窗口高点与当期回撤 |
| `MAXDD_60` | `窗口内最大回撤，60 日` |  |
| `MAX_RET_60` | `max(ret, 60)` |  |
| `PARKINSON_20` | `sqrt(mean(ln(high/low)^2, 20) / (4*ln2))` | 用高低价估波动，比收盘价更有效 |
| `RET_STAB_60` | `mean(|ret|, 60) / std(ret, 60)` | 收益稳定性 |
| `RS_VOL_20` | `Rogers-Satchell 估计量，20 日` |  |
| `SKEW_60` | `skewness(ret, 60)` |  |
| `UPSIDE_VOL_20` | `sqrt(mean(max(ret,0)^2, 20))` |  |
| `VOL_120` | `std(ret, 120)` |  |
| `VOL_60` | `std(ret, 60)` |  |
| `VOL_OF_VOL_60` | `std(VOL_20, 60)` | 波动率的波动，需多层写法 |
| `VOL_RATIO_20` | `上行波动 / 下行波动，窗口 20` | 值大说明涨得比跌得凶 |

## 流动性（`liquidity`）

### `AMIHUD_20`

**说明**：20 日 Amihud 非流动性：每亿元成交额带来的绝对收益，值越大越不流动。

**方向**：值越大越看多　**依赖字段**：`ret, amount_cny`　**回看窗口**：20 个交易日　**层数**：1

**公式**：`mean(|ret| / (amount / 1e8), 20)`

**计算步骤**

1. 把成交额换算成亿元：amount_cny / 1e8
2. 算每日的「|收益| ÷ 成交额」，即每成交一亿元价格变动多少
3. 对这 20 天的比值取平均，值越大越不流动

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
avg(abs(ret) / nullif(amount_cny / 100000000.0, 0)) over (partition by stock_code order by trade_date rows between 19 preceding and current row)
```

### `LN_AMT_20`

**说明**：20 日平均成交额取对数，小盘流动性差的股票长期有溢价。

**方向**：值越大越看空　**依赖字段**：`amount_cny`　**回看窗口**：20 个交易日　**层数**：1

**公式**：`ln(mean(amount, 20))`

**计算步骤**

1. 取最近 20 个交易日的成交额 amount_cny
2. 取平均后取自然对数，压缩量纲

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
ln(avg(amount_cny) over (partition by stock_code order by trade_date rows between 19 preceding and current row))
```

### `TURN_20`

**说明**：20 日平均换手率，高换手在 A 股通常对应未来收益偏弱。

**方向**：值越大越看空　**依赖字段**：`turnover_rate`　**回看窗口**：20 个交易日　**层数**：1

**公式**：`mean(turnover_rate, 20)`

**计算步骤**

1. 取最近 20 个交易日的换手率 turnover_rate
2. 取平均，代表近一个月的交易活跃度

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
avg(turnover_rate) over (partition by stock_code order by trade_date rows between 19 preceding and current row)
```

### 待实现（13 条）

| 因子 | 公式 | 实现要点 |
|---|---|---|
| `AMIHUD_120` | `mean(|ret| / (amount/1e8), 120)` |  |
| `AMIHUD_60` | `mean(|ret| / (amount/1e8), 60)` |  |
| `AMT_SHARE_20` | `mean(amount, 20) / 当日全市场成交额` | 需截面求和，用 cs_scale 之外的写法 |
| `FREE_FLOAT_RATIO` | `circ_mv / total_mv` | 自由流通比例 |
| `LN_AMT_60` | `ln(mean(amount, 60))` |  |
| `TURN_5` | `mean(turnover_rate, 5)` |  |
| `TURN_60` | `mean(turnover_rate, 60)` |  |
| `TURN_BIAS_20` | `turnover_rate / mean(turnover_rate, 20) - 1` | 当日换手相对近期均值的偏离 |
| `TURN_STD_20` | `std(turnover_rate, 20)` | 换手率波动，交易热度不稳 |
| `TURN_TREND` | `mean(turnover_rate,5) / mean(turnover_rate,60) - 1` | 换手率的短期趋势 |
| `VOL_STAB_60` | `std(ln(volume), 60)` | 成交量的稳定性 |
| `ZERO_VOL_20` | `20 日内停牌或零成交天数占比` | 用 is_suspended 或 volume=0 计数 |
| `ZERO_VOL_60` | `60 日内停牌或零成交天数占比` |  |

## 量价关系（`volume_price`）

### `PV_CORR_20`

**说明**：20 日收益与成交量的相关系数，量价背离（相关低）往往意味着抛压小。

**方向**：值越大越看空　**依赖字段**：`ret, volume_share`　**回看窗口**：20 个交易日　**层数**：1

**公式**：`corr(ret, volume, 20)`

**计算步骤**

1. 取最近 20 个交易日的日收益 ret 与成交量 volume_share
2. 算两者的相关系数，衡量「涨跌有没有成交量配合」

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
corr(ret, volume_share) over (partition by stock_code order by trade_date rows between 19 preceding and current row)
```

### `VWAP_BIAS_5`

**说明**：5 日收盘价相对成交均价的平均偏离，衡量近期买盘是否一直在更高价成交。

**方向**：值越大越看空　**依赖字段**：`close, vwap`　**回看窗口**：5 个交易日　**层数**：1

**公式**：`mean(close / vwap - 1, 5)`

**计算步骤**

1. 算每日收盘价相对当日成交均价的偏离：close / vwap - 1
2. 取最近 5 天的平均偏离

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
avg(close / nullif(vwap, 0) - 1) over (partition by stock_code order by trade_date rows between 4 preceding and current row)
```

### 待实现（11 条）

| 因子 | 公式 | 实现要点 |
|---|---|---|
| `AMT_RET_CORR_60` | `corr(|ret|, amount, 60)` | 放量是否伴随大波动 |
| `MFI_14` | `资金流量指标，14 日` | 典型价与成交额加权 |
| `OBV_SLOPE_20` | `OBV 序列对时间的 20 日回归斜率` | 需多层：先累积 OBV |
| `PV_CORR_120` | `corr(ret, volume, 120)` |  |
| `PV_CORR_60` | `corr(ret, volume, 60)` |  |
| `TURN_RET_CORR_60` | `corr(ret, turnover_rate, 60)` |  |
| `VOL_PRICE_DIVERGE` | `价格新高但成交量萎缩的标志` | 需多层：比较 20 日价格分位与量分位 |
| `VPT_20` | `sum(sign(ret) * volume, 20)` | 量价趋势累积 |
| `VPT_60` | `sum(sign(ret) * volume, 60)` |  |
| `VWAP_BIAS_20` | `mean(close / vwap - 1, 20)` |  |
| `VWAP_DEV_20` | `vwap / mean(vwap, 20) - 1` | 成交均价相对自身的偏离 |

## 技术形态（`technical`）

### `BIAS_20`

**说明**：20 日均线乖离率，价格偏离中枢越远，均值回复压力越大。

**方向**：值越大越看空　**依赖字段**：`close`　**回看窗口**：20 个交易日　**层数**：1

**公式**：`close / mean(close, 20) - 1`

**计算步骤**

1. 取最近 20 个交易日的收盘价均值，即 20 日均线
2. 用当日收盘价除以均线再减 1，得到偏离百分比

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
close / avg(close) over (partition by stock_code order by trade_date rows between 19 preceding and current row) - 1
```

### `BOLL_POS_20`

**说明**：布林带位置：0 贴着下轨、1 贴着上轨；上轨附近短期偏弱。

**方向**：值越大越看空　**依赖字段**：`close`　**回看窗口**：20 个交易日　**层数**：1

**公式**：`(close - (mid - 2*sd)) / (4*sd), mid=mean(close,20), sd=std(close,20)`

**计算步骤**

1. 算 20 日均线 mid 与 20 日收盘价标准差 sd
2. 上下轨取 mid ± 2sd
3. (close - 下轨) ÷ (上轨 - 下轨) 得到当日价格在带内的位置：0 贴下轨、1 贴上轨

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
(close - (avg(close) over (partition by stock_code order by trade_date rows between 19 preceding and current row) - 2 * stddev_samp(close) over (partition by stock_code order by trade_date rows between 19 preceding and current row))) / nullif(4 * stddev_samp(close) over (partition by stock_code order by trade_date rows between 19 preceding and current row), 0)
```

### `RSI_14`

**说明**：14 日相对强弱指标（SMA 口径，非 Wilder 平滑），高值代表短期超买。

**方向**：值越大越看空　**依赖字段**：`ret`　**回看窗口**：14 个交易日　**层数**：1

**公式**：`100 * mean(gain, 14) / (mean(gain, 14) + mean(loss, 14))`

**计算步骤**

1. 把每日收益拆成上涨部分 greatest(ret, 0) 与下跌部分 greatest(-ret, 0)
2. 两部分各取 14 日均值，得到平均涨幅与平均跌幅
3. RSI = 100 × 平均涨幅 ÷（平均涨幅 + 平均跌幅）；两边都是 0 时置空

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
100 * (avg(greatest(ret, 0)) over (partition by stock_code order by trade_date rows between 13 preceding and current row)) / nullif((avg(greatest(ret, 0)) over (partition by stock_code order by trade_date rows between 13 preceding and current row)) + (avg(greatest(-ret, 0)) over (partition by stock_code order by trade_date rows between 13 preceding and current row)), 0)
```

### 待实现（22 条）

| 因子 | 公式 | 实现要点 |
|---|---|---|
| `ADX_14` | `14 日平均趋向指数` | 需多层写法 |
| `AMPLITUDE_20` | `mean((high - low) / pre_close, 20)` | 平均振幅 |
| `BIAS_10` | `close / mean(close,10) - 1` |  |
| `BIAS_5` | `close / mean(close,5) - 1` |  |
| `BIAS_60` | `close / mean(close,60) - 1` |  |
| `BOLL_WIDTH_20` | `4 * std(close,20) / mean(close,20)` | 带宽，波动状态代理 |
| `CCI_14` | `(TP - mean(TP,14)) / (0.015 * 平均绝对偏差)` | TP 为典型价 |
| `CLOSE_POS_20` | `(close - min(low,20)) / (max(high,20) - min(low,20))` | 价格在近期区间中的位置 |
| `KDJ_D` | `K 值再做 2/3 平滑` | 需多层写法 |
| `KDJ_J` | `3K - 2D` | 需多层写法 |
| `KDJ_K` | `RSV 的 2/3 平滑 + 1/3 前值` | 递归平滑，需多层写法 |
| `MACD_DEA` | `ema(MACD_DIF, 9)` | 需多层写法 |
| `MACD_DIF` | `ema(close,12) - ema(close,26)` | EMA 需多层或近似 |
| `MACD_HIST` | `2 * (MACD_DIF - MACD_DEA)` | 需多层写法 |
| `MA_ALIGN` | `MA5>MA10>MA20>MA60 的得分` | 均线多头排列打分 |
| `MA_CROSS_5_20` | `mean(close,5) / mean(close,20) - 1` | 金叉死叉的连续版本 |
| `PSY_12` | `12 日内上涨天数占比` |  |
| `ROC_12` | `close / delay(close,12) - 1` |  |
| `RSI_24` | `100 * mean(gain,24) / (mean(gain,24) + mean(loss,24))` |  |
| `RSI_6` | `100 * mean(gain,6) / (mean(gain,6) + mean(loss,6))` |  |
| `VR_26` | `26 日上涨日成交额 / 下跌日成交额` |  |
| `WR_14` | `(max(high,14) - close) / (max(high,14) - min(low,14))` | 威廉指标 |

## 估值与规模（`valuation`）

### `BP`

**说明**：账面市值比（市净率倒数），最经典的价值因子。

**方向**：值越大越看多　**依赖字段**：`pb`　**回看窗口**：0 个交易日　**层数**：1

**公式**：`1 / pb, 仅当 pb > 0`

**计算步骤**

1. 取市净率 pb
2. 只在 pb > 0 时取倒数 1 / pb，得到账面市值比

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
case when pb > 0 then 1.0 / pb else null end
```

### `EP_TTM`

**说明**：盈利收益率（市盈率倒数），价值因子；亏损股置空值而不是记成负的 EP。

**方向**：值越大越看多　**依赖字段**：`pe_ttm`　**回看窗口**：0 个交易日　**层数**：1

**公式**：`1 / pe_ttm, 仅当 pe_ttm > 0`

**计算步骤**

1. 取滚动市盈率 pe_ttm
2. 只在 pe_ttm > 0（盈利）时取倒数 1 / pe_ttm
3. 亏损股置空，不记成负值

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
case when pe_ttm > 0 then 1.0 / pe_ttm else null end
```

### `LN_MV`

**说明**：总市值取对数，规模因子；A 股长期小盘溢价，但 2017 年后明显减弱。

**方向**：值越大越看空　**依赖字段**：`total_mv`　**回看窗口**：0 个交易日　**层数**：1

**公式**：`ln(total_mv)`

**计算步骤**

1. 取总市值 total_mv（万元）
2. 取自然对数，压缩量纲，方便做截面比较

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
ln(total_mv)
```

### 待实现（9 条）

| 因子 | 公式 | 实现要点 |
|---|---|---|
| `BP_CHG_60` | `BP - delay(BP, 60)` | 需多层写法 |
| `CIRC_MV_RATIO` | `circ_mv / total_mv` |  |
| `DP_TTM` | `dv_ttm` | 股息率，覆盖约六成行 |
| `EP_CHG_60` | `EP_TTM - delay(EP_TTM, 60)` | 估值变化动量，需多层写法 |
| `LN_CIRC_MV` | `ln(circ_mv)` | 流通市值，小盘因子的常用口径 |
| `MV_CHG_20` | `ln(total_mv) - delay(ln(total_mv), 20)` | 市值变化，需注意股本变动 |
| `PB_PCT_250` | `pb 在自身 250 日中的分位` | 用 ts_rank |
| `PE_PCT_250` | `pe_ttm 在自身 250 日中的分位` | 用 ts_rank |
| `SP_TTM` | `1 / ps_ttm，仅 ps_ttm > 0` |  |

## 财务与质量（`fundamental`）

### `ACCRUAL`

**说明**：应计利润占资产比例，越高说明利润的现金含量越差。

**方向**：值越大越看空　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_income, fin_balance, fin_cashflow

**公式**：`(净利润 - 经营现金流) / 总资产`

**计算步骤**

1. 取利润表净利润与现金流量表经营现金流净额
2. 两者相减后除以资产负债表总资产
3. 只保留 [-1, 1] 内的数值

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN bs_total_assets > 0 THEN CASE WHEN isfinite((inc_n_income - cf_n_cashflow_act) / bs_total_assets) AND ((inc_n_income - cf_n_cashflow_act) / bs_total_assets) BETWEEN -1 AND 1 THEN (inc_n_income - cf_n_cashflow_act) / bs_total_assets ELSE NULL END ELSE NULL END
```

### `ASSET_GROWTH`

**说明**：总资产同比增速，衡量扩张速度。

**方向**：值越大越看多　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_indicator

**公式**：`assets_yoy / 100`

**计算步骤**

1. 取财务指标中的总资产同比增速
2. 把百分数除以 100 转成小数
3. 绝对值超过 1000% 的异常值置空

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN isfinite(fi_assets_yoy) AND abs(fi_assets_yoy) <= 1000 THEN fi_assets_yoy / 100.0 ELSE NULL END
```

### `ASSET_TURNOVER`

**说明**：总资产周转率，衡量资产产生收入的能力。

**方向**：值越大越看多　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_indicator

**公式**：`assets_turn`

**计算步骤**

1. 取财务指标中的总资产周转率
2. 只保留 [0, 10] 内的数值

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN isfinite(fi_assets_turn) AND (fi_assets_turn) BETWEEN 0 AND 10 THEN fi_assets_turn ELSE NULL END
```

### `CURRENT_RATIO`

**说明**：流动比率，衡量短期偿债能力。

**方向**：值越大越看多　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_indicator

**公式**：`current_ratio`

**计算步骤**

1. 取财务指标中的流动比率
2. 只保留 [0, 20] 内的数值

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN isfinite(fi_current_ratio) AND (fi_current_ratio) BETWEEN 0 AND 20 THEN fi_current_ratio ELSE NULL END
```

### `DEBT_TO_ASSETS`

**说明**：资产负债率，越高通常财务风险越大。

**方向**：值越大越看空　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_indicator

**公式**：`debt_to_assets / 100`

**计算步骤**

1. 取财务指标中的资产负债率
2. 把百分数除以 100 转成小数
3. 只保留 [0, 2] 内的数值

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN isfinite(fi_debt_to_assets / 100.0) AND (fi_debt_to_assets / 100.0) BETWEEN 0 AND 2 THEN fi_debt_to_assets / 100.0 ELSE NULL END
```

### `EQUITY_GROWTH`

**说明**：净资产同比增速，衡量内生积累与扩张。

**方向**：值越大越看多　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_indicator

**公式**：`eqt_yoy / 100`

**计算步骤**

1. 取财务指标中的净资产同比增速
2. 把百分数除以 100 转成小数
3. 绝对值超过 1000% 的异常值置空

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN isfinite(fi_eqt_yoy) AND abs(fi_eqt_yoy) <= 1000 THEN fi_eqt_yoy / 100.0 ELSE NULL END
```

### `FCF_YIELD`

**说明**：自由现金流收益率，衡量现金回报相对市值的吸引力。

**方向**：值越大越看多　**依赖字段**：`circ_mv`　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_cashflow

**公式**：`(经营现金流 - 资本开支) / 流通市值`

**计算步骤**

1. 取经营现金流净额与购建长期资产支付的现金
2. 两者相减得到自由现金流
3. 除以流通市值（万元转元），只保留 [-1, 1] 内的数值

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN circ_mv > 0 THEN CASE WHEN isfinite((cf_n_cashflow_act - cf_c_pay_acq_const_fiolta) / (circ_mv * 10000.0)) AND ((cf_n_cashflow_act - cf_c_pay_acq_const_fiolta) / (circ_mv * 10000.0)) BETWEEN -1 AND 1 THEN (cf_n_cashflow_act - cf_c_pay_acq_const_fiolta) / (circ_mv * 10000.0) ELSE NULL END ELSE NULL END
```

### `GROSS_MARGIN`

**说明**：毛利率，反映产品定价能力与成本控制。

**方向**：值越大越看多　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_indicator

**公式**：`grossprofit_margin / 100`

**计算步骤**

1. 取财务指标中的毛利率
2. 把百分数除以 100 转成小数
3. 绝对值超过 500% 的异常值置空

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN isfinite(fi_grossprofit_margin) AND abs(fi_grossprofit_margin) <= 500 THEN fi_grossprofit_margin / 100.0 ELSE NULL END
```

### `NET_MARGIN`

**说明**：净利率，反映收入最终转化为净利润的比例。

**方向**：值越大越看多　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_indicator

**公式**：`netprofit_margin / 100`

**计算步骤**

1. 取财务指标中的净利率
2. 把百分数除以 100 转成小数
3. 绝对值超过 500% 的异常值置空

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN isfinite(fi_netprofit_margin) AND abs(fi_netprofit_margin) <= 500 THEN fi_netprofit_margin / 100.0 ELSE NULL END
```

### `NP_GROWTH`

**说明**：归母净利润同比增速，衡量盈利增长。

**方向**：值越大越看多　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_indicator

**公式**：`netprofit_yoy / 100`

**计算步骤**

1. 取财务指标中的净利润同比增速
2. 把百分数除以 100 转成小数
3. 绝对值超过 1000% 的异常值置空

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN isfinite(fi_netprofit_yoy) AND abs(fi_netprofit_yoy) <= 1000 THEN fi_netprofit_yoy / 100.0 ELSE NULL END
```

### `OCF_GROWTH`

**说明**：经营现金流同比增速，衡量现金创造能力的改善。

**方向**：值越大越看多　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_indicator

**公式**：`ocf_yoy / 100`

**计算步骤**

1. 取财务指标中的经营现金流同比增速
2. 把百分数除以 100 转成小数
3. 绝对值超过 1000% 的异常值置空

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN isfinite(fi_ocf_yoy) AND abs(fi_ocf_yoy) <= 1000 THEN fi_ocf_yoy / 100.0 ELSE NULL END
```

### `OCF_TO_NP`

**说明**：净利润的现金含量，越高说明盈利质量越好。

**方向**：值越大越看多　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_income, fin_cashflow

**公式**：`经营现金流净额 / 净利润`

**计算步骤**

1. 取现金流量表经营现金流净额与利润表净利润
2. 两者相除，只对正净利润计算
3. 只保留 [-10, 10] 内的数值

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN inc_n_income > 0 THEN CASE WHEN isfinite(cf_n_cashflow_act / inc_n_income) AND (cf_n_cashflow_act / inc_n_income) BETWEEN -10 AND 10 THEN cf_n_cashflow_act / inc_n_income ELSE NULL END ELSE NULL END
```

### `OP_GROWTH`

**说明**：营业利润同比增速，剔除营业外项目后的增长。

**方向**：值越大越看多　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_indicator

**公式**：`op_yoy / 100`

**计算步骤**

1. 取财务指标中的营业利润同比增速
2. 把百分数除以 100 转成小数
3. 绝对值超过 1000% 的异常值置空

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN isfinite(fi_op_yoy) AND abs(fi_op_yoy) <= 1000 THEN fi_op_yoy / 100.0 ELSE NULL END
```

### `QUICK_RATIO`

**说明**：速动比率，剔除存货后的短期偿债能力。

**方向**：值越大越看多　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_indicator

**公式**：`quick_ratio`

**计算步骤**

1. 取财务指标中的速动比率
2. 只保留 [0, 20] 内的数值

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN isfinite(fi_quick_ratio) AND (fi_quick_ratio) BETWEEN 0 AND 20 THEN fi_quick_ratio ELSE NULL END
```

### `REV_GROWTH`

**说明**：营业收入同比增速，衡量业务扩张。

**方向**：值越大越看多　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_indicator

**公式**：`or_yoy / 100`

**计算步骤**

1. 取财务指标中的营业收入同比增速
2. 把百分数除以 100 转成小数
3. 绝对值超过 1000% 的异常值置空

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN isfinite(fi_or_yoy) AND abs(fi_or_yoy) <= 1000 THEN fi_or_yoy / 100.0 ELSE NULL END
```

### `ROA_YEARLY`

**说明**：年化总资产收益率，衡量全部资产的盈利效率。

**方向**：值越大越看多　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_indicator

**公式**：`roa_yearly / 100`

**计算步骤**

1. 取财务指标中的年化 ROA
2. 把百分数除以 100 转成小数
3. 绝对值超过 500% 的异常值置空

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN isfinite(fi_roa_yearly) AND abs(fi_roa_yearly) <= 500 THEN fi_roa_yearly / 100.0 ELSE NULL END
```

### `ROE_YEARLY`

**说明**：年化净资产收益率，衡量股东资金的使用效率。

**方向**：值越大越看多　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_indicator

**公式**：`roe_yearly / 100`

**计算步骤**

1. 取财务指标中的年化 ROE
2. 把百分数除以 100 转成小数
3. 绝对值超过 500% 的异常值置空

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN isfinite(fi_roe_yearly) AND abs(fi_roe_yearly) <= 500 THEN fi_roe_yearly / 100.0 ELSE NULL END
```

### `ROIC`

**说明**：投入资本回报率，兼顾债权与股权资本的使用效率。

**方向**：值越大越看多　**依赖字段**：``　**回看窗口**：0 个交易日　**层数**：1；附加数据源：fin_indicator

**公式**：`roic / 100`

**计算步骤**

1. 取财务指标中的 ROIC
2. 把百分数除以 100 转成小数
3. 绝对值超过 500% 的异常值置空

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
CASE WHEN isfinite(fi_roic) AND abs(fi_roic) <= 500 THEN fi_roic / 100.0 ELSE NULL END
```

## A 股特色（`cn_special`）

### `LIMIT_UP_CNT_20`

**说明**：20 日内涨停天数，衡量资金关注度；涨停越多短期越容易反转。

**方向**：值越大越看空　**依赖字段**：`is_limit_up`　**回看窗口**：20 个交易日　**层数**：1

**公式**：`sum(is_limit_up, 20)`

**计算步骤**

1. 逐日判断是否涨停（is_limit_up）
2. 统计最近 20 个有效交易日里的涨停天数

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
sum(case when is_limit_up then 1 else 0 end) over (partition by stock_code order by trade_date rows between 19 preceding and current row)
```

### `OVERNIGHT_RET_20`

**说明**：20 日隔夜跳空均值，隔夜收益在 A 股长期偏负，反映散户开盘追高。

**方向**：值越大越看空　**依赖字段**：`open, pre_close`　**回看窗口**：20 个交易日　**层数**：1

**公式**：`mean(open / pre_close - 1, 20)`

**计算步骤**

1. 算每日隔夜跳空：open / 前收 - 1
2. 取最近 20 天的平均值

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
avg(open / nullif(pre_close, 0) - 1) over (partition by stock_code order by trade_date rows between 19 preceding and current row)
```

### 待实现（12 条）

| 因子 | 公式 | 实现要点 |
|---|---|---|
| `BOMB_RATE_20` | `炸板天数 / 触及涨停天数，窗口 20` | 用 touched_limit_up 与 is_limit_up 之差 |
| `CONSEC_LIMIT_UP` | `当前连续涨停天数` | 需多层：连续计数 |
| `EX_DIV_CNT_250` | `250 日内除权除息次数` | 用 is_ex_div 计数 |
| `INTRADAY_RET_20` | `mean(close/open - 1, 20)` | 日内收益，与隔夜收益互补 |
| `LIMIT_DOWN_CNT_20` | `sum(is_limit_down, 20)` |  |
| `LIMIT_UP_CNT_60` | `sum(is_limit_up, 60)` |  |
| `LIST_DAYS` | `trade_date - list_date` | 上市天数，做次新股因子用 |
| `ONEWORD_CNT_20` | `sum(is_one_word_board, 20)` | 一字板数量 |
| `OVERNIGHT_RET_5` | `mean(open/pre_close - 1, 5)` |  |
| `RESUME_GAP` | `复牌首日的跳空幅度` | 用 is_resume 与 open/pre_close 组合 |
| `ST_DAYS_250` | `250 日内 ST 天数` | 用 is_st 计数 |
| `SUSPEND_DAYS_60` | `60 个自然交易日内的停牌天数` | 用 status='suspended' 计数 |

## 市场与相对强度（`market`）

### `BETA_60`

**说明**：60 日市场贝塔，低贝塔异象：贝塔低的股票风险调整后收益更好；窗口内少于 40 个交易日则置空。

**方向**：值越大越看空　**依赖字段**：`ret`　**回看窗口**：60 个交易日　**层数**：1；附加数据源：index_000300

**公式**：`regr_slope(ret, idx_ret, 60)`

**计算步骤**

1. 取最近 60 个交易日的个股日收益与沪深300 日收益
2. 对两者做一元回归，取斜率即为贝塔
3. 窗口内有效交易日不足 40 天时置空，避免次新股算出上千的伪贝塔

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
case when count(ret) over (partition by stock_code order by trade_date rows between 59 preceding and current row) >= 40 then regr_slope(ret, idx_ret_300) over (partition by stock_code order by trade_date rows between 59 preceding and current row) else null end
```

### `RS_20`

**说明**：20 日相对沪深300 的超额收益，相对强度因子。

**方向**：值越大越看多　**依赖字段**：`ret`　**回看窗口**：20 个交易日　**层数**：1；附加数据源：index_000300

**公式**：`sum(ret, 20) - sum(idx_ret, 20)`

**计算步骤**

1. 算个股最近 20 个交易日的累计收益
2. 算沪深300 同期累计收益
3. 两者相减，得到 20 日超额收益

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
sum(ret) over (partition by stock_code order by trade_date rows between 19 preceding and current row) - sum(idx_ret_300) over (partition by stock_code order by trade_date rows between 19 preceding and current row)
```

### 待实现（8 条）

| 因子 | 公式 | 实现要点 |
|---|---|---|
| `BETA_120` | `regr_slope(ret, idx_ret, 120)` |  |
| `BETA_250` | `regr_slope(ret, idx_ret, 250)` |  |
| `CORR_MKT_60` | `corr(ret, idx_ret, 60)` |  |
| `DOWNSIDE_BETA_60` | `仅用指数下跌日的收益回归，60 日` | 需多层：先筛指数下跌日 |
| `IDIO_MOM_20` | `20 日收益 - beta * 指数 20 日收益` | 需多层写法 |
| `IDIO_VOL_60` | `回归残差的标准差，60 日` | 用 regr_r2 反推，或多层写法 |
| `RS_120` | `sum(ret,120) - sum(idx_ret,120)` |  |
| `RS_60` | `sum(ret,60) - sum(idx_ret,60)` |  |

## WorldQuant Alpha101 复现（`alpha101`）

### `ALPHA101_001`

**说明**：原始 Alpha#1：先按 20 日波动放大价格，再看最近 5 日极值出现在几天前，截面排序。

**方向**：值越大越看多　**依赖字段**：`close, ret`　**回看窗口**：24 个交易日　**层数**：5

**公式**：`rank(5 日内极大值出现在几天前) - 0.5，极大值口径为 power(ret<0 ? std(ret,20) : close, 2)`

**计算步骤**

1. 第 1 层：给每行编号 row_number，用来记录极值出现的位置
2. 第 2 层：算 20 日日收益标准差 std(ret, 20)
3. 第 3 层：当日收益为负时取这个标准差、否则取收盘价，再平方，得到放大后的价格序列
4. 第 4 层：在最近 5 日里找出该序列最大的一行，返回它的行号
5. 第 5 层：用「当前行号 − 极值行号」得到极值距今几天，做截面排名后减 0.5

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
-- 第 1 层
row_number() over w
-- 第 2 层
stddev_samp(ret) over (partition by stock_code order by trade_date rows between 19 preceding and current row)
-- 第 3 层
power(case when ret < 0 then v2 else close end, 2)
-- 第 4 层
arg_max(v1, v3) over (partition by stock_code order by trade_date rows between 4 preceding and current row)
-- 第 5 层
percent_rank() over (partition by trade_date order by (v1 - v4)) - 0.5
```

### `ALPHA101_006`

**说明**：原始 Alpha#6：开盘价与成交量的负相关，量价背离的经典写法。

**方向**：值越大越看多　**依赖字段**：`open, volume_share`　**回看窗口**：10 个交易日　**层数**：1

**公式**：`-corr(open, volume, 10)`

**计算步骤**

1. 取最近 10 个交易日的开盘价与成交量
2. 算两者的相关系数
3. 取负号，把「量价正相关」变成反向信号

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
(-(corr(open, volume_share) over (partition by stock_code order by trade_date rows between 9 preceding and current row)))
```

### `ALPHA101_012`

**说明**：原始 Alpha#12：放量则记当日跌幅，缩量则记当日涨幅的取负。

**方向**：值越大越看多　**依赖字段**：`close, volume_share`　**回看窗口**：1 个交易日　**层数**：1

**公式**：`sign(delta(volume, 1)) * (-delta(close, 1))`

**计算步骤**

1. 算成交量的一日变动并取符号：放量为正、缩量为负
2. 算收盘价的一日变动并取负
3. 两者相乘

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
sign((volume_share - lag(volume_share, 1) over w)) * (-((close - lag(close, 1) over w)))
```

### 待实现（17 条）

| 因子 | 公式 | 实现要点 |
|---|---|---|
| `ALPHA101_002` | `-1 * corr(rank(delta(ln(volume),2)), rank((close-open)/open), 6)` |  |
| `ALPHA101_003` | `-1 * corr(rank(open), rank(volume), 10)` |  |
| `ALPHA101_004` | `-1 * ts_rank(rank(low), 9)` | 需要 ts_rank 双层写法 |
| `ALPHA101_013` | `-1 * rank(cov(rank(close), rank(volume), 5))` | 需多层写法 |
| `ALPHA101_014` | `-1 * rank(delta(ret,3)) * corr(open, volume, 10)` | 需多层写法 |
| `ALPHA101_015` | `-1 * sum(rank(corr(rank(high), rank(volume), 3)), 3)` | 需多层写法 |
| `ALPHA101_016` | `-1 * rank(cov(rank(high), rank(volume), 5))` | 需多层写法 |
| `ALPHA101_018` | `-1 * rank(std(abs(close-open),5) + (close-open) + corr(close, open, 10))` |  |
| `ALPHA101_020` | `-1 * rank(open - delay(high,1)) * rank(open - delay(close,1)) * rank(open - delay(low,1))` |  |
| `ALPHA101_023` | `sum(high,20)/20 < high ? -delta(high,2) : 0` |  |
| `ALPHA101_024` | `delta(mean(close,100),100)/delay(close,100) <= 0.05 ? -(close - min(close,100)) : -delta(close,3)` | 长窗口，注意 lookback |
| `ALPHA101_026` | `-1 * ts_max(corr(ts_rank(volume,5), ts_rank(high,5), 5), 3)` | 需要 ts_rank 双层写法 |
| `ALPHA101_028` | `scale(corr(adv20, low, 5) + (high+low)/2 - close)` |  |
| `ALPHA101_033` | `rank(-(1 - open/close))` |  |
| `ALPHA101_034` | `rank(1 - rank(std(ret,2)/std(ret,5)) + 1 - rank(delta(close,1)))` |  |
| `ALPHA101_041` | `sqrt(high * low) - vwap` |  |
| `ALPHA101_101` | `(close - open) / (high - low + 0.001) * volume / adv20` |  |

## 国泰君安 Alpha191 复现（`gtja191`）

### `GTJA_191_001`

**说明**：原始 Alpha#1：成交量变化的分位与日内涨跌幅的分位做 6 日相关，再取负。

**方向**：值越大越看多　**依赖字段**：`open, close, volume_share`　**回看窗口**：6 个交易日　**层数**：5

**公式**：`-corr(rank(delta(ln(volume),1)), rank((close-open)/open), 6)`

**计算步骤**

1. 第 1 层：算每日日内涨跌幅 (close - open) / open
2. 第 2 层：算成交量的对数变化 ln(volume) - lag(ln(volume), 1)
3. 第 3 层：把日内涨跌幅放到当日截面上排名
4. 第 4 层：把成交量变化放到当日截面上排名
5. 第 5 层：算这两个排名最近 6 日的相关系数，再取负

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
-- 第 1 层
(close - open) / nullif(open, 0)
-- 第 2 层
ln(volume_share) - lag(ln(volume_share), 1) over w
-- 第 3 层
percent_rank() over (partition by trade_date order by v1)
-- 第 4 层
percent_rank() over (partition by trade_date order by v2)
-- 第 5 层
-corr(v3, v4) over (partition by stock_code order by trade_date rows between 5 preceding and current row)
```

### `GTJA_191_003`

**说明**：原始 Alpha#3：6 日内收在最高价与最低价的天数之差，衡量收盘强弱。

**方向**：值越大越看多　**依赖字段**：`close, high, low`　**回看窗口**：6 个交易日　**层数**：1

**公式**：`sum(close == high ? 1 : (close == low ? -1 : 0), 6)`

**计算步骤**

1. 逐日比较：收在最高价记 +1、收在最低价记 −1、其余记 0
2. 把最近 6 天的得分求和，衡量收盘强弱

**实现表达式**（构建时真实执行的 SQL；`w` 是按 `stock_code` 分组、`trade_date` 排序的窗口）

```sql
sum(case when close = high then 1 when close = low then -1 else 0 end) over (partition by stock_code order by trade_date rows between 5 preceding and current row)
```

### 待实现（18 条）

| 因子 | 公式 | 实现要点 |
|---|---|---|
| `GTJA_191_004` | `收盘价与均线关系的条件式，见原始公式表` | 条件分支多，注意可读性 |
| `GTJA_191_006` | `rank(open - close) 与 rank(ret - open) 的组合式` | 见原始公式表 |
| `GTJA_191_007` | `(rank(max(vwap-close,3)) + rank(min(vwap-close,3))) * rank(delta(volume,3))` |  |
| `GTJA_191_008` | `rank(delta((high+low)/2*0.2 + vwap*0.8, 4) * -1)` |  |
| `GTJA_191_009` | `SMA(((high+low)/2 - (delay(high,1)+delay(low,1))/2) * (high-low)/volume, 7, 2)` | 递归平滑，需多层写法 |
| `GTJA_191_010` | `rank(max(ret<0 ? std(ret,20) : close, 5) ^ 2)` | 多层写法 |
| `GTJA_191_011` | `sum(((close-low) - (high-close)) / (high-low) * volume, 6)` | 多空力量对比 |
| `GTJA_191_012` | `rank(open - mean(vwap,10)) * (-1 * rank(abs(close - vwap)))` |  |
| `GTJA_191_013` | `sqrt(high * low) - vwap` |  |
| `GTJA_191_014` | `close - delay(close, 5)` |  |
| `GTJA_191_016` | `-1 * ts_max(rank(corr(rank(volume), rank(vwap), 5)), 5)` | 多层写法 |
| `GTJA_191_017` | `rank(vwap - max(vwap,15)) ^ delta(close,5)` |  |
| `GTJA_191_018` | `close / delay(close, 5)` |  |
| `GTJA_191_019` | `close < delay(close,5) ? (close-delay(close,5))/delay(close,5) : (close-delay(close,5))/close` |  |
| `GTJA_191_020` | `(close - delay(close,6)) / delay(close,6) * 100` |  |
| `GTJA_191_021` | `regr_slope(mean(close,6), 序列, 6)` | 用 regr_slope 实现 REGBETA |
| `GTJA_191_023` | `SMA(close > delay(close,1) ? std(close,20) : 0, 20, 1) / (该式 + 下跌方向的同式)` | 多层写法 |
| `GTJA_191_024` | `SMA(close - delay(close,5), 5, 1)` | 多层写法 |

