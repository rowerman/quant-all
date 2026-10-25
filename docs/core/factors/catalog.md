# 因子清单

本文件由 `python -m scripts.build_factors.catalog --write-docs` 生成，不要手工编辑；
改因子定义后重新生成即可。因子的唯一来源是 `factors/` 包里的注册表。

- 规划条目：186 条
- 已实现：47 条
- 覆盖因子族：11 个

状态含义：**已实现** 表示已经注册并能参与构建；**待实现** 表示公式已列在这里，
照着写一个 `define(...)` 就能纳入构建。

## 动量与反转（`momentum`）

共 18 条，其中已实现 5 条。

| 因子 | 状态 | 方向 | 公式 | 说明 |
|---|---|---|---|---|
| `MOM_20` | 已实现 | 越大越看多 | `close_t / close_(t-20) - 1` | 20 交易日累计收益，短中期动量；A 股短端更偏反转，方向以 IC 实测为准。 |
| `MOM_60` | 已实现 | 越大越看多 | `close_t / close_(t-60) - 1` | 60 交易日累计收益，季度动量。 |
| `MOM_60_20` | 已实现 | 越大越看多 | `close_(t-20) / close_(t-60) - 1` | 跳过最近 20 日的中期动量，剔除短期反转对动量因子的稀释。 |
| `REV_20` | 已实现 | 越大越看多 | `-(close_t / close_(t-20) - 1)` | 20 日反转：过去一个月跌幅越大，因子值越大。 |
| `REV_5` | 已实现 | 越大越看多 | `-(close_t / close_(t-5) - 1)` | 5 日反转：过去一周跌得多的股票，因子值更大。 |
| `EMA_MOM` | 待实现 |  | `ema(close,12) / ema(close,26) - 1` | EMA 无内置算子，用多层或近似 |
| `HIGH_52W_DIST` | 待实现 |  | `close_t / max(high, 250) - 1` | 距 52 周高点的距离，越接近 0 越强 |
| `LOW_52W_DIST` | 待实现 |  | `close_t / min(low, 250) - 1` | 距 52 周低点的距离 |
| `MOM_10` | 待实现 |  | `close_t / close_(t-10) - 1` |  |
| `MOM_120` | 待实现 |  | `close_t / close_(t-120) - 1` | 半年动量 |
| `MOM_250` | 待实现 |  | `close_t / close_(t-250) - 1` | 一年动量 |
| `MOM_250_20` | 待实现 |  | `close_(t-20) / close_(t-250) - 1` | 12-1 动量，国际通用口径 |
| `MOM_5` | 待实现 |  | `close_t / close_(t-5) - 1` | 一周动量，实际更常作为反转用 |
| `MOM_ACC` | 待实现 |  | `MOM_20 - delay(MOM_20, 20)` | 动量加速度，需多层写法 |
| `REV_10` | 待实现 |  | `-(close_t / close_(t-10) - 1)` |  |
| `REV_60` | 待实现 |  | `-(close_t / close_(t-60) - 1)` | 长端反转，A 股通常不显著 |
| `TREND_SLOPE_T` | 待实现 |  | `close 对时间的滚动回归 t 值，窗口 60` | 趋势强度，用 regr_slope 与 regr_r2 组合 |
| `WGT_MOM_60` | 待实现 |  | `sum(w_i * ret_i, 60)，w_i 线性递增` | 线性加权动量，需多层写法 |

## 波动与风险（`volatility`）

共 20 条，其中已实现 4 条。

| 因子 | 状态 | 方向 | 公式 | 说明 |
|---|---|---|---|---|
| `DOWNSIDE_VOL_20` | 已实现 | 越大越看空 | `sqrt(mean(min(ret, 0)^2, 20))` | 20 日下行波动，只统计亏损方向的波动，比总波动更贴近风险厌恶。 |
| `MAX_RET_20` | 已实现 | 越大越看空 | `max(ret, 20)` | 20 日最大单日收益，彩票效应：极值收益高的股票未来偏弱。 |
| `SKEW_20` | 已实现 | 越大越看空 | `skewness(ret, 20)` | 20 日收益偏度，右偏（博彩型）股票预期收益偏低。 |
| `VOL_20` | 已实现 | 越大越看空 | `std(ret, 20)` | 20 日收益波动率，低波动异象：波动越低未来收益越好。 |
| `ATR_14` | 待实现 |  | `mean(true_range, 14) / close` | 真实波幅占价格比例 |
| `DOWNSIDE_VOL_60` | 待实现 |  | `sqrt(mean(min(ret,0)^2, 60))` |  |
| `GK_VOL_20` | 待实现 |  | `Garman-Klass 高低开收估计量，20 日` |  |
| `KURT_20` | 待实现 |  | `kurtosis(ret, 20)` |  |
| `MAXDD_20` | 待实现 |  | `窗口内最大回撤，20 日` | 需多层：窗口高点与当期回撤 |
| `MAXDD_60` | 待实现 |  | `窗口内最大回撤，60 日` |  |
| `MAX_RET_60` | 待实现 |  | `max(ret, 60)` |  |
| `PARKINSON_20` | 待实现 |  | `sqrt(mean(ln(high/low)^2, 20) / (4*ln2))` | 用高低价估波动，比收盘价更有效 |
| `RET_STAB_60` | 待实现 |  | `mean(|ret|, 60) / std(ret, 60)` | 收益稳定性 |
| `RS_VOL_20` | 待实现 |  | `Rogers-Satchell 估计量，20 日` |  |
| `SKEW_60` | 待实现 |  | `skewness(ret, 60)` |  |
| `UPSIDE_VOL_20` | 待实现 |  | `sqrt(mean(max(ret,0)^2, 20))` |  |
| `VOL_120` | 待实现 |  | `std(ret, 120)` |  |
| `VOL_60` | 待实现 |  | `std(ret, 60)` |  |
| `VOL_OF_VOL_60` | 待实现 |  | `std(VOL_20, 60)` | 波动率的波动，需多层写法 |
| `VOL_RATIO_20` | 待实现 |  | `上行波动 / 下行波动，窗口 20` | 值大说明涨得比跌得凶 |

## 流动性（`liquidity`）

共 16 条，其中已实现 3 条。

| 因子 | 状态 | 方向 | 公式 | 说明 |
|---|---|---|---|---|
| `AMIHUD_20` | 已实现 | 越大越看多 | `mean(|ret| / (amount / 1e8), 20)` | 20 日 Amihud 非流动性：每亿元成交额带来的绝对收益，值越大越不流动。 |
| `LN_AMT_20` | 已实现 | 越大越看空 | `ln(mean(amount, 20))` | 20 日平均成交额取对数，小盘流动性差的股票长期有溢价。 |
| `TURN_20` | 已实现 | 越大越看空 | `mean(turnover_rate, 20)` | 20 日平均换手率，高换手在 A 股通常对应未来收益偏弱。 |
| `AMIHUD_120` | 待实现 |  | `mean(|ret| / (amount/1e8), 120)` |  |
| `AMIHUD_60` | 待实现 |  | `mean(|ret| / (amount/1e8), 60)` |  |
| `AMT_SHARE_20` | 待实现 |  | `mean(amount, 20) / 当日全市场成交额` | 需截面求和，用 cs_scale 之外的写法 |
| `FREE_FLOAT_RATIO` | 待实现 |  | `circ_mv / total_mv` | 自由流通比例 |
| `LN_AMT_60` | 待实现 |  | `ln(mean(amount, 60))` |  |
| `TURN_5` | 待实现 |  | `mean(turnover_rate, 5)` |  |
| `TURN_60` | 待实现 |  | `mean(turnover_rate, 60)` |  |
| `TURN_BIAS_20` | 待实现 |  | `turnover_rate / mean(turnover_rate, 20) - 1` | 当日换手相对近期均值的偏离 |
| `TURN_STD_20` | 待实现 |  | `std(turnover_rate, 20)` | 换手率波动，交易热度不稳 |
| `TURN_TREND` | 待实现 |  | `mean(turnover_rate,5) / mean(turnover_rate,60) - 1` | 换手率的短期趋势 |
| `VOL_STAB_60` | 待实现 |  | `std(ln(volume), 60)` | 成交量的稳定性 |
| `ZERO_VOL_20` | 待实现 |  | `20 日内停牌或零成交天数占比` | 用 is_suspended 或 volume=0 计数 |
| `ZERO_VOL_60` | 待实现 |  | `60 日内停牌或零成交天数占比` |  |

## 量价关系（`volume_price`）

共 13 条，其中已实现 2 条。

| 因子 | 状态 | 方向 | 公式 | 说明 |
|---|---|---|---|---|
| `PV_CORR_20` | 已实现 | 越大越看空 | `corr(ret, volume, 20)` | 20 日收益与成交量的相关系数，量价背离（相关低）往往意味着抛压小。 |
| `VWAP_BIAS_5` | 已实现 | 越大越看空 | `mean(close / vwap - 1, 5)` | 5 日收盘价相对成交均价的平均偏离，衡量近期买盘是否一直在更高价成交。 |
| `AMT_RET_CORR_60` | 待实现 |  | `corr(|ret|, amount, 60)` | 放量是否伴随大波动 |
| `MFI_14` | 待实现 |  | `资金流量指标，14 日` | 典型价与成交额加权 |
| `OBV_SLOPE_20` | 待实现 |  | `OBV 序列对时间的 20 日回归斜率` | 需多层：先累积 OBV |
| `PV_CORR_120` | 待实现 |  | `corr(ret, volume, 120)` |  |
| `PV_CORR_60` | 待实现 |  | `corr(ret, volume, 60)` |  |
| `TURN_RET_CORR_60` | 待实现 |  | `corr(ret, turnover_rate, 60)` |  |
| `VOL_PRICE_DIVERGE` | 待实现 |  | `价格新高但成交量萎缩的标志` | 需多层：比较 20 日价格分位与量分位 |
| `VPT_20` | 待实现 |  | `sum(sign(ret) * volume, 20)` | 量价趋势累积 |
| `VPT_60` | 待实现 |  | `sum(sign(ret) * volume, 60)` |  |
| `VWAP_BIAS_20` | 待实现 |  | `mean(close / vwap - 1, 20)` |  |
| `VWAP_DEV_20` | 待实现 |  | `vwap / mean(vwap, 20) - 1` | 成交均价相对自身的偏离 |

## 技术形态（`technical`）

共 25 条，其中已实现 3 条。

| 因子 | 状态 | 方向 | 公式 | 说明 |
|---|---|---|---|---|
| `BIAS_20` | 已实现 | 越大越看空 | `close / mean(close, 20) - 1` | 20 日均线乖离率，价格偏离中枢越远，均值回复压力越大。 |
| `BOLL_POS_20` | 已实现 | 越大越看空 | `(close - (mid - 2*sd)) / (4*sd), mid=mean(close,20), sd=std(close,20)` | 布林带位置：0 贴着下轨、1 贴着上轨；上轨附近短期偏弱。 |
| `RSI_14` | 已实现 | 越大越看空 | `100 * mean(gain, 14) / (mean(gain, 14) + mean(loss, 14))` | 14 日相对强弱指标（SMA 口径，非 Wilder 平滑），高值代表短期超买。 |
| `ADX_14` | 待实现 |  | `14 日平均趋向指数` | 需多层写法 |
| `AMPLITUDE_20` | 待实现 |  | `mean((high - low) / pre_close, 20)` | 平均振幅 |
| `BIAS_10` | 待实现 |  | `close / mean(close,10) - 1` |  |
| `BIAS_5` | 待实现 |  | `close / mean(close,5) - 1` |  |
| `BIAS_60` | 待实现 |  | `close / mean(close,60) - 1` |  |
| `BOLL_WIDTH_20` | 待实现 |  | `4 * std(close,20) / mean(close,20)` | 带宽，波动状态代理 |
| `CCI_14` | 待实现 |  | `(TP - mean(TP,14)) / (0.015 * 平均绝对偏差)` | TP 为典型价 |
| `CLOSE_POS_20` | 待实现 |  | `(close - min(low,20)) / (max(high,20) - min(low,20))` | 价格在近期区间中的位置 |
| `KDJ_D` | 待实现 |  | `K 值再做 2/3 平滑` | 需多层写法 |
| `KDJ_J` | 待实现 |  | `3K - 2D` | 需多层写法 |
| `KDJ_K` | 待实现 |  | `RSV 的 2/3 平滑 + 1/3 前值` | 递归平滑，需多层写法 |
| `MACD_DEA` | 待实现 |  | `ema(MACD_DIF, 9)` | 需多层写法 |
| `MACD_DIF` | 待实现 |  | `ema(close,12) - ema(close,26)` | EMA 需多层或近似 |
| `MACD_HIST` | 待实现 |  | `2 * (MACD_DIF - MACD_DEA)` | 需多层写法 |
| `MA_ALIGN` | 待实现 |  | `MA5>MA10>MA20>MA60 的得分` | 均线多头排列打分 |
| `MA_CROSS_5_20` | 待实现 |  | `mean(close,5) / mean(close,20) - 1` | 金叉死叉的连续版本 |
| `PSY_12` | 待实现 |  | `12 日内上涨天数占比` |  |
| `ROC_12` | 待实现 |  | `close / delay(close,12) - 1` |  |
| `RSI_24` | 待实现 |  | `100 * mean(gain,24) / (mean(gain,24) + mean(loss,24))` |  |
| `RSI_6` | 待实现 |  | `100 * mean(gain,6) / (mean(gain,6) + mean(loss,6))` |  |
| `VR_26` | 待实现 |  | `26 日上涨日成交额 / 下跌日成交额` |  |
| `WR_14` | 待实现 |  | `(max(high,14) - close) / (max(high,14) - min(low,14))` | 威廉指标 |

## 估值与规模（`valuation`）

共 12 条，其中已实现 3 条。

| 因子 | 状态 | 方向 | 公式 | 说明 |
|---|---|---|---|---|
| `BP` | 已实现 | 越大越看多 | `1 / pb, 仅当 pb > 0` | 账面市值比（市净率倒数），最经典的价值因子。 |
| `EP_TTM` | 已实现 | 越大越看多 | `1 / pe_ttm, 仅当 pe_ttm > 0` | 盈利收益率（市盈率倒数），价值因子；亏损股置空值而不是记成负的 EP。 |
| `LN_MV` | 已实现 | 越大越看空 | `ln(total_mv)` | 总市值取对数，规模因子；A 股长期小盘溢价，但 2017 年后明显减弱。 |
| `BP_CHG_60` | 待实现 |  | `BP - delay(BP, 60)` | 需多层写法 |
| `CIRC_MV_RATIO` | 待实现 |  | `circ_mv / total_mv` |  |
| `DP_TTM` | 待实现 |  | `dv_ttm` | 股息率，覆盖约六成行 |
| `EP_CHG_60` | 待实现 |  | `EP_TTM - delay(EP_TTM, 60)` | 估值变化动量，需多层写法 |
| `LN_CIRC_MV` | 待实现 |  | `ln(circ_mv)` | 流通市值，小盘因子的常用口径 |
| `MV_CHG_20` | 待实现 |  | `ln(total_mv) - delay(ln(total_mv), 20)` | 市值变化，需注意股本变动 |
| `PB_PCT_250` | 待实现 |  | `pb 在自身 250 日中的分位` | 用 ts_rank |
| `PE_PCT_250` | 待实现 |  | `pe_ttm 在自身 250 日中的分位` | 用 ts_rank |
| `SP_TTM` | 待实现 |  | `1 / ps_ttm，仅 ps_ttm > 0` |  |

## 财务与质量（`fundamental`）

共 18 条，其中已实现 18 条。

| 因子 | 状态 | 方向 | 公式 | 说明 |
|---|---|---|---|---|
| `ACCRUAL` | 已实现 | 越大越看空 | `(净利润 - 经营现金流) / 总资产` | 应计利润占资产比例，越高说明利润的现金含量越差。 |
| `ASSET_GROWTH` | 已实现 | 越大越看多 | `assets_yoy / 100` | 总资产同比增速，衡量扩张速度。 |
| `ASSET_TURNOVER` | 已实现 | 越大越看多 | `assets_turn` | 总资产周转率，衡量资产产生收入的能力。 |
| `CURRENT_RATIO` | 已实现 | 越大越看多 | `current_ratio` | 流动比率，衡量短期偿债能力。 |
| `DEBT_TO_ASSETS` | 已实现 | 越大越看空 | `debt_to_assets / 100` | 资产负债率，越高通常财务风险越大。 |
| `EQUITY_GROWTH` | 已实现 | 越大越看多 | `eqt_yoy / 100` | 净资产同比增速，衡量内生积累与扩张。 |
| `FCF_YIELD` | 已实现 | 越大越看多 | `(经营现金流 - 资本开支) / 流通市值` | 自由现金流收益率，衡量现金回报相对市值的吸引力。 |
| `GROSS_MARGIN` | 已实现 | 越大越看多 | `grossprofit_margin / 100` | 毛利率，反映产品定价能力与成本控制。 |
| `NET_MARGIN` | 已实现 | 越大越看多 | `netprofit_margin / 100` | 净利率，反映收入最终转化为净利润的比例。 |
| `NP_GROWTH` | 已实现 | 越大越看多 | `netprofit_yoy / 100` | 归母净利润同比增速，衡量盈利增长。 |
| `OCF_GROWTH` | 已实现 | 越大越看多 | `ocf_yoy / 100` | 经营现金流同比增速，衡量现金创造能力的改善。 |
| `OCF_TO_NP` | 已实现 | 越大越看多 | `经营现金流净额 / 净利润` | 净利润的现金含量，越高说明盈利质量越好。 |
| `OP_GROWTH` | 已实现 | 越大越看多 | `op_yoy / 100` | 营业利润同比增速，剔除营业外项目后的增长。 |
| `QUICK_RATIO` | 已实现 | 越大越看多 | `quick_ratio` | 速动比率，剔除存货后的短期偿债能力。 |
| `REV_GROWTH` | 已实现 | 越大越看多 | `or_yoy / 100` | 营业收入同比增速，衡量业务扩张。 |
| `ROA_YEARLY` | 已实现 | 越大越看多 | `roa_yearly / 100` | 年化总资产收益率，衡量全部资产的盈利效率。 |
| `ROE_YEARLY` | 已实现 | 越大越看多 | `roe_yearly / 100` | 年化净资产收益率，衡量股东资金的使用效率。 |
| `ROIC` | 已实现 | 越大越看多 | `roic / 100` | 投入资本回报率，兼顾债权与股权资本的使用效率。 |

## A 股特色（`cn_special`）

共 14 条，其中已实现 2 条。

| 因子 | 状态 | 方向 | 公式 | 说明 |
|---|---|---|---|---|
| `LIMIT_UP_CNT_20` | 已实现 | 越大越看空 | `sum(is_limit_up, 20)` | 20 日内涨停天数，衡量资金关注度；涨停越多短期越容易反转。 |
| `OVERNIGHT_RET_20` | 已实现 | 越大越看空 | `mean(open / pre_close - 1, 20)` | 20 日隔夜跳空均值，隔夜收益在 A 股长期偏负，反映散户开盘追高。 |
| `BOMB_RATE_20` | 待实现 |  | `炸板天数 / 触及涨停天数，窗口 20` | 用 touched_limit_up 与 is_limit_up 之差 |
| `CONSEC_LIMIT_UP` | 待实现 |  | `当前连续涨停天数` | 需多层：连续计数 |
| `EX_DIV_CNT_250` | 待实现 |  | `250 日内除权除息次数` | 用 is_ex_div 计数 |
| `INTRADAY_RET_20` | 待实现 |  | `mean(close/open - 1, 20)` | 日内收益，与隔夜收益互补 |
| `LIMIT_DOWN_CNT_20` | 待实现 |  | `sum(is_limit_down, 20)` |  |
| `LIMIT_UP_CNT_60` | 待实现 |  | `sum(is_limit_up, 60)` |  |
| `LIST_DAYS` | 待实现 |  | `trade_date - list_date` | 上市天数，做次新股因子用 |
| `ONEWORD_CNT_20` | 待实现 |  | `sum(is_one_word_board, 20)` | 一字板数量 |
| `OVERNIGHT_RET_5` | 待实现 |  | `mean(open/pre_close - 1, 5)` |  |
| `RESUME_GAP` | 待实现 |  | `复牌首日的跳空幅度` | 用 is_resume 与 open/pre_close 组合 |
| `ST_DAYS_250` | 待实现 |  | `250 日内 ST 天数` | 用 is_st 计数 |
| `SUSPEND_DAYS_60` | 待实现 |  | `60 个自然交易日内的停牌天数` | 用 status='suspended' 计数 |

## 市场与相对强度（`market`）

共 10 条，其中已实现 2 条。

| 因子 | 状态 | 方向 | 公式 | 说明 |
|---|---|---|---|---|
| `BETA_60` | 已实现 | 越大越看空 | `regr_slope(ret, idx_ret, 60)` | 60 日市场贝塔，低贝塔异象：贝塔低的股票风险调整后收益更好；窗口内少于 40 个交易日则置空。 |
| `RS_20` | 已实现 | 越大越看多 | `sum(ret, 20) - sum(idx_ret, 20)` | 20 日相对沪深300 的超额收益，相对强度因子。 |
| `BETA_120` | 待实现 |  | `regr_slope(ret, idx_ret, 120)` |  |
| `BETA_250` | 待实现 |  | `regr_slope(ret, idx_ret, 250)` |  |
| `CORR_MKT_60` | 待实现 |  | `corr(ret, idx_ret, 60)` |  |
| `DOWNSIDE_BETA_60` | 待实现 |  | `仅用指数下跌日的收益回归，60 日` | 需多层：先筛指数下跌日 |
| `IDIO_MOM_20` | 待实现 |  | `20 日收益 - beta * 指数 20 日收益` | 需多层写法 |
| `IDIO_VOL_60` | 待实现 |  | `回归残差的标准差，60 日` | 用 regr_r2 反推，或多层写法 |
| `RS_120` | 待实现 |  | `sum(ret,120) - sum(idx_ret,120)` |  |
| `RS_60` | 待实现 |  | `sum(ret,60) - sum(idx_ret,60)` |  |

## WorldQuant Alpha101 复现（`alpha101`）

共 20 条，其中已实现 3 条。

| 因子 | 状态 | 方向 | 公式 | 说明 |
|---|---|---|---|---|
| `ALPHA101_001` | 已实现 | 越大越看多 | `rank(5 日内极大值出现在几天前) - 0.5，极大值口径为 power(ret<0 ? std(ret,20) : close, 2)` | 原始 Alpha#1：先按 20 日波动放大价格，再看最近 5 日极值出现在几天前，截面排序。 |
| `ALPHA101_006` | 已实现 | 越大越看多 | `-corr(open, volume, 10)` | 原始 Alpha#6：开盘价与成交量的负相关，量价背离的经典写法。 |
| `ALPHA101_012` | 已实现 | 越大越看多 | `sign(delta(volume, 1)) * (-delta(close, 1))` | 原始 Alpha#12：放量则记当日跌幅，缩量则记当日涨幅的取负。 |
| `ALPHA101_002` | 待实现 |  | `-1 * corr(rank(delta(ln(volume),2)), rank((close-open)/open), 6)` |  |
| `ALPHA101_003` | 待实现 |  | `-1 * corr(rank(open), rank(volume), 10)` |  |
| `ALPHA101_004` | 待实现 |  | `-1 * ts_rank(rank(low), 9)` | 需要 ts_rank 双层写法 |
| `ALPHA101_013` | 待实现 |  | `-1 * rank(cov(rank(close), rank(volume), 5))` | 需多层写法 |
| `ALPHA101_014` | 待实现 |  | `-1 * rank(delta(ret,3)) * corr(open, volume, 10)` | 需多层写法 |
| `ALPHA101_015` | 待实现 |  | `-1 * sum(rank(corr(rank(high), rank(volume), 3)), 3)` | 需多层写法 |
| `ALPHA101_016` | 待实现 |  | `-1 * rank(cov(rank(high), rank(volume), 5))` | 需多层写法 |
| `ALPHA101_018` | 待实现 |  | `-1 * rank(std(abs(close-open),5) + (close-open) + corr(close, open, 10))` |  |
| `ALPHA101_020` | 待实现 |  | `-1 * rank(open - delay(high,1)) * rank(open - delay(close,1)) * rank(open - delay(low,1))` |  |
| `ALPHA101_023` | 待实现 |  | `sum(high,20)/20 < high ? -delta(high,2) : 0` |  |
| `ALPHA101_024` | 待实现 |  | `delta(mean(close,100),100)/delay(close,100) <= 0.05 ? -(close - min(close,100)) : -delta(close,3)` | 长窗口，注意 lookback |
| `ALPHA101_026` | 待实现 |  | `-1 * ts_max(corr(ts_rank(volume,5), ts_rank(high,5), 5), 3)` | 需要 ts_rank 双层写法 |
| `ALPHA101_028` | 待实现 |  | `scale(corr(adv20, low, 5) + (high+low)/2 - close)` |  |
| `ALPHA101_033` | 待实现 |  | `rank(-(1 - open/close))` |  |
| `ALPHA101_034` | 待实现 |  | `rank(1 - rank(std(ret,2)/std(ret,5)) + 1 - rank(delta(close,1)))` |  |
| `ALPHA101_041` | 待实现 |  | `sqrt(high * low) - vwap` |  |
| `ALPHA101_101` | 待实现 |  | `(close - open) / (high - low + 0.001) * volume / adv20` |  |

## 国泰君安 Alpha191 复现（`gtja191`）

共 20 条，其中已实现 2 条。

| 因子 | 状态 | 方向 | 公式 | 说明 |
|---|---|---|---|---|
| `GTJA_191_001` | 已实现 | 越大越看多 | `-corr(rank(delta(ln(volume),1)), rank((close-open)/open), 6)` | 原始 Alpha#1：成交量变化的分位与日内涨跌幅的分位做 6 日相关，再取负。 |
| `GTJA_191_003` | 已实现 | 越大越看多 | `sum(close == high ? 1 : (close == low ? -1 : 0), 6)` | 原始 Alpha#3：6 日内收在最高价与最低价的天数之差，衡量收盘强弱。 |
| `GTJA_191_004` | 待实现 |  | `收盘价与均线关系的条件式，见原始公式表` | 条件分支多，注意可读性 |
| `GTJA_191_006` | 待实现 |  | `rank(open - close) 与 rank(ret - open) 的组合式` | 见原始公式表 |
| `GTJA_191_007` | 待实现 |  | `(rank(max(vwap-close,3)) + rank(min(vwap-close,3))) * rank(delta(volume,3))` |  |
| `GTJA_191_008` | 待实现 |  | `rank(delta((high+low)/2*0.2 + vwap*0.8, 4) * -1)` |  |
| `GTJA_191_009` | 待实现 |  | `SMA(((high+low)/2 - (delay(high,1)+delay(low,1))/2) * (high-low)/volume, 7, 2)` | 递归平滑，需多层写法 |
| `GTJA_191_010` | 待实现 |  | `rank(max(ret<0 ? std(ret,20) : close, 5) ^ 2)` | 多层写法 |
| `GTJA_191_011` | 待实现 |  | `sum(((close-low) - (high-close)) / (high-low) * volume, 6)` | 多空力量对比 |
| `GTJA_191_012` | 待实现 |  | `rank(open - mean(vwap,10)) * (-1 * rank(abs(close - vwap)))` |  |
| `GTJA_191_013` | 待实现 |  | `sqrt(high * low) - vwap` |  |
| `GTJA_191_014` | 待实现 |  | `close - delay(close, 5)` |  |
| `GTJA_191_016` | 待实现 |  | `-1 * ts_max(rank(corr(rank(volume), rank(vwap), 5)), 5)` | 多层写法 |
| `GTJA_191_017` | 待实现 |  | `rank(vwap - max(vwap,15)) ^ delta(close,5)` |  |
| `GTJA_191_018` | 待实现 |  | `close / delay(close, 5)` |  |
| `GTJA_191_019` | 待实现 |  | `close < delay(close,5) ? (close-delay(close,5))/delay(close,5) : (close-delay(close,5))/close` |  |
| `GTJA_191_020` | 待实现 |  | `(close - delay(close,6)) / delay(close,6) * 100` |  |
| `GTJA_191_021` | 待实现 |  | `regr_slope(mean(close,6), 序列, 6)` | 用 regr_slope 实现 REGBETA |
| `GTJA_191_023` | 待实现 |  | `SMA(close > delay(close,1) ? std(close,20) : 0, 20, 1) / (该式 + 下跌方向的同式)` | 多层写法 |
| `GTJA_191_024` | 待实现 |  | `SMA(close - delay(close,5), 5, 1)` | 多层写法 |

