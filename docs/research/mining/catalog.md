# 自动因子挖掘字段与算子目录

- schema：`mining_catalog_v1`
- 数据版本：`tushare_fullA_20000104_20261015`
- 目录哈希：`c7fd70ec06466143508e35eec2b9615d9e0db5a72ee997205149d305155e6a8e`
- 字段总数：95，允许使用：84
- 算子总数：34

## 字段

| 来源 | 字段 | 中文名 | 类型 | 维度 | 单位 | 可用时点 | 允许 | 说明 |
|---|---|---|---|---|---|---|---|---|
| `daily_hfq` | `amount_cny` | 成交额 | number | money | 元 | close | 是 | 可空 |
| `fin_balance` | `bs_accounts_receiv` | 应收账款 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_balance` | `bs_goodwill` | 商誉 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_balance` | `bs_inventories` | 存货 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_balance` | `bs_money_cap` | 货币资金 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_balance` | `bs_total_assets` | 总资产 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_balance` | `bs_total_cur_assets` | 流动资产合计 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_balance` | `bs_total_cur_liab` | 流动负债合计 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_balance` | `bs_total_hldr_eqy_exc_min_int` | 归母股东权益 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_balance` | `bs_total_liab` | 总负债 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_balance` | `bs_total_share` | 总股本 | number | shares | 股 | next_trading_day | 是 | 可空 |
| `daily_hfq` | `can_buy_close` | 收盘可买 | boolean | boolean | 布尔 | close | 是 | 可空 |
| `daily_hfq` | `can_buy_open_next` | 次日开盘可买 | boolean | boolean | 布尔 | close | 是 | 可空 |
| `daily_hfq` | `can_sell_close` | 收盘可卖 | boolean | boolean | 布尔 | close | 是 | 可空 |
| `daily_hfq` | `can_sell_open_next` | 次日开盘可卖 | boolean | boolean | 布尔 | close | 是 | 可空 |
| `fin_cashflow` | `cf_c_pay_acq_const_fiolta` | 购建长期资产现金 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_cashflow` | `cf_depr_fa_coga_dpba` | 折旧与摊销 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_cashflow` | `cf_free_cashflow` | 自由现金流 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_cashflow` | `cf_n_cash_flows_fnc_act` | 筹资活动现金流净额 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_cashflow` | `cf_n_cashflow_act` | 经营活动现金流净额 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_cashflow` | `cf_n_cashflow_inv_act` | 投资活动现金流净额 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_cashflow` | `cf_n_incr_cash_cash_equ` | 现金及等价物净增加 | number | money | 元 | next_trading_day | 是 | 可空 |
| `daily_hfq` | `change` | 未复权涨跌额 | number | price | 元 | close | 否 | 与后复权价格混用会产生尺度错误 |
| `daily_hfq` | `circ_mv` | 流通市值 | number | money | 万元 | close | 是 | 可空 |
| `daily_hfq` | `close` | 后复权收盘价 | number | price | 元 | close | 是 | 可空 |
| `daily_hfq` | `delist_date` | 退市日 | date | time | 日期 | close | 否 | 日期不能作为因子输入 |
| `daily_hfq` | `dv_ttm` | 股息率 TTM | number | ratio | % | close | 是 | 可空 |
| `daily_hfq` | `factor` | 后复权因子 | number | scalar | 无量纲 | close | 是 | 可空 |
| `fin_indicator` | `fi_assets_turn` | 总资产周转率 | number | scalar | 倍 | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_assets_yoy` | 总资产同比 | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_cash_ratio` | 现金比率 | number | scalar | 倍 | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_current_ratio` | 流动比率 | number | scalar | 倍 | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_debt_to_assets` | 资产负债率 | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_eqt_yoy` | 净资产同比 | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_grossprofit_margin` | 毛利率 | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_netprofit_margin` | 净利率 | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_netprofit_yoy` | 净利润同比 | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_ocf_yoy` | 经营现金流同比 | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_op_yoy` | 营业利润同比 | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_or_yoy` | 营业收入同比 | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_profit_dedt` | 扣非净利润 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_q_op_qoq` | 单季营业利润环比 | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_q_sales_yoy` | 单季营收同比 | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_quick_ratio` | 速动比率 | number | scalar | 倍 | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_roa` | ROA | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_roa2_yearly` | 年化 ROA2 | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_roa_yearly` | 年化 ROA | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_roe` | ROE | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_roe_dt` | 扣非 ROE | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_roe_waa` | 加权平均 ROE | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_roe_yearly` | 年化 ROE | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_roic` | ROIC | number | ratio | % | next_trading_day | 是 | 可空 |
| `fin_indicator` | `fi_tr_yoy` | 营业总收入同比 | number | ratio | % | next_trading_day | 是 | 可空 |
| `daily_hfq` | `high` | 后复权最高价 | number | price | 元 | close | 是 | 可空 |
| `index_000300` | `idx_ret_300` | 沪深300日收益 | number | ratio | 小数 | close | 是 | 可空 |
| `index_000905` | `idx_ret_500` | 中证500日收益 | number | ratio | 小数 | close | 是 | 可空 |
| `index_000985` | `idx_ret_all` | 中证全指日收益 | number | ratio | 小数 | close | 是 | 可空 |
| `fin_income` | `inc_n_income` | 净利润 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_income` | `inc_n_income_attr_p` | 归母净利润 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_income` | `inc_oper_cost` | 营业成本 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_income` | `inc_operate_profit` | 营业利润 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_income` | `inc_rd_exp` | 研发费用 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_income` | `inc_revenue` | 营业收入 | number | money | 元 | next_trading_day | 是 | 可空 |
| `fin_income` | `inc_total_revenue` | 营业总收入 | number | money | 元 | next_trading_day | 是 | 可空 |
| `daily_hfq` | `is_delisting` | 退市整理期 | boolean | boolean | 布尔 | close | 是 | 可空 |
| `daily_hfq` | `is_ex_div` | 当日除权除息 | boolean | boolean | 布尔 | close | 是 | 可空 |
| `daily_hfq` | `is_limit_down` | 收盘跌停 | boolean | boolean | 布尔 | close | 是 | 可空 |
| `daily_hfq` | `is_limit_up` | 收盘涨停 | boolean | boolean | 布尔 | close | 是 | 可空 |
| `daily_hfq` | `is_one_word_board` | 一字板 | boolean | boolean | 布尔 | close | 是 | 可空 |
| `daily_hfq` | `is_price_limit_applicable` | 适用涨跌停规则 | boolean | boolean | 布尔 | close | 是 | 可空 |
| `daily_hfq` | `is_resume` | 当日复牌 | boolean | boolean | 布尔 | close | 是 | 可空 |
| `daily_hfq` | `is_st` | 当日 ST | boolean | boolean | 布尔 | close | 是 | 可空 |
| `daily_hfq` | `is_suspended` | 当日停牌 | boolean | boolean | 布尔 | close | 是 | 可空 |
| `daily_hfq` | `limit_down` | 未复权跌停价 | number | price | 元 | close | 否 | 与后复权价格混用会产生尺度错误 |
| `daily_hfq` | `limit_ratio` | 涨跌停幅度 | number | ratio | 小数 | close | 是 | 可空 |
| `daily_hfq` | `limit_up` | 未复权涨停价 | number | price | 元 | close | 否 | 与后复权价格混用会产生尺度错误 |
| `daily_hfq` | `list_date` | 上市日 | date | time | 日期 | close | 否 | 日期不能作为因子输入 |
| `daily_hfq` | `low` | 后复权最低价 | number | price | 元 | close | 是 | 可空 |
| `daily_hfq` | `open` | 后复权开盘价 | number | price | 元 | close | 是 | 可空 |
| `daily_hfq` | `pb` | 市净率 | number | ratio | 倍 | close | 是 | 可空 |
| `daily_hfq` | `pe_ttm` | 市盈率 TTM | number | ratio | 倍 | close | 是 | 可空 |
| `daily_hfq` | `pre_close` | 未复权前收盘 | number | price | 元 | close | 否 | 与后复权价格混用会产生尺度错误 |
| `daily_hfq` | `pre_close_source` | 前收来源 | string | audit | 文本 | close | 否 | 审计字段不能作为因子输入 |
| `daily_hfq` | `ps_ttm` | 市销率 TTM | number | ratio | 倍 | close | 是 | 可空 |
| `daily_hfq` | `ret` | 日收益 | number | ratio | 小数 | close | 是 | 可空 |
| `daily_hfq` | `st_source` | ST 来源 | string | audit | 文本 | close | 否 | 审计字段不能作为因子输入 |
| `daily_hfq` | `status` | 交易状态 | string | enum | 枚举 | close | 否 | 状态不能作为数值因子输入 |
| `daily_hfq` | `stock_code` | 股票代码 | string | identifier | 代码 | close | 否 | 代码是键，不是因子输入 |
| `daily_hfq` | `total_mv` | 总市值 | number | money | 万元 | close | 是 | 可空 |
| `daily_hfq` | `touched_limit_down` | 盘中触及跌停 | boolean | boolean | 布尔 | close | 是 | 可空 |
| `daily_hfq` | `touched_limit_up` | 盘中触及涨停 | boolean | boolean | 布尔 | close | 是 | 可空 |
| `daily_hfq` | `trade_date` | 交易日 | date | time | 日期 | close | 否 | 日期是键，不允许进入因子值 |
| `daily_hfq` | `turnover_rate` | 换手率 | number | ratio | % | close | 是 | 可空 |
| `daily_hfq` | `volume_share` | 成交量 | number | shares | 股 | close | 是 | 可空 |
| `daily_hfq` | `vwap` | 后复权成交量加权价 | number | price | 元 | close | 是 | 可空 |

## 算子

| 算子 | 类型 | 参数数 | 窗口范围 | 说明 |
|---|---|---|---|---|
| `abs` | scalar | 1～1 | 无 | 绝对值 |
| `add` | arithmetic | 2～2 | 无 | 加法 |
| `corr` | time_series | 3～3 | 2～252 | 滚动相关 |
| `count` | time_series | 2～2 | 1～252 | 滚动非空计数 |
| `cov` | time_series | 3～3 | 2～252 | 滚动样本协方差 |
| `delta` | time_series | 2～2 | 1～252 | 滞后差分 |
| `demean` | cross_section | 1～1 | 无 | 截面去均值 |
| `div` | arithmetic | 2～2 | 无 | 安全除法，分母为零返回空值 |
| `downside_std` | time_series | 2～2 | 2～252 | 滚动下行波动 |
| `greatest` | scalar | 2～8 | 无 | 逐行最大值 |
| `kurt` | time_series | 2～2 | 1～252 | 滚动峰度 |
| `lag` | time_series | 2～2 | 1～252 | 滞后期数 |
| `least` | scalar | 2～8 | 无 | 逐行最小值 |
| `ln` | scalar | 1～1 | 无 | 自然对数 |
| `max` | time_series | 2～2 | 1～252 | 滚动最大值 |
| `mean` | time_series | 2～2 | 1～252 | 滚动均值 |
| `median` | time_series | 2～2 | 1～252 | 滚动中位数 |
| `min` | time_series | 2～2 | 1～252 | 滚动最小值 |
| `mul` | arithmetic | 2～2 | 无 | 乘法 |
| `neg` | arithmetic | 1～1 | 无 | 取负 |
| `product` | time_series | 2～2 | 1～252 | 滚动乘积 |
| `r2` | time_series | 3～3 | 2～252 | 滚动回归决定系数 |
| `rank` | cross_section | 1～1 | 无 | 截面分位 |
| `scale` | cross_section | 1～1 | 无 | 截面绝对值归一 |
| `sign` | scalar | 1～1 | 无 | 符号 |
| `signed_power` | parameterized | 2～2 | 无 | 保号幂 |
| `skew` | time_series | 2～2 | 1～252 | 滚动偏度 |
| `slope` | time_series | 3～3 | 2～252 | 滚动回归斜率 |
| `sqrt` | scalar | 1～1 | 无 | 平方根 |
| `std` | time_series | 2～2 | 1～252 | 样本标准差 |
| `sub` | arithmetic | 2～2 | 无 | 减法 |
| `sum` | time_series | 2～2 | 1～252 | 滚动求和 |
| `ts_rank` | time_series | 2～2 | 2～252 | 当前值在最近窗口中的分位 |
| `zscore` | cross_section | 1～1 | 无 | 截面标准化 |

## 安全边界

- 财务字段只在公告日次日及以后可用，连接条件固定为 `trade_date > available_date`。
- 表达式只接受本目录中的字段和算子，不接受字符串、原始 SQL、注释或多语句。
- 股票代码、日期、状态、标签、审计列和未复权价格字段不能进入表达式。
