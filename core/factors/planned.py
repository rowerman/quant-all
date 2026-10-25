"""因子规划清单：可做的经典因子与构建方法。

这份清单是「因子库要做到多大」的路线图，条目比已实现的因子多得多：
其中已经注册（出现在 ``factors.registry`` 里）的会在文档里标成「已实现」，
其余是待补齐项，公式与依赖都写在下面，照着写一个 ``define(...)`` 即可。

约定：

- 量价类因子一律用后复权价（``md.daily_hfq``）；
- ``adv20`` 指 20 日平均成交额，本项目用 ``ts_mean('amount_cny', 20)``；
- ``rank`` 指截面分位 ``cs_rank``，``delay`` 指 ``ts_delay``；
- 财务类因子已接入 ``fin.*`` 时点数据，条目列在「财务与质量」分组。
"""

from __future__ import annotations


def _p(name: str, family: str, formula: str, note: str = "") -> dict:
    """构造一条规划条目。

    参数:
        name: 因子名。
        family: 因子族。
        formula: 公式文本。
        note: 实现要点或依赖说明。
    """
    return {"name": name, "family": family, "formula": formula, "note": note}


PLANNED: tuple[dict, ...] = (
    # ---------- 动量与反转 ----------
    _p("MOM_5", "momentum", "close_t / close_(t-5) - 1", "一周动量，实际更常作为反转用"),
    _p("MOM_10", "momentum", "close_t / close_(t-10) - 1", ""),
    _p("MOM_20", "momentum", "close_t / close_(t-20) - 1", ""),
    _p("MOM_60", "momentum", "close_t / close_(t-60) - 1", ""),
    _p("MOM_120", "momentum", "close_t / close_(t-120) - 1", "半年动量"),
    _p("MOM_250", "momentum", "close_t / close_(t-250) - 1", "一年动量"),
    _p("MOM_60_20", "momentum", "close_(t-20) / close_(t-60) - 1", "跳过最近一月的动量"),
    _p("MOM_250_20", "momentum", "close_(t-20) / close_(t-250) - 1", "12-1 动量，国际通用口径"),
    _p("WGT_MOM_60", "momentum", "sum(w_i * ret_i, 60)，w_i 线性递增", "线性加权动量，需多层写法"),
    _p("HIGH_52W_DIST", "momentum", "close_t / max(high, 250) - 1", "距 52 周高点的距离，越接近 0 越强"),
    _p("LOW_52W_DIST", "momentum", "close_t / min(low, 250) - 1", "距 52 周低点的距离"),
    _p("EMA_MOM", "momentum", "ema(close,12) / ema(close,26) - 1", "EMA 无内置算子，用多层或近似"),
    _p("TREND_SLOPE_T", "momentum", "close 对时间的滚动回归 t 值，窗口 60", "趋势强度，用 regr_slope 与 regr_r2 组合"),
    _p("MOM_ACC", "momentum", "MOM_20 - delay(MOM_20, 20)", "动量加速度，需多层写法"),
    _p("REV_5", "momentum", "-(close_t / close_(t-5) - 1)", ""),
    _p("REV_10", "momentum", "-(close_t / close_(t-10) - 1)", ""),
    _p("REV_20", "momentum", "-(close_t / close_(t-20) - 1)", ""),
    _p("REV_60", "momentum", "-(close_t / close_(t-60) - 1)", "长端反转，A 股通常不显著"),

    # ---------- 波动与风险 ----------
    _p("VOL_20", "volatility", "std(ret, 20)", ""),
    _p("VOL_60", "volatility", "std(ret, 60)", ""),
    _p("VOL_120", "volatility", "std(ret, 120)", ""),
    _p("DOWNSIDE_VOL_20", "volatility", "sqrt(mean(min(ret,0)^2, 20))", ""),
    _p("DOWNSIDE_VOL_60", "volatility", "sqrt(mean(min(ret,0)^2, 60))", ""),
    _p("UPSIDE_VOL_20", "volatility", "sqrt(mean(max(ret,0)^2, 20))", ""),
    _p("VOL_RATIO_20", "volatility", "上行波动 / 下行波动，窗口 20", "值大说明涨得比跌得凶"),
    _p("SKEW_20", "volatility", "skewness(ret, 20)", ""),
    _p("SKEW_60", "volatility", "skewness(ret, 60)", ""),
    _p("KURT_20", "volatility", "kurtosis(ret, 20)", ""),
    _p("MAX_RET_20", "volatility", "max(ret, 20)", "彩票效应"),
    _p("MAX_RET_60", "volatility", "max(ret, 60)", ""),
    _p("MAXDD_20", "volatility", "窗口内最大回撤，20 日", "需多层：窗口高点与当期回撤"),
    _p("MAXDD_60", "volatility", "窗口内最大回撤，60 日", ""),
    _p("PARKINSON_20", "volatility", "sqrt(mean(ln(high/low)^2, 20) / (4*ln2))", "用高低价估波动，比收盘价更有效"),
    _p("GK_VOL_20", "volatility", "Garman-Klass 高低开收估计量，20 日", ""),
    _p("RS_VOL_20", "volatility", "Rogers-Satchell 估计量，20 日", ""),
    _p("ATR_14", "volatility", "mean(true_range, 14) / close", "真实波幅占价格比例"),
    _p("RET_STAB_60", "volatility", "mean(|ret|, 60) / std(ret, 60)", "收益稳定性"),
    _p("VOL_OF_VOL_60", "volatility", "std(VOL_20, 60)", "波动率的波动，需多层写法"),

    # ---------- 流动性 ----------
    _p("AMIHUD_20", "liquidity", "mean(|ret| / (amount/1e8), 20)", ""),
    _p("AMIHUD_60", "liquidity", "mean(|ret| / (amount/1e8), 60)", ""),
    _p("AMIHUD_120", "liquidity", "mean(|ret| / (amount/1e8), 120)", ""),
    _p("TURN_5", "liquidity", "mean(turnover_rate, 5)", ""),
    _p("TURN_20", "liquidity", "mean(turnover_rate, 20)", ""),
    _p("TURN_60", "liquidity", "mean(turnover_rate, 60)", ""),
    _p("TURN_STD_20", "liquidity", "std(turnover_rate, 20)", "换手率波动，交易热度不稳"),
    _p("TURN_BIAS_20", "liquidity", "turnover_rate / mean(turnover_rate, 20) - 1", "当日换手相对近期均值的偏离"),
    _p("LN_AMT_20", "liquidity", "ln(mean(amount, 20))", ""),
    _p("LN_AMT_60", "liquidity", "ln(mean(amount, 60))", ""),
    _p("AMT_SHARE_20", "liquidity", "mean(amount, 20) / 当日全市场成交额", "需截面求和，用 cs_scale 之外的写法"),
    _p("ZERO_VOL_20", "liquidity", "20 日内停牌或零成交天数占比", "用 is_suspended 或 volume=0 计数"),
    _p("ZERO_VOL_60", "liquidity", "60 日内停牌或零成交天数占比", ""),
    _p("VOL_STAB_60", "liquidity", "std(ln(volume), 60)", "成交量的稳定性"),
    _p("FREE_FLOAT_RATIO", "liquidity", "circ_mv / total_mv", "自由流通比例"),
    _p("TURN_TREND", "liquidity", "mean(turnover_rate,5) / mean(turnover_rate,60) - 1", "换手率的短期趋势"),

    # ---------- 量价关系 ----------
    _p("PV_CORR_20", "volume_price", "corr(ret, volume, 20)", ""),
    _p("PV_CORR_60", "volume_price", "corr(ret, volume, 60)", ""),
    _p("PV_CORR_120", "volume_price", "corr(ret, volume, 120)", ""),
    _p("VPT_20", "volume_price", "sum(sign(ret) * volume, 20)", "量价趋势累积"),
    _p("VPT_60", "volume_price", "sum(sign(ret) * volume, 60)", ""),
    _p("OBV_SLOPE_20", "volume_price", "OBV 序列对时间的 20 日回归斜率", "需多层：先累积 OBV"),
    _p("MFI_14", "volume_price", "资金流量指标，14 日", "典型价与成交额加权"),
    _p("VWAP_BIAS_5", "volume_price", "mean(close / vwap - 1, 5)", ""),
    _p("VWAP_BIAS_20", "volume_price", "mean(close / vwap - 1, 20)", ""),
    _p("VWAP_DEV_20", "volume_price", "vwap / mean(vwap, 20) - 1", "成交均价相对自身的偏离"),
    _p("AMT_RET_CORR_60", "volume_price", "corr(|ret|, amount, 60)", "放量是否伴随大波动"),
    _p("TURN_RET_CORR_60", "volume_price", "corr(ret, turnover_rate, 60)", ""),
    _p("VOL_PRICE_DIVERGE", "volume_price", "价格新高但成交量萎缩的标志", "需多层：比较 20 日价格分位与量分位"),

    # ---------- 技术形态 ----------
    _p("RSI_6", "technical", "100 * mean(gain,6) / (mean(gain,6) + mean(loss,6))", ""),
    _p("RSI_14", "technical", "100 * mean(gain,14) / (mean(gain,14) + mean(loss,14))", ""),
    _p("RSI_24", "technical", "100 * mean(gain,24) / (mean(gain,24) + mean(loss,24))", ""),
    _p("MACD_DIF", "technical", "ema(close,12) - ema(close,26)", "EMA 需多层或近似"),
    _p("MACD_DEA", "technical", "ema(MACD_DIF, 9)", "需多层写法"),
    _p("MACD_HIST", "technical", "2 * (MACD_DIF - MACD_DEA)", "需多层写法"),
    _p("BIAS_5", "technical", "close / mean(close,5) - 1", ""),
    _p("BIAS_10", "technical", "close / mean(close,10) - 1", ""),
    _p("BIAS_20", "technical", "close / mean(close,20) - 1", ""),
    _p("BIAS_60", "technical", "close / mean(close,60) - 1", ""),
    _p("BOLL_POS_20", "technical", "(close - (mid - 2sd)) / (4sd)", ""),
    _p("BOLL_WIDTH_20", "technical", "4 * std(close,20) / mean(close,20)", "带宽，波动状态代理"),
    _p("KDJ_K", "technical", "RSV 的 2/3 平滑 + 1/3 前值", "递归平滑，需多层写法"),
    _p("KDJ_D", "technical", "K 值再做 2/3 平滑", "需多层写法"),
    _p("KDJ_J", "technical", "3K - 2D", "需多层写法"),
    _p("CCI_14", "technical", "(TP - mean(TP,14)) / (0.015 * 平均绝对偏差)", "TP 为典型价"),
    _p("WR_14", "technical", "(max(high,14) - close) / (max(high,14) - min(low,14))", "威廉指标"),
    _p("ADX_14", "technical", "14 日平均趋向指数", "需多层写法"),
    _p("MA_ALIGN", "technical", "MA5>MA10>MA20>MA60 的得分", "均线多头排列打分"),
    _p("MA_CROSS_5_20", "technical", "mean(close,5) / mean(close,20) - 1", "金叉死叉的连续版本"),
    _p("PSY_12", "technical", "12 日内上涨天数占比", ""),
    _p("VR_26", "technical", "26 日上涨日成交额 / 下跌日成交额", ""),
    _p("ROC_12", "technical", "close / delay(close,12) - 1", ""),
    _p("CLOSE_POS_20", "technical", "(close - min(low,20)) / (max(high,20) - min(low,20))", "价格在近期区间中的位置"),
    _p("AMPLITUDE_20", "technical", "mean((high - low) / pre_close, 20)", "平均振幅"),

    # ---------- 估值与规模 ----------
    _p("EP_TTM", "valuation", "1 / pe_ttm，仅 pe_ttm > 0", ""),
    _p("BP", "valuation", "1 / pb，仅 pb > 0", ""),
    _p("SP_TTM", "valuation", "1 / ps_ttm，仅 ps_ttm > 0", ""),
    _p("DP_TTM", "valuation", "dv_ttm", "股息率，覆盖约六成行"),
    _p("LN_MV", "valuation", "ln(total_mv)", ""),
    _p("LN_CIRC_MV", "valuation", "ln(circ_mv)", "流通市值，小盘因子的常用口径"),
    _p("EP_CHG_60", "valuation", "EP_TTM - delay(EP_TTM, 60)", "估值变化动量，需多层写法"),
    _p("BP_CHG_60", "valuation", "BP - delay(BP, 60)", "需多层写法"),
    _p("PE_PCT_250", "valuation", "pe_ttm 在自身 250 日中的分位", "用 ts_rank"),
    _p("PB_PCT_250", "valuation", "pb 在自身 250 日中的分位", "用 ts_rank"),
    _p("MV_CHG_20", "valuation", "ln(total_mv) - delay(ln(total_mv), 20)", "市值变化，需注意股本变动"),
    _p("CIRC_MV_RATIO", "valuation", "circ_mv / total_mv", ""),

    # ---------- 财务与质量（已实现） ----------
    _p("ROE_YEARLY", "fundamental", "roe_yearly / 100", "年化 ROE，按公告日次一交易日可用"),
    _p("ROA_YEARLY", "fundamental", "roa_yearly / 100", "年化 ROA"),
    _p("ROIC", "fundamental", "roic / 100", "投入资本回报率"),
    _p("GROSS_MARGIN", "fundamental", "grossprofit_margin / 100", "毛利率"),
    _p("NET_MARGIN", "fundamental", "netprofit_margin / 100", "净利率"),
    _p("REV_GROWTH", "fundamental", "or_yoy / 100", "营业收入同比增速"),
    _p("NP_GROWTH", "fundamental", "netprofit_yoy / 100", "净利润同比增速"),
    _p("OP_GROWTH", "fundamental", "op_yoy / 100", "营业利润同比增速"),
    _p("OCF_GROWTH", "fundamental", "ocf_yoy / 100", "经营现金流同比增速"),
    _p("ASSET_GROWTH", "fundamental", "assets_yoy / 100", "总资产同比增速"),
    _p("EQUITY_GROWTH", "fundamental", "eqt_yoy / 100", "净资产同比增速"),
    _p("ACCRUAL", "fundamental", "(净利润 - 经营现金流) / 总资产", "应计利润，越高盈利质量越差"),
    _p("OCF_TO_NP", "fundamental", "经营现金流净额 / 净利润", "净利润的现金含量"),
    _p("ASSET_TURNOVER", "fundamental", "assets_turn", "总资产周转率"),
    _p("CURRENT_RATIO", "fundamental", "current_ratio", "流动比率"),
    _p("QUICK_RATIO", "fundamental", "quick_ratio", "速动比率"),
    _p("DEBT_TO_ASSETS", "fundamental", "debt_to_assets / 100", "资产负债率"),
    _p("FCF_YIELD", "fundamental", "(经营现金流 - 资本开支) / 流通市值", "自由现金流收益率"),

    # ---------- A 股特色 ----------
    _p("LIMIT_UP_CNT_20", "cn_special", "sum(is_limit_up, 20)", ""),
    _p("LIMIT_UP_CNT_60", "cn_special", "sum(is_limit_up, 60)", ""),
    _p("LIMIT_DOWN_CNT_20", "cn_special", "sum(is_limit_down, 20)", ""),
    _p("BOMB_RATE_20", "cn_special", "炸板天数 / 触及涨停天数，窗口 20", "用 touched_limit_up 与 is_limit_up 之差"),
    _p("ONEWORD_CNT_20", "cn_special", "sum(is_one_word_board, 20)", "一字板数量"),
    _p("CONSEC_LIMIT_UP", "cn_special", "当前连续涨停天数", "需多层：连续计数"),
    _p("OVERNIGHT_RET_5", "cn_special", "mean(open/pre_close - 1, 5)", ""),
    _p("OVERNIGHT_RET_20", "cn_special", "mean(open/pre_close - 1, 20)", ""),
    _p("INTRADAY_RET_20", "cn_special", "mean(close/open - 1, 20)", "日内收益，与隔夜收益互补"),
    _p("SUSPEND_DAYS_60", "cn_special", "60 个自然交易日内的停牌天数", "用 status='suspended' 计数"),
    _p("EX_DIV_CNT_250", "cn_special", "250 日内除权除息次数", "用 is_ex_div 计数"),
    _p("ST_DAYS_250", "cn_special", "250 日内 ST 天数", "用 is_st 计数"),
    _p("LIST_DAYS", "cn_special", "trade_date - list_date", "上市天数，做次新股因子用"),
    _p("RESUME_GAP", "cn_special", "复牌首日的跳空幅度", "用 is_resume 与 open/pre_close 组合"),

    # ---------- 市场与相对强度 ----------
    _p("RS_20", "market", "sum(ret,20) - sum(idx_ret,20)", ""),
    _p("RS_60", "market", "sum(ret,60) - sum(idx_ret,60)", ""),
    _p("RS_120", "market", "sum(ret,120) - sum(idx_ret,120)", ""),
    _p("BETA_60", "market", "regr_slope(ret, idx_ret, 60)", ""),
    _p("BETA_120", "market", "regr_slope(ret, idx_ret, 120)", ""),
    _p("BETA_250", "market", "regr_slope(ret, idx_ret, 250)", ""),
    _p("DOWNSIDE_BETA_60", "market", "仅用指数下跌日的收益回归，60 日", "需多层：先筛指数下跌日"),
    _p("CORR_MKT_60", "market", "corr(ret, idx_ret, 60)", ""),
    _p("IDIO_VOL_60", "market", "回归残差的标准差，60 日", "用 regr_r2 反推，或多层写法"),
    _p("IDIO_MOM_20", "market", "20 日收益 - beta * 指数 20 日收益", "需多层写法"),

    # ---------- Alpha101 复现（不依赖财报） ----------
    _p("ALPHA101_001", "alpha101", "rank(ts_argmax(power(close<0?std(ret,20):close, 2), 5)) - 0.5", "已实现，多层示例"),
    _p("ALPHA101_002", "alpha101", "-1 * corr(rank(delta(ln(volume),2)), rank((close-open)/open), 6)", ""),
    _p("ALPHA101_003", "alpha101", "-1 * corr(rank(open), rank(volume), 10)", ""),
    _p("ALPHA101_004", "alpha101", "-1 * ts_rank(rank(low), 9)", "需要 ts_rank 双层写法"),
    _p("ALPHA101_006", "alpha101", "-1 * corr(open, volume, 10)", "已实现"),
    _p("ALPHA101_012", "alpha101", "sign(delta(volume,1)) * (-1 * delta(close,1))", "已实现"),
    _p("ALPHA101_013", "alpha101", "-1 * rank(cov(rank(close), rank(volume), 5))", "需多层写法"),
    _p("ALPHA101_014", "alpha101", "-1 * rank(delta(ret,3)) * corr(open, volume, 10)", "需多层写法"),
    _p("ALPHA101_015", "alpha101", "-1 * sum(rank(corr(rank(high), rank(volume), 3)), 3)", "需多层写法"),
    _p("ALPHA101_016", "alpha101", "-1 * rank(cov(rank(high), rank(volume), 5))", "需多层写法"),
    _p("ALPHA101_018", "alpha101", "-1 * rank(std(abs(close-open),5) + (close-open) + corr(close, open, 10))", ""),
    _p("ALPHA101_020", "alpha101", "-1 * rank(open - delay(high,1)) * rank(open - delay(close,1)) * rank(open - delay(low,1))", ""),
    _p("ALPHA101_023", "alpha101", "sum(high,20)/20 < high ? -delta(high,2) : 0", ""),
    _p("ALPHA101_024", "alpha101", "delta(mean(close,100),100)/delay(close,100) <= 0.05 ? -(close - min(close,100)) : -delta(close,3)", "长窗口，注意 lookback"),
    _p("ALPHA101_026", "alpha101", "-1 * ts_max(corr(ts_rank(volume,5), ts_rank(high,5), 5), 3)", "需要 ts_rank 双层写法"),
    _p("ALPHA101_028", "alpha101", "scale(corr(adv20, low, 5) + (high+low)/2 - close)", ""),
    _p("ALPHA101_033", "alpha101", "rank(-(1 - open/close))", ""),
    _p("ALPHA101_034", "alpha101", "rank(1 - rank(std(ret,2)/std(ret,5)) + 1 - rank(delta(close,1)))", ""),
    _p("ALPHA101_041", "alpha101", "sqrt(high * low) - vwap", ""),
    _p("ALPHA101_101", "alpha101", "(close - open) / (high - low + 0.001) * volume / adv20", ""),

    # ---------- GTJA191 复现（不依赖财报） ----------
    _p("GTJA_191_001", "gtja191", "-1 * corr(rank(delta(ln(volume),1)), rank((close-open)/open), 6)", "已实现，多层示例"),
    _p("GTJA_191_003", "gtja191", "sum(close == high ? 1 : (close == low ? -1 : 0), 6)", "已实现"),
    _p("GTJA_191_004", "gtja191", "收盘价与均线关系的条件式，见原始公式表", "条件分支多，注意可读性"),
    _p("GTJA_191_006", "gtja191", "rank(open - close) 与 rank(ret - open) 的组合式", "见原始公式表"),
    _p("GTJA_191_007", "gtja191", "(rank(max(vwap-close,3)) + rank(min(vwap-close,3))) * rank(delta(volume,3))", ""),
    _p("GTJA_191_008", "gtja191", "rank(delta((high+low)/2*0.2 + vwap*0.8, 4) * -1)", ""),
    _p("GTJA_191_009", "gtja191", "SMA(((high+low)/2 - (delay(high,1)+delay(low,1))/2) * (high-low)/volume, 7, 2)", "递归平滑，需多层写法"),
    _p("GTJA_191_010", "gtja191", "rank(max(ret<0 ? std(ret,20) : close, 5) ^ 2)", "多层写法"),
    _p("GTJA_191_011", "gtja191", "sum(((close-low) - (high-close)) / (high-low) * volume, 6)", "多空力量对比"),
    _p("GTJA_191_012", "gtja191", "rank(open - mean(vwap,10)) * (-1 * rank(abs(close - vwap)))", ""),
    _p("GTJA_191_013", "gtja191", "sqrt(high * low) - vwap", ""),
    _p("GTJA_191_014", "gtja191", "close - delay(close, 5)", ""),
    _p("GTJA_191_016", "gtja191", "-1 * ts_max(rank(corr(rank(volume), rank(vwap), 5)), 5)", "多层写法"),
    _p("GTJA_191_017", "gtja191", "rank(vwap - max(vwap,15)) ^ delta(close,5)", ""),
    _p("GTJA_191_018", "gtja191", "close / delay(close, 5)", ""),
    _p("GTJA_191_019", "gtja191", "close < delay(close,5) ? (close-delay(close,5))/delay(close,5) : (close-delay(close,5))/close", ""),
    _p("GTJA_191_020", "gtja191", "(close - delay(close,6)) / delay(close,6) * 100", ""),
    _p("GTJA_191_021", "gtja191", "regr_slope(mean(close,6), 序列, 6)", "用 regr_slope 实现 REGBETA"),
    _p("GTJA_191_023", "gtja191", "SMA(close > delay(close,1) ? std(close,20) : 0, 20, 1) / (该式 + 下跌方向的同式)", "多层写法"),
    _p("GTJA_191_024", "gtja191", "SMA(close - delay(close,5), 5, 1)", "多层写法"),
)


def planned_families() -> tuple[str, ...]:
    """规划清单里出现过的因子族，按首次出现顺序。"""
    seen: list[str] = []
    for item in PLANNED:
        if item["family"] not in seen:
            seen.append(item["family"])
    return tuple(seen)
