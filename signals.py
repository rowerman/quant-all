"""因子截面信号、调仓日和目标权重。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.data import MarketData
from backtest.errors import DataAlignmentError
from backtest.logging_conf import get_logger
from backtest.specs import PortfolioSpec, RebalanceSpec, SignalSpec
from backtest.universe import MIN_LISTED_DAYS

logger = get_logger("backtest.signals")


def rebalance_mask(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    """返回每个交易日是否为调仓信号日。

    参数:
        index: 交易日索引。
        frequency: ``daily`` / ``weekly`` / ``monthly``。
    """
    if frequency == "daily":
        return pd.Series(True, index=index)
    periods = pd.Series(index.to_period("W-FRI" if frequency == "weekly" else "M"), index=index)
    return periods.ne(periods.shift(-1)).fillna(True)


def _cross_section(frame: pd.DataFrame, standardize: bool) -> pd.DataFrame:
    """按日期对因子做截面标准化。"""
    if not standardize:
        return frame
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def build_scores(data: MarketData, spec: SignalSpec) -> pd.DataFrame:
    """生成因子线性组合分数。

    参数:
        data: 回测输入面板。
        spec: 信号定义，含因子名与线性权重。
    """
    wide = []
    for name, weight in zip(spec.factors, spec.weights):
        if name not in data.factors:
            raise DataAlignmentError(f"回测输入缺少因子 {name}；已加载：{sorted(data.factors)}")
        value = data.factors[name].astype("float64")
        if spec.winsorize:
            low = value.quantile(0.01, axis=1)
            high = value.quantile(0.99, axis=1)
            value = value.clip(lower=low, upper=high, axis=0)
        wide.append(_cross_section(value, spec.standardize) * float(weight))
    return sum(wide).replace([np.inf, -np.inf], np.nan)


def eligible_mask(data: MarketData) -> pd.DataFrame:
    """给出「信号日可入选」的布尔矩阵。

    口径与因子评估的股票池一致：可买入收盘、非 ST、非停牌、上市满
    ``MIN_LISTED_DAYS`` 个自然日；指数成分池再叠加当日成分归属。

    参数:
        data: 回测输入面板。
    """
    threshold = (data.list_date + pd.Timedelta(days=MIN_LISTED_DAYS)).to_numpy()
    days = data.index.to_numpy()[:, None]
    mask = (
        data.can_buy_close.to_numpy()
        & ~data.is_suspended.to_numpy()
        & ~data.is_st.to_numpy()
        & (days >= threshold[None, :])
    )
    if data.membership is not None:
        mask &= data.membership.to_numpy()
    return pd.DataFrame(mask, index=data.index, columns=data.columns)


def _weights_for(day: pd.Timestamp, chosen: pd.Index, data: MarketData, portfolio: PortfolioSpec) -> np.ndarray:
    """按权重方式给出选中股票的权重向量。

    参数:
        day: 信号日。
        chosen: 选中的股票。
        data: 回测输入面板。
        portfolio: 组合定义。
    """
    if portfolio.weighting == "equal":
        return np.full(len(chosen), 1.0 / len(chosen))
    market_cap = data.total_mv.loc[day, chosen].astype("float64").to_numpy()
    filled = np.where(np.isfinite(market_cap) & (market_cap > 0), market_cap, np.nan)
    if not np.isfinite(filled).any():
        logger.warning("%s 选中股票的市值全部缺失，本次退回等权", day)
        return np.full(len(chosen), 1.0 / len(chosen))
    filled = np.where(np.isfinite(filled), filled, np.nanmedian(filled))
    return filled / filled.sum()


def build_target_weights(
    data: MarketData,
    signal: SignalSpec,
    portfolio: PortfolioSpec,
    rebalance: RebalanceSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成信号日目标权重和选股排名。

    权重矩阵用 ``NaN`` 表示当天不提交订单，``0`` 表示清仓，正数表示目标仓位。

    参数:
        data: 回测输入面板。
        signal: 信号定义。
        portfolio: 组合定义。
        rebalance: 调仓定义。
    """
    scores = build_scores(data, signal)
    eligible = eligible_mask(data)
    mask = rebalance_mask(data.index, rebalance.frequency)
    weights = pd.DataFrame(np.nan, index=data.index, columns=data.columns, dtype=float)
    ranks = pd.DataFrame(np.nan, index=data.index, columns=data.columns, dtype=float)
    score_values = scores.to_numpy()
    eligible_values = eligible.to_numpy()
    columns = data.columns
    for position in np.flatnonzero(mask.to_numpy()):
        row = np.where(eligible_values[position] & np.isfinite(score_values[position]),
                       score_values[position], np.nan)
        series = pd.Series(row, index=columns).dropna()
        if series.empty:
            continue
        day = data.index[position]
        order = series.sort_values(ascending=False)
        ranks.loc[day] = order.rank(method="first", ascending=False).reindex(columns)
        chosen = order.head(portfolio.top_n).index
        weights.loc[day] = 0.0
        weights.loc[day, chosen] = _weights_for(day, chosen, data, portfolio)
    return weights, ranks

