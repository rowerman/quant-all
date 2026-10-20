"""信号路由：因子 Top-N 或外部目标权重提供者，以及调仓日和目标权重。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.data import MarketData
from backtest.errors import DataAlignmentError, SpecError
from backtest.execution_state import ProviderState
from backtest.logging_conf import get_logger
from backtest.providers import get_provider, provider_uses_position_state
from backtest.specs import PortfolioSpec, RebalanceSpec, SignalSpec, StrategySpec
from backtest.trading_rules import BOARD_BSE, BSE_TRADING_START
from backtest.universe import MIN_LISTED_DAYS

logger = get_logger("backtest.signals")

WEIGHT_EPS = 1e-9


def rebalance_mask(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    """返回每个交易日是否为调仓信号日。

    参数:
        index: 交易日索引。
        frequency: ``daily`` / ``weekly`` / ``monthly``。
    """
    if frequency == "daily":
        return pd.Series(True, index=index)
    if frequency == "custom":
        raise SpecError("frequency=custom 只用于目标权重提供者")
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

    口径与因子评估的股票池基本一致：可买入收盘、非 ST、非停牌、非退市整理期、
    上市满 ``MIN_LISTED_DAYS`` 个自然日；指数成分池再叠加当日成分归属。

    参数:
        data: 回测输入面板。
    """
    threshold = (data.list_date + pd.Timedelta(days=MIN_LISTED_DAYS)).to_numpy()
    days = data.index.to_numpy()[:, None]
    mask = (
        data.can_buy_close.to_numpy()
        & ~data.is_suspended.to_numpy()
        & ~data.is_st.to_numpy()
        & ~data.is_delisting.to_numpy()
        & (days >= threshold[None, :])
    )
    bse = data.board.reindex(data.columns).eq(BOARD_BSE).to_numpy(dtype=bool)
    if bse.any():
        before_bse = days < pd.Timestamp(BSE_TRADING_START).to_datetime64()
        mask &= ~(before_bse & bse[None, :])
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
    rebalance_days: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成信号日目标权重和选股排名。

    权重矩阵用 ``NaN`` 表示当天不提交订单，``0`` 表示清仓，正数表示目标仓位。

    参数:
        data: 回测输入面板。
        signal: 信号定义。
        portfolio: 组合定义。
        rebalance: 调仓定义。
        rebalance_days: 外部给定的调仓日掩码（整段区间算好后按块切片）；None 表示按
            本面板的日期索引现算。分块回测必须传入整段口径，否则块尾落在周中时
            会被当成周频调仓日。
    """
    scores = build_scores(data, signal)
    eligible = eligible_mask(data)
    mask = (
        rebalance_mask(data.index, rebalance.frequency)
        if rebalance_days is None
        else rebalance_days.reindex(data.index).fillna(False).astype(bool)
    )
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


def validate_provider_weights(
    frame: pd.DataFrame,
    data: MarketData,
    eligible: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """校验提供者返回的目标权重矩阵。

    整行 ``NaN`` 表示不调仓；非空行必须是有限、非负权重，单行合计不超过 1，
    且正权重只能落在 eligible 股票上。
    """
    if not isinstance(frame, pd.DataFrame):
        raise DataAlignmentError("目标权重提供者必须返回 DataFrame")
    if not frame.index.equals(data.index) or not frame.columns.equals(data.columns):
        raise DataAlignmentError("目标权重提供者的日期或股票列与行情面板不一致")
    try:
        values = frame.to_numpy(dtype="float64")
    except (TypeError, ValueError) as error:
        raise DataAlignmentError(f"目标权重提供者返回了非数值权重：{error}") from error
    active = ~np.isnan(values).all(axis=1)
    if np.isnan(values[active]).any():
        raise DataAlignmentError("目标权重提供者的非空调仓行不能包含部分 NaN")
    if not np.isfinite(values[active]).all():
        raise DataAlignmentError("目标权重提供者返回了无穷值")
    if (values[active] < -WEIGHT_EPS).any():
        raise DataAlignmentError("目标权重提供者返回了负权重；当前只支持多头")
    sums = values[active].sum(axis=1)
    if (sums > 1.0 + WEIGHT_EPS).any():
        raise DataAlignmentError("目标权重提供者单行权重合计不能超过 1")
    pool = eligible_mask(data) if eligible is None else eligible
    if not pool.index.equals(data.index) or not pool.columns.equals(data.columns):
        raise DataAlignmentError("eligible 矩阵与行情面板不一致")
    invalid = (values > WEIGHT_EPS) & ~pool.to_numpy(dtype=bool)
    if invalid.any():
        rows, columns = np.where(invalid)
        samples = [
            f"{data.index[row].date()}:{data.columns[column]}"
            for row, column in zip(rows[:5], columns[:5])
        ]
        raise DataAlignmentError(
            "目标权重提供者给不可买入股票分配了正权重："
            + ", ".join(samples)
            + ("..." if len(rows) > 5 else "")
        )
    return pd.DataFrame(values, index=data.index, columns=data.columns, dtype="float64")


def rank_provider_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """按正权重降序派生提供者排名。"""
    return weights.where(weights > 0).rank(axis=1, ascending=False, method="first")


def build_signal_targets(
    data: MarketData,
    strategy: StrategySpec,
    rebalance_days: pd.Series | None = None,
    provider_state: ProviderState | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按策略类型生成信号日目标权重和排名。

    因子策略沿用原有 Top-N 逻辑；提供者策略由注册对象返回权重，拒绝外部再传
    调仓日掩码，避免两套调仓口径同时生效。

    参数:
        provider_state: 上一遍执行账本给出的逐日持仓状态，仅状态提供者使用；
            None 表示首次运行，提供者只能给基准目标。
    """
    if strategy.signal.provider is None:
        return build_target_weights(
            data,
            strategy.signal,
            strategy.portfolio,
            strategy.rebalance,
            rebalance_days,
        )
    if rebalance_days is not None:
        raise SpecError("目标权重提供者自行决定调仓日，不能同时传 rebalance_days")
    provider = get_provider(strategy.signal.provider)
    pool = eligible_mask(data)
    params = dict(strategy.signal.provider.params)
    if provider_uses_position_state(provider):
        if provider_state is not None:
            provider_state = provider_state.slice(data.index)
        raw = provider.build(data, pool, params, state=provider_state)
    else:
        raw = provider.build(data, pool, params)
    weights = validate_provider_weights(raw, data, pool)
    return weights, rank_provider_weights(weights)

