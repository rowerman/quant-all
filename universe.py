"""回测股票池：全 A 池与指数成分池。

成分口径不在回测侧重写，直接复用计算包的 ``calculate.universe``：
指数成分用**上一个已结束月份**的月末权重快照，避免用到当月才披露的权重造成前视；
可交易性过滤同样是「可买入收盘、非 ST、上市满 60 个自然日」。

这里只把同一套口径落成回测需要的宽表布尔矩阵和取数用的股票并集。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import quant
from backtest.errors import ResourceLimitError, SpecError
from backtest.logging_conf import get_logger
from calculate.errors import UniverseError
from calculate.universe import (  # noqa: F401  对外转出，回测与因子评估共用同一套定义
    ALL_A,
    ALL_A_KEY,
    INDEX_PREFIX,
    INDEX_UNIVERSES,
    MIN_LISTED_DAYS,
    TRADABLE_FILTER,
    UniverseSpec,
    list_universes,
    resolve_universe as _resolve_universe,
)

logger = get_logger("backtest.universe")


def _calendar(end: str, db_path: Path | str | None) -> pd.DatetimeIndex:
    """读取交易日历（独立出来便于测试替换）。"""
    frame = quant.load_calendar(end=end, db_path=db_path)
    return pd.DatetimeIndex(sorted(frame["trade_date"]))


def _weights(index_code: str, end: str, db_path: Path | str | None) -> pd.DataFrame:
    """读取指数权重快照（独立出来便于测试替换）。"""
    return quant.load_index_weight(index_code=index_code, end=end, db_path=db_path)


def resolve_universe(key: str) -> UniverseSpec:
    """按标识取股票池，非法标识换成回测的 SpecError。

    参数:
        key: 股票池标识，``all_a`` 或 ``index:<指数代码>``。
    """
    try:
        return _resolve_universe(key)
    except UniverseError as error:
        raise SpecError(str(error)) from error


def _month_end_snapshot(calendar: pd.DatetimeIndex) -> dict[pd.Period, pd.Timestamp]:
    """按月给出「该月的最后交易日」。

    参数:
        calendar: 交易日历（升序）。
    """
    series = pd.Series(calendar, index=calendar)
    grouped = series.groupby(series.index.to_period("M")).max()
    return dict(grouped.items())


def _member_rows(spec: UniverseSpec, end: str, db_path: Path | str | None) -> pd.DataFrame:
    """取指数成分明细，并标记每条快照归属的生效月份。

    生效口径与计算包一致：第 M 个交易月的成分取第 M-1 个月的月末快照。

    参数:
        spec: 指数成分池定义。
        end: 回测截止日，用于截断日历。
        db_path: 行情库路径。

    返回:
        含 ``period``、``stock_code`` 两列的长表；快照缺失时返回空表。
    """
    calendar = _calendar(end, db_path)
    month_end = _month_end_snapshot(calendar)
    periods = sorted(month_end)
    # 第 M 个月的成分来自第 M-1 个月的月末快照
    snapshot_of_period = {
        period: month_end[periods[position - 1]]
        for position, period in enumerate(periods)
        if position > 0
    }
    weights = _weights(spec.index_code, end, db_path)
    weights = weights.loc[weights["weight"] > 0, ["trade_date", "stock_code"]].copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])
    period_of_snapshot = {stamp: period for period, stamp in snapshot_of_period.items()}
    weights = weights.loc[weights["trade_date"].isin(period_of_snapshot)]
    weights["period"] = pd.PeriodIndex(weights["trade_date"].map(period_of_snapshot), freq="M")
    return weights[["period", "stock_code"]].drop_duplicates()


def membership_frame(
    spec: UniverseSpec,
    index: pd.DatetimeIndex,
    columns: pd.Index,
    end: str,
    *,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """构造「交易日 × 股票」的成分布尔矩阵。

    参数:
        spec: 股票池定义，必须是指数成分池。
        index: 回测交易日索引。
        columns: 行情面板的股票列。
        end: 回测截止日。
        db_path: 行情库路径。
    """
    if not spec.is_index:
        raise SpecError(f"股票池 {spec.key} 不是指数成分池，没有成分数据")
    rows = _member_rows(spec, end, db_path)
    periods = pd.Series(index.to_period("M"), index=index)
    membership = pd.DataFrame(False, index=index, columns=columns)
    used: set[pd.Period] = set()
    for period, group in rows.groupby("period"):
        if period not in periods.to_numpy():
            continue
        codes = columns.intersection(pd.Index(group["stock_code"].unique()))
        membership.loc[periods == period, codes] = True
        used.add(period)
    missing = sorted(set(periods.unique()) - used)
    if missing:
        # 指数首月或权重快照断档时会没有成分，回测里这些交易日不选股
        logger.warning("指数 %s 成分快照缺失的月份：%s", spec.index_code, [str(item) for item in missing])
    return membership


def membership_rows(
    spec: UniverseSpec,
    end: str,
    *,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """返回指数池在截止日前的月末成分明细。

    这是 :func:`membership_frame` 的数据源，也用于回测输入快照；
    每行表示某个生效月份包含的一只股票。

    参数:
        spec: 指数成分池定义。
        end: 回测截止日。
        db_path: 行情库路径。
    """
    if not spec.is_index:
        raise SpecError(f"股票池 {spec.key} 不是指数成分池，没有成分数据")
    return _member_rows(spec, end, db_path)


def universe_stocks(
    spec: UniverseSpec,
    start: str,
    end: str,
    *,
    db_path: Path | str | None = None,
) -> tuple[str, ...] | None:
    """给出该股票池在区间内出现过的股票并集，用于收窄取数范围。

    参数:
        spec: 股票池定义。
        start: 回测起始日。
        end: 回测截止日。
        db_path: 行情库路径。

    返回:
        指数池返回成分并集；全 A 池返回 None（表示不限制）。
    """
    if not spec.is_index:
        return None
    rows = _member_rows(spec, end, db_path)
    first_period = pd.Period(start, freq="M")
    rows = rows.loc[rows["period"] >= first_period]
    stocks = tuple(sorted(set(rows["stock_code"])))
    if not stocks:
        raise ResourceLimitError(
            f"股票池 {spec.key} 在 {start} ~ {end} 没有成分数据，请检查区间是否早于指数首次披露"
        )
    return stocks


def check_limits(days: int, stocks: int, max_days: int, max_cells: int) -> None:
    """检查区间与面板规模是否超过资源上限。

    参数:
        days: 交易日数。
        stocks: 股票数。
        max_days: 交易日上限。
        max_cells: 交易日 × 股票数上限。
    """
    if days > max_days:
        raise ResourceLimitError(
            f"回测区间含 {days} 个交易日，超过上限 {max_days} 个（约十年）；"
            "请缩短区间，或改用指数成分池"
        )
    if days * stocks > max_cells:
        raise ResourceLimitError(
            f"面板规模 {days}×{stocks} 超过上限 {max_cells} 个单元格；"
            "请改用指数成分池（如 index:000300.SH）或缩短区间"
        )

