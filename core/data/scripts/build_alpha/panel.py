"""P0/P1 共用的按年面板读取与截面工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import quant
from alpha.config import DEFAULT_CONTRACT, dataset_versions
from alpha.errors import AlphaDataError
from backtest.strategies._factor_rank import weekly_last_dates

KEY_COLUMNS = ("stock_code", "trade_date")
DAILY_FIELDS = (
    "can_buy_close",
    "is_suspended",
    "is_st",
    "is_delisting",
    "list_date",
    "total_mv",
    "turnover_rate",
)


@dataclass(frozen=True)
class YearPanel:
    """一个自然年的因子、标签和交易状态面板。"""

    year: int
    frame: pd.DataFrame
    factor_names: tuple[str, ...]
    calendar: pd.DatetimeIndex


def implemented_factor_names() -> tuple[str, ...]:
    """返回因子目录里已落盘的全部因子名，顺序与目录一致。"""

    frame = quant.load_factor_list(status="implemented")
    if frame.empty:
        raise AlphaDataError("因子目录没有 implemented 因子")
    return tuple(frame["name"].astype(str))


def data_versions() -> dict:
    """读取行情与因子版本，用于数据集和实验清单。"""

    return dataset_versions()


def load_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """读取开市交易日。"""

    frame = quant.load_calendar(start=start, end=end)
    return pd.DatetimeIndex(pd.to_datetime(frame["trade_date"])).sort_values()


def weekly_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """返回日历中每周最后一个交易日。"""

    source = pd.DataFrame({"trade_date": pd.DatetimeIndex(calendar)})
    return weekly_last_dates(calendar, calendar=source)


def monthly_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """返回日历中每月最后一个交易日。"""

    stamps = pd.DatetimeIndex(calendar).sort_values()
    periods = pd.Series(stamps, index=stamps).dt.to_period("M")
    return stamps[periods.ne(periods.shift(-1)).fillna(True).to_numpy()]


def eligible_flags(frame: pd.DataFrame, min_listed_days: int = 60) -> pd.Series:
    """按回测股票池口径返回可交易布尔序列。"""

    listed_days = (pd.to_datetime(frame["trade_date"]) - pd.to_datetime(frame["list_date"])).dt.days
    flags = (
        frame["can_buy_close"].fillna(False).astype(bool)
        & ~frame["is_suspended"].fillna(True).astype(bool)
        & ~frame["is_st"].fillna(True).astype(bool)
        & ~frame["is_delisting"].fillna(True).astype(bool)
        & (listed_days >= min_listed_days)
    )
    return flags.rename("eligible")


def load_year_panel(year: int, factor_names: tuple[str, ...] | None = None) -> YearPanel:
    """读取一年因子、标签和交易状态，并计算可交易标记与上下文列。"""

    names = factor_names or implemented_factor_names()
    start, end = f"{year}-01-01", f"{year}-12-31"
    factors = quant.load_factor_panel(start=start, end=end, names=list(names))
    daily = quant.load_panel(start=start, end=end, fields=list(DAILY_FIELDS))
    labels = quant.load_label(
        start=start,
        end=end,
        horizons=[DEFAULT_CONTRACT.horizon],
        price=DEFAULT_CONTRACT.price,
    )
    label_column = DEFAULT_CONTRACT.label
    labels = labels[["stock_code", "trade_date", label_column]]
    frame = factors.merge(daily, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    frame = frame.merge(labels, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    if len(frame) != len(factors):
        raise AlphaDataError(f"{year} 年面板合并后行数从 {len(factors)} 变为 {len(frame)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["stock_code"] = frame["stock_code"].astype("string")
    for name in names:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float32")
    frame["eligible"] = eligible_flags(frame, DEFAULT_CONTRACT.min_listed_days)
    frame["listed_days"] = (
        frame["trade_date"] - pd.to_datetime(frame["list_date"])
    ).dt.days.astype("float32")
    frame["log_mv"] = np.log(
        pd.to_numeric(frame["total_mv"], errors="coerce").where(lambda value: value > 0)
    ).astype("float32")
    frame["turnover_rate"] = pd.to_numeric(
        frame["turnover_rate"], errors="coerce"
    ).astype("float32")
    calendar = load_calendar(start, end)
    return YearPanel(year=year, frame=frame, factor_names=tuple(names), calendar=calendar)


def to_wide(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """把长表的一列透视成交易日 × 股票宽表。"""

    return frame.pivot(index="trade_date", columns="stock_code", values=column)


def eligible_wide(frame: pd.DataFrame) -> pd.DataFrame:
    """把可交易标记透视成与因子面板同形的宽表。"""

    return to_wide(frame, "eligible").astype(bool)


def linear_zscore_score(
    frame: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """按日截面 z-score 后按给定权重合成分数。"""

    total: pd.DataFrame | None = None
    for name, weight in weights.items():
        wide = to_wide(frame, name).astype("float64")
        mean = wide.mean(axis=1)
        std = wide.std(axis=1, ddof=0).replace(0.0, np.nan)
        score = wide.sub(mean, axis=0).div(std, axis=0).mul(float(weight))
        total = score if total is None else total.add(score, fill_value=0.0)
    if total is None:
        raise AlphaDataError("线性分数没有任何因子")
    return total.replace([np.inf, -np.inf], np.nan)


def rank_score(
    frame: pd.DataFrame,
    factor_names: tuple[str, ...],
) -> pd.DataFrame:
    """按日可交易截面分位、方向调整后等权合成分数。"""

    import factors

    eligible = eligible_wide(frame)
    total = pd.DataFrame(0.0, index=eligible.index, columns=eligible.columns)
    count = pd.DataFrame(0.0, index=eligible.index, columns=eligible.columns)
    for name in factor_names:
        rank = to_wide(frame, name).astype("float64").where(eligible).rank(
            axis=1, pct=True
        )
        direction = int(factors.resolve(name).direction)
        total = total.add(rank * direction, fill_value=0.0)
        count = count.add(rank.notna().astype("float64"), fill_value=0.0)
    return total.div(count.replace(0.0, np.nan))
