"""Alpha 模型层使用的交易日历与调仓日期。"""

from __future__ import annotations

import pandas as pd

import quant


def load_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """读取开市交易日。"""

    frame = quant.load_calendar(start=start, end=end)
    return pd.DatetimeIndex(pd.to_datetime(frame["trade_date"])).sort_values()


def weekly_signal_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """返回每周最后一个交易日，与回测周频调仓口径一致。"""

    stamps = pd.DatetimeIndex(calendar).sort_values()
    if stamps.empty:
        return stamps
    periods = pd.Series(stamps, index=stamps).dt.to_period("W-FRI")
    return stamps[periods.ne(periods.shift(-1)).fillna(True).to_numpy()]


def monthly_last_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """返回每月最后一个交易日。"""

    stamps = pd.DatetimeIndex(calendar).sort_values()
    if stamps.empty:
        return stamps
    periods = pd.Series(stamps, index=stamps).dt.to_period("M")
    return stamps[periods.ne(periods.shift(-1)).fillna(True).to_numpy()]
