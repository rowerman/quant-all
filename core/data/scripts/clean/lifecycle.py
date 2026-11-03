"""生命周期（未上市 / 交易中 / 停牌 / 已终止）与 ST 状态构造。"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from quant.codes import board_of
from scripts.logging_conf import get_logger

logger = get_logger("clean.lifecycle")

STATUS_NOT_LISTED = 0
STATUS_TRADING = 1
STATUS_SUSPENDED = 2
STATUS_ENDED = 3

STATUS_TEXT = np.array(["not_listed", "trading", "suspended", "ended"])

ST_PATTERN = re.compile(r"^(S?\*?ST)")
DELIST_PATTERN = re.compile(r"(退$|退市|退\))")


def is_st_name(name: str) -> bool:
    """判断证券简称是否带 ST / *ST / SST / S*ST 前缀。

    参数:
        name: 证券简称。
    """
    if not isinstance(name, str):
        return False
    return bool(ST_PATTERN.match(name.strip().upper().replace(" ", "")))


def is_delisting_name(name: str) -> bool:
    """判断证券简称是否处于退市整理期。

    参数:
        name: 证券简称。
    """
    if not isinstance(name, str):
        return False
    return bool(DELIST_PATTERN.search(name.strip()))


def intervals_to_matrix(codes: list[str], dates: list[str], intervals: pd.DataFrame) -> np.ndarray:
    """把「代码 + 起止日期」区间表铺成布尔面板。

    参数:
        codes: 股票代码列表。
        dates: 交易日列表。
        intervals: 含 stock_code / start_date / end_date 的区间表。

    返回:
        布尔面板。
    """
    matrix = np.zeros((len(codes), len(dates)), dtype=bool)
    if intervals is None or intervals.empty:
        return matrix
    row_of = {code: position for position, code in enumerate(codes)}
    window_start, window_end = dates[0], dates[-1]
    ordered = np.array(dates)
    for item in intervals.to_dict("records"):
        position = row_of.get(item["stock_code"])
        if position is None:
            continue
        if item["end_date"] < window_start or item["start_date"] > window_end:
            continue
        low = max(int(np.searchsorted(ordered, item["start_date"])), 0)
        high = min(int(np.searchsorted(ordered, item["end_date"], side="right")) - 1, len(dates) - 1)
        if high >= low:
            matrix[position, low : high + 1] = True
    return matrix


def build_reference(codes: list[str]) -> pd.DataFrame:
    """抽取涨跌停规则需要的字段（上市日、退市日、板块）。

    返回行的顺序与传入的 codes 严格一致，缺失的股票用空值补齐。

    参数:
        codes: 股票代码列表。
    """
    from scripts.clean import sources

    basic = sources.stock_reference()
    info = {item["stock_code"]: item for item in basic.to_dict("records")}
    rows = []
    for code in codes:
        item = info.get(code)
        rows.append(
            {
                "stock_code": code,
                "name": item.get("name") if item else None,
                "list_date": item.get("list_date") if item else None,
                "delist_date": item.get("delist_date") if item else None,
                "board": board_of(code[2:]),
            }
        )
    return pd.DataFrame(rows, columns=["stock_code", "name", "list_date", "delist_date", "board"])


def build_status(dates: list[str], reference: pd.DataFrame, close: np.ndarray) -> np.ndarray:
    """按上市日、退市日与当日是否有行情构造状态面板。

    参数:
        dates: 该年的交易日列表。
        reference: 含 list_date / delist_date 的股票基础表，行序必须与 close 一致。
        close: 收盘价面板。

    返回:
        int8 状态面板。
    """
    n_rows, n_cols = close.shape
    index = np.arange(n_cols)[None, :]

    list_dates = np.array(
        [item if item else "1900-01-01" for item in reference["list_date"].tolist()],
        dtype="<U10",
    )
    start = np.searchsorted(np.array(dates), list_dates)
    before = index < start[:, None]

    has_delist = reference["delist_date"].notna().to_numpy()
    delist_dates = np.array(
        [item if item else "2999-12-31" for item in reference["delist_date"].tolist()],
        dtype="<U10",
    )
    stop = np.searchsorted(np.array(dates), delist_dates, side="right")
    after = (index >= stop[:, None]) & has_delist[:, None]

    is_price = ~np.isnan(close)
    status = np.where(is_price, STATUS_TRADING, STATUS_SUSPENDED)
    status = np.where(before, STATUS_NOT_LISTED, status)
    status = np.where(after, STATUS_ENDED, status)
    return status.astype(np.int8)


def build_resume_flags(status: np.ndarray) -> np.ndarray:
    """标记复牌首日：上一交易日停牌、当日恢复交易。

    参数:
        status: 状态码面板。
    """
    previous = np.full_like(status, STATUS_NOT_LISTED, dtype=np.int8)
    previous[:, 1:] = status[:, :-1]
    return (previous == STATUS_SUSPENDED) & (status == STATUS_TRADING)
