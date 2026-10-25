"""清洗输入层：把 Tushare 分片组装成按年的二维面板。

输出表只保留股票处于上市状态（list_date ~ delist_date）的行，上市前与退市后
不再铺满日历——那段区间在 ref.stock_basic 里有明确日期，铺出来只是空行。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scripts.clean import sources
from scripts.errors import DataReadError
from scripts.logging_conf import get_logger

logger = get_logger("clean.io")

# Tushare 字段 -> 面板列名
DAILY_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "change": "change",
    "pre_close": "pre_close_tushare",
    "vol": "volume_share",
    "amount": "amount_cny",
}
ADJ_MAP = {"adj_factor": "factor"}
LIMIT_MAP = {"up_limit": "limit_up_official", "down_limit": "limit_down_official"}
BASIC_MAP = {
    "pe_ttm": "pe_ttm",
    "pb": "pb",
    "ps_ttm": "ps_ttm",
    "dv_ttm": "dv_ttm",
    "total_mv": "total_mv",
    "circ_mv": "circ_mv",
    "turnover_rate": "turnover_rate",
}

VOLUME_TO_SHARE = 100.0
AMOUNT_TO_CNY = 1000.0


@dataclass
class Panel:
    """二维面板：行是股票、列是交易日。"""

    codes: list[str]
    dates: list[str]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        """返回面板形状 (股票数, 交易日数)。"""
        return len(self.codes), len(self.dates)

    def put(self, name: str, array: np.ndarray) -> None:
        """登记一个派生数组。

        参数:
            name: 列名。
            array: 形状与面板一致的数组。
        """
        if array.shape != self.shape:
            raise DataReadError(f"数组 {name} 形状 {array.shape} 与面板 {self.shape} 不符")
        self.arrays[name] = array


def _assign(
    arrays: dict[str, np.ndarray],
    row_of: dict[str, int],
    col_of: dict[str, int],
    frame: pd.DataFrame,
    mapping: dict[str, str],
    scale: dict[str, float] | None = None,
) -> None:
    """把长表按代码与日期铺进二维数组。

    参数:
        arrays: 目标数组字典。
        row_of: 代码到行下标。
        col_of: 日期到列下标。
        frame: 已含 stock_code 与 trade_date 的长表。
        mapping: 原始列名到面板列名。
        scale: 需要乘的换算系数。
    """
    rows = frame["stock_code"].map(row_of)
    cols = frame["trade_date"].map(col_of)
    keep = rows.notna() & cols.notna()
    rows = rows[keep].astype(int).to_numpy()
    cols = cols[keep].astype(int).to_numpy()
    scaled = scale or {}
    for source_name, target_name in mapping.items():
        if source_name not in frame.columns:
            continue
        values = pd.to_numeric(frame.loc[keep, source_name], errors="coerce").to_numpy(dtype=np.float64)
        factor = scaled.get(source_name, 1.0)
        arrays[target_name][rows, cols] = values * factor


def _prepare(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """统一代码与日期格式，认不出的历史遗留代码整行丢弃。

    参数:
        frame: 接口原始长表。
        label: 调用来源描述，用于告警定位。
    """
    result = frame.copy()
    result["stock_code"] = sources.normalize_codes(result["ts_code"], label)
    result["trade_date"] = result["trade_date"].astype(str)
    result["trade_date"] = (
        result["trade_date"].str.slice(0, 4)
        + "-"
        + result["trade_date"].str.slice(4, 6)
        + "-"
        + result["trade_date"].str.slice(6, 8)
    )
    return result


def build_year_panel(year: int, dates: list[str], codes: list[str]) -> Panel:
    """构建某一年的原始面板。

    参数:
        year: 年份。
        dates: 该年有效交易日列表。
        codes: 该年需要纳入的股票代码。

    返回:
        含原始字段的 Panel。
    """
    shape = (len(codes), len(dates))
    row_of = {code: position for position, code in enumerate(codes)}
    col_of = {day: position for position, day in enumerate(dates)}
    arrays: dict[str, np.ndarray] = {}
    for name in set(DAILY_MAP.values()) | set(ADJ_MAP.values()) | set(LIMIT_MAP.values()) | set(BASIC_MAP.values()):
        arrays[name] = np.full(shape, np.nan, dtype=np.float64)

    daily = _prepare(sources.load_api("daily", year), f"daily {year}")
    _assign(arrays, row_of, col_of, daily, DAILY_MAP,
            scale={"vol": VOLUME_TO_SHARE, "amount": AMOUNT_TO_CNY})

    adjust = _prepare(sources.load_api("adj_factor", year), f"adj_factor {year}")
    _assign(arrays, row_of, col_of, adjust, ADJ_MAP)

    limit = _prepare(sources.load_api("stk_limit", year), f"stk_limit {year}")
    _assign(arrays, row_of, col_of, limit, LIMIT_MAP)

    basic = _prepare(sources.load_api("daily_basic", year), f"daily_basic {year}")
    _assign(arrays, row_of, col_of, basic, BASIC_MAP)

    panel = Panel(codes=codes, dates=dates, arrays=arrays)
    return panel


def years_in_range(first: str, last: str) -> list[int]:
    """列出区间覆盖的年份。

    参数:
        first: 起始日期。
        last: 截止日期。
    """
    return list(range(int(first[:4]), int(last[:4]) + 1))


def load_st_matrix(codes: list[str], dates: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """构造 ST 与退市整理期面板。

    参数:
        codes: 股票代码列表。
        dates: 交易日列表。

    返回:
        (ST 面板, 退市整理期面板, 每只股票的来源标记)。
    """
    from scripts.clean import lifecycle

    intervals = sources.namechange_intervals()
    st = lifecycle.intervals_to_matrix(codes, dates, intervals[intervals["name"].map(lifecycle.is_st_name)])
    delisting = lifecycle.intervals_to_matrix(
        codes, dates, intervals[intervals["name"].map(lifecycle.is_delisting_name)]
    )
    covered = set(intervals["stock_code"])
    marks = ["tushare_namechange" if code in covered else "no_change_record" for code in codes]
    return st, delisting, marks


def load_suspend_matrix(codes: list[str], dates: list[str]) -> np.ndarray:
    """构造官方停牌面板。

    参数:
        codes: 股票代码列表。
        dates: 交易日列表。
    """
    records = sources.suspend_records()
    result = np.zeros((len(codes), len(dates)), dtype=bool)
    if records.empty:
        return result
    row_of = {code: position for position, code in enumerate(codes)}
    col_of = {day: position for position, day in enumerate(dates)}
    rows = records["stock_code"].map(row_of)
    cols = records["trade_date"].map(col_of)
    keep = rows.notna() & cols.notna()
    result[rows[keep].astype(int).to_numpy(), cols[keep].astype(int).to_numpy()] = True
    return result
