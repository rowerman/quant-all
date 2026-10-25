"""复权与收益计算。

数据给的是「未复权四价 + 后复权因子」，所以复权只有乘除两步：
    后复权价 = 原始价 × factor
    前复权价 = 原始价 × factor / factor_T      （factor_T 取该股最后一个有效交易日的因子）

收益主口径用交易所给的涨跌额反推：
    pre_close = close - change
    ret       = change / pre_close
因子口径 (close × factor).pct_change() 只用于交叉校验。
"""

from __future__ import annotations

import numpy as np

from scripts.config import RET_TOLERANCE
from scripts.errors import AdjustError, MissingFactorError
from scripts.logging_conf import get_logger

logger = get_logger("clean.adjust")


def shift_down(array: np.ndarray) -> np.ndarray:
    """按股票（行）向下平移一行，首列填 NaN。
    """
    shifted = np.full_like(array, np.nan, dtype=np.float64)
    shifted[:, 1:] = array[:, :-1]
    return shifted


def forward_fill_close(close: np.ndarray) -> np.ndarray:
    """把停牌（NaN）位置的收盘价填成上一有效交易日的收盘价。

    参数:
        close: 收盘价面板，停牌处为 NaN。

    返回:
        与输入同形状的数组；序列开头没有有效值时仍为 NaN。
    """
    filled = close.copy()
    n_rows, n_cols = filled.shape
    if n_cols == 0:
        return filled
    index = np.where(np.isnan(filled), 0, np.arange(n_cols)[None, :])
    np.maximum.accumulate(index, axis=1, out=index)
    return filled[np.arange(n_rows)[:, None], index]


def last_valid_close(close: np.ndarray) -> np.ndarray:
    """取每只股票最后一个有效交易日的收盘价。

    参数:
        close: 收盘价面板，停牌处为 NaN。

    返回:
        形状 (股票数,) 的数组，全年没有有效交易日时为 NaN。
    """
    n_rows, n_cols = close.shape
    valid = ~np.isnan(close)
    positions = np.where(valid, np.arange(n_cols)[None, :], -1)
    last = positions.max(axis=1)
    result = np.full(n_rows, np.nan, dtype=np.float64)
    has_value = last >= 0
    if np.any(has_value):
        result[has_value] = close[np.arange(n_rows)[has_value], last[has_value]]
    return result


def compute_pre_close(close: np.ndarray, change: np.ndarray) -> np.ndarray:
    """用官方涨跌额反推前收盘价。
    参数:
        close: 未复权收盘价面板。
        change: 官方涨跌额面板。
    返回:
        前收盘价面板，停牌处为 NaN。
    """
    return close - change


def compute_return(pre_close: np.ndarray, change: np.ndarray) -> np.ndarray:
    """按官方涨跌额与反推的前收计算日收益。
    参数:
        pre_close: 前收盘价面板。
        change: 官方涨跌额面板。
    返回:
        日收益面板（小数，不是百分数）。
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return change / pre_close


def mark_ex_dividend(close, change, prev_close) -> np.ndarray:
    """标记除权除息日：涨跌额与「今收－昨收」对不上即为除权。
    参数:
        close: 未复权收盘价面板。
        change: 官方涨跌额面板。
        prev_close: 上一交易日收盘价面板。
    返回:
        布尔面板。
    """
    diff = np.abs(change - (close - prev_close))
    tol = RET_TOLERANCE * np.abs(close)
    with np.errstate(invalid="ignore"):
        return diff > tol


def last_valid_factor(factor: np.ndarray, close: np.ndarray) -> np.ndarray:
    """取每只股票最后一个有效交易日的因子，用于前复权基准。
    参数:
        factor: 后复权因子面板。
        close: 收盘价面板，用于判断交易日。
    返回:
        形状 (股票数, 1) 的因子数组。
    """
    valid = ~np.isnan(close)
    if np.any(np.isnan(factor[valid])):
        raise MissingFactorError("有效交易日内存在复权因子缺失，无法复权")
    filled = np.where(valid, np.nan_to_num(factor, nan=np.nan), np.nan)
    result = np.full((factor.shape[0], 1), np.nan, dtype=np.float64)
    for row in range(factor.shape[0]):
        row_values = filled[row][valid[row]]
        if row_values.size == 0:
            raise MissingFactorError(f"第 {row} 行没有任何有效交易日")
        result[row, 0] = row_values[-1]
    return result


def adjust_scale(mode: str, factor: np.ndarray, base: np.ndarray) -> np.ndarray:
    """返回把未复权价换算成目标口径的乘数面板。
    参数:
        mode: raw / hfq / qfq。
        factor: 后复权因子面板。
        base: 每只股票的前复权基准因子，形状 (股票数, 1)。
    返回:
        与输入同形状的乘数面板。
    """
    if mode == "raw":
        return np.ones_like(factor, dtype=np.float64)
    if mode == "hfq":
        return factor
    if mode == "qfq":
        return factor / base
    raise AdjustError(f"未知复权口径: {mode}")


def factor_scale(close: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """因子口径的前复权乘积序列，用于和收益主口径交叉校验。
    参数:
        close: 未复权收盘价面板。
        factor: 后复权因子面板。
    返回:
        close × factor 面板。
    """
    return close * factor


def return_mismatch(official: np.ndarray, factor_based: np.ndarray) -> np.ndarray:
    """标记两种收益口径差异过大的位置。
    参数:
        official: 官方涨跌额口径的收益。
        factor_based: 因子口径的收益。
    返回:
        布尔面板。
    """
    with np.errstate(invalid="ignore"):
        return np.abs(official - factor_based) > RET_TOLERANCE
