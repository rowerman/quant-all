"""截面标准化工具：MAD 去极值、分位、zscore。

都按「每个交易日一组」使用，输入是一维数组，缺失值（NaN）原样保留，
由调用方在对齐后再丢空值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 正态分布下 MAD 与标准差的比例，用于把 MAD 换算成等价标准差
MAD_SCALE = 1.4826


def mad_winsorize(values, n_mad: float = 3.0) -> np.ndarray:
    """按中位数 ± n 倍 MAD 做裁剪。

    参数:
        values: 一维取值数组，允许含 NaN。
        n_mad: 倍数，默认 3。

    返回:
        裁剪后的数组；有效样本不足 3 个或 MAD 为 0 时原样返回。
    """
    array = np.asarray(values, dtype="float64")
    valid = array[np.isfinite(array)]
    if valid.size < 3:
        return array.copy()
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    if mad <= 0:
        return array.copy()
    limit = n_mad * MAD_SCALE * mad
    return np.clip(array, median - limit, median + limit)


def winsorize_by_day(dates, values, n_mad: float = 3.0) -> np.ndarray:
    """按交易日分别做 MAD 裁剪。

    参数:
        dates: 与 values 等长、表示交易日的序列。
        values: 一维取值序列。
        n_mad: 倍数，默认 3。

    返回:
        裁剪后的数组，顺序与输入一致。
    """
    frame = pd.DataFrame({"day": np.asarray(dates), "value": np.asarray(values, dtype="float64")})
    grouped = frame.groupby("day", sort=False)["value"]
    median = grouped.transform("median").to_numpy(dtype="float64")
    deviation = np.abs(frame["value"].to_numpy(dtype="float64") - median)
    mad = pd.Series(deviation).groupby(frame["day"].to_numpy(), sort=False).transform("median").to_numpy()
    limit = n_mad * MAD_SCALE * mad
    clipped = np.clip(frame["value"].to_numpy(dtype="float64"), median - limit, median + limit)
    # MAD 为 0（当天取值集中）时不做裁剪，保持原值
    return np.where(mad > 0, clipped, frame["value"].to_numpy(dtype="float64"))


def rank_pct(values) -> np.ndarray:
    """截面分位（0~1），并列取平均名次，NaN 保持 NaN。"""
    return pd.Series(np.asarray(values, dtype="float64")).rank(pct=True).to_numpy()


def zscore(values) -> np.ndarray:
    """截面标准化；有效样本不足 2 个或标准差为 0 时返回全 NaN。"""
    array = np.asarray(values, dtype="float64")
    valid = array[np.isfinite(array)]
    if valid.size < 2:
        return np.full(array.shape, np.nan)
    std = float(valid.std(ddof=1))
    if std <= 0:
        return np.full(array.shape, np.nan)
    return (array - float(valid.mean())) / std
