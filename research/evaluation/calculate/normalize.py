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


def _winsorize_by_day_numpy(
    dates,
    values,
    n_mad: float = 3.0,
) -> np.ndarray:
    """用 pandas 金标准按交易日做 MAD 裁剪。"""

    frame = pd.DataFrame({"day": np.asarray(dates), "value": np.asarray(values, dtype="float64")})
    grouped = frame.groupby("day", sort=False)["value"]
    median = grouped.transform("median").to_numpy(dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        deviation = np.abs(frame["value"].to_numpy(dtype="float64") - median)
        mad = pd.Series(deviation).groupby(
            frame["day"].to_numpy(),
            sort=False,
        ).transform("median").to_numpy()
        limit = n_mad * MAD_SCALE * mad
        clipped = np.clip(
            frame["value"].to_numpy(dtype="float64"),
            median - limit,
            median + limit,
        )
    # MAD 为 0（当天取值集中）时不做裁剪，保持原值
    return np.where(mad > 0, clipped, frame["value"].to_numpy(dtype="float64"))


def _zscore_by_day_numpy(dates, values) -> np.ndarray:
    """用 pandas 金标准按交易日做截面 z-score。"""

    value = np.asarray(values, dtype="float64")
    frame = pd.DataFrame({
        "day": np.asarray(dates),
        "value": np.where(np.isfinite(value), value, np.nan),
    })
    frame["square"] = frame["value"] ** 2
    grouped = frame.groupby("day", sort=False)
    count = grouped["value"].transform("count").to_numpy(dtype="float64")
    mean = grouped["value"].transform("mean").to_numpy(dtype="float64")
    square_sum = grouped["square"].transform("sum").to_numpy(dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = (square_sum - count * mean ** 2) / np.where(count > 1, count - 1, np.nan)
        std = np.sqrt(np.clip(variance, 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (value - mean) / std
    return np.where((count >= 2) & (std > 0), result, np.nan)


def _grouped_values(
    dates,
    values,
    *,
    finite_dates_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """把日期和取值整理成按日连续数组；无效日期返回 None。"""

    value = np.ascontiguousarray(np.asarray(values, dtype="float64"))
    date_values = np.asarray(dates)
    if date_values.size != value.size:
        raise ValueError("dates 与 values 长度必须一致")
    if value.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.array([0], dtype=np.int64),
            value,
        )
    codes, _ = pd.factorize(date_values, sort=False)
    if finite_dates_only and np.any(codes < 0):
        return None
    order = np.argsort(codes, kind="stable")
    ordered_codes = codes[order]
    offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(np.bincount(ordered_codes), dtype=np.int64),
        )
    ).astype(np.int64, copy=False)
    return order, offsets, np.ascontiguousarray(value[order])


def _apply_group_kernel(
    dates,
    values,
    kernel: str,
    *,
    n_mad: float = 3.0,
) -> np.ndarray:
    """按交易日分组后调用统一后端，并把结果恢复到原顺序。"""

    from kernels import get_backend

    grouped = _grouped_values(dates, values, finite_dates_only=True)
    if grouped is None:
        if kernel == "daily_winsorize_mad":
            return _winsorize_by_day_numpy(dates, values, n_mad)
        if kernel == "daily_zscore":
            return _zscore_by_day_numpy(dates, values)
        return _zscore_by_day_numpy(
            dates,
            _winsorize_by_day_numpy(dates, values, n_mad),
        )
    order, offsets, ordered = grouped
    if ordered.size == 0:
        return ordered.copy()
    selection = get_backend()
    if kernel == "daily_winsorize_mad":
        computed = selection.backend.daily_winsorize_mad(
            ordered,
            offsets,
            n_mad,
        )
    elif kernel == "daily_zscore":
        computed = selection.backend.daily_zscore(ordered, offsets)
    else:
        computed = selection.backend.daily_winsorize_zscore(
            ordered,
            offsets,
            n_mad,
        )
    result = np.empty_like(computed)
    result[order] = computed
    return result


def winsorize_by_day(dates, values, n_mad: float = 3.0) -> np.ndarray:
    """按交易日分别做 MAD 裁剪。

    参数:
        dates: 与 values 等长、表示交易日的序列。
        values: 一维取值序列。
        n_mad: 倍数，默认 3。

    返回:
        裁剪后的数组，顺序与输入一致。
    """
    if isinstance(n_mad, bool) or not np.isfinite(float(n_mad)) or float(n_mad) <= 0:
        return _winsorize_by_day_numpy(dates, values, n_mad)
    return _apply_group_kernel(
        dates,
        values,
        "daily_winsorize_mad",
        n_mad=float(n_mad),
    )


def standardize_by_day(dates, values, n_mad: float = 3.0) -> np.ndarray:
    """按交易日融合 MAD 去极值和 z-score。"""

    return _apply_group_kernel(
        dates,
        values,
        "daily_winsorize_zscore",
        n_mad=float(n_mad),
    )


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


def zscore_by_day(dates, values) -> np.ndarray:
    """按交易日分别做截面标准化。

    参数:
        dates: 与 values 等长、表示交易日的序列。
        values: 一维取值序列，允许含 NaN。

    返回:
        标准化后的数组，顺序与输入一致；有效样本不足 2 个或标准差为 0 的
        交易日返回空值。
    """
    return _apply_group_kernel(dates, values, "daily_zscore")


def _winsorize_zscore_by_day_numpy(dates, values, n_mad: float = 3.0) -> np.ndarray:
    """用 pandas 金标准串联逐日 MAD 去极值和 z-score。"""

    return _zscore_by_day_numpy(
        dates,
        _winsorize_by_day_numpy(dates, values, n_mad),
    )
