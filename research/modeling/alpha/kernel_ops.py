"""模型层逐日排名与标准化的统一核入口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from kernels.backend import NativeKernelBackend


@dataclass(frozen=True)
class DateGroups:
    """按日期稳定排序后的有效行位置和连续分组边界。"""

    positions: np.ndarray
    offsets: np.ndarray


def get_model_backend(
    backend: "NativeKernelBackend | None" = None,
) -> "NativeKernelBackend":
    """返回调用方指定后端，或按统一环境配置选择一次后端。"""

    if backend is not None:
        return backend
    from kernels import get_backend

    return get_backend().backend


def date_groups(dates: object, size: int) -> DateGroups:
    """把任意顺序的日期整理为连续分组，并排除缺失日期。"""

    date_values = np.asarray(dates)
    if date_values.ndim != 1:
        raise ValueError("dates 必须是一维数组")
    if date_values.size != size:
        raise ValueError("dates 长度必须与样本数一致")
    if size == 0:
        return DateGroups(
            np.empty(0, dtype=np.int64),
            np.array([0], dtype=np.int64),
        )
    codes, _ = pd.factorize(date_values, sort=False, use_na_sentinel=True)
    valid = codes >= 0
    valid_codes = codes[valid]
    order = np.argsort(valid_codes, kind="stable")
    ordered_codes = valid_codes[order]
    offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(np.bincount(ordered_codes), dtype=np.int64),
        )
    ).astype(np.int64, copy=False)
    positions = np.flatnonzero(valid)[order].astype(np.int64, copy=False)
    return DateGroups(positions, offsets)


def _ordered_values(
    array: np.ndarray,
    groups: DateGroups,
) -> np.ndarray:
    """按预构造日期分组提取连续一维数组。"""

    return np.ascontiguousarray(
        array[groups.positions]
    )


def _restore(
    size: int,
    groups: DateGroups,
    values: np.ndarray,
) -> np.ndarray:
    """把连续分组结果恢复到原行位置，未参与分组的行保持 NaN。"""

    if size == 0:
        return np.empty(0, dtype=np.float64)
    result = np.full(size, np.nan, dtype=np.float64)
    result[groups.positions] = values
    return result


def rank_pct_by_date(
    values: object,
    dates: object,
    *,
    backend: "NativeKernelBackend | None" = None,
    groups: DateGroups | None = None,
) -> np.ndarray:
    """按交易日计算平均并列名次分位，并保持原行序。"""

    array = np.asarray(values, dtype=np.float64)
    grouping = groups or date_groups(dates, array.size)
    ordered = _ordered_values(array, grouping)
    if array.size == 0 or grouping.positions.size == 0:
        return _restore(array.size, grouping, np.empty(0, dtype=np.float64))
    computed = get_model_backend(backend).daily_rank_pct(
        ordered,
        grouping.offsets,
    )
    return _restore(array.size, grouping, computed)


def winsorize_quantile_by_date(
    values: object,
    dates: object,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    backend: "NativeKernelBackend | None" = None,
    groups: DateGroups | None = None,
) -> np.ndarray:
    """按交易日做分位裁剪，并保持原行序。"""

    array = np.asarray(values, dtype=np.float64)
    grouping = groups or date_groups(dates, array.size)
    ordered = _ordered_values(array, grouping)
    if array.size == 0 or grouping.positions.size == 0:
        return _restore(array.size, grouping, np.empty(0, dtype=np.float64))
    computed = get_model_backend(backend).daily_winsorize_quantile(
        ordered,
        grouping.offsets,
        lower_quantile,
        upper_quantile,
    )
    return _restore(array.size, grouping, computed)


def zscore_population_by_date(
    values: object,
    dates: object,
    *,
    backend: "NativeKernelBackend | None" = None,
    groups: DateGroups | None = None,
) -> np.ndarray:
    """按交易日计算总体标准差 z-score，并保持原行序。"""

    array = np.asarray(values, dtype=np.float64)
    grouping = groups or date_groups(dates, array.size)
    ordered = _ordered_values(array, grouping)
    if array.size == 0 or grouping.positions.size == 0:
        return _restore(array.size, grouping, np.empty(0, dtype=np.float64))
    computed = get_model_backend(backend).daily_zscore_population(
        ordered,
        grouping.offsets,
    )
    return _restore(array.size, grouping, computed)
