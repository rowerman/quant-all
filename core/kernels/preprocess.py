"""模型预处理使用的 Python 金标准批量核。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kernels.backend import FloatArray, IntArray, group_codes


def winsorize_quantile(
    values: FloatArray,
    group_offsets: IntArray,
    lower_quantile: float,
    upper_quantile: float,
) -> FloatArray:
    """用 pandas 线性插值分位作为逐日 1%/99% 裁剪金标准。"""

    frame = pd.DataFrame(
        {
            "group": group_codes(group_offsets),
            "value": np.where(np.isfinite(values), values, np.nan),
        }
    )
    grouped = frame.groupby("group", sort=False)["value"]
    lower = grouped.transform(lambda item: item.quantile(lower_quantile))
    upper = grouped.transform(lambda item: item.quantile(upper_quantile))
    return frame["value"].clip(lower=lower, upper=upper).to_numpy(dtype=np.float64)


def zscore_population(
    values: FloatArray,
    group_offsets: IntArray,
) -> FloatArray:
    """用 pandas 总体标准差作为逐日 z-score 金标准。"""

    value = np.where(np.isfinite(values), values, np.nan)
    frame = pd.DataFrame(
        {
            "group": group_codes(group_offsets),
            "value": value,
        }
    )
    grouped = frame.groupby("group", sort=False)["value"]
    mean = grouped.transform("mean")
    standard_deviation = grouped.transform("std", ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (frame["value"] - mean) / standard_deviation.replace(0.0, np.nan)
    return result.to_numpy(dtype=np.float64)
