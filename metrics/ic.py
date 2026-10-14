"""IC 家族：IC / RankIC 的汇总、时序与分布。

口径：

- RankIC = 逐日截面 Spearman（双向分位后求相关），与 ``scripts/build_factors/validate.py``
  的方向检查一致；
- IC = 逐日截面 Pearson，默认对因子与标签各做 3 倍 MAD 去极值；
- 单日有效配对少于 5 只的截面不参与统计；
- 有效交易日少于 20 天时只报「样本不足」，不给数值，避免拿噪声当结论。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calculate.daily import MIN_CROSS_SECTION
from calculate.metrics.base import Metric, register_metric
from calculate.stats import series_stats

INSUFFICIENT_DAYS = 20
ROLLING_WINDOW = 20
HIST_BINS = 40


def _cumulative(values: np.ndarray) -> np.ndarray:
    """计算累计和；缺失日保留空值，但不污染后续累计值。

    参数:
        values: 日度 IC 或 RankIC 序列，可含 NaN/无穷值。
    """
    values = np.asarray(values, dtype="float64")
    finite = np.isfinite(values)
    cumulative = np.cumsum(np.where(finite, values, 0.0))
    cumulative[~finite] = np.nan
    return cumulative


def _blank(stats: dict) -> dict:
    """样本不足时只保留天数，其余数值一律置空。

    参数:
        stats: ``series_stats`` 的输出。
    """
    return {key: (value if key == "days" else None) for key, value in stats.items()}


def compute_stats(context, options) -> dict:
    """算 IC / RankIC 汇总指标与方向一致性。

    参数:
        context: 评估上下文。
        options: 指标参数（本指标无参数）。
    """
    series = context.daily_ic()
    ic = series_stats(series["ic"].to_numpy())
    rank_ic = series_stats(series["rank_ic"].to_numpy())
    days = max(ic["days"], rank_ic["days"])
    insufficient = days < INSUFFICIENT_DAYS
    if insufficient:
        ic = _blank(ic)
        rank_ic = _blank(rank_ic)
    direction = context.spec.direction
    aligned = None
    if insufficient:
        hint = (f"有效交易日只有 {days} 天，少于 {INSUFFICIENT_DAYS} 天，结论不可用；"
                "请拉长评估区间")
    elif rank_ic["mean"] is None:
        hint = "区间内没有可用的截面，无法判定方向"
    else:
        aligned = (rank_ic["mean"] > 0) == (direction > 0)
        hint = (
            f"因子定义方向「{'越大越看多' if direction > 0 else '越大越看空'}」，"
            f"实测 RankIC 均值 {rank_ic['mean']:+.4f}，{'一致' if aligned else '相反'}"
        )
    return {
        "ic": ic,
        "rank_ic": rank_ic,
        "valid_days": days,
        "insufficient": insufficient,
        "min_days": INSUFFICIENT_DAYS,
        "min_cross_section": MIN_CROSS_SECTION,
        "winsorize": context.params.winsorize,
        "direction": direction,
        "direction_aligned": aligned,
        "start": series.index.min() if len(series) else None,
        "end": series.index.max() if len(series) else None,
        "hint": hint,
    }


def compute_series(context, options) -> dict:
    """输出逐日 IC 序列、累计和与 20 日滚动均值，供前端画曲线。

    参数:
        context: 评估上下文。
        options: 指标参数（本指标无参数）。
    """
    series = context.daily_ic()
    ic = series["ic"].to_numpy(dtype="float64")
    rank_ic = series["rank_ic"].to_numpy(dtype="float64")
    rolling_ic = series["ic"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    rolling_rank = series["rank_ic"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    return {
        "dates": [day.strftime("%Y-%m-%d") for day in series.index],
        "ic": ic,
        "rank_ic": rank_ic,
        "n": series["n"].to_numpy(),
        "cumulative": {
            "ic": _cumulative(ic),
            "rank_ic": _cumulative(rank_ic),
        },
        "rolling": {
            "window": ROLLING_WINDOW,
            "ic": rolling_ic.to_numpy(),
            "rank_ic": rolling_rank.to_numpy(),
        },
    }


def _histogram(values: np.ndarray) -> dict:
    """算直方图边界、计数与分布特征。

    参数:
        values: 已经去掉空值的一维数组。
    """
    if values.size == 0:
        return {"edges": [], "counts": [], "count": 0}
    counts, edges = np.histogram(values, bins=HIST_BINS)
    frame = {
        "edges": edges,
        "counts": counts,
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else None,
        "skew": float(pd.Series(values).skew()) if values.size > 2 else None,
        "kurt": float(pd.Series(values).kurt()) if values.size > 3 else None,
    }
    for point in (0.05, 0.25, 0.50, 0.75, 0.95):
        frame[f"p{int(point * 100):02d}"] = float(np.quantile(values, point))
    return frame


def compute_dist(context, options) -> dict:
    """输出 IC 与 RankIC 的分布，供前端画直方图。

    参数:
        context: 评估上下文。
        options: 指标参数（本指标无参数）。
    """
    series = context.daily_ic()
    ic = series["ic"].to_numpy(dtype="float64")
    rank_ic = series["rank_ic"].to_numpy(dtype="float64")
    return {
        "bins": HIST_BINS,
        "ic": _histogram(ic[np.isfinite(ic)]),
        "rank_ic": _histogram(rank_ic[np.isfinite(rank_ic)]),
    }


register_metric(
    Metric(key="ic_stats", label="IC / RankIC 汇总", stage="overview", compute=compute_stats)
)
register_metric(
    Metric(key="ic_series", label="IC 时序", stage="ic_series", compute=compute_series)
)
register_metric(
    Metric(key="ic_dist", label="IC 分布", stage="ic_dist", compute=compute_dist)
)
