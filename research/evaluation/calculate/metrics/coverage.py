"""覆盖率与取值分布：判断一个因子「有没有值、值长什么样」。

分三层看覆盖率：全样本、池内、以及因子与标签都能对上的有效配对率——
最后一个才真正决定 IC 有多少样本可用。
"""

from __future__ import annotations

import numpy as np

from calculate.metrics.base import Metric, register_metric

# 分布统计里要报的分位点
QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def _distribution(values: np.ndarray) -> dict:
    """算取值分布：均值、标准差、偏度、峰度、极值与分位点。

    参数:
        values: 已经去掉空值的一维数组；非有限值会先被剔除。
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    frame = {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else None,
        "min": float(values.min()),
        "max": float(values.max()),
    }
    if values.size > 2:
        center = float(values.mean())
        scale = float(values.std())
        if scale > 0:
            centered = values - center
            frame["skew"] = float((centered ** 3).mean() / scale ** 3)
            frame["kurt"] = float((centered ** 4).mean() / scale ** 4 - 3.0)
    for point in QUANTILES:
        frame[f"p{int(point * 100):02d}"] = float(np.quantile(values, point))
    return frame


def compute(context, options) -> dict:
    """算覆盖率、池内分布与每日有效样本数。

    参数:
        context: 评估上下文。
        options: 指标参数（本指标无参数）。
    """
    panel = context.panel
    pool = panel["in_universe"]
    factor_values = panel["factor"]
    pool_factor = factor_values[pool]
    pool_rows = int(pool.sum())
    raw = factor_values.to_numpy(dtype="float64")
    daily_pairs = (
        panel.loc[pool & factor_values.notna() & panel["label"].notna(), "trade_date"]
        .value_counts()
        .to_numpy(dtype="float64")
    )
    return {
        "rows": int(len(panel)),
        "pool_rows": pool_rows,
        "stocks": context.summary["stocks"],
        "days": context.summary["days"],
        "coverage_all": float(factor_values.notna().mean()),
        "coverage_universe": float(pool_factor.notna().mean()) if pool_rows else None,
        "pair_coverage": context.summary["pair_coverage"],
        "pool_share": context.summary["universe_share"],
        "missing": int(np.isnan(raw).sum()),
        "non_finite": int(np.isinf(raw).sum()),
        "daily_pairs": {
            "mean": float(daily_pairs.mean()) if daily_pairs.size else None,
            "median": float(np.median(daily_pairs)) if daily_pairs.size else None,
            "min": float(daily_pairs.min()) if daily_pairs.size else None,
            "max": float(daily_pairs.max()) if daily_pairs.size else None,
        },
        "distribution": _distribution(pool_factor.dropna().to_numpy(dtype="float64")),
    }


register_metric(
    Metric(key="coverage", label="覆盖率与取值分布", stage="overview", compute=compute)
)
