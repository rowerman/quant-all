"""换手与稳定性：分位换手、全池换手与因子自相关。

换手说的是「这个因子的分位组合要买多少」，自相关说的是「因子值本身多久换一次」；
两者都是判断交易成本与信号衰减的第一手材料。
"""

from __future__ import annotations


from calculate.binning import AUTOCORR_LAGS, bin_turnover, factor_autocorrelation
from calculate.metrics.base import Metric, register_metric
from calculate.stats import series_stats


def compute_turnover(context, options) -> dict:
    """算分位换手、全池换手与因子自相关。

    参数:
        context: 评估上下文。
        options: 指标参数（本指标无参数）。
    """
    quantiles = context.params.quantiles
    frame = context.bins()
    if frame.empty:
        return {"quantiles": quantiles, "empty": True, "hint": "池内没有非空因子值，无法算换手"}

    matrix, ranks = context.bin_matrices()
    per_bin, overall = bin_turnover(matrix, quantiles)
    autocorrelation = factor_autocorrelation(ranks, AUTOCORR_LAGS)
    return {
        "quantiles": quantiles,
        "bins": list(range(1, quantiles + 1)),
        "lags": list(AUTOCORR_LAGS),
        "bin_turnover": {
            "mean": [
                None if per_bin.empty or bin_id not in per_bin
                else series_stats(per_bin[bin_id].to_numpy())["mean"]
                for bin_id in range(1, quantiles + 1)
            ],
            "days": [
                0 if per_bin.empty or bin_id not in per_bin
                else series_stats(per_bin[bin_id].to_numpy())["days"]
                for bin_id in range(1, quantiles + 1)
            ],
        },
        "daily_turnover": {
            "dates": [day.strftime("%Y-%m-%d") for day in overall.index],
            "mean": overall.to_numpy(),
        },
        "autocorrelation": {
            f"lag{lag}": series_stats(
                autocorrelation[f"lag{lag}"].to_numpy()
                if f"lag{lag}" in autocorrelation else []
            )
            for lag in AUTOCORR_LAGS
        },
        "note": (
            "分位换手 = 今天我们组里、上一交易日在池内但不在同组的股票占比；"
            "因子自相关 = 当日与 lag 个交易日前在同一批股票上的截面秩相关"
        ),
    }


register_metric(
    Metric(key="turnover", label="分位换手与因子自相关", stage="turnover",
           compute=compute_turnover, requires_quantiles=True)
)
