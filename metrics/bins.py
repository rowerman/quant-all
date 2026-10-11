"""binplot 家族：分位平均收益、分位净值、多空价差与分位收益分布。

两套权重贯穿每一张图：等权（组内对标签取均值）与分位内市值加权（按 `total_mv`）。
净值与多空曲线固定用 **1 日前瞻收益**累乘：期收益（例如 5 日）逐日重叠，
直接累乘会把同一个区间重复计入，口径上说不通，这一点在页面与 payload 里都标注出来。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calculate.binning import bin_daily_returns
from calculate.metrics.base import Metric, register_metric
from calculate.stats import annualised_sharpe, compound, max_drawdown, series_stats

# 两种权重口径，顺序即前端展示顺序
WEIGHTINGS: tuple[str, ...] = ("equal", "market_cap")
WEIGHT_LABELS: dict[str, str] = {"equal": "等权", "market_cap": "分位内市值加权"}
WEIGHT_NOTES: dict[str, str] = {
    "equal": "组内等权平均",
    "market_cap": "按 total_mv 加权，缺市值的样本不参与",
}

# 净值与多空曲线用的日频标签列（面板里的 label_1d）
CURVE_LABEL_COLUMN = "label_1d"
CURVE_BASIS = "1 日前瞻收益累乘"


def _weight_column(weight: str) -> str | None:
    """把权重口径换成加权列名，等权返回 None。

    参数:
        weight: ``equal`` 或 ``market_cap``。
    """
    return None if weight == "equal" else "total_mv"


def _daily_for(frame: pd.DataFrame, quantiles: int, weight: str,
               label_column: str) -> pd.DataFrame:
    """按指定标签列与权重口径算逐日分位收益。

    参数:
        frame: 分位分组表。
        quantiles: 分组数。
        weight: 权重口径。
        label_column: 用哪一列算收益。
    """
    return bin_daily_returns(
        frame, quantiles, label_column=label_column, weight_column=_weight_column(weight)
    )


def _monotonicity(means: np.ndarray) -> dict:
    """单调性：分位序号与分位平均收益的秩相关。

    参数:
        means: 各分位的平均收益，缺失值会被剔除。
    """
    valid = np.isfinite(means)
    if valid.sum() < 2:
        return {"spearman": None, "bins_used": int(valid.sum())}
    ranks_bin = pd.Series(np.arange(1, means.size + 1)[valid]).rank()
    ranks_mean = pd.Series(means[valid]).rank()
    return {
        "spearman": float(np.corrcoef(ranks_bin, ranks_mean)[0, 1]),
        "bins_used": int(valid.sum()),
    }


def _box_stats(values: np.ndarray) -> dict:
    """箱线图要用的五数概括与均值。

    参数:
        values: 取值数组，NaN 会被剔除。
    """
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    quantiles = np.quantile(array, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else None,
        "min": float(array.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(array.max()),
    }


def _empty_payload(context, stage_note: str) -> dict:
    """池内没有可用样本时的空载荷。

    参数:
        context: 评估上下文。
        stage_note: 给用户看的说明。
    """
    return {
        "quantiles": context.params.quantiles,
        "empty": True,
        "hint": f"池内没有非空因子值，{stage_note}",
    }


def compute_bin_returns(context, options) -> dict:
    """算分位平均期收益、t 值与单调性（两种权重）。

    参数:
        context: 评估上下文。
        options: 指标参数（本指标无参数）。
    """
    quantiles = context.params.quantiles
    frame = context.bins()
    if frame.empty:
        return _empty_payload(context, "无法分层")
    payload = {
        "quantiles": quantiles,
        "bins": list(range(1, quantiles + 1)),
        "bin_labels": {1: "最低", quantiles: "最高"},
        "weightings": list(WEIGHTINGS),
        "weight_labels": dict(WEIGHT_LABELS),
        "weight_notes": dict(WEIGHT_NOTES),
        "period_label": context.params.label_column,
        "monotonicity": {},
    }
    for weight in WEIGHTINGS:
        daily = _daily_for(frame, quantiles, weight, "label")
        stats = {key: [] for key in ("mean", "std", "stderr", "t", "win_rate", "days")}
        for bin_id in range(1, quantiles + 1):
            item = series_stats(daily[bin_id].to_numpy() if bin_id in daily else [])
            for key in stats:
                stats[key].append(item[key])
        payload[weight] = stats
        payload["monotonicity"][weight] = _monotonicity(np.asarray(stats["mean"], dtype="float64"))
    return payload


def compute_bin_curves(context, options) -> dict:
    """算分位累计净值曲线（两种权重，1 日前瞻收益累乘）。

    参数:
        context: 评估上下文。
        options: 指标参数（本指标无参数）。
    """
    quantiles = context.params.quantiles
    frame = context.bins()
    if frame.empty:
        return _empty_payload(context, "无法画分位净值")
    dates = pd.Index(sorted(frame["trade_date"].unique()), name="trade_date")
    payload = {
        "quantiles": quantiles,
        "bins": list(range(1, quantiles + 1)),
        "dates": [day.strftime("%Y-%m-%d") for day in dates],
        "base": 1.0,
        "basis": CURVE_BASIS,
        "basis_note": "期收益逐日重叠，累乘会把同一区间重复计入，所以净值曲线固定用 1 日收益",
        "weightings": list(WEIGHTINGS),
        "weight_labels": dict(WEIGHT_LABELS),
    }
    for weight in WEIGHTINGS:
        daily = _daily_for(frame, quantiles, weight, CURVE_LABEL_COLUMN).reindex(
            index=dates, columns=range(1, quantiles + 1)
        )
        curves = np.vstack([
            compound(daily[bin_id].to_numpy()) for bin_id in range(1, quantiles + 1)
        ])
        payload[weight] = {
            "curves": np.round(curves, 6),
            "final": np.round(curves[:, -1], 6),
        }
    return payload


def compute_spread(context, options) -> dict:
    """算多空价差：最高分位减最低分位的逐日收益、净值与统计量。

    参数:
        context: 评估上下文。
        options: 指标参数（本指标无参数）。
    """
    quantiles = context.params.quantiles
    frame = context.bins()
    if frame.empty:
        return _empty_payload(context, "无法算多空价差")
    dates = pd.Index(sorted(frame["trade_date"].unique()), name="trade_date")
    payload = {
        "quantiles": quantiles,
        "long_bin": quantiles,
        "short_bin": 1,
        "dates": [day.strftime("%Y-%m-%d") for day in dates],
        "basis": CURVE_BASIS,
        "weightings": list(WEIGHTINGS),
        "weight_labels": dict(WEIGHT_LABELS),
        "note": "多空 = 最高分位 − 最低分位；忽略 T+1 成交限制、融券约束与交易费用",
    }
    for weight in WEIGHTINGS:
        daily = _daily_for(frame, quantiles, weight, CURVE_LABEL_COLUMN).reindex(index=dates)
        spread = daily[quantiles] - daily[1]
        curve = compound(spread.to_numpy())
        stats = series_stats(spread.to_numpy())
        payload[weight] = {
            "daily": spread.to_numpy(),
            "cumulative": np.round(curve, 6),
            "mean": stats["mean"],
            "std": stats["std"],
            "t": stats["t"],
            "win_rate": stats["win_rate"],
            "days": stats["days"],
            "annual_sharpe": annualised_sharpe(spread.to_numpy()),
            "total_return": float(curve[-1] - 1.0),
            "max_drawdown": max_drawdown(curve),
        }
    return payload


def compute_bin_dist(context, options) -> dict:
    """算分位收益分布：每日分位均值的箱线数据 + 分位内个股收益分位点。

    参数:
        context: 评估上下文。
        options: 指标参数（本指标无参数）。
    """
    quantiles = context.params.quantiles
    frame = context.bins()
    if frame.empty:
        return _empty_payload(context, "无法画分位收益分布")
    payload = {
        "quantiles": quantiles,
        "bins": list(range(1, quantiles + 1)),
        "weightings": list(WEIGHTINGS),
        "weight_labels": dict(WEIGHT_LABELS),
        "daily_note": "箱线样本 = 每个交易日该分位的期收益均值",
        "stock_note": "个股口径不受权重影响：池内全部个股期收益（跨交易日汇总）",
        "daily": {},
    }
    for weight in WEIGHTINGS:
        daily = _daily_for(frame, quantiles, weight, "label")
        payload["daily"][weight] = [
            {"bin": bin_id, **_box_stats(daily[bin_id].to_numpy() if bin_id in daily else [])}
            for bin_id in range(1, quantiles + 1)
        ]
    labelled = frame.loc[frame["label"].notna()]
    payload["stocks"] = [
        {"bin": bin_id,
         **_box_stats(labelled.loc[labelled["bin"] == bin_id, "label"].to_numpy())}
        for bin_id in range(1, quantiles + 1)
    ]
    return payload


register_metric(
    Metric(key="bin_returns", label="分位平均收益与单调性", stage="bins",
           compute=compute_bin_returns, requires_quantiles=True)
)
register_metric(
    Metric(key="bin_curves", label="分位累计净值", stage="bin_curves",
           compute=compute_bin_curves, requires_quantiles=True)
)
register_metric(
    Metric(key="spread", label="多空价差", stage="spread",
           compute=compute_spread, requires_quantiles=True)
)
register_metric(
    Metric(key="bin_dist", label="分位收益分布", stage="bin_dist",
           compute=compute_bin_dist, requires_quantiles=True)
)
