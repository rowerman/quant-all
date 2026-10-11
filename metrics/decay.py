"""多周期衰减：同一个因子在 1/3/5/10/20 日标签上的表现。

口径：

- 各档独立取样本：某档标签为空的行（停牌、末端不足 N 个交易日）只从该档剔除，
  不做任何填充，所以长周期档的 `days` / `samples` 天然更少；
- 每档报 RankIC 均值、ICIR、t 值、胜率与衰减曲线；
- 多头组超额「最高分位 − 池内均值」为主口径，另报「最高 − 最低」多空价差，
  两者都出等权与分位内市值加权两套权重；
- 期收益逐日重叠，跨档的 t 值不可直接比较，这一点写在 payload 里。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calculate.binning import bin_daily_returns
from calculate.daily import daily_rank_ic_multi
from calculate.metrics.base import Metric, register_metric
from calculate.params import DECAY_HORIZONS
from calculate.stats import series_stats

# 有效交易日少于这个数只报「样本不足」，与 IC 家族保持同一口径
INSUFFICIENT_DAYS = 20

# 两种权重口径，顺序即前端展示顺序
WEIGHTINGS: tuple[str, ...] = ("equal", "market_cap")
WEIGHT_LABELS: dict[str, str] = {"equal": "等权", "market_cap": "分位内市值加权"}


def _curve_item(horizon: int, column: str, item: pd.DataFrame) -> dict:
    """把一个档位的 RankIC 序列压成衰减曲线上的一个点。

    参数:
        horizon: 标签周期。
        column: 面板标签列名。
        item: ``daily_rank_ic_multi`` 里该档的输出。
    """
    stats = series_stats(item["rank_ic"].to_numpy(dtype="float64"))
    insufficient = stats["days"] < INSUFFICIENT_DAYS
    return {
        "horizon": horizon,
        "column": column,
        "label": f"{horizon} 日",
        # 样本不足时只报天数，数值一律置空，避免拿噪声当结论
        "rank_ic_mean": None if insufficient else stats["mean"],
        "rank_ic_ir": None if insufficient else stats["ir"],
        "t": None if insufficient else stats["t"],
        "win_rate": None if insufficient else stats["win_rate"],
        "days": stats["days"],
        "samples": int(item["n"].sum()),
        "insufficient": insufficient,
    }


def _weight_column(weight: str) -> str | None:
    """把权重口径换成加权列名，等权返回 None。

    参数:
        weight: ``equal`` 或 ``market_cap``。
    """
    return None if weight == "equal" else "total_mv"


def _pool_mean(frame: pd.DataFrame, column: str, weight_column: str | None) -> pd.Series:
    """池内该档的平均收益，权重要与分位口径一致。

    参数:
        frame: 分位分组表（池内、因子非空）。
        column: 用哪一档标签算收益。
        weight_column: 加权列；为空表示等权。
    """
    work = frame.loc[frame[column].notna()]
    if weight_column is None:
        return work.groupby("trade_date")[column].mean()
    work = work.loc[work[weight_column].notna() & (work[weight_column] > 0)]
    weighted = work.assign(_w=work[column] * work[weight_column])
    grouped = weighted.groupby("trade_date")
    return grouped["_w"].sum() / grouped[weight_column].sum()


def _spread_stats(series: pd.Series) -> dict:
    """把一条多空/超额序列压成统计量。

    参数:
        series: 逐日收益序列。
    """
    stats = series_stats(series.to_numpy(dtype="float64"))
    return {key: stats[key] for key in ("mean", "std", "t", "win_rate", "days")}


def compute_decay(context, options) -> dict:
    """算多周期衰减：各档 RankIC 与分位收益。

    参数:
        context: 评估上下文。
        options: 指标参数（本指标无参数）。
    """
    horizons = list(DECAY_HORIZONS)
    columns = {horizon: context.horizon_label(horizon) for horizon in horizons}
    series = daily_rank_ic_multi(context.panel, list(columns.values()))

    curve = [
        _curve_item(horizon, columns[horizon], series[columns[horizon]])
        for horizon in horizons
    ]

    frame = context.bins()
    if frame.empty:
        return {
            "horizons": horizons,
            "price": context.params.price,
            "quantiles": context.params.quantiles,
            "curve": curve,
            "long_excess": {},
            "long_short": {},
            "weightings": list(WEIGHTINGS),
            "weight_labels": dict(WEIGHT_LABELS),
            "empty": True,
            "hint": "池内没有非空因子值，无法算分位收益，衰减曲线只给了 RankIC",
        }

    quantiles = context.params.quantiles
    long_excess: dict[str, list] = {weight: [] for weight in WEIGHTINGS}
    long_short: dict[str, list] = {weight: [] for weight in WEIGHTINGS}
    for horizon in horizons:
        column = columns[horizon]
        for weight in WEIGHTINGS:
            weight_column = _weight_column(weight)
            daily = bin_daily_returns(
                frame, quantiles, label_column=column, weight_column=weight_column
            )
            top = daily[quantiles]
            bottom = daily[1]
            pool = _pool_mean(frame, column, weight_column)
            long_excess[weight].append({
                "horizon": horizon,
                **_spread_stats(top - pool.reindex(top.index)),
            })
            long_short[weight].append({
                "horizon": horizon,
                **_spread_stats(top - bottom),
            })

    # 主档优先取参数里的周期；参数用了 15 档（不在衰减档位里）时取最接近的一档
    primary_horizon = context.params.horizon
    if primary_horizon not in columns:
        primary_horizon = min(horizons, key=lambda horizon: abs(horizon - context.params.horizon))
    primary = next(item for item in curve if item["horizon"] == primary_horizon)
    if primary["insufficient"]:
        hint = (f"主档（{primary_horizon} 日）有效交易日只有 {primary['days']} 天，"
                f"少于 {INSUFFICIENT_DAYS} 天，结论不可用；请拉长评估区间")
    else:
        hint = (
            f"主档 {primary_horizon} 日 RankIC 均值 {primary['rank_ic_mean']:+.4f}，"
            f"ICIR {primary['rank_ic_ir']:.3f}；各档独立取样本，长周期档天数更少"
        )
    return {
        "horizons": horizons,
        "primary_horizon": primary_horizon,
        "eval_horizon": context.params.horizon,
        "price": context.params.price,
        "quantiles": quantiles,
        "long_bin": quantiles,
        "short_bin": 1,
        "weightings": list(WEIGHTINGS),
        "weight_labels": dict(WEIGHT_LABELS),
        "curve": curve,
        "long_excess": long_excess,
        "long_short": long_short,
        "curve_basis": "各档 RankIC 均值随标签周期的变化",
        "long_excess_basis": "最高分位 − 池内均值（池内因子非空样本的该档平均收益）",
        "long_short_basis": "最高分位 − 最低分位",
        "basis": "各档标签独立取样本：停牌与末端不足 N 个交易日的行只从该档剔除，不做填充",
        "note": "期收益逐日重叠（1 日与 20 日的持有区间互相套嵌），跨档的 t 值不可直接比较",
        "min_days": INSUFFICIENT_DAYS,
        "empty": False,
        "hint": hint,
    }


register_metric(
    Metric(key="decay", label="多周期衰减", stage="decay", compute=compute_decay)
)
