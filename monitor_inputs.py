"""模型分数面板、缺失和滚动 RankIC 监控。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from alpha.artifacts import validate_dev_range
from alpha.errors import AlphaMonitorError
from alpha.evaluate import rank_ic_series
from alpha.monitor_policy import MonitorPolicy
from alpha.portfolio import smooth_rank_scores

DEPLOYED_SMOOTHING_WEEKS = 4


def normalize_date_frame(
    frame: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> pd.DataFrame:
    """复制输入并校验必需列、日期和键唯一性。"""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise AlphaMonitorError(f"{label}不能为空")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise AlphaMonitorError(f"{label}缺少列：{missing}")
    work = frame.copy()
    try:
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="raise")
    except (TypeError, ValueError) as error:
        raise AlphaMonitorError(f"{label}包含无效交易日期") from error
    if work["trade_date"].isna().any():
        raise AlphaMonitorError(f"{label}交易日期不能为空")
    if {"trade_date", "stock_code"} <= set(work.columns):
        if work.duplicated(["trade_date", "stock_code"]).any():
            raise AlphaMonitorError(f"{label}存在重复日期和股票")
    return work.sort_values(
        [column for column in ("trade_date", "stock_code") if column in work.columns],
        kind="mergesort",
    ).reset_index(drop=True)


def prepare_signal_frames(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    rebalance_dates: Sequence[object],
    *,
    start: str | None,
    end: str | None,
    allow_holdout: bool = False,
    holdout_window: object | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DatetimeIndex,
]:
    """对齐预测、标签、股票池和显式信号日。"""

    pred = normalize_date_frame(
        predictions,
        ("trade_date", "stock_code", "score"),
        "预测",
    )
    scores = pd.to_numeric(pred["score"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(scores).all():
        raise AlphaMonitorError("预测分数必须全部为有限值")
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(rebalance_dates)).unique()))
    if dates.empty:
        raise AlphaMonitorError("监控没有信号日")
    start_stamp = pd.Timestamp(start) if start is not None else dates.min()
    end_stamp = pd.Timestamp(end) if end is not None else dates.max()
    validate_dev_range(
        str(start_stamp.date()),
        str(end_stamp.date()),
        allow_holdout=allow_holdout,
        holdout_window=holdout_window,
    )
    dates = dates[(dates >= start_stamp) & (dates <= end_stamp)]
    if len(dates) < 2:
        raise AlphaMonitorError("监控区间至少需要 2 个信号日")

    pred_dates = pd.DatetimeIndex(pred["trade_date"].unique())
    if len(pred_dates) != len(dates) or not pred_dates.difference(dates).empty:
        missing = dates.difference(pred_dates)
        extra = pred_dates.difference(dates)
        raise AlphaMonitorError(
            "预测日期与回测信号日不一致："
            f"缺少 {len(missing)} 日，多余 {len(extra)} 日"
        )

    panel = normalize_date_frame(
        data,
        ("trade_date", "stock_code", "eligible", "label_raw", "industry_l1"),
        "数据集",
    )
    panel = panel.loc[panel["trade_date"].isin(dates)].copy()
    panel_dates = pd.DatetimeIndex(panel["trade_date"].unique())
    if len(panel_dates) != len(dates) or not panel_dates.difference(dates).empty:
        raise AlphaMonitorError("数据集没有完整覆盖监控信号日")

    stock_columns = pd.Index(panel["stock_code"].astype(str).unique())
    score_panel = pred.pivot(
        index="trade_date",
        columns="stock_code",
        values="score",
    ).reindex(index=dates, columns=stock_columns)
    score_panel.index.name = "trade_date"
    score_panel.columns.name = "stock_code"
    eligible = (
        panel.pivot(index="trade_date", columns="stock_code", values="eligible")
        .reindex(index=dates, columns=score_panel.columns)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    labels = (
        panel.pivot(index="trade_date", columns="stock_code", values="label_raw")
        .reindex(index=dates, columns=score_panel.columns)
        .astype("float64")
    )
    eligible.index.name = "trade_date"
    eligible.columns.name = "stock_code"
    labels.index.name = "trade_date"
    labels.columns.name = "stock_code"
    return panel, score_panel.astype("float64"), eligible, labels, dates


def rank_ic(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    eligible: pd.DataFrame,
) -> pd.Series:
    """按信号日计算股票池内 RankIC。"""

    score_rows = (
        scores.where(eligible).stack(future_stack=True).rename("score").reset_index()
    )
    label_rows = (
        labels.where(eligible).stack(future_stack=True).rename("label").reset_index()
    )
    merged = score_rows.merge(label_rows, on=["trade_date", "stock_code"]).dropna()
    if merged.empty:
        return pd.Series(dtype="float64")
    return rank_ic_series(merged).reindex(scores.index)


def score_statistics(
    scores: pd.DataFrame,
    eligible: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """计算每日截面分数分位数和离散度。"""

    rows: list[dict] = []
    for day in scores.index:
        values = scores.loc[day].where(eligible.loc[day]).dropna().to_numpy(
            dtype="float64"
        )
        if values.size:
            quantiles = np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        else:
            quantiles = np.full(5, np.nan)
            std = np.nan
        rows.append(
            {
                "trade_date": day,
                f"{prefix}_score_p01": quantiles[0],
                f"{prefix}_score_p25": quantiles[1],
                f"{prefix}_score_median": quantiles[2],
                f"{prefix}_score_p75": quantiles[3],
                f"{prefix}_score_p99": quantiles[4],
                f"{prefix}_score_std": std,
            }
        )
    return pd.DataFrame(rows).set_index("trade_date")


def score_timeseries(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    rebalance_dates: Sequence[object],
    policy: MonitorPolicy,
    *,
    start: str | None,
    end: str | None,
    allow_holdout: bool = False,
    holdout_window: object | None = None,
    smoothing_weeks: int = DEPLOYED_SMOOTHING_WEEKS,
) -> pd.DataFrame:
    """计算分数、缺失、滚动 RankIC 和行业对齐辅助数据。"""

    panel, raw, eligible, labels, dates = prepare_signal_frames(
        predictions,
        data,
        rebalance_dates,
        start=start,
        end=end,
        allow_holdout=allow_holdout,
        holdout_window=holdout_window,
    )
    deployed = smooth_rank_scores(
        raw,
        eligible,
        window=smoothing_weeks,
        min_periods=1,
    )
    result = pd.DataFrame(index=dates)
    result.index.name = "trade_date"
    eligible_count = eligible.sum(axis=1).astype(int)
    present_count = raw.where(eligible).notna().sum(axis=1).astype(int)
    result["signal_index"] = np.arange(1, len(dates) + 1)
    result["eligible_count"] = eligible_count.to_numpy()
    result["prediction_count"] = present_count.to_numpy()
    result["missing_rate"] = (
        (eligible_count - present_count).clip(lower=0)
        / eligible_count.replace(0, np.nan)
    ).to_numpy()
    result = result.join(score_statistics(raw, eligible, "raw"))
    result = result.join(score_statistics(deployed, eligible, "deployed"))
    result["raw_rank_ic"] = rank_ic(raw, labels, eligible).reindex(dates)
    result["deployed_rank_ic"] = rank_ic(deployed, labels, eligible).reindex(dates)
    rolling = lambda series: series.rolling(
        policy.rolling_weeks,
        min_periods=policy.min_periods,
    ).mean()
    result["rolling_raw_rank_ic"] = rolling(result["raw_rank_ic"])
    result["rolling_deployed_rank_ic"] = rolling(result["deployed_rank_ic"])
    industry_map = (
        panel.loc[:, ["trade_date", "stock_code", "industry_l1"]]
        .assign(industry_l1=lambda frame: frame["industry_l1"].fillna("未知"))
        .set_index(["trade_date", "stock_code"])["industry_l1"]
    )
    universe = panel.loc[panel["eligible"].astype(bool)]
    universe_weights = {
        day: group["industry_l1"].fillna("未知").value_counts(normalize=True)
        for day, group in universe.groupby("trade_date", sort=False)
    }
    result.attrs["industry_map"] = industry_map
    result.attrs["universe_weights"] = universe_weights
    return result


__all__ = [
    "DEPLOYED_SMOOTHING_WEEKS",
    "normalize_date_frame",
    "prepare_signal_frames",
    "rank_ic",
    "score_statistics",
    "score_timeseries",
]
