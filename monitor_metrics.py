"""模型账本、暴露、告警和衰减演练计算。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alpha.errors import AlphaMonitorError
from alpha.logging_conf import get_logger
from alpha.monitor_inputs import (
    normalize_date_frame,
    score_timeseries,
)
from alpha.monitor_policy import (
    MonitorPolicy,
    MonitorResult,
    SEVERITY_ORDER,
    first_alert_date,
    number,
)

TRADING_DAYS = 252

logger = get_logger("alpha.monitor")


def _ledger_timeseries(
    ledger: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    policy: MonitorPolicy,
) -> pd.DataFrame:
    """把日频完整账本汇总到周频信号日。"""

    required = {
        "trade_date",
        "return",
        "benchmark_return",
        "turnover",
        "drawdown",
    }
    daily = normalize_date_frame(ledger, required, "完整账本")
    daily = daily.set_index("trade_date").sort_index()
    columns = ["return", "benchmark_return", "turnover", "drawdown"]
    for name in columns:
        daily[name] = pd.to_numeric(daily[name], errors="coerce")
    if not np.isfinite(daily.loc[:, columns].to_numpy(dtype="float64")).all():
        raise AlphaMonitorError("完整账本核心指标必须为有限数值")

    rows: list[dict] = []
    for index, day in enumerate(signal_dates):
        if index == 0:
            rows.append(
                {
                    "trade_date": day,
                    "period_start": pd.NaT,
                    "period_days": 0,
                    "period_return": np.nan,
                    "period_benchmark_return": np.nan,
                    "period_excess_return": np.nan,
                    "period_turnover": np.nan,
                    "drawdown": np.nan,
                }
            )
            continue
        previous = signal_dates[index - 1]
        segment = daily.loc[(daily.index > previous) & (daily.index <= day)]
        if segment.empty:
            raise AlphaMonitorError(
                f"完整账本缺少 {previous.date()}～{day.date()} 的监控区间"
            )
        returns = segment["return"].to_numpy(dtype="float64")
        benchmark = segment["benchmark_return"].to_numpy(dtype="float64")
        total_return = float(np.prod(1.0 + returns) - 1.0)
        benchmark_return = float(np.prod(1.0 + benchmark) - 1.0)
        rows.append(
            {
                "trade_date": day,
                "period_start": previous,
                "period_days": int(len(segment)),
                "period_return": total_return,
                "period_benchmark_return": benchmark_return,
                "period_excess_return": total_return - benchmark_return,
                "period_turnover": float(segment["turnover"].sum()),
                "drawdown": float(segment["drawdown"].iloc[-1]),
            }
        )
    result = pd.DataFrame(rows).set_index("trade_date")
    result["rolling_excess"] = result["period_excess_return"].rolling(
        policy.rolling_weeks,
        min_periods=policy.min_periods,
    ).sum()
    turnover_sum = result["period_turnover"].rolling(
        policy.rolling_weeks,
        min_periods=policy.min_periods,
    ).sum()
    day_sum = result["period_days"].rolling(
        policy.rolling_weeks,
        min_periods=policy.min_periods,
    ).sum()
    result["rolling_turnover_two_way_annual"] = (
        turnover_sum * TRADING_DAYS / day_sum.replace(0, np.nan)
    )
    return result


def _style_timeseries(
    style_series: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    policy: MonitorPolicy,
) -> pd.DataFrame:
    """按信号日计算绝对风格暴露的滚动均值。"""

    required = {"trade_date", "style", "exposure"}
    missing = sorted(required - set(style_series.columns))
    if missing:
        raise AlphaMonitorError(f"风格暴露缺少列：{missing}")
    work = normalize_date_frame(style_series, required, "风格暴露")
    if work.duplicated(["trade_date", "style"]).any():
        raise AlphaMonitorError("风格暴露存在重复日期和风格")
    work["exposure"] = pd.to_numeric(work["exposure"], errors="coerce")
    if not np.isfinite(work["exposure"].to_numpy(dtype="float64")).all():
        raise AlphaMonitorError("风格暴露必须全部为有限值")
    pivot = work.pivot(index="trade_date", columns="style", values="exposure")
    if not pd.DatetimeIndex(pivot.index).equals(signal_dates):
        raise AlphaMonitorError("风格暴露日期与监控信号日不一致")
    rolling = pivot.abs().rolling(
        policy.rolling_weeks,
        min_periods=policy.min_periods,
    ).mean()
    rows: list[dict] = []
    for day in signal_dates:
        values = rolling.loc[day].dropna()
        if values.empty:
            rows.append(
                {
                    "trade_date": day,
                    "max_style_abs_exposure": np.nan,
                    "top_style": None,
                }
            )
            continue
        top_style = str(values.idxmax())
        rows.append(
            {
                "trade_date": day,
                "max_style_abs_exposure": float(values.loc[top_style]),
                "top_style": top_style,
            }
        )
    return pd.DataFrame(rows).set_index("trade_date")


def _industry_timeseries(
    holdings: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    industry_map: pd.Series,
    universe_weights: dict,
) -> pd.DataFrame:
    """按信号日后首次持仓计算实际行业主动暴露。"""

    required = {"trade_date", "stock_code", "weight"}
    missing = sorted(required - set(holdings.columns))
    if missing:
        raise AlphaMonitorError(f"持仓缺少列：{missing}")
    work = normalize_date_frame(holdings, required, "完整账本持仓")
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
    if not np.isfinite(work["weight"].to_numpy(dtype="float64")).all():
        raise AlphaMonitorError("持仓权重必须全部为有限值")
    holding_dates = pd.DatetimeIndex(sorted(work["trade_date"].unique()))
    rows: list[dict] = []
    for index, day in enumerate(signal_dates):
        upper = signal_dates[index + 1] if index + 1 < len(signal_dates) else None
        candidates = holding_dates[holding_dates >= day]
        if upper is not None:
            candidates = candidates[candidates < upper]
        if candidates.empty:
            raise AlphaMonitorError(f"信号日 {day.date()} 后没有可对齐的实际持仓")
        holding_day = candidates[0]
        group = work.loc[work["trade_date"].eq(holding_day)].copy()
        codes = pd.MultiIndex.from_arrays(
            [np.repeat(day, len(group)), group["stock_code"].astype(str)]
        )
        group["industry"] = industry_map.reindex(codes).to_numpy()
        group["industry"] = group["industry"].fillna("未知")
        portfolio = group.groupby("industry")["weight"].sum()
        universe = universe_weights.get(day, pd.Series(dtype="float64"))
        names = portfolio.index.union(universe.index)
        active = portfolio.reindex(names, fill_value=0.0) - universe.reindex(
            names,
            fill_value=0.0,
        )
        top = str(active.abs().idxmax()) if len(active) else None
        rows.append(
            {
                "trade_date": day,
                "holding_date": holding_day,
                "max_abs_industry_active_weight": (
                    float(active.abs().max()) if len(active) else np.nan
                ),
                "top_industry": top,
                "top_industry_active_weight": (
                    float(active.loc[top]) if top is not None else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("trade_date")


def _add_alerts(
    timeseries: pd.DataFrame,
    policy: MonitorPolicy,
    reference_turnover: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按固定策略生成逐期状态和长表告警。"""

    result = timeseries.copy()
    alerts: list[dict] = []
    statuses: list[str] = []
    triggered: list[str] = []
    previous_rank_ic: float | None = None
    for day, row in result.iterrows():
        day_alerts: list[dict] = []

        def add(
            metric: str,
            value: object,
            threshold: float,
            severity: str,
            reason: str,
        ) -> None:
            day_alerts.append(
                {
                    "trade_date": day,
                    "metric": metric,
                    "value": number(value),
                    "threshold": float(threshold),
                    "severity": severity,
                    "reason": reason,
                }
            )

        rank_ic = number(row.get("rolling_deployed_rank_ic"))
        if rank_ic is not None and rank_ic < policy.rank_ic_warning:
            consecutive = (
                previous_rank_ic is not None
                and previous_rank_ic < policy.rank_ic_warning
            )
            add(
                "rolling_deployed_rank_ic",
                rank_ic,
                policy.rank_ic_warning,
                "alert" if consecutive else "warning",
                "部署分数滚动 RankIC 低于阈值"
                + ("，且连续两期发生" if consecutive else ""),
            )
        excess = number(row.get("rolling_excess"))
        if excess is not None:
            if excess <= policy.excess_alert:
                add(
                    "rolling_excess",
                    excess,
                    policy.excess_alert,
                    "alert",
                    "滚动净超额达到告警下限",
                )
            elif excess < policy.excess_warning:
                add(
                    "rolling_excess",
                    excess,
                    policy.excess_warning,
                    "warning",
                    "滚动净超额为负",
                )
        drawdown = number(row.get("drawdown"))
        if drawdown is not None:
            if drawdown <= policy.drawdown_alert:
                add(
                    "drawdown",
                    drawdown,
                    policy.drawdown_alert,
                    "alert",
                    "当前回撤达到告警下限",
                )
            elif drawdown <= policy.drawdown_warning:
                add(
                    "drawdown",
                    drawdown,
                    policy.drawdown_warning,
                    "warning",
                    "当前回撤达到预警线",
                )
        turnover = number(row.get("rolling_turnover_two_way_annual"))
        if turnover is not None:
            if turnover >= reference_turnover * policy.turnover_alert_multiple:
                add(
                    "rolling_turnover_two_way_annual",
                    turnover,
                    reference_turnover * policy.turnover_alert_multiple,
                    "alert",
                    "滚动换手达到账本基准的告警倍数",
                )
            elif turnover >= reference_turnover * policy.turnover_warning_multiple:
                add(
                    "rolling_turnover_two_way_annual",
                    turnover,
                    reference_turnover * policy.turnover_warning_multiple,
                    "warning",
                    "滚动换手达到账本基准的预警倍数",
                )
        missing = number(row.get("missing_rate"))
        if missing is not None:
            if missing > policy.missing_alert:
                add(
                    "missing_rate",
                    missing,
                    policy.missing_alert,
                    "alert",
                    "可交易股票分数缺失率过高",
                )
            elif missing > policy.missing_warning:
                add(
                    "missing_rate",
                    missing,
                    policy.missing_warning,
                    "warning",
                    "可交易股票出现分数缺失",
                )
        style = number(row.get("max_style_abs_exposure"))
        if style is not None:
            if style > policy.style_alert:
                add(
                    "max_style_abs_exposure",
                    style,
                    policy.style_alert,
                    "alert",
                    "滚动绝对风格暴露过高",
                )
            elif style > policy.style_warning:
                add(
                    "max_style_abs_exposure",
                    style,
                    policy.style_warning,
                    "warning",
                    "滚动绝对风格暴露偏高",
                )
        industry = number(row.get("max_abs_industry_active_weight"))
        if industry is not None:
            if industry > policy.industry_alert:
                add(
                    "max_abs_industry_active_weight",
                    industry,
                    policy.industry_alert,
                    "alert",
                    "实际组合行业主动暴露过高",
                )
            elif industry > policy.industry_warning:
                add(
                    "max_abs_industry_active_weight",
                    industry,
                    policy.industry_warning,
                    "warning",
                    "实际组合行业主动暴露偏高",
                )

        alerts.extend(day_alerts)
        if day_alerts:
            status = max(
                (item["severity"] for item in day_alerts),
                key=SEVERITY_ORDER.__getitem__,
            )
        elif number(row.get("rolling_deployed_rank_ic")) is None:
            status = "insufficient_data"
        else:
            status = "ok"
        statuses.append(status)
        triggered.append(",".join(sorted({item["metric"] for item in day_alerts})))
        if rank_ic is not None:
            previous_rank_ic = rank_ic
    result["status"] = statuses
    result["alert_metrics"] = triggered
    alert_frame = pd.DataFrame(
        alerts,
        columns=("trade_date", "metric", "value", "threshold", "severity", "reason"),
    )
    return result, alert_frame


def evaluate_monitor_frames(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    ledger: pd.DataFrame,
    rebalance_dates: Sequence[object],
    style_series: pd.DataFrame,
    holdings: pd.DataFrame,
    *,
    reference_turnover: float,
    policy: MonitorPolicy | None = None,
    start: str | None = None,
    end: str | None = None,
) -> MonitorResult:
    """计算一组已对齐的模型、账本和暴露监控结果。"""

    settings = policy or MonitorPolicy()
    reference = number(reference_turnover)
    if reference is None or reference <= 0:
        raise AlphaMonitorError("账本基准换手必须为正数")
    score = score_timeseries(
        predictions,
        data,
        rebalance_dates,
        settings,
        start=start,
        end=end,
    )
    dates = pd.DatetimeIndex(score.index)
    ledger_frame = _ledger_timeseries(ledger, dates, settings)
    style_frame = _style_timeseries(style_series, dates, settings)
    industry_frame = _industry_timeseries(
        holdings,
        dates,
        score.attrs["industry_map"],
        score.attrs["universe_weights"],
    )
    timeseries = pd.concat(
        [score, ledger_frame, style_frame, industry_frame],
        axis=1,
    )
    timeseries, alerts = _add_alerts(timeseries, settings, reference)
    style_summary = (
        style_series.loc[:, ["style", "exposure", "coverage", "holdings"]]
        .groupby("style", as_index=False)
        .agg(
            mean_exposure=("exposure", "mean"),
            mean_abs_exposure=(
                "exposure",
                lambda values: float(np.mean(np.abs(values))),
            ),
            min_exposure=("exposure", "min"),
            max_exposure=("exposure", "max"),
            mean_coverage=("coverage", "mean"),
            days=("exposure", "size"),
        )
        .sort_values("mean_abs_exposure", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    industry_rows = [
        {
            "industry": row.top_industry,
            "max_abs_active_weight": row.max_abs_industry_active_weight,
            "active_weight": row.top_industry_active_weight,
            "trade_date": row.Index,
        }
        for row in timeseries.loc[
            timeseries["max_abs_industry_active_weight"].notna()
        ].itertuples()
    ]
    industry_summary = pd.DataFrame(industry_rows)
    if not industry_summary.empty:
        industry_summary = (
            industry_summary.groupby("industry", as_index=False)
            .agg(
                max_abs_active_weight=("max_abs_active_weight", "max"),
                mean_top_active_weight=("active_weight", "mean"),
                appearances=("trade_date", "size"),
            )
            .sort_values("max_abs_active_weight", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
    latest = timeseries.iloc[-1]
    summary = {
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "signal_days": int(len(dates)),
        "status": str(latest["status"]),
        "latest_date": str(dates.max().date()),
        "latest_rolling_deployed_rank_ic": number(
            latest["rolling_deployed_rank_ic"]
        ),
        "latest_rolling_excess": number(latest["rolling_excess"]),
        "latest_turnover_two_way_annual": number(
            latest["rolling_turnover_two_way_annual"]
        ),
        "latest_drawdown": number(latest["drawdown"]),
        "latest_missing_rate": number(latest["missing_rate"]),
        "latest_max_style_abs_exposure": number(
            latest["max_style_abs_exposure"]
        ),
        "latest_top_style": (
            None if pd.isna(latest["top_style"]) else str(latest["top_style"])
        ),
        "latest_max_abs_industry_active_weight": number(
            latest["max_abs_industry_active_weight"]
        ),
        "latest_top_industry": (
            None if pd.isna(latest["top_industry"]) else str(latest["top_industry"])
        ),
        "warning_count": int((alerts["severity"] == "warning").sum()),
        "alert_count": int((alerts["severity"] == "alert").sum()),
        "first_alert_date": first_alert_date(alerts),
    }
    logger.info(
        "模型监控计算完成：%d 个信号日，%d warning，%d alert",
        len(dates),
        summary["warning_count"],
        summary["alert_count"],
    )
    return MonitorResult(
        timeseries=timeseries,
        alerts=alerts,
        style_summary=style_summary,
        industry_summary=industry_summary,
        summary=summary,
    )


def simulate_score_decay(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    rebalance_dates: Sequence[object],
    *,
    policy: MonitorPolicy | None = None,
    weeks: int = 13,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """反转最近若干周原始截面排名，验证 RankIC 告警链路。"""

    if type(weeks) is not int or weeks < 2:
        raise AlphaMonitorError("衰减演练周数必须是不小于 2 的整数")
    settings = policy or MonitorPolicy()
    original = score_timeseries(
        predictions,
        data,
        rebalance_dates,
        settings,
        start=start,
        end=end,
    )
    dates = pd.DatetimeIndex(original.index)[-min(weeks, len(original.index)) :]
    if len(dates) < 2:
        raise AlphaMonitorError("衰减演练至少需要 2 个信号日")
    work = predictions.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["stock_code"] = work["stock_code"].astype(str)
    selected = work.loc[work["trade_date"].isin(dates)].copy()
    selected["score"] = selected.groupby("trade_date", sort=False)["score"].rank(
        method="average",
        pct=True,
    )
    selected["score"] = -selected["score"]
    decayed = pd.concat(
        [work.loc[~work["trade_date"].isin(dates)], selected],
        ignore_index=True,
    )
    changed = score_timeseries(
        decayed,
        data,
        rebalance_dates,
        settings,
        start=start,
        end=end,
    )
    records = pd.DataFrame(
        {
            "trade_date": dates,
            "original_raw_rank_ic": original.loc[dates, "raw_rank_ic"].to_numpy(),
            "decayed_raw_rank_ic": changed.loc[dates, "raw_rank_ic"].to_numpy(),
            "original_deployed_rank_ic": original.loc[
                dates, "deployed_rank_ic"
            ].to_numpy(),
            "decayed_deployed_rank_ic": changed.loc[
                dates, "deployed_rank_ic"
            ].to_numpy(),
            "decayed_rolling_deployed_rank_ic": changed.loc[
                dates, "rolling_deployed_rank_ic"
            ].to_numpy(),
        }
    )
    rolling = records["decayed_rolling_deployed_rank_ic"].dropna()
    consecutive = (
        len(rolling) >= settings.rank_ic_alert_consecutive
        and (
            rolling.iloc[-settings.rank_ic_alert_consecutive :]
            < settings.rank_ic_warning
        ).all()
    )
    status = "alert" if consecutive else (
        "warning"
        if len(rolling) and rolling.iloc[-1] < settings.rank_ic_warning
        else "ok"
    )
    summary = {
        "weeks": int(len(dates)),
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "status": status,
        "alert_triggered": status == "alert",
        "latest_decayed_rolling_deployed_rank_ic": (
            number(rolling.iloc[-1]) if len(rolling) else None
        ),
    }
    return {**summary, "timeseries": records}


__all__ = ["evaluate_monitor_frames", "simulate_score_decay"]
