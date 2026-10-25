"""模型生命周期因果观测、状态机和重训日程。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alpha.artifacts import validate_dev_range
from alpha.config import DEFAULT_CONTRACT
from alpha.errors import AlphaLifecycleError
from alpha.lifecycle_policy import LifecyclePolicy, LifecycleResult, _number

REQUIRED_COLUMNS = (
    "trade_date",
    "rolling_deployed_rank_ic",
    "rolling_excess",
    "drawdown",
    "rolling_turnover_two_way_annual",
    "missing_rate",
    "max_style_abs_exposure",
    "max_abs_industry_active_weight",
)


def _add_trigger(
    triggers: list[str],
    severity: list[str],
    metric: str,
    level: str,
    reason: str,
) -> None:
    """记录一个指标触发项。"""

    triggers.append(f"{metric}:{reason}")
    severity.append(level)


def _classify_row(
    row: pd.Series,
    previous_rank_ic: float | None,
    policy: LifecyclePolicy,
    reference_turnover: float,
) -> dict:
    """按固定阈值分类一个已对齐的因果观测。"""

    triggers: list[str] = []
    severity: list[str] = []
    rank_ic = _number(row["rolling_deployed_rank_ic"])
    if rank_ic is not None and rank_ic < policy.monitor.rank_ic_warning:
        consecutive = (
            previous_rank_ic is not None
            and previous_rank_ic < policy.monitor.rank_ic_warning
        )
        _add_trigger(
            triggers,
            severity,
            "rolling_deployed_rank_ic",
            "alert" if consecutive else "warning",
            "连续两期低于零" if consecutive else "低于零",
        )

    excess = _number(row["rolling_excess"])
    if excess is not None:
        if excess <= policy.monitor.excess_alert:
            _add_trigger(
                triggers, severity, "rolling_excess", "alert", "达到告警下限"
            )
        elif excess < policy.monitor.excess_warning:
            _add_trigger(triggers, severity, "rolling_excess", "warning", "为负")

    drawdown = _number(row["drawdown"])
    if drawdown is not None:
        if drawdown <= policy.monitor.drawdown_alert:
            _add_trigger(
                triggers, severity, "drawdown", "alert", "达到告警下限"
            )
        elif drawdown <= policy.monitor.drawdown_warning:
            _add_trigger(triggers, severity, "drawdown", "warning", "达到预警线")

    turnover = _number(row["rolling_turnover_two_way_annual"])
    if turnover is not None:
        if turnover >= reference_turnover * policy.monitor.turnover_alert_multiple:
            _add_trigger(
                triggers,
                severity,
                "rolling_turnover_two_way_annual",
                "alert",
                "达到告警倍数",
            )
        elif turnover >= reference_turnover * policy.monitor.turnover_warning_multiple:
            _add_trigger(
                triggers,
                severity,
                "rolling_turnover_two_way_annual",
                "warning",
                "达到预警倍数",
            )

    missing = _number(row["missing_rate"])
    hard_pause = False
    if missing is not None:
        if missing > policy.monitor.missing_alert:
            _add_trigger(
                triggers, severity, "missing_rate", "alert", "超过 5% 完整性上限"
            )
            hard_pause = True
        elif missing > policy.monitor.missing_warning:
            _add_trigger(triggers, severity, "missing_rate", "warning", "出现缺失")

    style = _number(row["max_style_abs_exposure"])
    if style is not None:
        if style > policy.monitor.style_alert:
            _add_trigger(
                triggers, severity, "max_style_abs_exposure", "alert", "超过告警线"
            )
        elif style > policy.monitor.style_warning:
            _add_trigger(
                triggers, severity, "max_style_abs_exposure", "warning", "超过预警线"
            )

    industry = _number(row["max_abs_industry_active_weight"])
    if industry is not None:
        if industry > policy.monitor.industry_alert:
            _add_trigger(
                triggers,
                severity,
                "max_abs_industry_active_weight",
                "alert",
                "超过告警线",
            )
        elif industry > policy.monitor.industry_warning:
            _add_trigger(
                triggers,
                severity,
                "max_abs_industry_active_weight",
                "warning",
                "超过预警线",
            )

    if "alert" in severity:
        level = "alert"
    elif "warning" in severity:
        level = "warning"
    elif rank_ic is None:
        level = "insufficient_data"
    else:
        level = "ok"
    return {
        "observable_rank_ic": rank_ic,
        "observation_severity": level,
        "alert_metrics": ",".join(
            sorted({item.split(":", 1)[0] for item in triggers})
        ),
        "triggers": ";".join(triggers),
        "hard_pause": hard_pause,
    }


def _state_action(
    state: str,
    policy: LifecyclePolicy,
) -> tuple[str | None, float]:
    """返回状态对应的活动和仓位乘数。"""

    if state in {"normal", "reduced"}:
        return policy.primary_model_id, (
            1.0 if state == "normal" else policy.reduced_multiplier
        )
    if state == "rollback":
        return policy.fallback_model_id, 1.0
    return None, 0.0


def build_retrain_schedule(
    signal_dates: Sequence[object],
    folds: Sequence[dict],
    *,
    retrain_start: str = "2012-01-04",
) -> pd.DataFrame:
    """根据模型折表生成年度训练和重训到期事件。"""

    dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(signal_dates)).unique()))
    if dates.empty:
        raise AlphaLifecycleError("生命周期没有信号日")
    if not folds:
        raise AlphaLifecycleError("模型 manifest 没有训练折")

    rows: list[dict] = []
    train_starts: set[str] = set()
    years: set[int] = set()
    for fold in sorted(folds, key=lambda item: str(item.get("test_start") or "")):
        required = {
            "fold_id",
            "train_start",
            "train_end",
            "valid_start",
            "valid_end",
            "test_start",
            "test_end",
            "purge_days",
            "embargo_days",
        }
        missing = sorted(required - set(fold))
        if missing:
            raise AlphaLifecycleError(f"训练折缺少字段：{missing}")
        try:
            train_start = pd.Timestamp(fold["train_start"])
            train_end = pd.Timestamp(fold["train_end"])
            valid_start = pd.Timestamp(fold["valid_start"])
            valid_end = pd.Timestamp(fold["valid_end"])
            test_start = pd.Timestamp(fold["test_start"])
            test_end = pd.Timestamp(fold["test_end"])
        except (TypeError, ValueError) as error:
            raise AlphaLifecycleError("训练折日期字段非法") from error
        if not train_start < train_end < valid_start <= valid_end < test_start:
            raise AlphaLifecycleError(f"{fold['fold_id']} 的训练、验证和考试窗口不连续")
        if int(fold["purge_days"]) != DEFAULT_CONTRACT.purge_days:
            raise AlphaLifecycleError(f"{fold['fold_id']} 的 purge_days 不符合契约")
        if int(fold["embargo_days"]) != DEFAULT_CONTRACT.embargo_days:
            raise AlphaLifecycleError(f"{fold['fold_id']} 的 embargo_days 不符合契约")
        if str(train_start.date()) != retrain_start:
            raise AlphaLifecycleError(
                f"{fold['fold_id']} 的训练起点不是 {retrain_start}"
            )
        if valid_end.year != test_start.year - 1:
            raise AlphaLifecycleError(f"{fold['fold_id']} 未使用前一完整验证年")
        train_starts.add(str(train_start.date()))
        if test_start.year in years:
            raise AlphaLifecycleError(f"训练折存在重复测试年份：{test_start.year}")
        years.add(test_start.year)
        candidates = dates[dates.year == test_start.year]
        if candidates.empty:
            raise AlphaLifecycleError(f"没有覆盖 {test_start.year} 的信号日")
        event_date = candidates.min()
        rows.append(
            {
                "trade_date": event_date,
                "event_type": (
                    "initial_fit" if not rows else "retrain_due"
                ),
                "fold_id": str(fold["fold_id"]),
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "valid_start": valid_start.date().isoformat(),
                "valid_end": valid_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "purge_days": int(fold["purge_days"]),
                "embargo_days": int(fold["embargo_days"]),
            }
        )
    if len(train_starts) != 1:
        raise AlphaLifecycleError("年度重训必须使用同一个扩展训练起点")
    schedule = pd.DataFrame(rows).sort_values("trade_date", kind="mergesort")
    return schedule.reset_index(drop=True)


def _transition_event(
    day: pd.Timestamp,
    previous: str,
    current: str,
    row: pd.Series,
) -> dict:
    """构造一次状态切换事件。"""

    event_type = {
        "reduced": "reduce",
        "rollback": "rollback",
        "paused": "pause",
        "normal": "recover",
    }[current]
    return {
        "trade_date": day,
        "event_type": event_type,
        "fold_id": None,
        "train_start": None,
        "train_end": None,
        "valid_start": None,
        "valid_end": None,
        "test_start": None,
        "test_end": None,
        "purge_days": None,
        "embargo_days": None,
        "from_state": previous,
        "to_state": current,
        "active_model_id": row["active_model_id"],
        "exposure_multiplier": row["exposure_multiplier"],
        "reason": row["triggers"] or row["observation_severity"],
    }


def evaluate_lifecycle(
    timeseries: pd.DataFrame,
    *,
    reference_turnover: float,
    folds: Sequence[dict],
    policy: LifecyclePolicy | None = None,
    start: str | None = None,
    end: str | None = None,
    allow_holdout: bool = False,
    holdout_window: object | None = None,
    include_retrain_schedule: bool = True,
) -> LifecycleResult:
    """计算因果生命周期状态、动作和年度重训日程。"""

    settings = policy or LifecyclePolicy()
    reference = _number(reference_turnover)
    if reference is None or reference <= 0:
        raise AlphaLifecycleError("账本基准换手必须为正数")
    if not isinstance(timeseries, pd.DataFrame) or timeseries.empty:
        raise AlphaLifecycleError("监控时序不能为空")
    missing = sorted(set(REQUIRED_COLUMNS) - set(timeseries.columns))
    if missing:
        raise AlphaLifecycleError(f"监控时序缺少列：{missing}")
    work = timeseries.copy()
    try:
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="raise")
    except (TypeError, ValueError) as error:
        raise AlphaLifecycleError("监控时序包含非法交易日期") from error
    work = work.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    if work["trade_date"].duplicated().any():
        raise AlphaLifecycleError("监控时序存在重复交易日期")
    dates = pd.DatetimeIndex(work["trade_date"])
    start_text = start or str(dates.min().date())
    end_text = end or str(dates.max().date())
    validate_dev_range(
        start_text,
        end_text,
        allow_holdout=allow_holdout,
        holdout_window=holdout_window,
    )
    work = work.loc[
        (work["trade_date"] >= pd.Timestamp(start_text))
        & (work["trade_date"] <= pd.Timestamp(end_text))
    ].reset_index(drop=True)
    if len(work) < 2:
        raise AlphaLifecycleError("生命周期区间至少需要 2 个信号日")
    dates = pd.DatetimeIndex(work["trade_date"])
    schedule = (
        build_retrain_schedule(
            dates,
            folds,
            retrain_start=settings.retrain_start,
        )
        if include_retrain_schedule
        else pd.DataFrame(
            columns=(
                "trade_date",
                "event_type",
                "fold_id",
                "train_start",
                "train_end",
                "valid_start",
                "valid_end",
                "test_start",
                "test_end",
                "purge_days",
                "embargo_days",
            )
        )
    )

    observable_rank_ic = pd.to_numeric(
        work["rolling_deployed_rank_ic"], errors="coerce"
    ).shift(1)
    state = "normal"
    warning_run = 0
    alert_run = 0
    ok_run = 0
    previous_rank_ic: float | None = None
    state_rows: list[dict] = []
    events: list[dict] = []
    for index, row in work.iterrows():
        observation = _classify_row(
            row,
            previous_rank_ic,
            settings,
            reference,
        )
        level = observation["observation_severity"]
        if level in {"warning", "alert"}:
            warning_run += 1
            ok_run = 0
        elif level == "ok":
            ok_run += 1
            warning_run = 0
        alert_run = alert_run + 1 if level == "alert" else (
            0 if level in {"ok", "warning"} else alert_run
        )

        previous_state = state
        if observation["hard_pause"]:
            state = "paused"
        elif alert_run >= settings.pause_after:
            state = "paused"
        elif alert_run >= settings.rollback_after:
            state = "rollback"
        elif state == "normal" and warning_run >= settings.reduce_after:
            state = "reduced"
        if (
            state in {"reduced", "rollback", "paused"}
            and ok_run >= settings.recover_after
        ):
            state = "normal"

        active_model, multiplier = _state_action(state, settings)
        row_payload = {
            "trade_date": row["trade_date"],
            "observable_rank_ic": observation["observable_rank_ic"],
            "observation_severity": level,
            "alert_metrics": observation["alert_metrics"],
            "triggers": observation["triggers"],
            "warning_run": warning_run,
            "alert_run": alert_run,
            "ok_run": ok_run,
            "state": state,
            "active_model_id": active_model,
            "exposure_multiplier": multiplier,
        }
        state_rows.append(row_payload)
        as_row = pd.Series(row_payload)
        if state != previous_state:
            events.append(
                _transition_event(
                    pd.Timestamp(row["trade_date"]),
                    previous_state,
                    state,
                    as_row,
                )
            )
        if observation["observable_rank_ic"] is not None:
            previous_rank_ic = float(observation["observable_rank_ic"])

    state_frame = pd.DataFrame(state_rows)
    schedule_events = schedule.assign(
        from_state=None,
        to_state=None,
        active_model_id=None,
        exposure_multiplier=None,
        reason="年度重训窗口到期；挑战者必须通过既有 P3-5 决策门",
    )
    schedule_events = schedule_events.loc[
        :,
        [
            "trade_date",
            "event_type",
            "fold_id",
            "train_start",
            "train_end",
            "valid_start",
            "valid_end",
            "test_start",
            "test_end",
            "purge_days",
            "embargo_days",
            "from_state",
            "to_state",
            "active_model_id",
            "exposure_multiplier",
            "reason",
        ],
    ]
    combined_events = [
        *events,
        *schedule_events.to_dict(orient="records"),
    ]
    event_frame = pd.DataFrame(combined_events)
    event_frame = event_frame.sort_values(
        ["trade_date", "event_type"], kind="mergesort"
    )
    event_frame = event_frame.reset_index(drop=True)

    state_counts = state_frame["state"].value_counts().to_dict()
    action_counts = event_frame["event_type"].value_counts().to_dict()
    latest = state_frame.iloc[-1]
    summary = {
        "start": str(pd.Timestamp(state_frame["trade_date"].min()).date()),
        "end": str(pd.Timestamp(state_frame["trade_date"].max()).date()),
        "signal_days": int(len(state_frame)),
        "state_counts": {
            name: int(state_counts.get(name, 0)) for name in ("normal", "reduced", "rollback", "paused")
        },
        "event_counts": {
            name: int(action_counts.get(name, 0))
            for name in ("initial_fit", "retrain_due", "reduce", "rollback", "pause", "recover")
        },
        "retrain_due_count": int(
            (event_frame["event_type"] == "retrain_due").sum()
        ),
        "latest_state": str(latest["state"]),
        "latest_active_model_id": (
            None if pd.isna(latest["active_model_id"]) else str(latest["active_model_id"])
        ),
        "latest_exposure_multiplier": float(latest["exposure_multiplier"]),
        "latest_observable_rank_ic": (
            None
            if pd.isna(latest["observable_rank_ic"])
            else float(latest["observable_rank_ic"])
        ),
    }
    return LifecycleResult(
        timeseries=state_frame,
        events=event_frame,
        retrain_schedule=schedule,
        summary=summary,
    )


__all__ = [
    "REQUIRED_COLUMNS",
    "build_retrain_schedule",
    "evaluate_lifecycle",
]
