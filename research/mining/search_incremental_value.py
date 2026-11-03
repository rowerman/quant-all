"""选择期增量价值与正式因子相关性计算。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from calculate import EvalParams, build_panel
from calculate.metrics.exposure import compute_incremental_ic
from mining.candidate_materialization import materialize_candidate
from mining.contract_candidates import CandidateDefinition
from mining.search_context import FrozenSearchContext
from mining.search_evaluation_adapter import selection_context
from mining.search_selection_contracts import (
    IncrementalValueReport,
    finite_or_none,
)


def formal_factor_incremental_value(
    technical_metrics: Mapping[str, Any],
) -> tuple[str, str | None, float | None, float | None]:
    """从技术门禁结果提取正式因子相关性。"""

    payload = technical_metrics.get("existing_correlation")
    if not isinstance(payload, Mapping):
        return "not_computable", None, None, None
    reason = payload.get("reason")
    peer = payload.get("peer")
    correlation = finite_or_none(payload.get("correlation"))
    maximum = finite_or_none(payload.get("max_absolute"))
    if peer is None and reason == "no_peers":
        return "no_formal_factors", None, None, None
    if peer is None or correlation is None:
        return "not_computable", None, None, maximum
    return "available", str(peer), correlation, maximum


def compute_incremental_value(
    candidate: CandidateDefinition,
    context: FrozenSearchContext,
    candidate_root: Path | str,
    technical_metrics: Mapping[str, Any],
    *,
    db_file: Path | str,
    factor_dir: Path | str,
) -> IncrementalValueReport:
    """计算选择期风格残差 RankIC，并合并正式因子相关性。"""

    formal_status, peer, correlation, maximum = formal_factor_incremental_value(
        technical_metrics
    )
    selection = selection_context(context)
    try:
        materialized = materialize_candidate(
            candidate,
            selection,
            candidate_root,
            start=context.windows.selection_start,
            end=context.windows.selection_end,
        )
        params = EvalParams(
            materialized.factor_view.name,
            context.windows.selection_start,
            context.windows.selection_end,
            universe=context.universe,
            horizon=5,
            price="open",
            quantiles=10,
            winsorize=True,
        )
        eval_context = build_panel(
            params,
            db_file=db_file,
            factor_dir=factor_dir,
            factor=materialized.factor_view,
        )
        payload = compute_incremental_ic(eval_context, {})
        rank_ic = eval_context.daily_ic()["rank_ic"].dropna()
        rank_ic_dates = tuple(str(index.date()) for index in rank_ic.index)
        rank_ic_values = tuple(float(value) for value in rank_ic.to_numpy())
    except Exception as exc:
        return IncrementalValueReport(
            formal_factor_status=formal_status,
            formal_factor_peer=peer,
            formal_factor_correlation=correlation,
            formal_factor_max_absolute=maximum,
            raw_rank_ic=None,
            incremental_rank_ic=None,
            survivor_share=None,
            valid_days=None,
            status="not_computable",
            reason=str(exc)[:500],
        )
    raw = payload.get("raw") or {}
    residual = payload.get("residual") or {}
    valid_days = payload.get("days")
    if isinstance(valid_days, bool) or not isinstance(valid_days, int):
        valid_days = None
    if payload.get("insufficient") or residual.get("mean") is None:
        status = "not_computable"
        reason = str(payload.get("hint") or "增量 IC 有效日不足")
    else:
        status = "available"
        reason = None
    return IncrementalValueReport(
        formal_factor_status=formal_status,
        formal_factor_peer=peer,
        formal_factor_correlation=correlation,
        formal_factor_max_absolute=maximum,
        raw_rank_ic=finite_or_none(raw.get("mean")),
        incremental_rank_ic=finite_or_none(residual.get("mean")),
        survivor_share=finite_or_none(payload.get("survivor_share")),
        valid_days=valid_days,
        status=status,
        rank_ic_dates=rank_ic_dates,
        rank_ic_values=rank_ic_values,
        reason=reason,
    )


__all__ = [
    "compute_incremental_value",
    "formal_factor_incremental_value",
]
