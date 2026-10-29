"""把 V2 分层窗口适配到现有确定性评估器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mining.candidate_evaluator import evaluate_candidate
from mining.candidate_stage_evaluator import (
    evaluate_full_candidate,
    evaluate_holdout_candidate,
)
from mining.contract_candidates import CandidateDefinition
from mining.contract_requests import FrozenContext, MiningConstraints
from mining.evaluation_contract import (
    CandidateEvaluationInput,
    CandidateEvaluationResult,
    FullEvaluationResult,
    HoldoutEvaluationResult,
)
from mining.search_context import FrozenSearchContext


def _legacy_context(
    context: FrozenSearchContext,
    *,
    research_start: str,
    research_end: str,
) -> FrozenContext:
    budget = context.budget
    return FrozenContext(
        data_version=context.data_version,
        data_fingerprint=context.data_fingerprint,
        label=context.label,
        universe=context.universe,
        price_channel=context.price_channel,
        research_start=research_start,
        research_end=research_end,
        full_eval_start=research_start,
        full_eval_end=research_end,
        holdout_start=context.windows.holdout_start,
        holdout_end=context.windows.holdout_end,
        llm_profile=context.llm_profile,
        llm_model=context.llm_model,
        field_catalog=context.field_catalog,
        operator_catalog=context.operator_catalog,
        existing_factors=context.existing_factors,
        budget=MiningConstraints(
            max_rounds=budget.max_search_turns,
            max_candidates=budget.max_candidates_per_run,
            full_eval_top_k=budget.max_full_evaluations,
            max_llm_calls=budget.max_search_turns,
            max_expression_depth=6,
            min_coverage=context.control_policy.min_coverage,
            max_failure_rate=context.control_policy.max_failure_rate,
            min_quality_checks_for_failure_stop=min(
                context.control_policy.min_quality_checks_for_failure_stop,
                budget.max_candidates_per_run,
            ),
        ),
    )


def development_context(context: FrozenSearchContext) -> FrozenContext:
    """返回开发期评估上下文。"""

    return _legacy_context(
        context,
        research_start=context.windows.development_start,
        research_end=context.windows.development_end,
    )


def selection_context(context: FrozenSearchContext) -> FrozenContext:
    """返回选择期评估上下文，仅供确定性选择器使用。"""

    return _legacy_context(
        context,
        research_start=context.windows.selection_start,
        research_end=context.windows.selection_end,
    )


def final_context(context: FrozenSearchContext) -> FrozenContext:
    """返回最终研究期评估上下文。"""

    return _legacy_context(
        context,
        research_start=context.windows.final_start,
        research_end=context.windows.final_end,
    )


def holdout_context(context: FrozenSearchContext) -> FrozenContext:
    """返回模块测试段执行器可见的上下文。"""

    return _legacy_context(
        context,
        research_start=context.windows.final_start,
        research_end=context.windows.final_end,
    )


def _input(
    *,
    run_id: str,
    candidate: CandidateDefinition,
    context: FrozenContext,
    stage_root: Path | str,
) -> CandidateEvaluationInput:
    return CandidateEvaluationInput(
        run_id=run_id,
        candidate_id=candidate.candidate_id,
        candidate=candidate,
        context=context,
        candidate_root=stage_root,
    )


def evaluate_development(
    *,
    run_id: str,
    candidate: CandidateDefinition,
    context: FrozenSearchContext,
    stage_root: Path | str,
) -> CandidateEvaluationResult:
    """执行技术门禁和开发期快速评估。"""

    return evaluate_candidate(
        _input(
            run_id=run_id,
            candidate=candidate,
            context=development_context(context),
            stage_root=stage_root,
        )
    )


def evaluate_selection(
    *,
    run_id: str,
    candidate: CandidateDefinition,
    context: FrozenSearchContext,
    stage_root: Path | str,
) -> CandidateEvaluationResult:
    """执行选择期快速评估。"""

    return evaluate_candidate(
        _input(
            run_id=run_id,
            candidate=candidate,
            context=selection_context(context),
            stage_root=stage_root,
        )
    )


def evaluate_final(
    *,
    run_id: str,
    candidate: CandidateDefinition,
    context: FrozenSearchContext,
    stage_root: Path | str,
) -> FullEvaluationResult:
    """执行最终研究期完整评估。"""

    selected = final_context(context)
    evaluation_input = _input(
        run_id=run_id,
        candidate=candidate,
        context=selected,
        stage_root=stage_root,
    )
    object.__setattr__(
        evaluation_input,
        "quick_window",
        (selected.research_start, selected.research_end),
    )
    object.__setattr__(
        evaluation_input,
        "full_window",
        (selected.full_eval_start, selected.full_eval_end),
    )
    return evaluate_full_candidate(evaluation_input)


def evaluate_holdout(
    *,
    run_id: str,
    candidate: CandidateDefinition,
    context: FrozenSearchContext,
    stage_root: Path | str,
) -> HoldoutEvaluationResult:
    """执行一次性模块测试段。"""

    selected = holdout_context(context)
    return evaluate_holdout_candidate(
        _input(
            run_id=run_id,
            candidate=candidate,
            context=selected,
            stage_root=stage_root,
        )
    )


def candidate_from_record(record: dict[str, Any]) -> CandidateDefinition:
    """从 V2 候选载荷恢复不可变候选定义。"""

    payload = dict(record["payload"])
    payload.pop("status", None)
    return CandidateDefinition(**payload)
