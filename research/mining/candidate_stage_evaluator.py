"""S6 使用的完整评估和模块测试段评估入口。"""

from __future__ import annotations

from pathlib import Path

from calculate import EvalParams, build_panel, evaluate_context, planned_stages
from calculate.jsonutil import json_safe
from mining.candidate_evaluator import (
    build_evaluation_report,
    full_metrics,
    validate_frozen_inputs,
    write_evaluation_artifact,
)
from mining.candidate_materialization import materialize_candidate
from mining.evaluation_contract import (
    CandidateEvaluationInput,
    FullEvaluationResult,
    HoldoutEvaluationResult,
    QualityPolicy,
    load_quality_policy,
    stage_input_hash,
)
from mining.contract_types import utc_now_iso


def evaluate_full_candidate(
    evaluation_input: CandidateEvaluationInput,
    *,
    policy: QualityPolicy | None = None,
) -> FullEvaluationResult:
    """在研究期已有产物上只执行完整评估。"""

    policy = policy or load_quality_policy()
    validate_frozen_inputs(evaluation_input)
    materialized = materialize_candidate(
        evaluation_input.candidate,
        evaluation_input.context,
        evaluation_input.candidate_root,
        start=evaluation_input.quick_window[0],
        end=evaluation_input.quick_window[1],
    )
    input_hash = stage_input_hash(
        stage="full",
        evaluation_input=evaluation_input,
        factor_sha256=materialized.artifact_sha256,
        params=evaluation_input.full_params_material,
        policy=policy,
    )
    started_at = utc_now_iso()
    try:
        metrics = full_metrics(evaluation_input, materialized.factor_view)
        artifact_path, artifact_sha = write_evaluation_artifact(
            Path(evaluation_input.candidate_root) / "evaluations" / "full.json",
            json_safe(metrics),
        )
        report = build_evaluation_report(
            evaluation_input=evaluation_input,
            stage="full",
            status="passed",
            input_hash=input_hash,
            metrics=metrics,
            warnings=(),
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha,
            started_at=started_at,
        )
        warnings: tuple[str, ...] = ()
    except Exception as exc:
        report = build_evaluation_report(
            evaluation_input=evaluation_input,
            stage="full",
            status="error",
            input_hash=input_hash,
            metrics={},
            warnings=(),
            artifact_path=None,
            artifact_sha256=None,
            error_code="MINING-106",
            error_message=str(exc)[:2000],
            started_at=started_at,
        )
        warnings = (f"完整评估失败：{exc}",)
    return FullEvaluationResult(
        report=report,
        warnings=warnings,
        factor_glob=materialized.factor_view.parquet_glob,
        factor_sha256=materialized.artifact_sha256,
    )


def evaluate_holdout_candidate(
    evaluation_input: CandidateEvaluationInput,
    *,
    policy: QualityPolicy | None = None,
) -> HoldoutEvaluationResult:
    """在隔离产物上执行一次性模块测试段。"""

    policy = policy or load_quality_policy()
    validate_frozen_inputs(evaluation_input)
    holdout_root = Path(evaluation_input.candidate_root) / "holdout"
    materialized = materialize_candidate(
        evaluation_input.candidate,
        evaluation_input.context,
        holdout_root,
        start=evaluation_input.context.holdout_start,
        end=evaluation_input.context.holdout_end,
    )
    input_hash = stage_input_hash(
        stage="holdout",
        evaluation_input=evaluation_input,
        factor_sha256=materialized.artifact_sha256,
        params={
            "start": evaluation_input.context.holdout_start,
            "end": evaluation_input.context.holdout_end,
            "universe": evaluation_input.context.universe,
            "label": evaluation_input.context.label,
            "price_channel": evaluation_input.context.price_channel,
            "horizon": 5,
            "quantiles": 10,
            "winsorize": True,
            "data_version": evaluation_input.context.data_version,
            "data_fingerprint": evaluation_input.context.data_fingerprint,
        },
        policy=policy,
    )
    started_at = utc_now_iso()
    try:
        params = EvalParams(
            materialized.factor_view.name,
            evaluation_input.context.holdout_start,
            evaluation_input.context.holdout_end,
            universe=evaluation_input.context.universe,
            horizon=5,
            price="open",
            quantiles=10,
            winsorize=True,
        )
        context = build_panel(params, factor=materialized.factor_view)
        valid_days = int(context.daily_ic()["rank_ic"].dropna().shape[0])
        expected_stages = set(planned_stages())
        events = {
            stage: payload
            for stage, payload in evaluate_context(context)
            if stage not in {"meta", "done"}
        }
        complete = set(events) == expected_stages
        passed = complete and valid_days >= policy.quick_min_valid_days
        metrics = {
            "valid_days": valid_days,
            "min_valid_days": policy.quick_min_valid_days,
            "complete": complete,
            "stages": events,
        }
        artifact_path, artifact_sha = write_evaluation_artifact(
            Path(evaluation_input.candidate_root) / "evaluations" / "holdout.json",
            json_safe(metrics),
        )
        warnings = () if passed else (
            "模块测试段有效交易日不足或评估阶段不完整",
        )
        report = build_evaluation_report(
            evaluation_input=evaluation_input,
            stage="holdout",
            status="passed" if passed else "failed",
            input_hash=input_hash,
            metrics=metrics,
            warnings=warnings,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha,
            started_at=started_at,
        )
    except Exception as exc:
        passed = False
        warnings = (f"模块测试段执行失败：{exc}",)
        report = build_evaluation_report(
            evaluation_input=evaluation_input,
            stage="holdout",
            status="error",
            input_hash=input_hash,
            metrics={},
            warnings=warnings,
            artifact_path=None,
            artifact_sha256=None,
            error_code="MINING-106",
            error_message=str(exc)[:2000],
            started_at=started_at,
        )
    return HoldoutEvaluationResult(
        report=report,
        passed=passed,
        warnings=warnings,
        factor_glob=materialized.factor_view.parquet_glob,
        factor_sha256=materialized.artifact_sha256,
    )
