"""候选技术门禁、快速筛选和可选完整评估编排。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from calculate import EvalParams, build_panel, evaluate_context, load_daily_ic
from calculate.correlation import available_factors
from calculate.jsonutil import json_safe
from common.paths import FACTOR_DIR, MINING_DATA_DIR
from mining.candidate_materialization import materialize_candidate
from mining.catalog import load_catalog
from mining.contract_evidence import EvaluationReport, TechnicalGate
from mining.contract_types import canonical_json, sha256_json, utc_now_iso
from mining.errors import MiningEvaluationError
from mining.evaluation_contract import (
    CandidateEvaluationInput,
    CandidateEvaluationResult,
    QualityPolicy,
    load_quality_policy,
    stage_input_hash,
)
from mining.evaluation_gates import evaluate_technical_checks
from mining.logging_conf import format_event, get_mining_logger

_LOGGER = get_mining_logger("mining.evaluation.candidate")


def validate_frozen_inputs(evaluation_input: CandidateEvaluationInput) -> None:
    """确认当前目录和正式因子清单仍与冻结上下文一致。"""

    catalog = load_catalog()
    context = evaluation_input.context
    if context.field_catalog and (
        _catalog_hash(context.field_catalog)
        != _catalog_hash(catalog.frozen_field_mappings())
    ):
        raise MiningEvaluationError("冻结字段目录与当前目录不一致")
    if context.operator_catalog and (
        _catalog_hash(context.operator_catalog)
        != _catalog_hash(catalog.frozen_operator_mappings())
    ):
        raise MiningEvaluationError("冻结算子目录与当前目录不一致")
    if context.existing_factors:
        current = tuple(
            sorted(
                name
                for names in available_factors(FACTOR_DIR).values()
                for name in names
            )
        )
        if current != context.existing_factors:
            raise MiningEvaluationError("正式因子清单与冻结上下文不一致")


def _catalog_hash(values) -> str:
    """忽略输入顺序比较目录内容。"""

    normalized = sorted(
        (dict(item) for item in values),
        key=canonical_json,
    )
    return sha256_json(normalized)


def _relative_artifact(path: Path) -> str:
    """优先记录相对候选数据根的产物路径。"""

    resolved = path.resolve()
    try:
        return resolved.relative_to(MINING_DATA_DIR.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_evaluation_artifact(
    path: Path,
    payload: dict[str, Any],
) -> tuple[str, str]:
    """原子写入评估载荷并返回路径和 SHA-256。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(payload), encoding="utf-8")
    os.replace(temporary, path)
    return _relative_artifact(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _quick_metrics(
    evaluation_input: CandidateEvaluationInput,
    factor_view,
    policy: QualityPolicy,
) -> tuple[dict[str, Any], tuple[str, ...], bool]:
    """研究期轻量 RankIC 筛选，不构建完整评估面板。"""

    start, end = evaluation_input.quick_window
    params = EvalParams(
        factor_view.name,
        start,
        end,
        universe=evaluation_input.context.universe,
        horizon=5,
        price="open",
        quantiles=10,
        winsorize=True,
    )
    started = time.perf_counter()
    series = load_daily_ic(params, factor=factor_view)["rank_ic"].dropna()
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    valid_days = int(len(series))
    if valid_days < policy.quick_min_valid_days:
        metrics = {
            "valid_days": valid_days,
            "min_valid_days": policy.quick_min_valid_days,
            "rank_ic_mean": None,
            "rank_ic_ir": None,
            "positive_ratio": None,
            "year_sign_consistency": None,
            "elapsed_ms": elapsed_ms,
        }
        return (
            metrics,
            (f"研究期有效 RankIC 不足 {policy.quick_min_valid_days} 个交易日",),
            False,
        )
    mean = float(series.mean())
    standard_deviation = float(series.std(ddof=1))
    expected_sign = 1 if mean > 0 else -1
    yearly = series.groupby(series.index.year).mean()
    consistency = float((np.sign(yearly) == expected_sign).mean())
    metrics = {
        "valid_days": valid_days,
        "min_valid_days": policy.quick_min_valid_days,
        "rank_ic_mean": mean,
        "rank_ic_ir": mean / standard_deviation if standard_deviation else None,
        "positive_ratio": float(
            (series > 0).mean() if mean > 0 else (series < 0).mean()
        ),
        "year_sign_consistency": consistency,
        "elapsed_ms": elapsed_ms,
    }
    warnings: list[str] = []
    if abs(mean) < policy.warning_rank_ic:
        warnings.append(f"研究期 |RankIC| 低于 {policy.warning_rank_ic:g}")
    if metrics["rank_ic_ir"] is None or abs(metrics["rank_ic_ir"]) < policy.warning_icir:
        warnings.append(f"研究期 |ICIR| 低于 {policy.warning_icir:g}")
    if consistency < policy.warning_year_sign_ratio:
        warnings.append(
            f"年度 RankIC 同号率低于 {policy.warning_year_sign_ratio:.0%}"
        )
    return metrics, tuple(warnings), True


def full_metrics(
    evaluation_input: CandidateEvaluationInput,
    factor_view,
) -> dict[str, Any]:
    """在候选临时因子路径上执行现有全部评估 stage。"""

    start, end = evaluation_input.full_window
    params = EvalParams(
        factor_view.name,
        start,
        end,
        universe=evaluation_input.context.universe,
        horizon=5,
        price="open",
        quantiles=10,
        winsorize=True,
    )
    started = time.perf_counter()
    context = build_panel(params, factor=factor_view)
    events: dict[str, Any] = {}
    for stage, payload in evaluate_context(context):
        events[stage] = payload
    return {
        "events": events,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def build_evaluation_report(
    *,
    evaluation_input: CandidateEvaluationInput,
    stage: str,
    status: str,
    input_hash: str,
    metrics: dict[str, Any],
    warnings: tuple[str, ...],
    artifact_path: str | None,
    artifact_sha256: str | None,
    error_code: str | None = None,
    error_message: str | None = None,
    started_at: str | None = None,
) -> EvaluationReport:
    """构造 S2 可持久化的阶段报告。"""

    return EvaluationReport(
        run_id=evaluation_input.run_id,
        candidate_id=evaluation_input.candidate_id,
        stage=stage,
        candidate_content_hash=evaluation_input.candidate.content_hash,
        context_hash=evaluation_input.context.context_hash,
        input_hash=input_hash,
        status=status,
        metrics=metrics,
        warnings=warnings,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        error_code=error_code,
        error_message=error_message,
        started_at=started_at or utc_now_iso(),
        finished_at=utc_now_iso(),
    )


def evaluate_candidate(
    evaluation_input: CandidateEvaluationInput,
    *,
    policy: QualityPolicy | None = None,
) -> CandidateEvaluationResult:
    """评估单个候选；不修改正式注册表、正式数据或候选元数据库。"""

    if not isinstance(evaluation_input, CandidateEvaluationInput):
        raise MiningEvaluationError("evaluation_input 必须是 CandidateEvaluationInput")
    policy = policy or load_quality_policy()
    validate_frozen_inputs(evaluation_input)
    start = evaluation_input.quick_window[0]
    end = max(
        evaluation_input.quick_window[1],
        evaluation_input.full_window[1] if evaluation_input.run_full else start,
    )
    materialized = materialize_candidate(
        evaluation_input.candidate,
        evaluation_input.context,
        evaluation_input.candidate_root,
        start=start,
        end=end,
    )
    technical_input = stage_input_hash(
        stage="technical",
        evaluation_input=evaluation_input,
        factor_sha256=materialized.artifact_sha256,
        params=evaluation_input.quick_params_material,
        policy=policy,
    )
    technical_started = utc_now_iso()
    technical_timer = time.perf_counter()
    try:
        passed, checks, technical_warnings = evaluate_technical_checks(
            materialized.computation,
            parquet_glob=materialized.factor_view.parquet_glob,
            target_name=materialized.factor_view.name,
            peer_factor_dir=FACTOR_DIR,
            start=evaluation_input.quick_window[0],
            end=evaluation_input.quick_window[1],
            min_coverage=evaluation_input.context.budget.min_coverage,
            policy=policy,
            correlation_cache=evaluation_input.correlation_cache,
        )
        checks["elapsed_ms"] = round(
            (time.perf_counter() - technical_timer) * 1000,
            1,
        )
    except Exception as exc:
        technical = build_evaluation_report(
            evaluation_input=evaluation_input,
            stage="technical",
            status="error",
            input_hash=technical_input,
            metrics={},
            warnings=(),
            artifact_path=None,
            artifact_sha256=None,
            error_code="MINING-106",
            error_message=str(exc)[:2000],
            started_at=technical_started,
        )
        gate = TechnicalGate(
            run_id=evaluation_input.run_id,
            candidate_id=evaluation_input.candidate_id,
            candidate_content_hash=evaluation_input.candidate.content_hash,
            context_hash=evaluation_input.context.context_hash,
            rule_version=policy.schema_version,
            passed=False,
            checks={"error": str(exc)[:2000]},
        )
        return CandidateEvaluationResult(
            gate=gate,
            technical=technical,
            quick=None,
            full=None,
            warnings=(f"技术门禁执行失败：{exc}",),
            factor_glob=materialized.factor_view.parquet_glob,
            factor_sha256=materialized.artifact_sha256,
        )
    technical = build_evaluation_report(
        evaluation_input=evaluation_input,
        stage="technical",
        status="passed" if passed else "failed",
        input_hash=technical_input,
        metrics=checks,
        warnings=technical_warnings,
        artifact_path=None,
        artifact_sha256=None,
        started_at=technical_started,
    )
    gate = TechnicalGate(
        run_id=evaluation_input.run_id,
        candidate_id=evaluation_input.candidate_id,
        candidate_content_hash=evaluation_input.candidate.content_hash,
        context_hash=evaluation_input.context.context_hash,
        rule_version=policy.schema_version,
        passed=passed,
        checks=checks,
    )
    if not passed:
        return CandidateEvaluationResult(
            gate=gate,
            technical=technical,
            quick=None,
            full=None,
            warnings=technical_warnings,
            factor_glob=materialized.factor_view.parquet_glob,
            factor_sha256=materialized.artifact_sha256,
        )

    quick_input = stage_input_hash(
        stage="quick",
        evaluation_input=evaluation_input,
        factor_sha256=materialized.artifact_sha256,
        params=evaluation_input.quick_params_material,
        policy=policy,
    )
    quick_started = utc_now_iso()
    quick_metrics, quick_warnings, quick_passed = _quick_metrics(
        evaluation_input, materialized.factor_view, policy
    )
    quick_metrics["rows"] = technical.metrics["values"]["rows"]
    quick_metrics["coverage"] = technical.metrics["values"]["coverage"]
    quick_payload = {"metrics": quick_metrics, "stage": "quick"}
    artifact_path, artifact_sha = write_evaluation_artifact(
        Path(evaluation_input.candidate_root) / "evaluations" / "quick.json",
        quick_payload,
    )
    quick = build_evaluation_report(
        evaluation_input=evaluation_input,
        stage="quick",
        status="passed" if quick_passed else "failed",
        input_hash=quick_input,
        metrics=quick_metrics,
        warnings=quick_warnings,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha,
        started_at=quick_started,
    )
    warnings = (*technical_warnings, *quick_warnings)
    full = None
    if evaluation_input.run_full and quick_passed:
        full_input = stage_input_hash(
            stage="full",
            evaluation_input=evaluation_input,
            factor_sha256=materialized.artifact_sha256,
            params=evaluation_input.full_params_material,
            policy=policy,
        )
        full_started = utc_now_iso()
        try:
            full_result = full_metrics(evaluation_input, materialized.factor_view)
            artifact_path, artifact_sha = write_evaluation_artifact(
                Path(evaluation_input.candidate_root) / "evaluations" / "full.json",
                json_safe(full_result),
            )
            full = build_evaluation_report(
                evaluation_input=evaluation_input,
                stage="full",
                status="passed",
                input_hash=full_input,
                metrics=full_result,
                warnings=(),
                artifact_path=artifact_path,
                artifact_sha256=artifact_sha,
                started_at=full_started,
            )
        except Exception as exc:
            full = build_evaluation_report(
                evaluation_input=evaluation_input,
                stage="full",
                status="error",
                input_hash=full_input,
                metrics={},
                warnings=(),
                artifact_path=None,
                artifact_sha256=None,
                error_code="MINING-106",
                error_message=str(exc)[:2000],
                started_at=full_started,
            )
            warnings = (*warnings, f"完整评估失败：{exc}")
            _LOGGER.warning(
                format_event(
                    {
                        "operation": "full_evaluation_failed",
                        "run_id": evaluation_input.run_id,
                        "candidate_id": evaluation_input.candidate_id,
                        "error_code": "MINING-106",
                    }
                )
            )
    return CandidateEvaluationResult(
        gate=gate,
        technical=technical,
        quick=quick,
        full=full,
        warnings=warnings,
        factor_glob=materialized.factor_view.parquet_glob,
        factor_sha256=materialized.artifact_sha256,
    )
