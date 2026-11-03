"""在独立进程中测量一次受控 DSL 候选流程。"""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import resource
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable

from calculate import open_source
from common.paths import FACTOR_DIR
from mining import CandidateEvaluationInput
from mining.candidate_batch import CandidateBatchItem, _publish_from_combined
from mining.candidate_evaluator import evaluate_candidate
from mining.candidate_materialization import _computation
from mining.catalog import load_catalog
from mining.contract_requests import FrozenContext, MiningConstraints
from mining.evaluation_correlation import CandidateCorrelationCache
from mining.expression_batch import compile_candidate_batch
from mining.expression_dag import build_candidate_expression_plan
from mining.expression_validation import (
    validate_candidate_expression,
    validate_expression,
)
from tools.monitor.baseline_contract import (
    atomic_write_json,
    load_golden,
    strip_volatile,
)
from tools.monitor.mining_dsl_analysis import (
    SCHEMA_VERSION,
    stable_digest,
    summarize_profile,
)
from tools.monitor.mining_native_common import (
    EXPRESSIONS,
    QUICK_END,
    QUICK_START,
)


def _peak_rss_mb() -> float:
    """返回当前进程峰值 RSS。"""

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3)


def _context(input_hash: str) -> FrozenContext:
    """构造固定四候选的冻结上下文。"""

    catalog = load_catalog()
    return FrozenContext(
        data_fingerprint=input_hash,
        field_catalog=catalog.frozen_field_mappings(),
        operator_catalog=catalog.frozen_operator_mappings(),
        budget=MiningConstraints(
            max_rounds=1,
            max_candidates=len(EXPRESSIONS),
            full_eval_top_k=1,
            max_llm_calls=1,
            max_failure_rate=1.0,
            min_quality_checks_for_failure_stop=1,
        ),
    )


def _candidates():
    """构造与 P2-02 相同的固定四候选。"""

    return tuple(
        validate_expression(expression, direction=1).to_candidate(
            mechanism="DSL 原生执行器决策",
            description=f"固定候选 {index}",
            steps=("执行受控表达式",),
            discovery_round=1,
            discovery_index=index,
        )
        for index, expression in enumerate(EXPRESSIONS, start=1)
    )


def _timed(function: Callable[[], Any]) -> tuple[Any, float]:
    """执行函数并返回结果与耗时。"""

    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


def _stable_evaluation(evaluation) -> dict[str, Any]:
    """返回跨轮可比较的候选评估摘要。"""

    if evaluation.quick is None:
        raise RuntimeError("固定 DSL 决策候选缺少快速筛选报告")

    def report(value: dict[str, Any]) -> dict[str, Any]:
        stable = strip_volatile(value)
        stable.pop("artifact_sha256", None)
        return stable

    return {
        "factor_sha256": evaluation.factor_sha256,
        "gate": report(asdict(evaluation.gate)),
        "technical": report(asdict(evaluation.technical)),
        "quick": report(asdict(evaluation.quick)),
    }


def _stable_value(value: Any) -> Any:
    """移除报告身份字段，并把浮点归一到证据比较精度。"""

    if isinstance(value, dict):
        return {
            key: _stable_value(item)
            for key, item in value.items()
            if key != "artifact_sha256"
        }
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


def _publish_materialized(
    items: list[CandidateBatchItem],
    validated,
    computations,
    context: FrozenContext,
    *,
    combined: Path,
    years: tuple[int, ...],
    connection,
):
    """拆分并发布全部候选，供基准单独计时。"""

    materialized = []
    for item in items:
        materialized.append(
            _publish_from_combined(
                item,
                validated[item.candidate.candidate_id],
                computations[item.candidate.candidate_id],
                context,
                combined=combined,
                years=years,
                start=QUICK_START,
                end=QUICK_END,
                connection=connection,
            )
        )
    return materialized


def run_worker(input_hash: str, output: Path) -> None:
    """在一个独立进程中执行一次完整阶段剖析。"""

    catalog = load_catalog()
    context = _context(input_hash)
    candidates = _candidates()
    with TemporaryDirectory(prefix="mining-dsl-decision-") as directory:
        root = Path(directory)
        items = [
            CandidateBatchItem(candidate, root / candidate.candidate_id)
            for candidate in candidates
        ]
        plan, plan_s = _timed(
            lambda: build_candidate_expression_plan(
                candidates,
                context,
                catalog=catalog,
            )
        )
        compiled, compile_s = _timed(
            lambda: compile_candidate_batch(
                plan,
                candidates,
                context,
                start=QUICK_START,
                end=QUICK_END,
                catalog=catalog,
            )
        )

        combined = root / "combined.parquet"
        profile_path = root / "duckdb-profile.json"
        connection = open_source()
        try:
            duckdb_threads = int(
                connection.execute("SELECT current_setting('threads')").fetchone()[0]
            )
            escaped_profile = str(profile_path).replace("'", "''")
            connection.execute("PRAGMA enable_profiling='json'")
            connection.execute(f"PRAGMA profiling_output='{escaped_profile}'")
            escaped_combined = str(combined).replace("'", "''")
            _, query_s = _timed(
                lambda: connection.execute(
                    f"COPY ({compiled.sql}) TO '{escaped_combined}' "
                    "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 200000)"
                )
            )
            profile = load_golden(profile_path)
            years = tuple(
                int(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT year(trade_date) FROM read_parquet(?) ORDER BY 1",
                    [str(combined)],
                ).fetchall()
            )
            validated = {
                candidate.candidate_id: validate_candidate_expression(
                    candidate,
                    context,
                    catalog=catalog,
                )
                for candidate in candidates
            }
            computations = {
                candidate.candidate_id: _computation(
                    candidate,
                    validated[candidate.candidate_id],
                )
                for candidate in candidates
            }
            materialized, publish_s = _timed(
                lambda: _publish_materialized(
                    items,
                    validated,
                    computations,
                    context,
                    combined=combined,
                    years=years,
                    connection=connection,
                )
            )
        finally:
            connection.close()

        correlation_cache = CandidateCorrelationCache(FACTOR_DIR)
        evaluations = []
        evaluation_started = time.perf_counter()
        for candidate in candidates:
            evaluation = evaluate_candidate(
                CandidateEvaluationInput(
                    run_id="p2-03-dsl-decision",
                    candidate_id=candidate.candidate_id,
                    candidate=candidate,
                    context=context,
                    candidate_root=root / candidate.candidate_id,
                    quick_window=(QUICK_START, QUICK_END),
                    correlation_cache=correlation_cache,
                )
            )
            evaluations.append(_stable_evaluation(evaluation))
        evaluation_s = time.perf_counter() - evaluation_started

        profile_summary = summarize_profile(profile)
        materialization_s = plan_s + compile_s + query_s + publish_s
        total_candidate_s = materialization_s + evaluation_s
        payload = {
            "schema_version": SCHEMA_VERSION,
            "input_tree_sha256": input_hash,
            "threads": {
                "duckdb": duckdb_threads,
                "native": int(os.environ.get("QUANT_NATIVE_THREADS", "1")),
            },
            "candidate_count": len(candidates),
            "years": [int(year) for year in years],
            "temporary_bytes": combined.stat().st_size,
            "stages_s": {
                "plan": plan_s,
                "compile": compile_s,
                "query_copy": query_s,
                "split_publish": publish_s,
                "materialization": materialization_s,
                "evaluation": evaluation_s,
                "total_candidate": total_candidate_s,
            },
            "profile": profile_summary,
            "stable_evaluations": sorted(
                evaluations,
                key=lambda item: item["factor_sha256"],
            ),
            "stable_payload_sha256": stable_digest(_stable_value(evaluations)),
            "peak_rss_mb": _peak_rss_mb(),
        }
    atomic_write_json(output, payload)
