"""在独立进程中测量一个挖掘相关性计算后端。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import resource
from tempfile import TemporaryDirectory
import time
from typing import Any

import numpy as np
import pandas as pd

from calculate import EvalParams, resolve_factor
from calculate.config import DB_FILE, FACTOR_DIR
from calculate.history import load_daily_ic
from calculate.panel import open_source
from calculate.universe import TRADABLE_FILTER, membership_cte, resolve_universe
from kernels import get_backend, native_threads
from mining import CandidateEvaluationInput, FrozenContext
from mining.candidate_batch import CandidateBatchItem, materialize_candidate_batch
from mining.candidate_evaluator import evaluate_candidate
from mining.catalog import load_catalog
from mining.contract_requests import MiningConstraints
from mining.evaluation_correlation import CandidateCorrelationCache
from mining.evaluation_contract import load_quality_policy
from mining.evaluation_gates import _target_correlation
from mining.expression_validation import validate_expression
from tools.monitor.baseline_contract import (
    atomic_write_json,
    strip_volatile,
)
from tools.monitor.mining_native_common import (
    END,
    EXPRESSIONS,
    QUICK_END,
    QUICK_START,
    REPEATS,
    SCHEMA_VERSION,
    START,
)


def _peak_rss_mb() -> float:
    """返回当前进程峰值 RSS。"""

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3)


def _context(input_hash: str) -> FrozenContext:
    """构造固定四候选使用的冻结上下文。"""

    catalog = load_catalog()
    budget = MiningConstraints(
        max_rounds=1,
        max_candidates=len(EXPRESSIONS),
        full_eval_top_k=1,
        max_llm_calls=1,
        max_failure_rate=1.0,
        min_quality_checks_for_failure_stop=1,
    )
    return FrozenContext(
        data_fingerprint=input_hash,
        field_catalog=catalog.frozen_field_mappings(),
        operator_catalog=catalog.frozen_operator_mappings(),
        budget=budget,
    )


def _candidates():
    """构造固定四候选。"""

    return tuple(
        validate_expression(expression, direction=1).to_candidate(
            mechanism="挖掘相关性原生基准",
            description=f"固定候选 {index}",
            steps=("执行受控表达式",),
            discovery_round=1,
            discovery_index=index,
        )
        for index, expression in enumerate(EXPRESSIONS, start=1)
    )


def _stable_evaluations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """移除计时和临时路径，只保留可跨后端比较的评估内容。"""

    def report(value: dict[str, Any]) -> dict[str, Any]:
        stable = strip_volatile(value)
        stable.pop("artifact_sha256", None)
        return stable

    return [
        {
            "candidate": item["candidate"],
            "artifact_sha256": item["artifact_sha256"],
            "gate": report(item["gate"]),
            "technical": report(item["technical"]),
            "quick": report(item["quick"]),
        }
        for item in records
    ]


def _rankic_query(params: EvalParams) -> str:
    """返回只取目标与标签原始值的轻量 RankIC 查询。"""

    spec = resolve_factor(params.factor, FACTOR_DIR)
    universe = resolve_universe(params.universe)
    membership = (
        membership_cte(universe, params.start, params.end) + ", "
        if universe.is_index
        else ""
    )
    member_join = (
        "JOIN member m ON m.stock_code=f.stock_code "
        "AND m.trade_date=f.trade_date"
        if universe.is_index
        else ""
    )
    path = spec.parquet_glob.replace("'", "''")
    return f"""
        WITH {membership} joined AS (
            SELECT f.trade_date,
                   CAST(f.{spec.name} AS DOUBLE) AS factor,
                   CAST(y.{params.label_column} AS DOUBLE) AS label
            FROM read_parquet('{path}', hive_partitioning=1) f
            JOIN src.md.fwd_return y
              ON y.stock_code=f.stock_code AND y.trade_date=f.trade_date
            JOIN src.md.daily_hfq b
              ON b.stock_code=f.stock_code AND b.trade_date=f.trade_date
            {member_join}
            WHERE b.trade_date BETWEEN DATE '{params.start}' AND DATE '{params.end}'
              AND f.trade_date BETWEEN DATE '{params.start}' AND DATE '{params.end}'
              AND {TRADABLE_FILTER}
              AND isfinite(f.{spec.name})
              AND isfinite(y.{params.label_column})
        )
        SELECT trade_date, factor, label FROM joined ORDER BY trade_date
    """


def _rank_ic_native(params: EvalParams) -> tuple[pd.DataFrame, float]:
    """用 C++ 逐日排名和 Pearson 核计算轻量 RankIC。"""

    sql = _rankic_query(params)
    started = time.perf_counter()
    conn = open_source(DB_FILE)
    try:
        frame = conn.execute(sql).df()
    finally:
        conn.close()
    codes = pd.factorize(frame["trade_date"], sort=True)[0]
    offsets = np.r_[
        0,
        np.flatnonzero(np.diff(codes) != 0) + 1,
        len(codes),
    ].astype(np.int64)
    backend = get_backend("native").backend
    left = backend.daily_rank_pct(
        np.ascontiguousarray(frame["factor"].to_numpy(dtype=np.float64)),
        offsets,
    )
    right = backend.daily_rank_pct(
        np.ascontiguousarray(frame["label"].to_numpy(dtype=np.float64)),
        offsets,
    )
    correlation, count = backend.daily_pearson(left, right, offsets, 5)
    index = pd.DatetimeIndex(
        sorted(pd.unique(frame["trade_date"])),
        name="trade_date",
    )
    result = pd.DataFrame({"rank_ic": correlation, "n": count}, index=index)
    result.loc[result["n"] < 5, "rank_ic"] = np.nan
    return result, time.perf_counter() - started


def _rankic_comparison() -> dict[str, Any]:
    """比较 DuckDB SQL 与 C++ 轻量 RankIC。"""

    params = EvalParams(
        "MOM_20",
        START,
        END,
        horizon=5,
        price="open",
        quantiles=10,
        winsorize=True,
    )
    sql_runs = []
    sql_result = None
    for _ in range(REPEATS):
        started = time.perf_counter()
        sql_result = load_daily_ic(params)
        sql_runs.append(time.perf_counter() - started)
    native_runs = []
    native_result = None
    for _ in range(REPEATS):
        native_result, elapsed = _rank_ic_native(params)
        native_runs.append(elapsed)
    assert sql_result is not None and native_result is not None
    maximum_difference = float(
        np.nanmax(
            np.abs(
                sql_result["rank_ic"].to_numpy(dtype=np.float64)
                - native_result["rank_ic"].to_numpy(dtype=np.float64)
            )
        )
    )
    counts_equal = bool(
        np.array_equal(
            sql_result["n"].to_numpy(dtype=np.int64),
            native_result["n"].to_numpy(dtype=np.int64),
        )
    )
    sql_median = float(np.median(sql_runs))
    native_median = float(np.median(native_runs))
    return {
        "sql_s": sql_runs,
        "native_s": native_runs,
        "sql_median_s": sql_median,
        "native_median_s": native_median,
        "native_speedup": sql_median / native_median,
        "max_rank_ic_difference": maximum_difference,
        "counts_equal": counts_equal,
        "retain_sql": native_median >= sql_median,
    }


def run_worker(
    backend: str,
    input_hash: str,
    output: Path,
    *,
    cache_enabled: bool = True,
) -> None:
    """在独立进程中执行一个后端的完整挖掘筛选与微基准。"""

    os.environ["QUANT_KERNEL_BACKEND"] = backend
    os.environ["QUANT_NATIVE_THREADS"] = "1"
    selection = get_backend()
    context = _context(input_hash)
    candidates = _candidates()
    started = time.perf_counter()
    with TemporaryDirectory(prefix=f"mining-native-{backend}-") as root:
        root_path = Path(root)
        batch = materialize_candidate_batch(
            [
                CandidateBatchItem(
                    candidate,
                    root_path / candidate.candidate_id,
                )
                for candidate in candidates
            ],
            context,
            start=QUICK_START,
            end=QUICK_END,
        )
        if batch.failures:
            raise RuntimeError(
                "固定候选批量物化失败："
                + ", ".join(
                    f"{failure.candidate_id}:{failure.error_code}"
                    for failure in batch.failures
                )
            )
        correlation_cache = (
            CandidateCorrelationCache(FACTOR_DIR)
            if cache_enabled
            else None
        )
        records = []
        for materialized in batch.materialized:
            candidate = next(
                item for item in candidates
                if item.name == materialized.factor_view.name
            )
            evaluation = evaluate_candidate(
                CandidateEvaluationInput(
                    run_id="p2-02-native-baseline",
                    candidate_id=candidate.candidate_id,
                    candidate=candidate,
                    context=context,
                    candidate_root=root_path / candidate.candidate_id,
                    quick_window=(QUICK_START, QUICK_END),
                    correlation_cache=correlation_cache,
                )
            )
            if evaluation.quick is None:
                raise RuntimeError(f"候选 {candidate.name} 缺少快速筛选报告")
            records.append(
                {
                    "candidate": sorted(candidate.dependencies),
                    "artifact_sha256": evaluation.factor_sha256,
                    "gate": asdict(evaluation.gate),
                    "technical": asdict(evaluation.technical),
                    "quick": asdict(evaluation.quick),
                    "technical_elapsed_ms": float(
                        evaluation.technical.metrics["elapsed_ms"]
                    ),
                    "quick_elapsed_ms": float(
                        evaluation.quick.metrics["elapsed_ms"]
                    ),
                }
            )
    total_elapsed_s = time.perf_counter() - started
    records.sort(key=lambda item: item["artifact_sha256"])

    policy = load_quality_policy()
    target = resolve_factor("LN_MV", FACTOR_DIR)
    conn = open_source(DB_FILE)
    try:
        cache = CandidateCorrelationCache(FACTOR_DIR)
        correlation_runs = []
        correlation_result = None
        for _ in range(REPEATS):
            run_started = time.perf_counter()
            correlation_result = _target_correlation(
                conn,
                target_name=target.name,
                target_glob=target.parquet_glob,
                peer_factor_dir=FACTOR_DIR,
                start=START,
                end=END,
                policy=policy,
                correlation_cache=cache,
            )
            correlation_runs.append(time.perf_counter() - run_started)
    finally:
        conn.close()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "requested_backend": backend,
        "actual_backend": selection.name,
        "cache_enabled": cache_enabled,
        "fallback_reason": selection.fallback_reason,
        "threads": native_threads() if selection.name == "native" else 1,
        "candidate_count": len(records),
        "total_elapsed_s": total_elapsed_s,
        "technical_elapsed_s": sum(
            item["technical_elapsed_ms"] for item in records
        )
        / 1000.0,
        "quick_elapsed_s": sum(
            item["quick_elapsed_ms"] for item in records
        )
        / 1000.0,
        "correlation_runs_s": correlation_runs,
        "correlation_median_s": float(np.median(correlation_runs)),
        "correlation_result": correlation_result,
        "stable_evaluations": _stable_evaluations(records),
        "rankic": _rankic_comparison() if backend == "native" else None,
        "peak_rss_mb": _peak_rss_mb(),
    }
    atomic_write_json(output, payload)


def main(argv: list[str] | None = None) -> int:
    """解析并执行 worker。"""

    parser = argparse.ArgumentParser(description="测量一个挖掘相关性后端")
    parser.add_argument("--backend", required=True, choices=("python", "native", "auto"))
    parser.add_argument("--input-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    run_worker(
        args.backend,
        args.input_hash,
        args.output,
        cache_enabled=not args.no_cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
