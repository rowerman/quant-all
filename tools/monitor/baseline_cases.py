"""执行算子优化基线的五类真实案例和确定性边界案例。"""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from calculate import EvalParams, evaluate, load_daily_ic, resolve_factor
from calculate.correlation import daily_spearman_matrix
from calculate.daily import _daily_pearson, daily_ic, rank_pct_by_day
from common.paths import PROJECT_ROOT
from mining import (
    CandidateBatchItem,
    CandidateEvaluationInput,
    FrozenContext,
    MiningConstraints,
    MiningRunRequest,
    evaluate_candidate,
    load_catalog,
    materialize_candidate_batch,
    validate_expression,
)
from mining.candidate_materialization import materialize_candidate
from scripts.build_factors import build as factor_build
from tools.monitor.baseline_contract import (
    CASE_DEFINITIONS,
    canonical_json,
    canonical_value,
    file_record,
    strip_volatile,
)

_BACKEND_ROOT = PROJECT_ROOT / "apps" / "monitor"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from backend import service


def _event_case(definition: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """通过生产后端适配层执行完整评估或单 stage 评估。"""

    query = {
        "factor": definition["factor"],
        "start": definition["start"],
        "end": definition["end"],
        "universe": definition["universe"],
        "horizon": str(definition["horizon"]),
        "price": definition["price"],
        "quantiles": str(definition["quantiles"]),
        "winsorize": "true" if definition["winsorize"] else "false",
    }
    if definition.get("stages"):
        query["stages"] = ",".join(definition["stages"])
    if "top_n" in definition:
        query["metric_options"] = canonical_json(
            {"correlation": {"top_n": definition["top_n"]}}
        )

    prepare_started = time.perf_counter()
    spec = service.prepare(query)
    prepare_elapsed_ms = (time.perf_counter() - prepare_started) * 1000
    events: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    started = time.perf_counter()
    last = started
    for stage, payload in service.run(spec):
        received = time.perf_counter()
        generation_elapsed_ms = (received - last) * 1000
        serialization_started = time.perf_counter()
        event = {"stage": stage, "payload": payload}
        events.append(event)
        payload_bytes = len(canonical_json(payload).encode("utf-8"))
        serialization_ms = (time.perf_counter() - serialization_started) * 1000
        timings.append(
            {
                "stage": stage,
                "generation_elapsed_ms": round(generation_elapsed_ms, 3),
                "reported_elapsed_ms": payload.get("elapsed_ms"),
                "serialization_ms": round(serialization_ms, 3),
                "payload_bytes": payload_bytes,
            }
        )
        last = time.perf_counter()
    return (
        {
            "query": definition,
            "events": strip_volatile(events),
        },
        {
            "prepare_elapsed_ms": round(prepare_elapsed_ms, 3),
            "run_elapsed_s": round(time.perf_counter() - started, 3),
            "events": timings,
        },
    )


def _target_correlation_case(
    definition: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """执行非正式候选的目标行相关性路径。"""

    params = EvalParams(
        factor=definition["factor"],
        start=definition["start"],
        end=definition["end"],
        universe=definition["universe"],
        horizon=definition["horizon"],
        price=definition["price"],
        quantiles=definition["quantiles"],
        winsorize=definition["winsorize"],
    )
    factor_view = replace(
        resolve_factor(definition["factor"]),
        formal=definition["formal"],
    )
    events: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    started = time.perf_counter()
    last = started
    for stage, payload in evaluate(
        definition["factor"],
        params,
        factor_view=factor_view,
        stages=definition["stages"],
        metric_options={"correlation": {"top_n": definition["top_n"]}},
    ):
        received = time.perf_counter()
        serialization_started = time.perf_counter()
        events.append({"stage": stage, "payload": payload})
        payload_bytes = len(canonical_json(payload).encode("utf-8"))
        serialization_ms = (time.perf_counter() - serialization_started) * 1000
        timings.append(
            {
                "stage": stage,
                "generation_elapsed_ms": round((received - last) * 1000, 3),
                "reported_elapsed_ms": payload.get("elapsed_ms"),
                "serialization_ms": round(serialization_ms, 3),
                "payload_bytes": payload_bytes,
            }
        )
        last = time.perf_counter()
    return (
        {
            "query": definition,
            "events": strip_volatile(events),
        },
        {
            "run_elapsed_s": round(time.perf_counter() - started, 3),
            "events": timings,
        },
    )


def _history_case(definition: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """执行全历史轻量 IC 并保存逐日序列。"""

    params = EvalParams(
        factor=definition["factor"],
        start=definition["start"],
        end=definition["end"],
        universe=definition["universe"],
        horizon=definition["horizon"],
        price=definition["price"],
        quantiles=definition["quantiles"],
        winsorize=definition["winsorize"],
    )
    started = time.perf_counter()
    frame = load_daily_ic(params)
    elapsed_s = time.perf_counter() - started
    series = frame.reset_index().loc[:, ["trade_date", "ic", "rank_ic", "n"]]
    rank_ic = frame["rank_ic"].dropna()
    golden = {
        "params": params.describe(),
        "summary": {
            "days": int(len(frame)),
            "valid_ic_days": int(frame["ic"].notna().sum()),
            "valid_rank_ic_days": int(rank_ic.size),
            "paired_rows": int(frame["n"].sum()),
            "ic_mean": float(frame["ic"].mean()) if frame["ic"].notna().any() else None,
            "rank_ic_mean": float(rank_ic.mean()) if rank_ic.size else None,
        },
        "series": canonical_value(series),
    }
    return golden, {
        "elapsed_s": round(elapsed_s, 3),
        "load_elapsed_ms": float(frame.attrs["elapsed_ms"]),
    }


def _factor_build_case(
    definition: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """在临时目录构建因子，只保留稳定摘要和全部年度文件哈希。"""

    with TemporaryDirectory(prefix="operator-baseline-factor-") as directory:
        root = Path(directory)
        factor_dir = root / "factors"
        catalog_file = root / "factors.duckdb"
        started = time.perf_counter()
        result = factor_build.build(
            list(definition["families"]),
            definition["start"],
            definition["end"],
            factor_dir=factor_dir,
            catalog_file=catalog_file,
            write_catalog=True,
            build_mode=definition.get("build_mode", "full_history"),
        )
        elapsed_s = time.perf_counter() - started
        family = definition["families"][0]
        paths = sorted((factor_dir / family).glob("year=*/part.parquet"))
        if not paths:
            raise FileNotFoundError(f"因子构建没有生成任何年度文件：{factor_dir / family}")

        conn = duckdb.connect()
        outputs = []
        try:
            for path in paths:
                year = int(path.parent.name.split("=")[1])
                columns = [
                    row[0]
                    for row in conn.execute(
                        "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
                    ).fetchall()
                ]
                factor_columns = [
                    name for name in columns
                    if name not in {"stock_code", "trade_date", "year"}
                ]
                counts = conn.execute(
                    "SELECT count(*), count(DISTINCT stock_code), "
                    "count(DISTINCT trade_date) FROM read_parquet(?)",
                    [str(path)],
                ).fetchone()
                non_finite = {
                    name: int(
                        conn.execute(
                            f'SELECT count(*) FROM read_parquet(?) '
                            f'WHERE isinf("{name}") OR isnan("{name}")',
                            [str(path)],
                        ).fetchone()[0]
                    )
                    for name in factor_columns
                }
                output_file = file_record(path)
                output_file["path"] = f"{family}/year={year}/part.parquet"
                outputs.append({
                    "family": family,
                    "year": year,
                    "columns": columns,
                    "factor_columns": factor_columns,
                    "rows": int(counts[0]),
                    "stocks": int(counts[1]),
                    "trade_days": int(counts[2]),
                    "non_finite": non_finite,
                    "file": output_file,
                })
        finally:
            conn.close()

        golden = {
            "definition": definition,
            "build": strip_volatile(result),
            "output": outputs[0],
        }
        if len(outputs) > 1:
            golden["outputs"] = outputs
        return golden, {
            "elapsed_s": round(elapsed_s, 3),
            "rows_written": int(result["rows_written"]),
            "factor_count": int(result["factor_count"]),
        }


def _mining_case(
    definition: dict[str, Any],
    common: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """执行固定候选的技术门禁和快速筛选。"""

    validated = validate_expression(
        definition["expression"],
        direction=definition["direction"],
        max_depth=6,
    )
    candidate = validated.to_candidate(
        mechanism="短期价格趋势",
        description="收盘价五日均值",
        steps=("计算收盘价五日均值",),
        discovery_round=1,
        discovery_index=1,
    )
    constraints = MiningConstraints(
        max_rounds=1,
        max_candidates=1,
        full_eval_top_k=1,
        max_llm_calls=1,
        max_failure_rate=1.0,
        min_quality_checks_for_failure_stop=1,
    )
    request = MiningRunRequest(
        "p0-01-baseline-20261027",
        "短期价格趋势可能带来未来五日超额收益",
        constraints=constraints,
    )
    context = FrozenContext(
        data_fingerprint=common["inputs"]["input_tree_sha256"],
        research_start=definition["research_start"],
        research_end=definition["research_end"],
        full_eval_start=definition["full_eval_start"],
        full_eval_end=definition["full_eval_end"],
        holdout_start=definition["holdout_start"],
        holdout_end=definition["holdout_end"],
        budget=constraints,
    )
    started = time.perf_counter()
    with TemporaryDirectory(prefix="operator-baseline-mining-") as directory:
        evaluation = evaluate_candidate(
            CandidateEvaluationInput(
                run_id=request.run_id,
                candidate_id=candidate.candidate_id,
                candidate=candidate,
                context=context,
                candidate_root=Path(directory) / "candidate",
                run_full=definition["run_full"],
                quick_window=(definition["quick_start"], definition["quick_end"]),
            )
        )
    elapsed_s = time.perf_counter() - started
    if not evaluation.gate.passed:
        raise RuntimeError("固定挖掘基准候选未通过技术门禁")
    if evaluation.quick is None or evaluation.quick.status != "passed":
        raise RuntimeError("固定挖掘基准候选未完成快速筛选")

    technical = asdict(evaluation.technical)
    quick = asdict(evaluation.quick)
    gate = asdict(evaluation.gate)
    golden = {
        "definition": definition,
        "candidate": {
            "name": candidate.name,
            "canonical_expression": candidate.canonical_expression,
            "content_hash": candidate.content_hash,
            "semantic_hash": candidate.semantic_hash,
        },
        "context": {
            "data_version": context.data_version,
            "data_fingerprint": context.data_fingerprint,
            "context_hash": context.context_hash,
            "quick_window": [
                definition["quick_start"],
                definition["quick_end"],
            ],
        },
        "gate": strip_volatile(gate),
        "technical": strip_volatile(technical),
        "quick": strip_volatile(quick),
        "factor_sha256": evaluation.factor_sha256,
    }
    return golden, {
        "elapsed_s": round(elapsed_s, 3),
        "technical_elapsed_ms": evaluation.technical.metrics["elapsed_ms"],
        "quick_elapsed_ms": evaluation.quick.metrics["elapsed_ms"],
    }


def _edge_case(definition: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """生成确定性边界金标准，覆盖后续原生核最易漂移的语义。"""

    started = time.perf_counter()
    rank_codes = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.int64)
    rank_values = np.array([3.0, 1.0, 1.0, np.nan, 2.0, 4.0, 7.0])
    ranks = rank_pct_by_day(rank_codes, rank_values)

    pearson_codes = np.array([0] * 6 + [1] + [2] * 5, dtype=np.int64)
    pearson_left = np.array([1., 2., 3., 4., 5., 6., 10., 5., 5., 5., 5., 5.])
    pearson_right = np.array([2., 4., 6., 8., 10., 12., 10., 1., 2., 3., 4., 5.])
    pearson, count = _daily_pearson(
        pearson_codes,
        pearson_left,
        pearson_right,
        int(definition["min_cross_section"]),
    )

    matrix_values = np.array(
        [
            [1., 2., np.nan, np.inf, 7.],
            [2., 1., np.nan, 1., 7.],
            [2., 1., 1., 2., 7.],
            [4., 3., 2., 3., 7.],
            [5., 6., 3., 4., 7.],
            [6., 5., 4., 5., 7.],
        ]
    )
    matrix = daily_spearman_matrix(matrix_values)

    empty_panel = pd.DataFrame(
        {
            "trade_date": pd.to_datetime([]),
            "factor": pd.Series(dtype="float64"),
            "label": pd.Series(dtype="float64"),
            "in_universe": pd.Series(dtype="bool"),
        }
    )
    empty_ic = daily_ic(empty_panel)

    checks = {
        "rank_ticks_are_average": bool(
            math.isclose(ranks[1], ranks[2])
            and ranks[1] < ranks[4] < ranks[0] < ranks[5]
        ),
        "rank_nan_preserved": bool(np.isnan(ranks[3])),
        "rank_single_sample_preserved": bool(ranks[6] == 1.0),
        "pearson_perfect_correlation": bool(math.isclose(pearson[0], 1.0)),
        "pearson_single_sample_is_nan": bool(np.isnan(pearson[1])),
        "pearson_constant_is_nan": bool(np.isnan(pearson[2])),
        "constant_rank_column_has_no_correlation": bool(np.isnan(matrix[:, 4]).all()),
        "insufficient_missing_column_is_nan": bool(np.isnan(matrix[:, 2]).all()),
        "empty_ic_has_no_rows": bool(empty_ic.empty),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"边界语义不符合预期：{failed}")
    return (
        {
            "definition": definition,
            "rank": {
                "codes": rank_codes,
                "values": rank_values,
                "result": ranks,
            },
            "pearson": {
                "codes": pearson_codes,
                "left": pearson_left,
                "right": pearson_right,
                "correlation": pearson,
                "count": count,
            },
            "spearman": {
                "values": matrix_values,
                "matrix": matrix,
            },
            "empty_ic": {
                "rows": len(empty_ic),
                "columns": list(empty_ic.columns),
            },
            "checks": checks,
        },
        {
            "elapsed_s": round(time.perf_counter() - started, 6),
        },
    )


def _mining_batch_case(
    definition: dict[str, Any],
    common: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """对照同轮候选的串行物化与 DAG 批量物化。"""

    catalog = load_catalog()
    constraints = MiningConstraints(
        max_rounds=1,
        max_candidates=4,
        full_eval_top_k=1,
        max_llm_calls=1,
        max_failure_rate=1.0,
        min_quality_checks_for_failure_stop=1,
    )
    context = FrozenContext(
        data_fingerprint=common["inputs"]["input_tree_sha256"],
        field_catalog=catalog.frozen_field_mappings(),
        operator_catalog=catalog.frozen_operator_mappings(),
        budget=constraints,
    )
    candidates = tuple(
        validate_expression(expression, direction=1).to_candidate(
            mechanism="批量物化基准",
            description=f"候选 {index}",
            steps=("执行受控表达式",),
            discovery_round=1,
            discovery_index=index,
        )
        for index, expression in enumerate(definition["expressions"], start=1)
    )
    started = time.perf_counter()
    with TemporaryDirectory(prefix=f"operator-baseline-{definition['mode']}-") as root:
        root_path = Path(root)
        if definition["mode"] == "serial":
            results = [
                materialize_candidate(
                    candidate,
                    context,
                    root_path / candidate.candidate_id,
                    start=definition["start"],
                    end=definition["end"],
                )
                for candidate in candidates
            ]
            diagnostics = {
                "candidate_count": len(candidates),
                "shared_node_count": 0,
                "materialized_node_count": 0,
                "analysis_sql_count": len(candidates),
                "daily_source_reference_count": len(candidates) * 2,
                "temporary_bytes": 0,
                "fallback_reason": "serial",
            }
        else:
            result = materialize_candidate_batch(
                [
                    CandidateBatchItem(
                        candidate,
                        root_path / candidate.candidate_id,
                    )
                    for candidate in candidates
                ],
                context,
                start=definition["start"],
                end=definition["end"],
                catalog=catalog,
            )
            if result.failures:
                raise RuntimeError(
                    "批量基准出现候选失败："
                    + ", ".join(
                        f"{failure.candidate_id}:{failure.error_code}"
                        for failure in result.failures
                    )
                )
            results = list(result.materialized)
            diagnostics = {
                "candidate_count": result.diagnostics.candidate_count,
                "shared_node_count": result.diagnostics.shared_node_count,
                "materialized_node_count": result.diagnostics.materialized_node_count,
                "analysis_sql_count": result.diagnostics.analysis_sql_count,
                "daily_source_reference_count": (
                    result.diagnostics.daily_source_reference_count
                ),
                "temporary_bytes": result.diagnostics.temporary_bytes,
                "fallback_reason": result.diagnostics.fallback_reason,
            }
        elapsed_s = time.perf_counter() - started
        artifacts = []
        for materialized in sorted(
            results,
            key=lambda item: item.factor_view.name,
        ):
            factor_root = Path(materialized.factor_view.parquet_glob).parent.parent
            artifacts.append(
                {
                    "name": materialized.factor_view.name,
                    "artifact_sha256": materialized.artifact_sha256,
                    "files": [
                        {
                            "path": path.relative_to(factor_root).as_posix(),
                            "bytes": path.stat().st_size,
                        }
                        for path in materialized.files
                    ],
                }
            )
    return (
        {
            "definition": definition,
            "candidates": artifacts,
            "diagnostics": diagnostics,
        },
        {
            "materialize_elapsed_s": round(elapsed_s, 3),
            **diagnostics,
        },
    )


def run_case(
    name: str,
    common: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """执行一个固定案例；参数 name 为案例名，common 为共享输入快照。"""

    if name == "evaluation":
        return _event_case(CASE_DEFINITIONS[name])
    if name == "correlation":
        return _event_case(CASE_DEFINITIONS[name])
    if name == "correlation_target":
        return _target_correlation_case(CASE_DEFINITIONS[name])
    if name == "history":
        return _history_case(CASE_DEFINITIONS[name])
    if name in {
        "factor_build",
        "factor_build_yearly",
        "factor_build_full_history",
        "factor_build_all_history",
    }:
        return _factor_build_case(CASE_DEFINITIONS[name])
    if name == "mining":
        return _mining_case(CASE_DEFINITIONS[name], common)
    if name in {"mining_batch_serial", "mining_batch_dag"}:
        return _mining_batch_case(CASE_DEFINITIONS[name], common)
    if name == "edge":
        return _edge_case(CASE_DEFINITIONS[name])
    raise ValueError(f"未知基准案例：{name}")
