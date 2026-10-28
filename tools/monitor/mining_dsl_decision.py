"""测量受控 DSL 剩余瓶颈并决定是否需要原生执行器。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable

from calculate import open_source
from common.paths import FACTOR_DIR, PROJECT_ROOT
from mining import CandidateEvaluationInput
from mining.candidate_batch import (
    CandidateBatchItem,
    _publish_from_combined,
)
from mining.candidate_materialization import _computation
from mining.candidate_evaluator import evaluate_candidate
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
    atomic_write_text,
    environment_snapshot,
    file_record,
    input_snapshot,
    load_golden,
    project_relative,
    strip_volatile,
    update_manifest,
)
from tools.monitor.mining_dsl_analysis import (
    FIXTURE_EXPRESSIONS,
    OPERATOR_COVERAGE_EXPRESSIONS,
    ROUTE_COSTS,
    decision_from_measurements,
    stable_digest,
    summarize_corpus,
    summarize_profile,
)
from tools.monitor.mining_native_common import (
    EXPRESSIONS,
    QUICK_END,
    QUICK_START,
)

SCHEMA_VERSION = "mining_dsl_decision_v1"
REPEATS = 3
EVIDENCE_ROOT = PROJECT_ROOT / "AiRecord/benchmarks/V122/p2-03-20261028"


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


def run_worker(input_hash: str, output: Path) -> None:
    """在一个独立进程中执行一次完整阶段剖析。"""

    catalog = load_catalog()
    context = _context(input_hash)
    candidates = _candidates()
    with TemporaryDirectory(prefix="mining-dsl-decision-") as directory:
        root = Path(directory)
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
                item.candidate.candidate_id: validate_candidate_expression(
                    item.candidate,
                    context,
                    catalog=catalog,
                )
                for item in (
                    CandidateBatchItem(candidate, root / candidate.candidate_id)
                    for candidate in candidates
                )
            }
            computations = {
                candidate.candidate_id: _computation(
                    candidate,
                    validated[candidate.candidate_id],
                )
                for candidate in candidates
            }
            items = [
                CandidateBatchItem(candidate, root / candidate.candidate_id)
                for candidate in candidates
            ]

            def publish_all():
                published = []
                for item in items:
                    published.append(
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
                return published

            materialized, publish_s = _timed(publish_all)
        finally:
            connection.close()

        correlation_cache = CandidateCorrelationCache(FACTOR_DIR)
        evaluations = []
        evaluation_started = time.perf_counter()
        for candidate, item in zip(candidates, materialized, strict=True):
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
            "stable_payload_sha256": stable_digest(
                _stable_value(evaluations)
            ),
            "peak_rss_mb": _peak_rss_mb(),
        }
    atomic_write_json(output, payload)


def _median(values: list[float]) -> float:
    """返回浮点中位数。"""

    return float(statistics.median(values))


def summarize_runs(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总三轮阶段耗时、收益上限和稳定结果。"""

    if not runs:
        raise ValueError("至少需要一轮测量")
    stage_names = tuple(runs[0]["stages_s"])
    stage_s = {
        name: _median([float(run["stages_s"][name]) for run in runs])
        for name in stage_names
    }
    profile_categories = tuple(runs[0]["profile"]["category_cpu_s"])
    category_cpu_s = {
        name: _median(
            [float(run["profile"]["category_cpu_s"][name]) for run in runs]
        )
        for name in profile_categories
    }
    category_wall_s = {
        name: _median(
            [float(run["profile"]["category_wall_equivalent_s"][name]) for run in runs]
        )
        for name in profile_categories
    }
    query_wall_s = _median(
        [float(run["profile"]["query_wall_s"]) for run in runs]
    )
    query_cpu_s = _median(
        [float(run["profile"]["query_cpu_s"]) for run in runs]
    )
    expression_s = category_wall_s["expression"]
    materialization_s = stage_s["materialization"]
    total_s = stage_s["total_candidate"]
    expression_pipeline_share = expression_s / total_s
    expression_materialization_share = expression_s / materialization_s
    decision = decision_from_measurements(
        expression_pipeline_share,
        expression_pipeline_share,
    )
    stable_hashes = {str(run["stable_payload_sha256"]) for run in runs}
    if len(stable_hashes) != 1:
        raise RuntimeError("三轮候选稳定 payload 哈希不一致")
    decision["stable_payload_sha256"] = stable_hashes.pop()
    return {
        "schema_version": SCHEMA_VERSION,
        "repeat_count": len(runs),
        "stage_s": {
            name: round(value, 6)
            for name, value in stage_s.items()
        },
        "profile": {
            "query_wall_s": round(query_wall_s, 6),
            "query_cpu_s": round(query_cpu_s, 6),
            "category_cpu_s": {
                name: round(value, 6)
                for name, value in category_cpu_s.items()
            },
            "category_wall_equivalent_s": {
                name: round(value, 6)
                for name, value in category_wall_s.items()
            },
            "expression_share_of_query_cpu": (
                category_cpu_s["expression"] / query_cpu_s
                if query_cpu_s
                else 0.0
            ),
            "expression_share_of_query_wall_equivalent": (
                expression_s / query_wall_s if query_wall_s else 0.0
            ),
            "expression_share_of_materialization": (
                expression_materialization_share
            ),
            "expression_share_of_pipeline": expression_pipeline_share,
            "zero_cost_improvement_upper_bound": expression_pipeline_share,
        },
        "peak_rss_mb": round(max(float(run["peak_rss_mb"]) for run in runs), 3),
        "threads": runs[0]["threads"],
        "decision": decision,
        "routes": ROUTE_COSTS,
    }


def _write_report(
    output: Path,
    corpus: dict[str, Any],
    summary: dict[str, Any],
    input_hash: str,
) -> None:
    """写出人工可读的决策报告。"""

    decision = summary["decision"]
    stage = summary["stage_s"]
    profile = summary["profile"]
    operator_counts = "、".join(
        f"{name}={count}"
        for name, count in corpus["operator_counts"].items()
    )
    lines = [
        "# P2-03 受控 DSL 原生执行器决策",
        "",
        "## 口径",
        "",
        f"- 区间：`{QUICK_START}`～`{QUICK_END}`；固定四候选、单进程、"
        f"{summary['repeat_count']} 轮中位数。",
        f"- 输入树 SHA-256：`{input_hash}`。",
        f"- DuckDB 线程：{summary['threads']['duckdb']}；native 记录线程："
        f"{summary['threads']['native']}。",
        f"- 覆盖语料：{corpus['candidate_count']} 个去重表达式，覆盖 "
        f"{corpus['coverage']['covered_operator_count']}/"
        f"{corpus['coverage']['catalog_operator_count']} 个冻结算子。",
        "- 当前无正式候选库，语料频率只代表仓库覆盖样本，不代表线上分布。",
        "",
        "## 阶段",
        "",
        "| 阶段 | 中位耗时(s) |",
        "|---|---:|",
        f"| DAG 计划 | {stage['plan']:.6f} |",
        f"| 批量 SQL 编译 | {stage['compile']:.6f} |",
        f"| DuckDB COPY | {stage['query_copy']:.6f} |",
        f"| 候选拆分发布 | {stage['split_publish']:.6f} |",
        f"| 候选物化合计 | {stage['materialization']:.6f} |",
        f"| 技术门禁与快速筛选 | {stage['evaluation']:.6f} |",
        f"| 完整候选流程 | {stage['total_candidate']:.6f} |",
        "",
        "## Profile",
        "",
        "算子时间为 DuckDB 多线程累计 CPU 时间；墙钟等效值按查询 CPU/墙钟比例折算。",
        "",
        "| 分类 | 中位 CPU(s) | 墙钟等效(s) |",
        "|---|---:|---:|",
    ]
    for name, value in profile["category_cpu_s"].items():
        lines.append(
            f"| {name} | {value:.6f} | "
            f"{profile['category_wall_equivalent_s'][name]:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- 查询墙钟：{profile['query_wall_s']:.6f}s；累计 CPU："
            f"{profile['query_cpu_s']:.6f}s。",
            f"- 表达式占批量 SQL CPU："
            f"{profile['expression_share_of_query_cpu']:.2%}；墙钟等效占比："
            f"{profile['expression_share_of_query_wall_equivalent']:.2%}。",
            f"- 表达式占候选物化：{profile['expression_share_of_materialization']:.2%}。",
            f"- 表达式占完整候选流程：{profile['expression_share_of_pipeline']:.2%}。",
            "- 若表达式耗时降为零，完整流程改善上限："
            f"{profile['zero_cost_improvement_upper_bound']:.2%}。",
            "",
            "## 算子复杂度",
            "",
            f"- 深度：均值 {corpus['depth']['mean']:.3f}，中位 "
            f"{corpus['depth']['median']:.3f}，P95 {corpus['depth']['p95']:.3f}，"
            f"最大 {corpus['depth']['max']}。",
            f"- 节点数：均值 {corpus['node_count']['mean']:.3f}，中位 "
            f"{corpus['node_count']['median']:.3f}，P95 {corpus['node_count']['p95']:.3f}，"
            f"最大 {corpus['node_count']['max']}。",
            f"- 算子出现次数：{operator_counts}。",
            "",
            "## 路线成本",
            "",
            "| 路线 | 版本锁定 | 构建 | 调试 | 崩溃隔离 | 测试 | 相对成本 | 结论 |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name, route in summary["routes"].items():
        lines.append(
            f"| {name} | {route['version_lock']} | {route['build']} | "
            f"{route['debug']} | {route['crash_isolation']} | {route['test']} | "
            f"{route['relative_cost']} | {route['summary']} |"
        )
    lines.extend(
        [
            "",
            "成本分值：0=无新增，1=中，2=高。",
            "",
            "## 结论",
            "",
            f"- 决策：`{decision['decision']}`。",
            f"- 原因：{', '.join(decision['reasons']) or '达到后续设计门槛'}。",
            "- 本轮未编写原生表达式 VM，未修改生产路由。",
            "- 实际 C++ 替换模块新增：0；C++ 替换台账不变。",
            "",
            "## 复现",
            "",
            "```bash",
            ".venv/bin/python -m tools.monitor.mining_dsl_decision "
            "--output AiRecord/benchmarks/V122/p2-03-20261028",
            "```",
        ]
    )
    atomic_write_text(output / "report.md", "\n".join(lines) + "\n")


def _worker_command(input_hash: str, output: Path) -> list[str]:
    """构造一次独立 worker 命令。"""

    return [
        sys.executable,
        "-m",
        "tools.monitor.mining_dsl_decision",
        "--worker",
        "--input-hash",
        input_hash,
        "--worker-output",
        str(output),
    ]


def run_suite(output: Path, *, repeats: int = REPEATS) -> int:
    """运行覆盖统计和三阶段重复测量，并写出证据。"""

    if repeats < 1:
        raise ValueError("repeats 必须为正整数")
    output.mkdir(parents=True, exist_ok=True)
    environment = environment_snapshot()
    inputs = input_snapshot()
    atomic_write_json(output / "environment.json", environment)
    atomic_write_json(output / "inputs.json", inputs)
    input_hash = str(inputs["input_tree_sha256"])

    catalog = load_catalog()
    corpus = summarize_corpus(
        {
            "benchmark": EXPRESSIONS,
            "operator_coverage": OPERATOR_COVERAGE_EXPRESSIONS,
            "fixture": FIXTURE_EXPRESSIONS,
        },
        catalog,
    )
    atomic_write_json(output / "corpus.json", corpus)

    worker_dir = output / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for index in range(1, repeats + 1):
        print(f"[mining-dsl-decision] 开始第 {index}/{repeats} 轮", flush=True)
        path = worker_dir / f"run-{index}.json"
        completed = subprocess.run(
            _worker_command(input_hash, path),
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"第 {index} 轮失败：{completed.stderr or completed.stdout}"
            )
        runs.append(load_golden(path))

    summary = summarize_runs(runs)
    result = {
        "schema_version": SCHEMA_VERSION,
        "input_tree_sha256": input_hash,
        "source_files": [
            file_record(PROJECT_ROOT / path)
            for path in (
                "tools/monitor/mining_dsl_analysis.py",
                "tools/monitor/mining_dsl_decision.py",
                "research/mining/expression_ast.py",
                "research/mining/expression_batch.py",
                "research/mining/expression_compiler.py",
                "research/mining/expression_dag.py",
                "research/mining/expression_parser.py",
                "research/mining/expression_validation.py",
            )
        ],
        "corpus": corpus,
        "runs": runs,
        "summary": summary,
    }
    atomic_write_json(output / "result.json", result)
    _write_report(output, corpus, summary, input_hash)
    update_manifest(output)
    print(
        json.dumps(
            {
                "output": project_relative(output),
                "decision": summary["decision"]["decision"],
                "reasons": summary["decision"]["reasons"],
                "expression_share_of_pipeline": summary["profile"][
                    "expression_share_of_pipeline"
                ],
                "total_candidate_s": summary["stage_s"]["total_candidate"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析参数并执行决策基准。"""

    parser = argparse.ArgumentParser(description="评估受控 DSL 是否需要原生执行器")
    parser.add_argument("--output", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input-hash", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.input_hash or args.worker_output is None:
            parser.error("worker 模式需要 input-hash 和 worker-output")
        run_worker(args.input_hash, args.worker_output.resolve())
        return 0
    return run_suite(args.output.resolve(), repeats=args.repeats)


if __name__ == "__main__":
    raise SystemExit(main())
