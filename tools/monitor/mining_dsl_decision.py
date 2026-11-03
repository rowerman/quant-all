"""运行受控 DSL 原生执行器决策基准。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from common.paths import PROJECT_ROOT
from mining.catalog import load_catalog
from tools.monitor.baseline_contract import (
    atomic_write_json,
    environment_snapshot,
    file_record,
    input_snapshot,
    load_golden,
    project_relative,
    update_manifest,
)
from tools.monitor.mining_dsl_analysis import (
    FIXTURE_EXPRESSIONS,
    OPERATOR_COVERAGE_EXPRESSIONS,
    SCHEMA_VERSION,
    summarize_corpus,
)
from tools.monitor.mining_dsl_report import summarize_runs, write_report
from tools.monitor.mining_dsl_worker import run_worker
from tools.monitor.mining_native_common import EXPRESSIONS

REPEATS = 3
EVIDENCE_ROOT = PROJECT_ROOT / "AiRecord/benchmarks/V122/p2-03-20261028"
_SOURCE_PATHS = (
    "tools/monitor/mining_dsl_analysis.py",
    "tools/monitor/mining_dsl_decision.py",
    "tools/monitor/mining_dsl_report.py",
    "tools/monitor/mining_dsl_worker.py",
    "research/mining/expression_ast.py",
    "research/mining/expression_batch.py",
    "research/mining/expression_compiler.py",
    "research/mining/expression_dag.py",
    "research/mining/expression_parser.py",
    "research/mining/expression_validation.py",
)


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


def _run_workers(
    output: Path,
    input_hash: str,
    repeats: int,
) -> list[dict]:
    """用独立进程运行既定轮数。"""

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
    return runs


def run_suite(output: Path, *, repeats: int = REPEATS) -> int:
    """运行覆盖统计和重复测量，并写出证据。"""

    if repeats < 1:
        raise ValueError("repeats 必须为正整数")
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "environment.json", environment_snapshot())
    inputs = input_snapshot()
    atomic_write_json(output / "inputs.json", inputs)
    input_hash = str(inputs["input_tree_sha256"])

    corpus = summarize_corpus(
        {
            "benchmark": EXPRESSIONS,
            "operator_coverage": OPERATOR_COVERAGE_EXPRESSIONS,
            "fixture": FIXTURE_EXPRESSIONS,
        },
        load_catalog(),
    )
    atomic_write_json(output / "corpus.json", corpus)

    runs = _run_workers(output, input_hash, repeats)
    summary = summarize_runs(runs)
    result = {
        "schema_version": SCHEMA_VERSION,
        "input_tree_sha256": input_hash,
        "source_files": [
            file_record(PROJECT_ROOT / path) for path in _SOURCE_PATHS
        ],
        "corpus": corpus,
        "runs": runs,
        "summary": summary,
    }
    atomic_write_json(output / "result.json", result)
    write_report(output, corpus, summary, input_hash)
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
