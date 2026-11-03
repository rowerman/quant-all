"""清洗层条件化剖析与 C++ 实施决策。

用法::

    .venv/bin/python -m tools.monitor.clean_profile --case all --repeats 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from common.paths import PROJECT_ROOT
from tools.monitor.baseline_contract import (
    atomic_write_json,
    atomic_write_text,
    load_golden,
    project_relative,
)
from tools.monitor.clean_profile_contract import (
    CASE_DEFINITIONS,
    CASE_ORDER,
    EVIDENCE_ROOT,
    PRIMARY_CASE,
    SCHEMA_VERSION,
    THRESHOLD,
    environment_snapshot,
    input_snapshot,
    median,
    update_manifest,
)
from tools.monitor.clean_profile_report import render_report
from tools.monitor.clean_profile_probe import stage_definitions


def _run_worker(case: str, repeat: int) -> dict[str, Any]:
    """启动独立进程运行一次目标年度剖析。"""

    definition = CASE_DEFINITIONS[case]
    with tempfile.TemporaryDirectory(prefix=f"clean-profile-result-{case}-") as directory:
        result_path = Path(directory) / f"run-{repeat}.json"
        command = [
            sys.executable,
            "-m",
            "tools.monitor.clean_profile_worker",
            "--year",
            str(definition["year"]),
            "--seed-year",
            str(definition["seed_year"]),
            "--result",
            str(result_path),
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"{case} 第 {repeat} 次剖析失败：\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        return load_golden(result_path)


def _median_stages(runs: list[dict[str, Any]], total_s: float) -> dict[str, Any]:
    """对多次运行的阶段耗时逐项取中位数并重算占比。"""

    names = sorted(
        {
            name
            for run in runs
            for name in run["profile"]["stages"]
        }
    )
    stages: dict[str, dict[str, Any]] = {}
    for name in names:
        observations = [
            run["profile"]["stages"][name]
            for run in runs
            if name in run["profile"]["stages"]
        ]
        elapsed = median([float(item["elapsed_s"]) for item in observations])
        stages[name] = {
            "category": observations[0]["category"],
            "pure_array": bool(observations[0]["pure_array"]),
            "count": int(round(median([float(item["count"]) for item in observations]))),
            "elapsed_s": elapsed,
            "share": round(elapsed / total_s, 6),
        }

    categories: dict[str, float] = {}
    for current in stages.values():
        category = str(current["category"])
        categories[category] = round(
            categories.get(category, 0.0) + float(current["elapsed_s"]),
            6,
        )
    categories["unattributed"] = median(
        [float(run["profile"]["unattributed_s"]) for run in runs]
    )
    pure = [
        (name, current)
        for name, current in stages.items()
        if current["pure_array"]
    ]
    largest_name = None
    largest_elapsed = 0.0
    largest_share = 0.0
    if pure:
        largest_name, largest_stage = max(
            pure,
            key=lambda item: float(item[1]["elapsed_s"]),
        )
        largest_elapsed = float(largest_stage["elapsed_s"])
        largest_share = float(largest_stage["share"])
    return {
        "elapsed_s": total_s,
        "stages": stages,
        "categories": dict(sorted(categories.items())),
        "unattributed_s": categories["unattributed"],
        "largest_pure_array_stage": largest_name,
        "largest_pure_array_elapsed_s": largest_elapsed,
        "largest_pure_array_share": largest_share,
    }


def aggregate_runs(case: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总一个案例的逐次观测、稳定性和中位阶段。"""

    if not runs:
        raise ValueError("案例至少需要一次运行")
    total_s = median([float(run["elapsed_s"]) for run in runs])
    profile = _median_stages(runs, total_s)
    digests = [str(run["content_digest"]["sha256"]) for run in runs]
    stable_stats = len(
        {
            json.dumps(run["stats"], ensure_ascii=False, sort_keys=True)
            for run in runs
        }
    ) == 1
    return {
        "case": case,
        "definition": CASE_DEFINITIONS[case],
        "repeat_count": len(runs),
        "median": {
            "elapsed_s": total_s,
            "peak_rss_mb": median([float(run["peak_rss_mb"]) for run in runs]),
            "seed_elapsed_s": median(
                [float(run["seed_elapsed_s"]) for run in runs]
            ),
            "duckdb_daily_s": median(
                [float(run["duckdb_daily"]["elapsed_s"]) for run in runs]
            ),
            "profile": profile,
        },
        "stability": {
            "content_sha256": digests[0] if len(set(digests)) == 1 else None,
            "content_exact": len(set(digests)) == 1,
            "stats_exact": stable_stats,
            "rows": int(runs[0]["content_digest"]["rows"]),
            "column_count": len(runs[0]["content_digest"]["columns"]),
        },
        "runs": [
            {
                "elapsed_s": run["elapsed_s"],
                "peak_rss_mb": run["peak_rss_mb"],
                "seed_elapsed_s": run["seed_elapsed_s"],
                "content_sha256": run["content_digest"]["sha256"],
                "profile": run["profile"],
                "duckdb_daily": run["duckdb_daily"],
                "stats": run["stats"],
            }
            for run in runs
        ],
    }


def decide_cpp(primary: dict[str, Any]) -> dict[str, Any]:
    """按主案例最大纯数组规则占比形成 C++ 路线决策。"""

    profile = primary["median"]["profile"]
    stage = profile["largest_pure_array_stage"]
    share = float(profile["largest_pure_array_share"])
    elapsed = float(profile["largest_pure_array_elapsed_s"])
    total = float(primary["median"]["elapsed_s"])
    qualifies = stage is not None and share >= THRESHOLD
    return {
        "status": "cpp_candidate" if qualifies else "no_cpp",
        "threshold": THRESHOLD,
        "primary_case": PRIMARY_CASE,
        "largest_stage": stage,
        "elapsed_s": elapsed,
        "total_elapsed_s": total,
        "share": share,
        "candidate_rule_reached_threshold": qualifies,
        "reason": (
            "主案例存在占比达到 20% 的纯数组规则，另立实现任务前需做 Python 优化对照。"
            if qualifies
            else "主案例没有任何纯数组规则达到清洗总耗时的 20%，不实施清洗层 C++。"
        ),
        "session_cpp_modules_added": 0,
        "existing_cpp_replacement_total": 14,
    }


def run_profile(
    selected_cases: tuple[str, ...],
    *,
    repeats: int,
    output: Path,
) -> dict[str, Any]:
    """运行所选案例并把证据、结果和报告写入输出目录。"""

    if repeats < 1:
        raise ValueError("repeats 必须至少为 1")
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "environment.json", environment_snapshot())
    atomic_write_json(output / "inputs.json", input_snapshot())
    atomic_write_json(
        output / "case_definitions.json",
        {
            "schema_version": SCHEMA_VERSION,
            "threshold": THRESHOLD,
            "primary_case": PRIMARY_CASE,
            "repeats": repeats,
            "cases": CASE_DEFINITIONS,
            "stage_targets": stage_definitions(),
        },
    )
    cases: dict[str, dict[str, Any]] = {}
    for case in selected_cases:
        runs = [_run_worker(case, repeat + 1) for repeat in range(repeats)]
        cases[case] = aggregate_runs(case, runs)
    if PRIMARY_CASE not in cases:
        raise ValueError(f"决策要求包含主案例 {PRIMARY_CASE}")
    result = {
        "schema_version": SCHEMA_VERSION,
        "started_cases": list(selected_cases),
        "cases": cases,
        "decision": decide_cpp(cases[PRIMARY_CASE]),
    }
    atomic_write_json(output / "result.json", result)
    atomic_write_text(output / "report.md", render_report(result))
    update_manifest(output)
    return result


def main(argv: list[str] | None = None) -> int:
    """解析参数并运行清洗层剖析。"""

    parser = argparse.ArgumentParser(description="清洗层条件化剖析与 C++ 决策")
    parser.add_argument(
        "--case",
        choices=(*CASE_ORDER, "all"),
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=EVIDENCE_ROOT)
    args = parser.parse_args(argv)
    selected = CASE_ORDER if args.case == "all" else (args.case,)
    if PRIMARY_CASE not in selected:
        selected = (*selected, PRIMARY_CASE)
    result = run_profile(
        selected,
        repeats=args.repeats,
        output=args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "cases": {
                    name: {
                        "median_elapsed_s": case["median"]["elapsed_s"],
                        "peak_rss_mb": case["median"]["peak_rss_mb"],
                        "duckdb_daily_s": case["median"]["duckdb_daily_s"],
                        "largest_pure_array_stage": case["median"]["profile"][
                            "largest_pure_array_stage"
                        ],
                        "largest_pure_array_share": case["median"]["profile"][
                            "largest_pure_array_share"
                        ],
                        "content_exact": case["stability"]["content_exact"],
                    }
                    for name, case in result["cases"].items()
                },
                "evidence": project_relative(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
