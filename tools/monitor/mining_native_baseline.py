"""验证挖掘候选相关性的 C++ 接入收益和不退化门槛。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from common.paths import PROJECT_ROOT
from tools.monitor.baseline_contract import (
    atomic_write_json,
    atomic_write_text,
    compare_golden,
    load_golden,
    prepare_common,
    project_relative,
    update_manifest,
)
from tools.monitor.mining_native_common import (
    BACKENDS,
    END,
    MAX_RSS_RATIO,
    MIN_CORRELATION_SPEEDUP,
    MIN_TECHNICAL_IMPROVEMENT,
    SCHEMA_VERSION,
    START,
)


def _worker_command(
    backend: str,
    input_hash: str,
    output: Path,
    *,
    cache_enabled: bool = True,
) -> list[str]:
    """构造独立后端测量命令。"""

    command = [
        sys.executable,
        "-m",
        "tools.monitor.mining_native_worker",
        "--backend",
        backend,
        "--input-hash",
        input_hash,
        "--output",
        str(output),
    ]
    if not cache_enabled:
        command.append("--no-cache")
    return command


def summarize_gate(
    workers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """按正确性、耗时、后端选择和内存生成接入门禁。"""

    reasons: list[str] = []
    python = workers["python"]
    native = workers["native"]
    auto = workers["auto"]
    uncached = workers["python_uncached"]
    stable_match = True
    for backend in ("native", "auto"):
        try:
            compare_golden(
                python["stable_evaluations"],
                workers[backend]["stable_evaluations"],
            )
        except AssertionError:
            stable_match = False
    correlation_speedup = (
        python["correlation_median_s"] / native["correlation_median_s"]
    )
    cached_python_technical_speedup = (
        python["technical_elapsed_s"] / native["technical_elapsed_s"]
    )
    technical_speedup = (
        uncached["technical_elapsed_s"] / auto["technical_elapsed_s"]
    )
    technical_improvement = 1.0 - 1.0 / technical_speedup
    auto_vs_cached = python["total_elapsed_s"] / auto["total_elapsed_s"]
    auto_vs_uncached = uncached["total_elapsed_s"] / auto["total_elapsed_s"]
    rss_ratios = {
        backend: workers[backend]["peak_rss_mb"] / python["peak_rss_mb"]
        for backend in ("native", "auto")
    }
    rankic = native.get("rankic") or {}
    if not stable_match:
        reasons.append("stable_payload_mismatch")
    if auto["actual_backend"] != "native":
        reasons.append("auto_did_not_select_native")
    if correlation_speedup < MIN_CORRELATION_SPEEDUP:
        reasons.append("correlation_speedup_below_1.2x")
    if technical_improvement < MIN_TECHNICAL_IMPROVEMENT:
        reasons.append("technical_improvement_below_10pct")
    if cached_python_technical_speedup < 1.0:
        reasons.append("native_slower_than_cached_python")
    if auto_vs_cached < 1.0:
        reasons.append("auto_total_regression")
    if any(ratio > MAX_RSS_RATIO for ratio in rss_ratios.values()):
        reasons.append("rss_above_1.2x")
    if rankic.get("max_rank_ic_difference", np.inf) > 1e-10:
        reasons.append("rankic_result_mismatch")
    if not rankic.get("counts_equal", False):
        reasons.append("rankic_count_mismatch")
    if not rankic.get("retain_sql", False):
        reasons.append("rankic_sql_not_retained_without_gain")
    gate_pass = not reasons
    return {
        "schema_version": SCHEMA_VERSION,
        "stable_payload_match": stable_match,
        "auto_selects_native": auto["actual_backend"] == "native",
        "correlation_speedup": correlation_speedup,
        "technical_speedup_vs_uncached_python": technical_speedup,
        "technical_improvement": technical_improvement,
        "native_vs_cached_python_technical_speedup": (
            cached_python_technical_speedup
        ),
        "auto_total_speedup_vs_cached_python": auto_vs_cached,
        "auto_total_speedup_vs_uncached_python": auto_vs_uncached,
        "rss_ratios": rss_ratios,
        "rankic": rankic,
        "gate_pass": gate_pass,
        "reason": reasons,
        "cpp_modules_replaced": 1 if gate_pass else 0,
    }


def _write_report(
    output: Path,
    workers: dict[str, dict[str, Any]],
    gate: dict[str, Any],
    input_hash: str,
) -> None:
    """写出人工可读的性能与正确性报告。"""

    lines = [
        "# P2-02 挖掘候选相关性 C++ 接入",
        "",
        "## 口径",
        "",
        f"- 区间：`{START}`～`{END}`；四候选批量物化后执行技术门禁和快速筛选。",
        "- 每个后端使用独立进程、单线程，相关性阶段重复 3 次取中位数。",
        f"- 输入树 SHA-256：`{input_hash}`。",
        "",
        "## 端到端",
        "",
        "| 后端 | 实际后端 | 总耗时(s) | 技术阶段(s) | 缓存后相关性中位(s) | 峰值RSS(MB) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for backend in (*BACKENDS, "python_uncached"):
        item = workers[backend]
        lines.append(
            f"| {backend} | {item['actual_backend']} | "
            f"{item['total_elapsed_s']:.3f} | {item['technical_elapsed_s']:.3f} | "
            f"{item['correlation_median_s']:.3f} | {item['peak_rss_mb']:.3f} |"
        )
    rankic = gate["rankic"]
    lines.extend(
        [
            "",
            "## 轻量 RankIC 决策",
            "",
            "| 路径 | 中位耗时(s) |",
            "|---|---:|",
            f"| DuckDB SQL | {rankic['sql_median_s']:.3f} |",
            f"| C++ 核 | {rankic['native_median_s']:.3f} |",
            "",
            f"- C++ 相对 SQL 加速：{rankic['native_speedup']:.3f}x。",
            f"- 最大 RankIC 差异：{rankic['max_rank_ic_difference']:.3e}；"
            f"计数一致：{'是' if rankic['counts_equal'] else '否'}。",
            f"- 结论：{'保留 DuckDB SQL' if rankic['retain_sql'] else '可评估接入 C++'}。",
            "",
            "## 门禁",
            "",
            f"- 稳定性：{'通过' if gate['stable_payload_match'] else '失败'}。",
            f"- 相关性加速：{gate['correlation_speedup']:.3f}x。",
            "- 技术阶段相对未缓存 Python 改善："
            f"{gate['technical_improvement']:.2%}。",
            "- native 相对缓存后 Python 的技术阶段加速："
            f"{gate['native_vs_cached_python_technical_speedup']:.3f}x。",
            "- auto 相对缓存后 Python 的总耗时加速："
            f"{gate['auto_total_speedup_vs_cached_python']:.3f}x。",
            "- auto 相对未缓存 Python 的总耗时加速："
            f"{gate['auto_total_speedup_vs_uncached_python']:.3f}x。",
            f"- RSS 比例：{json.dumps(gate['rss_ratios'], ensure_ascii=False)}。",
            f"- 结论：{'通过' if gate['gate_pass'] else '未通过'}；"
            f"原因：{', '.join(gate['reason']) or '无'}。",
            f"- 实际 C++ 替换模块新增：{gate['cpp_modules_replaced']}。",
            "",
        ]
    )
    atomic_write_text(output / "result.md", "\n".join(lines))


def run_suite(output: Path) -> int:
    """运行三后端和未缓存对照，并写出门禁证据。"""

    output.mkdir(parents=True, exist_ok=True)
    common = prepare_common(output, refresh=True)
    input_hash = common["inputs"]["input_tree_sha256"]
    with TemporaryDirectory(prefix="mining-native-workers-") as root:
        root_path = Path(root)
        workers = {}
        for backend in BACKENDS:
            worker_path = root_path / f"{backend}.json"
            completed = subprocess.run(
                _worker_command(backend, input_hash, worker_path),
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"{backend} 基准失败：{completed.stderr or completed.stdout}"
                )
            workers[backend] = load_golden(worker_path)
        uncached_path = root_path / "python_uncached.json"
        completed = subprocess.run(
            _worker_command(
                "python",
                input_hash,
                uncached_path,
                cache_enabled=False,
            ),
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                "python_uncached 基准失败："
                f"{completed.stderr or completed.stdout}"
            )
        workers["python_uncached"] = load_golden(uncached_path)
    gate = summarize_gate(workers)
    result = {
        "schema_version": SCHEMA_VERSION,
        "input_tree_sha256": input_hash,
        "workers": workers,
        "gate": gate,
    }
    atomic_write_json(output / "result.json", result)
    _write_report(output, workers, gate, input_hash)
    update_manifest(output)
    print(
        json.dumps(
            {
                "output": project_relative(output),
                "gate_pass": gate["gate_pass"],
                "reason": gate["reason"],
                "correlation_speedup": gate["correlation_speedup"],
                "technical_improvement": gate["technical_improvement"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if gate["gate_pass"] else 1


def main(argv: list[str] | None = None) -> int:
    """解析参数并执行挖掘原生基准。"""

    parser = argparse.ArgumentParser(description="验证挖掘候选相关性 C++ 接入")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_suite(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
