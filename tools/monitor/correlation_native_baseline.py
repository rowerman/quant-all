"""验证相关性 Spearman 后端接入的真实 stage 收益和回退语义。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np

from calculate.correlation import available_factors
from calculate.panel import build_panel
from common.paths import PROJECT_ROOT
from kernels import get_backend, native_threads
from tools.monitor.baseline_cases import run_case
from tools.monitor.baseline_contract import (
    atomic_write_json,
    atomic_write_text,
    compare_golden,
    load_golden,
    prepare_common,
    project_relative,
    update_manifest,
)
from tools.monitor.kernel_baseline import (
    CORRELATION_QUERY,
    _load_spearman_matrix,
)

_BACKEND_ROOT = PROJECT_ROOT / "apps" / "monitor"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from backend import service

SCHEMA_VERSION = "correlation_native_baseline_v1"
EVIDENCE_ROOT = PROJECT_ROOT / "AiRecord/benchmarks/V114/p1-03-20261027"
V105_INPUTS = PROJECT_ROOT / "AiRecord/benchmarks/V105/p0-01-20261027/inputs.json"
V107_ROOT = PROJECT_ROOT / "AiRecord/benchmarks/V107/p0-03-20261027"
BACKENDS = ("python", "native", "auto")
SCOPES = ("full", "target")
MIN_STAGE_IMPROVEMENT = 0.20
MIN_KERNEL_SPEEDUP = 1.50
MAX_RSS_RATIO = 1.20
REPEATS = 5


def _stage_elapsed_ms(case_timing: dict[str, Any], stage: str) -> float:
    """从固定 stage 计时中读取相关性耗时。"""

    for event in case_timing["events"]:
        if event["stage"] == stage and event["reported_elapsed_ms"] is not None:
            return float(event["reported_elapsed_ms"])
    raise RuntimeError(f"基准结果缺少 {stage} 阶段耗时")


def _panel_shape(golden: dict[str, Any]) -> tuple[int, int]:
    """从 meta 事件读取面板行数和交易日数。"""

    for event in golden["events"]:
        if event["stage"] == "meta":
            panel = event["payload"]["panel"]
            return int(panel["rows"]), int(panel["days"])
    raise RuntimeError("基准结果缺少 meta 面板信息")


def _case_name(scope: str) -> str:
    """把范围名称转换为固定基准案例名。"""

    if scope == "full":
        return "correlation"
    if scope == "target":
        return "correlation_target"
    raise ValueError(f"未知相关性范围：{scope}")


def run_stage_worker(backend: str, scope: str, output: Path) -> None:
    """在独立进程中执行一个后端与范围组合。"""

    os.environ["QUANT_KERNEL_BACKEND"] = backend
    os.environ["QUANT_NATIVE_THREADS"] = "1"
    selection = get_backend()
    started = time.perf_counter()
    golden, case_timing = run_case(_case_name(scope), {})
    elapsed_s = time.perf_counter() - started
    rows, groups = _panel_shape(golden)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "case": _case_name(scope),
        "scope": scope,
        "requested_backend": backend,
        "actual_backend": selection.name,
        "fallback_reason": selection.fallback_reason,
        "threads": native_threads() if selection.name == "native" else 1,
        "rows": rows,
        "groups": groups,
        "elapsed_s": round(elapsed_s, 3),
        "stage_elapsed_ms": _stage_elapsed_ms(case_timing, "correlation"),
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            3,
        ),
        "golden": golden,
        "case_timing": case_timing,
    }
    atomic_write_json(output, payload)


def _array_digest(array: np.ndarray) -> str:
    """返回数组内容和形状的稳定摘要。"""

    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    if np.issubdtype(array.dtype, np.floating):
        normalized = np.asarray(array, dtype=np.float64).copy()
        normalized[~np.isfinite(normalized)] = np.nan
        digest.update(normalized.tobytes())
    else:
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_kernel_worker(backend: str, output: Path) -> None:
    """测量固定真实截面的完整矩阵和目标行核速度。"""

    os.environ["QUANT_KERNEL_BACKEND"] = backend
    os.environ["QUANT_NATIVE_THREADS"] = "1"
    selection = get_backend()
    context = build_panel(service.prepare(CORRELATION_QUERY).params)
    values = np.ascontiguousarray(_load_spearman_matrix(context), dtype=np.float64)
    offsets = np.array([0, values.shape[0]], dtype=np.int64)
    names = [
        name
        for family_names in available_factors(context.factor_dir).values()
        for name in family_names
    ]
    target_index = names.index(context.params.factor)
    results: dict[str, Any] = {}
    for scope, target in (("full", None), ("target", target_index)):
        first_started = time.perf_counter()
        correlation, count = selection.backend.pairwise_spearman(
            values,
            offsets,
            5,
            target,
        )
        first_s = time.perf_counter() - first_started
        runs = []
        for _ in range(REPEATS):
            started = time.perf_counter()
            correlation, count = selection.backend.pairwise_spearman(
                values,
                offsets,
                5,
                target,
            )
            runs.append(time.perf_counter() - started)
        results[scope] = {
            "first_s": first_s,
            "runs_s": runs,
            "median_s": float(np.median(np.asarray(runs, dtype=np.float64))),
            "correlation_sha256": _array_digest(correlation),
            "count_sha256": _array_digest(count),
        }
        stem = output.with_suffix("")
        np.save(f"{stem}-{scope}-correlation.npy", correlation)
        np.save(f"{stem}-{scope}-count.npy", count)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "requested_backend": backend,
        "actual_backend": selection.name,
        "fallback_reason": selection.fallback_reason,
        "threads": native_threads() if selection.name == "native" else 1,
        "date": "2026-08-03",
        "rows": int(values.shape[0]),
        "columns": int(values.shape[1]),
        "target_index": target_index,
        "results": results,
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            3,
        ),
    }
    atomic_write_json(output, payload)


def _worker_command(mode: str, backend: str, scope: str | None, output: Path) -> list[str]:
    """构造独立基准子进程命令。"""

    command = [
        sys.executable,
        "-m",
        "tools.monitor.correlation_native_baseline",
        "--worker-mode",
        mode,
        "--worker-backend",
        backend,
        "--worker-output",
        str(output),
    ]
    if scope is not None:
        command.extend(["--worker-scope", scope])
    return command


def _run_worker(command: list[str]) -> None:
    """执行一个子进程，失败时携带原始输出退出。"""

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"基准子进程失败：{' '.join(command)}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )


def _payload_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """比较两个真实评估 payload，浮点使用固定金标准容差。"""

    try:
        compare_golden(expected["events"], actual["events"])
    except AssertionError:
        return False
    return True


def summarize_gate(
    stages: dict[str, dict[str, dict[str, Any]]],
    kernels: dict[str, dict[str, Any]],
    *,
    python_matches: bool,
    payloads_match: bool,
    kernel_results_match: bool,
) -> dict[str, Any]:
    """按正确性、stage 收益、候选不退化、核速度和内存生成门禁。"""

    full_python = stages["full"]["python"]
    full_native = stages["full"]["native"]
    full_auto = stages["full"]["auto"]
    target_python = stages["target"]["python"]
    target_native = stages["target"]["native"]
    target_auto = stages["target"]["auto"]
    full_stage_speedup = (
        full_python["stage_elapsed_ms"] / full_native["stage_elapsed_ms"]
    )
    full_improvement = 1.0 - 1.0 / full_stage_speedup
    full_auto_speedup = (
        full_python["stage_elapsed_ms"] / full_auto["stage_elapsed_ms"]
    )
    full_auto_improvement = 1.0 - 1.0 / full_auto_speedup
    target_speedup = (
        target_python["stage_elapsed_ms"] / target_native["stage_elapsed_ms"]
    )
    target_auto_speedup = (
        target_python["stage_elapsed_ms"] / target_auto["stage_elapsed_ms"]
    )
    kernel_speedup = (
        kernels["python"]["results"]["full"]["median_s"]
        / kernels["native"]["results"]["full"]["median_s"]
    )
    rss_ratios = {
        f"{backend}_{scope}": stages[scope][backend]["peak_rss_mb"]
        / stages[scope]["python"]["peak_rss_mb"]
        for scope in SCOPES
        for backend in ("native", "auto")
    }
    auto_native = all(
        stages[scope]["auto"]["actual_backend"] == "native"
        for scope in SCOPES
    )
    reasons = []
    if not python_matches:
        reasons.append("python_payload_mismatch")
    if not payloads_match:
        reasons.append("native_or_auto_payload_mismatch")
    if not kernel_results_match:
        reasons.append("kernel_result_mismatch")
    if not auto_native:
        reasons.append("auto_did_not_select_native")
    if full_improvement < MIN_STAGE_IMPROVEMENT - 1e-12:
        reasons.append("full_stage_improvement_below_20pct")
    if full_auto_improvement < MIN_STAGE_IMPROVEMENT - 1e-12:
        reasons.append("auto_full_stage_improvement_below_20pct")
    if target_speedup < 1.0:
        reasons.append("target_stage_regression")
    if target_auto_speedup < 1.0:
        reasons.append("auto_target_stage_regression")
    if kernel_speedup < MIN_KERNEL_SPEEDUP:
        reasons.append("kernel_speedup_below_1.5x")
    if any(ratio > MAX_RSS_RATIO for ratio in rss_ratios.values()):
        reasons.append("rss_above_1.2x")
    gate_pass = not reasons
    return {
        "schema_version": SCHEMA_VERSION,
        "correct": python_matches and payloads_match and kernel_results_match,
        "payloads_match": payloads_match,
        "kernel_results_match": kernel_results_match,
        "auto_selects_native": auto_native,
        "full_stage_speedup": full_stage_speedup,
        "full_stage_improvement": full_improvement,
        "auto_full_stage_speedup": full_auto_speedup,
        "auto_full_stage_improvement": full_auto_improvement,
        "target_stage_speedup": target_speedup,
        "auto_target_stage_speedup": target_auto_speedup,
        "kernel_speedup": kernel_speedup,
        "rss_ratios": rss_ratios,
        "full_auto_elapsed_ms": full_auto["stage_elapsed_ms"],
        "target_auto_elapsed_ms": target_auto["stage_elapsed_ms"],
        "gate_pass": gate_pass,
        "reason": reasons,
        "cpp_modules_replaced": 1 if gate_pass else 0,
    }


def _write_report(output: Path, stages: dict[str, Any], kernels: dict[str, Any],
                  result: dict[str, Any], input_hash: str) -> None:
    """写出便于人工复核的 Markdown 报告。"""

    lines = [
        "# P1-03 Spearman 相关性核接入",
        "",
        "## 口径",
        "",
        "- V105 固定三年区间、全 A、MOM_20、单线程。",
        "- Python、native、auto 分别在独立进程执行完整矩阵和候选目标行。",
        f"- 每项核热运行 {REPEATS} 次取中位数。",
        f"- 输入树 SHA-256：`{input_hash}`。",
        "",
        "## 三年 stage",
        "",
        "| 范围 | 后端 | 实际后端 | 回退原因 | 耗时(ms) | 峰值RSS(MB) |",
        "|---|---|---|---|---:|---:|",
    ]
    for scope in SCOPES:
        for backend in BACKENDS:
            item = stages[scope][backend]
            lines.append(
                f"| {scope} | {backend} | {item['actual_backend']} | "
                f"{item['fallback_reason'] or 'none'} | "
                f"{item['stage_elapsed_ms']:.1f} | {item['peak_rss_mb']:.3f} |"
            )
    lines.extend([
        "",
        "## 核微基准",
        "",
        "| 范围 | Python(s) | native(s) | 加速 |",
        "|---|---:|---:|---:|",
    ])
    for scope in SCOPES:
        python_s = kernels["python"]["results"][scope]["median_s"]
        native_s = kernels["native"]["results"][scope]["median_s"]
        lines.append(
            f"| {scope} | {python_s:.6f} | {native_s:.6f} | {python_s / native_s:.3f}x |"
        )
    lines.extend([
        "",
        "## 门禁",
        "",
        f"- 正式 stage 改善：{result['full_stage_improvement'] * 100:.2f}%",
        f"- 默认 auto 正式 stage 改善：{result['auto_full_stage_improvement'] * 100:.2f}%",
        f"- 候选 stage 加速：{result['target_stage_speedup']:.3f}x",
        f"- 默认 auto 候选 stage 加速：{result['auto_target_stage_speedup']:.3f}x",
        f"- 核加速：{result['kernel_speedup']:.3f}x",
        f"- auto 选择 native：{'是' if result['auto_selects_native'] else '否'}",
        f"- 门禁结论：{'通过' if result['gate_pass'] else '未通过'}",
        f"- 原因：{', '.join(result['reason']) or '满足全部门槛'}",
        f"- 实际 C++ 替换模块数：{result['cpp_modules_replaced']}",
        "",
    ])
    atomic_write_text(output / "result.md", "\n".join(lines))


def run_suite(
    output: Path,
    *,
    refresh: bool = False,
    refresh_common: bool = False,
) -> int:
    """运行全部后端和范围并生成 P1-03 证据。"""

    output.mkdir(parents=True, exist_ok=True)
    common = prepare_common(output, refresh=refresh or refresh_common)
    current_hash = common["inputs"]["input_tree_sha256"]
    expected_hash = load_golden(V105_INPUTS)["input_tree_sha256"]
    if current_hash != expected_hash:
        raise RuntimeError(
            "输入数据相对 V105 已漂移，停止性能对照："
            f"{current_hash} != {expected_hash}"
        )

    stages: dict[str, dict[str, dict[str, Any]]] = {}
    for scope in SCOPES:
        stages[scope] = {}
        for backend in BACKENDS:
            path = output / "workers" / f"{scope}-{backend}.json"
            if refresh or not path.is_file():
                _run_worker(_worker_command("stage", backend, scope, path))
            stages[scope][backend] = load_golden(path)

    kernels = {}
    for backend in ("python", "native"):
        path = output / "workers" / f"kernel-{backend}.json"
        if refresh or not path.is_file():
            _run_worker(_worker_command("kernel", backend, None, path))
        kernels[backend] = load_golden(path)

    python_matches = True
    for scope, filename in (
        ("full", "correlation.golden.json"),
        ("target", "correlation_target.golden.json"),
    ):
        expected = load_golden(V107_ROOT / filename)["golden"]
        try:
            compare_golden(expected["events"], stages[scope]["python"]["golden"]["events"])
        except AssertionError:
            python_matches = False

    payloads_match = all(
        _payload_matches(
            stages[scope]["python"]["golden"],
            stages[scope][backend]["golden"],
        )
        for scope in SCOPES
        for backend in ("native", "auto")
    )
    kernel_results_match = True
    for scope in SCOPES:
        python_stem = output / "workers" / f"kernel-python-{scope}"
        native_stem = output / "workers" / f"kernel-native-{scope}"
        python_correlation = np.load(f"{python_stem}-correlation.npy")
        native_correlation = np.load(f"{native_stem}-correlation.npy")
        python_count = np.load(f"{python_stem}-count.npy")
        native_count = np.load(f"{native_stem}-count.npy")
        if not np.allclose(
            native_correlation,
            python_correlation,
            atol=1e-10,
            rtol=1e-10,
            equal_nan=True,
        ):
            kernel_results_match = False
        if not np.array_equal(native_count, python_count):
            kernel_results_match = False
    result = summarize_gate(
        stages,
        kernels,
        python_matches=python_matches,
        payloads_match=payloads_match,
        kernel_results_match=kernel_results_match,
    )
    result["input_tree_sha256"] = current_hash
    result["workers"] = {
        scope: {
            backend: {
                key: stages[scope][backend][key]
                for key in (
                    "requested_backend",
                    "actual_backend",
                    "fallback_reason",
                    "threads",
                    "rows",
                    "groups",
                    "elapsed_s",
                    "stage_elapsed_ms",
                    "peak_rss_mb",
                )
            }
            for backend in BACKENDS
        }
        for scope in SCOPES
    }
    result["kernel_workers"] = {
        backend: {
            key: kernels[backend][key]
            for key in (
                "requested_backend",
                "actual_backend",
                "fallback_reason",
                "threads",
                "rows",
                "columns",
                "target_index",
                "peak_rss_mb",
                "results",
            )
        }
        for backend in ("python", "native")
    }
    atomic_write_json(output / "result.json", result)
    _write_report(output, stages, kernels, result, current_hash)
    update_manifest(output)
    print(
        json.dumps(
            {
                "output": project_relative(output),
                "gate_pass": result["gate_pass"],
                "reason": result["reason"],
                "full_stage_improvement": result["full_stage_improvement"],
                "auto_full_stage_improvement": result["auto_full_stage_improvement"],
                "target_stage_speedup": result["target_stage_speedup"],
                "auto_target_stage_speedup": result["auto_target_stage_speedup"],
                "kernel_speedup": result["kernel_speedup"],
                "rss_ratios": result["rss_ratios"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if result["gate_pass"] else 1


def main(argv: list[str] | None = None) -> int:
    """解析并执行相关性 native 接入基准。"""

    parser = argparse.ArgumentParser(description="验证相关性 Spearman 后端接入")
    parser.add_argument("--output", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--refresh-common", action="store_true")
    parser.add_argument("--worker-mode", choices=("stage", "kernel"))
    parser.add_argument("--worker-backend", choices=BACKENDS)
    parser.add_argument("--worker-scope", choices=SCOPES)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    if args.worker_mode:
        if args.worker_backend is None or args.worker_output is None:
            parser.error("worker 模式必须提供 --worker-backend 和 --worker-output")
        if args.worker_mode == "stage":
            if args.worker_scope is None:
                parser.error("stage worker 必须提供 --worker-scope")
            run_stage_worker(
                args.worker_backend,
                args.worker_scope,
                args.worker_output,
            )
        else:
            run_kernel_worker(args.worker_backend, args.worker_output)
        return 0
    return run_suite(
        args.output.resolve(),
        refresh=args.refresh,
        refresh_common=args.refresh_common,
    )


if __name__ == "__main__":
    raise SystemExit(main())
