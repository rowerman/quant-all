"""P1-07 暴露链路基准；按 1e-5 门槛验证 C++ OLS。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import numpy as np

_BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "monitor"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from backend import service
from calculate.evaluate import evaluate_context
from calculate.exposure import run_exposure
from calculate.panel import build_panel
from common.paths import PROJECT_ROOT
from kernels import get_backend
from kernels.fallback import PythonKernelBackend
from kernels.native import CppKernelBackend
from tools.monitor.baseline_contract import (
    GoldenMismatch,
    atomic_write_json,
    atomic_write_text,
    compare_golden,
    load_golden,
    prepare_common,
    project_relative,
    strip_volatile,
    update_manifest,
)
from tools.monitor.standardize_native_baseline import RssSampler

SCHEMA_VERSION = "ols_native_baseline_v1"
STAGES = ("exposure", "incremental_ic")
BACKENDS = ("python", "native", "auto")
BASE_QUERY = {
    "factor": "MOM_20",
    "start": "2023-10-15",
    "end": "2026-10-15",
    "universe": "all_a",
    "horizon": "5",
    "price": "close",
    "quantiles": "10",
    "winsorize": "true",
}
STAGE_REPEATS = 3
MAX_RSS_RATIO = 1.2
TARGET_IMPROVEMENT = 0.20
PAYLOAD_TOLERANCE = 1e-5
MIN_KERNEL_SPEEDUP = 1.5


def _worker_environment(backend: str) -> dict[str, str]:
    """返回固定后端和单线程的独立进程环境。"""

    return {
        **os.environ,
        "QUANT_KERNEL_BACKEND": backend,
        "QUANT_NATIVE_THREADS": "1",
    }


def _worker_command(
    backend: str,
    context_path: Path,
    output_dir: Path,
) -> list[str]:
    """构造 stage worker 命令。"""

    return [
        sys.executable,
        "-m",
        "tools.monitor.ols_native_baseline",
        "--worker-backend",
        backend,
        "--worker-context",
        str(context_path),
        "--worker-output",
        str(output_dir),
    ]


def _run_child(command: list[str], backend: str) -> None:
    """运行独立 worker；失败时抛出包含 stderr 的异常。"""

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_worker_environment(backend),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"worker 失败：{completed.stderr or completed.stdout}")


def _run_stage_worker(
    backend: str,
    context_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """在独立进程运行暴露与增量 IC，并写出结果。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    with context_path.open("rb") as stream:
        context = pickle.load(stream)
    actual_backend = get_backend(backend).name
    payloads: dict[str, dict] = {}
    stage_elapsed: dict[str, float] = {}
    with RssSampler() as sampler:
        started = time.perf_counter()
        for stage, payload in evaluate_context(context, stages=STAGES):
            if stage in STAGES:
                payloads[stage] = strip_volatile(payload)
                stage_elapsed[stage] = float(payload["elapsed_ms"])
        elapsed_s = time.perf_counter() - started
    if set(payloads) != set(STAGES):
        raise RuntimeError("评估入口没有产出全部暴露阶段")
    for stage, payload in payloads.items():
        atomic_write_json(output_dir / f"{stage}.json", payload)
    result = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "actual_backend": actual_backend,
        "elapsed_s": round(elapsed_s, 6),
        "stage_elapsed_ms": stage_elapsed,
        "combined_stage_elapsed_ms": float(sum(stage_elapsed.values())),
        "baseline_rss_mb": round(sampler.baseline_mb, 3),
        "peak_rss_mb": round(sampler.peak_mb, 3),
    }
    atomic_write_json(output_dir / "timing.json", result)
    return result


def _median_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总同一后端的多次独立运行。"""

    return {
        "actual_backend": records[0]["actual_backend"],
        "elapsed_s": float(np.median([item["elapsed_s"] for item in records])),
        "stage_elapsed_ms": {
            stage: float(
                np.median([item["stage_elapsed_ms"][stage] for item in records])
            )
            for stage in STAGES
        },
        "combined_stage_elapsed_ms": float(
            np.median([item["combined_stage_elapsed_ms"] for item in records])
        ),
        "peak_rss_mb": max(float(item["peak_rss_mb"]) for item in records),
        "runs": records,
    }


def _measure(output: Path) -> tuple[dict, dict]:
    """运行三后端各三次，并返回公共输入和汇总。"""

    output.mkdir(parents=True, exist_ok=True)
    common = prepare_common(output, refresh=True)
    with TemporaryDirectory(prefix="quant-ols-context-") as directory:
        context_path = Path(directory) / "context.pkl"
        context = build_panel(service.prepare(BASE_QUERY).params)
        with context_path.open("wb") as stream:
            pickle.dump(context, stream, protocol=pickle.HIGHEST_PROTOCOL)
        records: dict[str, list[dict[str, Any]]] = {}
        for backend in BACKENDS:
            records[backend] = []
            for repeat in range(STAGE_REPEATS):
                target = output / "workers" / backend / f"run-{repeat + 1}"
                _run_child(
                    _worker_command(backend, context_path, target),
                    backend,
                )
                records[backend].append(load_golden(target / "timing.json"))
    return common, {
        backend: _median_record(items)
        for backend, items in records.items()
    }


def _payloads_match(workers: Path) -> bool:
    """确认三后端两个 stage 的业务 payload 在 1e-5 内一致。"""

    for stage in STAGES:
        expected = load_golden(
            workers / "python" / "run-1" / f"{stage}.json"
        )
        for backend in ("native", "auto"):
            actual = load_golden(
                workers / backend / "run-1" / f"{stage}.json"
            )
            try:
                compare_golden(
                    expected,
                    actual,
                    abs_tol=PAYLOAD_TOLERANCE,
                    rel_tol=PAYLOAD_TOLERANCE,
                )
            except GoldenMismatch:
                return False
    return True


def _write_result(
    output: Path,
    common: dict,
    stages: dict,
    *,
    kind: str,
    baseline: dict | None,
) -> int:
    """写出 JSON 与 Markdown 汇总。"""

    payloads_match = _payloads_match(output / "workers")
    auto_selects_native = stages["auto"]["actual_backend"] == "native"
    rss_ratios = {
        backend: (
            stages[backend]["peak_rss_mb"]
            / stages["python"]["peak_rss_mb"]
        )
        for backend in ("native", "auto")
    }
    improvement = None
    if baseline is not None:
        previous = float(
            baseline["stages"]["auto"]["combined_stage_elapsed_ms"]
        )
        current = stages["auto"]["combined_stage_elapsed_ms"]
        improvement = 1.0 - current / previous
    python_current = stages["python"]["combined_stage_elapsed_ms"]
    native_current = stages["auto"]["combined_stage_elapsed_ms"]
    python_improvement = 1.0 - native_current / python_current
    rss_pass = all(ratio <= MAX_RSS_RATIO for ratio in rss_ratios.values())
    acceptance_pass = (
        payloads_match
        and auto_selects_native
        and rss_pass
        and python_improvement >= TARGET_IMPROVEMENT
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "input": {
            "query": BASE_QUERY,
            "input_tree_sha256": common["inputs"]["input_tree_sha256"],
            "stages": list(STAGES),
            "stage_repeats": STAGE_REPEATS,
            "threads": 1,
        },
        "stages": stages,
        "payloads_match": payloads_match,
        "auto_selects_native": auto_selects_native,
        "rss_ratios": rss_ratios,
        "stage_improvement": improvement,
        "python_stage_improvement": python_improvement,
        "acceptance_gate": {
            "pass": acceptance_pass if baseline is not None else None,
            "target_improvement": TARGET_IMPROVEMENT,
            "payload_tolerance": PAYLOAD_TOLERANCE,
            "cpp_ols_replaced": acceptance_pass,
        },
    }
    atomic_write_json(output / "result.json", result)
    lines = [
        f"# P1-07 暴露链路{kind}",
        "",
        "| 后端 | exposure | incremental_ic | 合计 | 峰值 RSS |",
        "|---|---:|---:|---:|---:|",
    ]
    for backend in BACKENDS:
        record = stages[backend]
        lines.append(
            f"| {backend} | "
            f"{record['stage_elapsed_ms']['exposure'] / 1000:.3f}s | "
            f"{record['stage_elapsed_ms']['incremental_ic'] / 1000:.3f}s | "
            f"{record['combined_stage_elapsed_ms'] / 1000:.3f}s | "
            f"{record['peak_rss_mb']:.1f}MB |"
        )
    lines.extend([
        "",
        f"- Python/native/auto payload 一致：{'是' if payloads_match else '否'}。",
        f"- auto 实际后端：{stages['auto']['actual_backend']}。",
        f"- native/auto RSS 比例："
        f"{rss_ratios['native']:.3f}x / {rss_ratios['auto']:.3f}x。",
    ])
    if improvement is not None:
        lines.extend([
            f"- 相对 V118 auto 合计改善（补充）：{improvement * 100:.2f}%。",
            f"- native 相对 Python 合计改善：{python_improvement * 100:.2f}%。",
            f"- C++ OLS 验收门槛：{'通过' if acceptance_pass else '未通过'}。",
        ])
    atomic_write_text(output / "result.md", "\n".join(lines))
    update_manifest(output)
    print(
        f"{kind} result written to {project_relative(output)}",
        flush=True,
    )
    return 0


def run_baseline(output: Path) -> int:
    """保存改动前的三后端暴露链路基线。"""

    common, stages = _measure(output)
    return _write_result(
        output,
        common,
        stages,
        kind="prechange",
        baseline=None,
    )


def run_accepted(output: Path, baseline: Path) -> int:
    """按 1e-5 门槛验收 C++ OLS 和暴露链路。"""

    baseline_result = load_golden(baseline / "result.json")
    common, stages = _measure(output)
    if (
        common["inputs"]["input_tree_sha256"]
        != baseline_result["input"]["input_tree_sha256"]
    ):
        raise RuntimeError("输入数据相对改动前基线已漂移")
    return _write_result(
        output,
        common,
        stages,
        kind="accepted",
        baseline=baseline_result,
    )


def _run_backend_micro(backend, design, response, offsets):
    """调用一个后端的 OLS 并记录 5 次热运行。"""

    runs = []
    with RssSampler() as sampler:
        started = time.perf_counter()
        result = backend.daily_ols(design, response, offsets)
        cold_s = time.perf_counter() - started
        for _ in range(5):
            started = time.perf_counter()
            result = backend.daily_ols(design, response, offsets)
            runs.append(time.perf_counter() - started)
    return result, {
        "cold_s": round(cold_s, 6),
        "runs_s": [round(value, 6) for value in runs],
        "median_s": float(np.median(runs)),
        "peak_rss_mb": round(sampler.peak_mb, 3),
    }


def run_micro(output: Path) -> int:
    """在真实 OLS 输入上比较 Python 与 native。"""

    output.mkdir(parents=True, exist_ok=True)
    context = build_panel(service.prepare(BASE_QUERY).params)
    captured: dict[str, np.ndarray] = {}
    original = PythonKernelBackend.daily_ols

    def capture(self, design, response, offsets, min_rows=30):
        if not captured:
            captured["design"] = np.ascontiguousarray(design)
            captured["response"] = np.ascontiguousarray(response)
            captured["offsets"] = np.ascontiguousarray(offsets)
        return original(self, design, response, offsets, min_rows)

    previous_backend = os.environ.get("QUANT_KERNEL_BACKEND")
    PythonKernelBackend.daily_ols = capture
    os.environ["QUANT_KERNEL_BACKEND"] = "python"
    try:
        run_exposure(context)
    finally:
        PythonKernelBackend.daily_ols = original
        if previous_backend is None:
            os.environ.pop("QUANT_KERNEL_BACKEND", None)
        else:
            os.environ["QUANT_KERNEL_BACKEND"] = previous_backend
    design = captured["design"]
    response = captured["response"]
    offsets = captured["offsets"]
    python_result, python_timing = _run_backend_micro(
        PythonKernelBackend(),
        design,
        response,
        offsets,
    )
    native_result, native_timing = _run_backend_micro(
        CppKernelBackend(),
        design,
        response,
        offsets,
    )
    matches = True
    comparison: dict[str, dict[str, Any]] = {}
    for key in ("days", "rows", "residual_rows", "active"):
        expected = np.asarray(python_result[key])
        actual = np.asarray(native_result[key])
        equal = np.array_equal(expected, actual)
        comparison[key] = {"exact": equal}
        if not equal:
            matches = False
    for key in ("coefficients", "t_stats", "r2", "residual_values"):
        expected = np.asarray(python_result[key], dtype=np.float64)
        actual = np.asarray(native_result[key], dtype=np.float64)
        if key == "t_stats" and expected.ndim == 2 and expected.shape[1] > 1:
            # 截距 t 不进入暴露 payload；风格系数的日度 t 才参与业务验收。
            expected = expected[:, 1:]
            actual = actual[:, 1:]
        close = np.isclose(
            actual,
            expected,
            atol=PAYLOAD_TOLERANCE,
            rtol=PAYLOAD_TOLERANCE,
            equal_nan=True,
        )
        finite = np.isfinite(actual) & np.isfinite(expected)
        difference = np.abs(actual[finite] - expected[finite])
        comparison[key] = {
            "exact": bool(close.all()),
            "mismatches": int((~close).sum()),
            "max_abs_diff": float(difference.max()) if difference.size else 0.0,
            "examples": [
                {
                    "index": [int(value) for value in index],
                    "python": float(expected[tuple(index)]),
                    "native": float(actual[tuple(index)]),
                }
                for index in np.argwhere(~close)[:5]
            ],
        }
        if not close.all():
            matches = False
    speedup = python_timing["median_s"] / native_timing["median_s"]
    passed = matches and speedup >= MIN_KERNEL_SPEEDUP
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "micro",
        "rows": int(design.shape[0]),
        "columns": int(design.shape[1]),
        "groups": int(offsets.size - 1),
        "tolerance": PAYLOAD_TOLERANCE,
        "python": python_timing,
        "native": native_timing,
        "match": matches,
        "comparison": comparison,
        "speedup": speedup,
        "pass": passed,
    }
    atomic_write_json(output / "result.json", result)
    atomic_write_text(
        output / "result.md",
        "\n".join([
            "# P1-07 逐日 OLS 微基准",
            "",
            f"- 输入：{design.shape[0]:,} 行、{design.shape[1]} 列、"
            f"{offsets.size - 1} 组。",
            f"- Python 中位：{python_timing['median_s']:.6f}s。",
            f"- native 中位：{native_timing['median_s']:.6f}s。",
            f"- 加速：{speedup:.3f}x。",
            f"- `{PAYLOAD_TOLERANCE:g}` 内一致：{'是' if matches else '否'}。",
            "- 日度 t 比较业务消费的风格列；截距 t 不进入暴露 payload。",
            f"- 门槛：{'通过' if passed else '未通过'}。",
            "",
        ]),
    )
    update_manifest(output)
    print(f"micro result written to {project_relative(output)}", flush=True)
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    """解析基准或 worker 参数。"""

    parser = argparse.ArgumentParser(description="P1-07 暴露链路基准")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--retained", action="store_true")
    parser.add_argument("--accepted", action="store_true")
    parser.add_argument("--micro", action="store_true")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "AiRecord/benchmarks/V118/p1-07-20261027/prechange"
        ),
    )
    parser.add_argument("--worker-backend", choices=BACKENDS)
    parser.add_argument("--worker-context", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)

    if args.worker_backend:
        if args.worker_context is None or args.worker_output is None:
            parser.error("worker 需要 --worker-context 和 --worker-output")
        _run_stage_worker(
            args.worker_backend,
            args.worker_context.resolve(),
            args.worker_output.resolve(),
        )
        return 0
    if args.output is None:
        parser.error("非 worker 模式需要 --output")
    if args.baseline_only:
        return run_baseline(args.output.resolve())
    if args.retained or args.accepted:
        return run_accepted(
            args.output.resolve(),
            args.baseline.resolve(),
        )
    if args.micro:
        return run_micro(args.output.resolve())
    parser.error("需要 --baseline-only、--accepted 或 --micro")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
