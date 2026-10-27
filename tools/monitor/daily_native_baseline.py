"""验证逐日排名与 Pearson C++ 核的正确性、速度和真实 stage 收益。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from typing import Any

import numpy as np
import pandas as pd

from calculate.daily import _grouped_arrays
from calculate.panel import build_panel
from calculate.evaluate import evaluate_context
from common.paths import PROJECT_ROOT
from kernels import get_backend
from tools.monitor.baseline_contract import (
    atomic_write_json,
    atomic_write_text,
    compare_golden,
    load_golden,
    prepare_common,
    project_relative,
    strip_volatile,
    update_manifest,
)

_BACKEND_ROOT = PROJECT_ROOT / "apps" / "monitor"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from backend import service

SCHEMA_VERSION = "daily_native_baseline_v1"
EVIDENCE_ROOT = PROJECT_ROOT / "AiRecord/benchmarks/V115/p1-04-20261027"
V105_INPUTS = (
    PROJECT_ROOT
    / "AiRecord/benchmarks/V105/p0-01-20261027/inputs.json"
)
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
STAGES = ("overview", "decay", "incremental_ic")
BACKENDS = ("python", "native", "auto")
REPEATS = 5
MIN_KERNEL_SPEEDUP = 1.5
MIN_STAGE_SPEEDUP = 1.2
MAX_RSS_RATIO = 1.2


def _current_rss_mb() -> float:
    """从 /proc 读取当前常驻内存，单位 MB。"""

    page_size = os.sysconf("SC_PAGE_SIZE")
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * page_size / (1024 * 1024)


class RssSampler:
    """后台采样当前进程峰值 RSS。"""

    def __init__(self, interval_s: float = 0.002):
        self.interval_s = interval_s
        self.baseline_mb = _current_rss_mb()
        self.peak_mb = self.baseline_mb
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.peak_mb = max(self.peak_mb, _current_rss_mb())

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self.peak_mb = max(self.peak_mb, _current_rss_mb())
        self._stop.set()
        self._thread.join(timeout=5)


def _worker_environment(backend: str) -> dict[str, str]:
    """返回固定后端和单线程的独立进程环境。"""

    return {
        **os.environ,
        "QUANT_KERNEL_BACKEND": backend,
        "QUANT_NATIVE_THREADS": "1",
    }


def _run_worker(
    backend: str,
    *,
    stage: str | None = None,
    input_dir: Path | None = None,
    context_path: Path | None = None,
    output_dir: Path,
) -> dict[str, Any]:
    """在独立进程运行微基准或一个真实 stage。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    if stage is not None:
        if context_path is None:
            raise ValueError("真实 stage 缺少固定面板")
        return _run_stage_worker(stage, context_path, output_dir)
    if input_dir is None:
        raise ValueError("微基准缺少输入目录")
    return _run_micro_worker(backend, input_dir, output_dir)


def _run_stage_worker(
    stage: str,
    context_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """在共享固定面板上运行一个 stage 并保存稳定 payload。"""

    with context_path.open("rb") as stream:
        context = pickle.load(stream)
    with RssSampler() as sampler:
        started = time.perf_counter()
        payload = None
        for event, value in evaluate_context(context, stages=(stage,)):
            if event == stage:
                payload = value
        elapsed_s = time.perf_counter() - started
    if payload is None:
        raise RuntimeError(f"评估入口没有产出 stage {stage}")
    stable_payload = strip_volatile(payload)
    atomic_write_json(output_dir / "payload.json", stable_payload)
    result = {
        "schema_version": SCHEMA_VERSION,
        "backend": os.environ["QUANT_KERNEL_BACKEND"],
        "stage": stage,
        "elapsed_s": round(elapsed_s, 6),
        "stage_elapsed_ms": float(payload["elapsed_ms"]),
        "baseline_rss_mb": round(sampler.baseline_mb, 3),
        "peak_rss_mb": round(sampler.peak_mb, 3),
    }
    atomic_write_json(output_dir / "timing.json", result)
    return result


def _run_micro_worker(
    backend: str,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """分别测量逐日排名和 Pearson 的热运行。"""

    left = np.load(input_dir / "left.npy")
    right = np.load(input_dir / "right.npy")
    offsets = np.load(input_dir / "offsets.npy")
    import_started = time.perf_counter()
    selected = get_backend(backend).backend
    import_s = time.perf_counter() - import_started

    with RssSampler() as sampler:
        rank_cold_started = time.perf_counter()
        rank_result = selected.daily_rank_pct(left, offsets)
        rank_cold_s = time.perf_counter() - rank_cold_started
        pearson_cold_started = time.perf_counter()
        correlation, count = selected.daily_pearson(left, right, offsets, 5)
        pearson_cold_s = time.perf_counter() - pearson_cold_started

        rank_runs = []
        pearson_runs = []
        for _ in range(REPEATS):
            started = time.perf_counter()
            rank_result = selected.daily_rank_pct(left, offsets)
            rank_runs.append(time.perf_counter() - started)
            started = time.perf_counter()
            correlation, count = selected.daily_pearson(left, right, offsets, 5)
            pearson_runs.append(time.perf_counter() - started)

    np.save(output_dir / "rank.npy", rank_result)
    np.save(output_dir / "correlation.npy", correlation)
    np.save(output_dir / "count.npy", count)
    result = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "rows": int(left.size),
        "groups": int(offsets.size - 1),
        "import_s": round(import_s, 6),
        "rank": {
            "cold_s": round(rank_cold_s, 6),
            "runs_s": [round(value, 6) for value in rank_runs],
            "median_s": float(np.median(rank_runs)),
        },
        "pearson": {
            "cold_s": round(pearson_cold_s, 6),
            "runs_s": [round(value, 6) for value in pearson_runs],
            "median_s": float(np.median(pearson_runs)),
        },
        "baseline_rss_mb": round(sampler.baseline_mb, 3),
        "peak_rss_mb": round(sampler.peak_mb, 3),
    }
    atomic_write_json(output_dir / "timing.json", result)
    return result


def _make_micro_inputs(
    output: Path,
    context_path: Path,
) -> dict[str, Any]:
    """从固定三年面板生成微基准输入。"""

    query = {**BASE_QUERY, "stages": "overview"}
    context = build_panel(service.prepare(query).params)
    with context_path.open("wb") as stream:
        pickle.dump(context, stream, protocol=pickle.HIGHEST_PROTOCOL)
    work = context.panel.loc[
        context.panel["in_universe"],
        ["trade_date", "factor", "label"],
    ].dropna()
    work = work.loc[
        np.isfinite(work["factor"]) & np.isfinite(work["label"])
    ]
    codes, dates = pd.factorize(work["trade_date"], sort=True)
    input_dir = output / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    offsets, (left, right) = _grouped_arrays(
        codes,
        work["factor"].to_numpy(dtype=np.float64),
        work["label"].to_numpy(dtype=np.float64),
    )
    np.save(input_dir / "left.npy", left)
    np.save(input_dir / "right.npy", right)
    np.save(input_dir / "offsets.npy", offsets)
    return {
        "rows": int(left.size),
        "groups": int(len(dates)),
        "left_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
        "right_sha256": hashlib.sha256(right.tobytes()).hexdigest(),
        "offsets_sha256": hashlib.sha256(offsets.tobytes()).hexdigest(),
    }


def _subprocess_command(
    *,
    backend: str,
    output_dir: Path,
    stage: str | None = None,
    input_dir: Path | None = None,
    context_path: Path | None = None,
) -> list[str]:
    """构造独立 worker 命令。"""

    command = [
        sys.executable,
        "-m",
        "tools.monitor.daily_native_baseline",
        "--worker-backend",
        backend,
        "--worker-output",
        str(output_dir),
    ]
    if stage is not None:
        command.extend(["--worker-stage", stage])
    if input_dir is not None:
        command.extend(["--worker-input", str(input_dir)])
    if context_path is not None:
        command.extend(["--worker-context", str(context_path)])
    return command


def _run_child(command: list[str], backend: str) -> None:
    """运行独立 worker 并在失败时保留错误信息。"""

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_worker_environment(backend),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"worker 失败：{completed.stderr or completed.stdout}"
        )


def _compare_micro(
    python_dir: Path,
    native_dir: Path,
) -> tuple[bool, bool]:
    """比较排名、Pearson 和有效计数。"""

    try:
        np.testing.assert_array_equal(
            np.load(native_dir / "rank.npy"),
            np.load(python_dir / "rank.npy"),
        )
        rank_exact = True
    except AssertionError:
        rank_exact = False
    try:
        np.testing.assert_allclose(
            np.load(native_dir / "correlation.npy"),
            np.load(python_dir / "correlation.npy"),
            atol=1e-10,
            rtol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            np.load(native_dir / "count.npy"),
            np.load(python_dir / "count.npy"),
        )
        pearson_exact = True
    except AssertionError:
        pearson_exact = False
    return rank_exact, pearson_exact


def _stage_speedup(python: dict[str, Any], native: dict[str, Any]) -> float:
    """按 stage 自报耗时计算加速比。"""

    return float(python["stage_elapsed_ms"]) / float(native["stage_elapsed_ms"])


def run_suite(output: Path) -> int:
    """运行固定微基准和三条真实业务路径。"""

    output.mkdir(parents=True, exist_ok=True)
    common = prepare_common(output, refresh=True)
    expected = load_golden(V105_INPUTS)
    if common["inputs"]["input_tree_sha256"] != expected["input_tree_sha256"]:
        raise RuntimeError("输入数据相对 V105 已漂移，停止性能对照")

    workers = output / "workers"
    with TemporaryDirectory(prefix="quant-daily-context-") as directory:
        context_path = Path(directory) / "context.pkl"
        micro_inputs = _make_micro_inputs(output, context_path)
        micro_results = {}
        for backend in ("python", "native"):
            target = workers / "micro" / backend
            _run_child(
                _subprocess_command(
                    backend=backend,
                    output_dir=target,
                    input_dir=output / "inputs",
                ),
                backend,
            )
            micro_results[backend] = load_golden(target / "timing.json")

        rank_exact, pearson_exact = _compare_micro(
            workers / "micro" / "python",
            workers / "micro" / "native",
        )
        rank_speedup = (
            micro_results["python"]["rank"]["median_s"]
            / micro_results["native"]["rank"]["median_s"]
        )
        pearson_speedup = (
            micro_results["python"]["pearson"]["median_s"]
            / micro_results["native"]["pearson"]["median_s"]
        )
        rank_pass = rank_exact and rank_speedup >= MIN_KERNEL_SPEEDUP
        pearson_pass = pearson_exact and pearson_speedup >= MIN_KERNEL_SPEEDUP

        stage_results: dict[str, dict[str, Any]] = {}
        payload_match: dict[str, bool] = {}
        for stage in STAGES:
            stage_results[stage] = {}
            for backend in BACKENDS:
                target = workers / "stages" / stage / backend
                _run_child(
                    _subprocess_command(
                        backend=backend,
                        output_dir=target,
                        stage=stage,
                        context_path=context_path,
                    ),
                    backend,
                )
                stage_results[stage][backend] = load_golden(
                    target / "timing.json"
                )
            try:
                compare_golden(
                    load_golden(
                        workers / "stages" / stage / "python" / "payload.json"
                    ),
                    load_golden(
                        workers / "stages" / stage / "native" / "payload.json"
                    ),
                    abs_tol=1e-10,
                    rel_tol=1e-10,
                )
                compare_golden(
                    load_golden(
                        workers / "stages" / stage / "python" / "payload.json"
                    ),
                    load_golden(
                        workers / "stages" / stage / "auto" / "payload.json"
                    ),
                    abs_tol=1e-10,
                    rel_tol=1e-10,
                )
                payload_match[stage] = True
            except AssertionError:
                payload_match[stage] = False

    overview_speedup = _stage_speedup(
        stage_results["overview"]["python"],
        stage_results["overview"]["native"],
    )
    decay_speedup = _stage_speedup(
        stage_results["decay"]["python"],
        stage_results["decay"]["native"],
    )
    incremental_speedup = _stage_speedup(
        stage_results["incremental_ic"]["python"],
        stage_results["incremental_ic"]["native"],
    )
    rss_ratios = {
        backend: {
            stage: (
                stage_results[stage][backend]["peak_rss_mb"]
                / stage_results[stage]["python"]["peak_rss_mb"]
            )
            for stage in STAGES
        }
        for backend in ("native", "auto")
    }
    stage_pass = (
        all(payload_match.values())
        and overview_speedup >= MIN_STAGE_SPEEDUP
        and decay_speedup >= MIN_STAGE_SPEEDUP
        and incremental_speedup >= 1.0
    )
    rss_pass = all(
        ratio <= MAX_RSS_RATIO
        for backend_ratios in rss_ratios.values()
        for ratio in backend_ratios.values()
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "input_tree_sha256": common["inputs"]["input_tree_sha256"],
            "query": BASE_QUERY,
            "micro": micro_inputs,
            "threads": 1,
            "repeats": REPEATS,
        },
        "micro": {
            "python": micro_results["python"],
            "native": micro_results["native"],
            "rank_exact": rank_exact,
            "pearson_exact": pearson_exact,
            "rank_speedup": rank_speedup,
            "pearson_speedup": pearson_speedup,
            "rank_pass": rank_pass,
            "pearson_pass": pearson_pass,
        },
        "stages": stage_results,
        "stage_speedup": {
            "overview": overview_speedup,
            "decay": decay_speedup,
            "incremental_ic": incremental_speedup,
        },
        "payload_match": payload_match,
        "rss_ratios": rss_ratios,
        "gate": {
            "micro_pass": rank_pass and pearson_pass,
            "stage_pass": stage_pass,
            "rss_pass": rss_pass,
            "pass": rank_pass and pearson_pass and stage_pass and rss_pass,
            "cpp_modules_replaced": (
                2
                if rank_pass and pearson_pass and stage_pass and rss_pass
                else 0
            ),
        },
    }
    atomic_write_json(output / "result.json", result)
    lines = [
        "# P1-04 逐日排名与 Pearson C++ 核基准",
        "",
        "## 固定口径",
        "",
        f"- 输入树 SHA-256：`{common['inputs']['input_tree_sha256']}`。",
        f"- 微基准：{micro_inputs['rows']:,} 行、{micro_inputs['groups']} 组，"
        "单线程，热运行 5 次取中位数。",
        "- 真实 stage 在独立进程中分别运行 Python、native、auto，共享同一固定面板；",
        "  只比较目标 stage 自身耗时，避免重新建面板引入无关浮点漂移。",
        "",
        "## 核函数",
        "",
        "| 核 | Python 中位 | C++ 中位 | 加速 | 正确 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| 逐日排名 | {micro_results['python']['rank']['median_s']:.6f}s | "
            f"{micro_results['native']['rank']['median_s']:.6f}s | "
            f"{rank_speedup:.3f}x | {'是' if rank_exact else '否'} |"
        ),
        (
            f"| 逐日 Pearson | {micro_results['python']['pearson']['median_s']:.6f}s | "
            f"{micro_results['native']['pearson']['median_s']:.6f}s | "
            f"{pearson_speedup:.3f}x | {'是' if pearson_exact else '否'} |"
        ),
        "",
        "## 真实 stage",
        "",
        "| stage | Python | native | auto | native 加速 | payload一致 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        values = stage_results[stage]
        lines.append(
            f"| {stage} | {float(values['python']['stage_elapsed_ms']) / 1000:.3f}s | "
            f"{float(values['native']['stage_elapsed_ms']) / 1000:.3f}s | "
            f"{float(values['auto']['stage_elapsed_ms']) / 1000:.3f}s | "
            f"{result['stage_speedup'][stage]:.3f}x | "
            f"{'是' if payload_match[stage] else '否'} |"
        )
    lines.extend(
        [
            "",
            f"- 门禁：{'通过' if result['gate']['pass'] else '未通过'}。",
            f"- 微基准：{'通过' if result['gate']['micro_pass'] else '未通过'}；"
            f"stage：{'通过' if stage_pass else '未通过'}；"
            f"RSS：{'通过' if rss_pass else '未通过'}。",
            "- 增量 IC 以不退化作为业务门槛；该 stage 主要耗时仍来自风格回归。",
            "- 本轮满足条件时按两个原生核分别登记，实际 C++ 替换模块数为 2。",
            "",
        ]
    )
    atomic_write_text(output / "result.md", "\n".join(lines))
    update_manifest(output)
    print(
        json.dumps(
            {
                "output": project_relative(output),
                "rank_speedup": rank_speedup,
                "pearson_speedup": pearson_speedup,
                "stage_speedup": result["stage_speedup"],
                "rss_ratios": rss_ratios,
                "gate_pass": result["gate"]["pass"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if result["gate"]["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    """解析命令；默认运行完整套件。"""

    parser = argparse.ArgumentParser(description="验证逐日排名与 Pearson C++ 核")
    parser.add_argument("--output", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--worker-backend", choices=BACKENDS)
    parser.add_argument("--worker-stage", choices=STAGES)
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--worker-context", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()

    if args.worker_backend:
        if args.worker_output is None:
            parser.error("worker 需要 --worker-output")
        _run_worker(
            args.worker_backend,
            stage=args.worker_stage,
            input_dir=args.worker_input,
            context_path=args.worker_context,
            output_dir=args.worker_output.resolve(),
        )
        return 0
    return run_suite(output)


if __name__ == "__main__":
    raise SystemExit(main())
