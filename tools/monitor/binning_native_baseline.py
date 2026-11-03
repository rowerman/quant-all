"""验证 P1-06 分位分组、收益、换手和自相关 C++ 核。"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

_BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "monitor"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from backend import service
from calculate.binning import (
    assign_bins,
    bin_matrix,
    rank_matrix,
)
from calculate.daily import _grouped_arrays
from calculate.evaluate import evaluate_context
from calculate.panel import build_panel
from common.paths import PROJECT_ROOT
from kernels import get_backend
from tools.monitor.baseline_contract import (
    atomic_write_json,
    atomic_write_text,
    compare_golden,
    load_golden,
    prepare_common,
    strip_volatile,
    update_manifest,
)

SCHEMA_VERSION = "binning_native_baseline_v1"
EVIDENCE_ROOT = PROJECT_ROOT / "AiRecord/benchmarks/V117/p1-06-20261027"
V105_INPUTS = PROJECT_ROOT / "AiRecord/benchmarks/V105/p0-01-20261027/inputs.json"
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
STAGES = ("bins", "bin_curves", "spread", "bin_dist", "turnover", "decay")
BACKENDS = ("python", "native", "auto")
AGGREGATE_STAGES = ("bins", "turnover")
MICRO_REPEATS = 5
STAGE_REPEATS = 3
MIN_KERNEL_SPEEDUP = 1.5
MIN_STAGE_IMPROVEMENT = 0.20
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


def _sha256_array(values: np.ndarray) -> str:
    """返回数组原始字节的 SHA-256。"""

    return hashlib.sha256(values.tobytes()).hexdigest()


def _timed_call(function: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    """测量一次冷运行和五次热运行的中位耗时。"""

    cold_started = time.perf_counter()
    result = function()
    cold_s = time.perf_counter() - cold_started
    runs = []
    for _ in range(MICRO_REPEATS):
        started = time.perf_counter()
        result = function()
        runs.append(time.perf_counter() - started)
    return result, {
        "cold_s": round(cold_s, 6),
        "runs_s": [round(value, 6) for value in runs],
        "median_s": float(np.median(runs)),
    }


def _run_micro_worker(backend: str, input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """在独立进程中测量四个新增核。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_golden(input_dir / "metadata.json")
    quantile_values = np.load(input_dir / "quantile_values.npy")
    quantile_offsets = np.load(input_dir / "quantile_offsets.npy")
    returns_values = np.load(input_dir / "returns_values.npy")
    day_codes = np.load(input_dir / "day_codes.npy")
    bin_codes = np.load(input_dir / "bin_codes.npy")
    equal_weights = np.load(input_dir / "equal_weights.npy")
    market_weights = np.load(input_dir / "market_weights.npy")
    turnover_matrix = np.load(input_dir / "turnover_matrix.npy")
    rank_values = np.load(input_dir / "rank_matrix.npy")
    lags = tuple(int(value) for value in metadata["lags"])
    selected = get_backend(backend).backend
    measurements: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}

    result, measurements["quantile_bins"] = _timed_call(
        lambda: selected.daily_quantile_bins(
            quantile_values,
            quantile_offsets,
            int(metadata["quantiles"]),
        )
    )
    arrays["quantile_bins"] = result

    result, measurements["returns_equal"] = _timed_call(
        lambda: selected.bin_returns_multi(
            returns_values,
            day_codes,
            bin_codes,
            equal_weights,
            int(metadata["day_count"]),
            int(metadata["quantiles"]),
        )
    )
    arrays["returns_equal"] = result["returns"]
    arrays["pools_equal"] = result["pools"]

    result, measurements["returns_market_cap"] = _timed_call(
        lambda: selected.bin_returns_multi(
            returns_values,
            day_codes,
            bin_codes,
            market_weights,
            int(metadata["day_count"]),
            int(metadata["quantiles"]),
        )
    )
    arrays["returns_market_cap"] = result["returns"]
    arrays["pools_market_cap"] = result["pools"]

    result, measurements["turnover"] = _timed_call(
        lambda: selected.bin_turnover(
            turnover_matrix,
            int(metadata["quantiles"]),
        )
    )
    arrays["turnover_per_bin"] = result["per_bin"]
    arrays["turnover_overall"] = result["overall"]

    def row_spearman():
        correlations = np.full(
            (len(lags), rank_values.shape[0]),
            np.nan,
            dtype="float64",
        )
        counts = np.zeros(
            (len(lags), rank_values.shape[0]),
            dtype="int64",
        )
        for index, lag in enumerate(lags):
            correlation, count = selected.daily_row_spearman(
                np.ascontiguousarray(rank_values[lag:]),
                np.ascontiguousarray(rank_values[:-lag]),
                5,
            )
            correlations[index, lag:] = correlation
            counts[index, lag:] = count
        return correlations, counts

    with RssSampler() as sampler:
        result, measurements["row_spearman"] = _timed_call(row_spearman)
    arrays["row_spearman"] = result[0]
    arrays["row_spearman_count"] = result[1]
    for name, values in arrays.items():
        np.save(output_dir / f"{name}.npy", values)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "rows": int(quantile_values.size),
        "groups": int(quantile_offsets.size - 1),
        "returns_rows": int(returns_values.shape[0]),
        "days": int(metadata["day_count"]),
        "quantiles": int(metadata["quantiles"]),
        "lags": list(lags),
        "kernels": measurements,
        "baseline_rss_mb": round(sampler.baseline_mb, 3),
        "peak_rss_mb": round(sampler.peak_mb, 3),
    }
    atomic_write_json(output_dir / "timing.json", payload)
    return payload


def _run_stage_worker(stage: str, context_path: Path, output_dir: Path) -> dict[str, Any]:
    """在固定面板上运行一个评估 stage。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    with context_path.open("rb") as stream:
        context = pickle.load(stream)
    actual_backend = get_backend(os.environ["QUANT_KERNEL_BACKEND"]).name
    with RssSampler() as sampler:
        started = time.perf_counter()
        payload = None
        for event, value in evaluate_context(context, stages=(stage,)):
            if event == stage:
                payload = value
        elapsed_s = time.perf_counter() - started
    if payload is None:
        raise RuntimeError(f"评估入口没有产出 stage {stage}")
    atomic_write_json(output_dir / "payload.json", strip_volatile(payload))
    result = {
        "schema_version": SCHEMA_VERSION,
        "backend": os.environ["QUANT_KERNEL_BACKEND"],
        "actual_backend": actual_backend,
        "stage": stage,
        "elapsed_s": round(elapsed_s, 6),
        "stage_elapsed_ms": float(payload["elapsed_ms"]),
        "baseline_rss_mb": round(sampler.baseline_mb, 3),
        "peak_rss_mb": round(sampler.peak_mb, 3),
    }
    atomic_write_json(output_dir / "timing.json", result)
    return result


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
        "tools.monitor.binning_native_baseline",
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
    """运行 worker 并在失败时保留 stderr。"""

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


def _make_micro_inputs(output: Path, context_path: Path) -> dict[str, Any]:
    """从固定三年面板生成可重复的微基准输入。"""

    context = build_panel(service.prepare(BASE_QUERY).params)
    with context_path.open("wb") as stream:
        pickle.dump(context, stream, protocol=pickle.HIGHEST_PROTOCOL)
    frame = assign_bins(
        context.panel,
        context.params.quantiles,
        label_columns=context.label_columns,
    )
    subset = context.panel.loc[
        context.panel["in_universe"] & context.panel["factor"].notna(),
        ["trade_date", "factor"],
    ]
    codes, _ = pd.factorize(subset["trade_date"], sort=True)
    offsets, (quantile_values,) = _grouped_arrays(
        codes,
        subset["factor"].to_numpy(dtype="float64"),
    )
    day_codes, days = pd.factorize(frame["trade_date"], sort=True)
    labels = [
        context.horizon_label(horizon)
        for horizon in (1, 3, 5, 10, 20)
    ]
    returns_values = np.ascontiguousarray(
        frame[labels].to_numpy(dtype="float64")
    )
    bin_codes = np.ascontiguousarray(
        frame["bin"].to_numpy(dtype="int64") - 1
    )
    equal_weights = np.ones(frame.shape[0], dtype="float64")
    market_weights = np.ascontiguousarray(
        frame["total_mv"].to_numpy(dtype="float64")
    )
    turnover_matrix = np.ascontiguousarray(
        bin_matrix(frame).to_numpy(dtype="float64") - 1.0
    )
    rank_values = np.ascontiguousarray(
        rank_matrix(frame).to_numpy(dtype="float64")
    )
    input_dir = output / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "quantile_values": quantile_values,
        "quantile_offsets": offsets,
        "returns_values": returns_values,
        "day_codes": np.ascontiguousarray(day_codes, dtype=np.int64),
        "bin_codes": bin_codes,
        "equal_weights": equal_weights,
        "market_weights": market_weights,
        "turnover_matrix": turnover_matrix,
        "rank_matrix": rank_values,
    }
    for name, values in arrays.items():
        np.save(input_dir / f"{name}.npy", values)
    metadata = {
        "quantiles": context.params.quantiles,
        "day_count": int(days.size),
        "lags": [1, 5, 20],
        "labels": labels,
        "rows": int(quantile_values.size),
        "bins_rows": int(frame.shape[0]),
        "stocks": int(turnover_matrix.shape[1]),
        "arrays": {
            name: {
                "shape": list(values.shape),
                "sha256": _sha256_array(values),
            }
            for name, values in arrays.items()
        },
    }
    atomic_write_json(input_dir / "metadata.json", metadata)
    return metadata


def _compare_micro(python_dir: Path, native_dir: Path) -> dict[str, bool]:
    """比较分位组、收益、换手和自相关输出。"""

    matches: dict[str, bool] = {}
    for name in (
        "returns_equal",
        "pools_equal",
        "returns_market_cap",
        "pools_market_cap",
        "turnover_per_bin",
        "turnover_overall",
        "row_spearman",
    ):
        try:
            np.testing.assert_allclose(
                np.load(native_dir / f"{name}.npy"),
                np.load(python_dir / f"{name}.npy"),
                rtol=1e-10,
                atol=1e-10,
                equal_nan=True,
            )
            matches[name] = True
        except AssertionError:
            matches[name] = False
    for name in ("quantile_bins", "row_spearman_count"):
        try:
            np.testing.assert_array_equal(
                np.load(native_dir / f"{name}.npy"),
                np.load(python_dir / f"{name}.npy"),
            )
            matches[name] = True
        except AssertionError:
            matches[name] = False
    return matches


def _median_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总同一后端的多次独立 stage 运行。"""

    return {
        "actual_backend": records[0]["actual_backend"],
        "elapsed_s": float(np.median([item["elapsed_s"] for item in records])),
        "stage_elapsed_ms": float(
            np.median([item["stage_elapsed_ms"] for item in records])
        ),
        "peak_rss_mb": max(float(item["peak_rss_mb"]) for item in records),
        "runs": records,
    }


def summarize_gate(
    python_micro: dict[str, Any],
    native_micro: dict[str, Any],
    micro_match: dict[str, bool],
    python_stages: dict[str, dict[str, Any]],
    native_stages: dict[str, dict[str, Any]],
    auto_stages: dict[str, dict[str, Any]],
    golden_matches: dict[str, dict[str, bool]],
) -> dict[str, Any]:
    """汇总核、stage、内存和 auto 选择门禁。"""

    kernel_names = (
        "quantile_bins",
        "returns_equal",
        "returns_market_cap",
        "turnover",
        "row_spearman",
    )
    speedups = {
        name: (
            python_micro["kernels"][name]["median_s"]
            / native_micro["kernels"][name]["median_s"]
        )
        for name in kernel_names
    }
    stage_speedups = {
        stage: (
            python_stages[stage]["stage_elapsed_ms"]
            / native_stages[stage]["stage_elapsed_ms"]
        )
        for stage in STAGES
    }
    python_total = sum(
        python_stages[stage]["stage_elapsed_ms"]
        for stage in AGGREGATE_STAGES
    )
    native_total = sum(
        native_stages[stage]["stage_elapsed_ms"]
        for stage in AGGREGATE_STAGES
    )
    aggregate_speedup = python_total / native_total
    rss_ratios = {
        backend: {
            stage: (
                records[stage]["peak_rss_mb"]
                / python_stages[stage]["peak_rss_mb"]
            )
            for stage in STAGES
        }
        for backend, records in (("native", native_stages), ("auto", auto_stages))
    }
    auto_selects_native = all(
        records["actual_backend"] == "native"
        for records in auto_stages.values()
    )
    kernel_pass = all(micro_match.values()) and all(
        speedups[name] >= MIN_KERNEL_SPEEDUP
        for name in (
            "quantile_bins",
            "returns_equal",
            "returns_market_cap",
            "turnover",
            "row_spearman",
        )
    )
    native_not_worse_than_python_gold = all(
        matches["native"] or not matches["python"]
        for matches in golden_matches.values()
    )
    stage_pass = (
        aggregate_speedup >= 1.0 / (1.0 - MIN_STAGE_IMPROVEMENT)
        and all(speedup >= 1.0 for speedup in stage_speedups.values())
        and native_not_worse_than_python_gold
    )
    rss_pass = all(
        ratio <= MAX_RSS_RATIO
        for stage_ratios in rss_ratios.values()
        for ratio in stage_ratios.values()
    )
    return {
        "kernel_speedups": speedups,
        "micro_matches": micro_match,
        "kernel_pass": kernel_pass,
        "stage_speedups": stage_speedups,
        "aggregate_stages": list(AGGREGATE_STAGES),
        "aggregate_speedup": aggregate_speedup,
        "stage_improvement": 1.0 - 1.0 / aggregate_speedup,
        "stage_pass": stage_pass,
        "golden_matches": golden_matches,
        "native_not_worse_than_python_gold": native_not_worse_than_python_gold,
        "rss_ratios": rss_ratios,
        "rss_pass": rss_pass,
        "auto_selects_native": auto_selects_native,
        "gate_pass": bool(
            kernel_pass and stage_pass and rss_pass and auto_selects_native
        ),
    }


def run_suite(output: Path) -> int:
    """运行固定微基准和六个相关评估 stage。"""

    output.mkdir(parents=True, exist_ok=True)
    common = prepare_common(output, refresh=True)
    expected = load_golden(V105_INPUTS)
    if common["inputs"]["input_tree_sha256"] != expected["input_tree_sha256"]:
        raise RuntimeError("输入数据相对 V105 已漂移，停止性能对照")
    evaluation_golden = load_golden(
        PROJECT_ROOT
        / "AiRecord/benchmarks/V105/p0-01-20261027/evaluation.golden.json"
    )
    golden_events = {
        event["stage"]: event["payload"]
        for event in evaluation_golden["golden"]["events"]
    }

    workers = output / "workers"
    with TemporaryDirectory(prefix="quant-binning-context-") as directory:
        context_path = Path(directory) / "context.pkl"
        metadata = _make_micro_inputs(output, context_path)
        micro_results: dict[str, dict[str, Any]] = {}
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
        micro_match = _compare_micro(
            workers / "micro" / "python",
            workers / "micro" / "native",
        )

        stage_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
        golden_matches: dict[str, dict[str, bool]] = {}
        for stage in STAGES:
            stage_records[stage] = {}
            for backend in BACKENDS:
                records = []
                for repeat in range(STAGE_REPEATS):
                    target = (
                        workers
                        / "stages"
                        / stage
                        / backend
                        / f"run-{repeat + 1}"
                    )
                    _run_child(
                        _subprocess_command(
                            backend=backend,
                            output_dir=target,
                            stage=stage,
                            context_path=context_path,
                        ),
                        backend,
                    )
                    records.append(load_golden(target / "timing.json"))
                stage_records[stage][backend] = records
            for backend in ("native", "auto"):
                compare_golden(
                    load_golden(
                        workers / "stages" / stage / "python" / "run-1" / "payload.json"
                    ),
                    load_golden(
                        workers / "stages" / stage / backend / "run-1" / "payload.json"
                    ),
                    abs_tol=1e-10,
                    rel_tol=1e-10,
                )
            expected_payload = dict(golden_events[stage])
            expected_payload.pop("factor", None)
            golden_matches[stage] = {}
            for backend in ("python", "native"):
                try:
                    compare_golden(
                        expected_payload,
                        load_golden(
                            workers
                            / "stages"
                            / stage
                            / backend
                            / "run-1"
                            / "payload.json"
                        ),
                        abs_tol=1e-10,
                        rel_tol=1e-10,
                    )
                    golden_matches[stage][backend] = True
                except Exception:
                    golden_matches[stage][backend] = False

        stage_results = {
            stage: {
                backend: _median_record(records)
                for backend, records in backends.items()
            }
            for stage, backends in stage_records.items()
        }
        gate = summarize_gate(
            micro_results["python"],
            micro_results["native"],
            micro_match,
            {
                name: records["python"]
                for name, records in stage_results.items()
            },
            {
                name: records["native"]
                for name, records in stage_results.items()
            },
            {
                name: records["auto"]
                for name, records in stage_results.items()
            },
            golden_matches,
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "input": {
                "input_tree_sha256": common["inputs"]["input_tree_sha256"],
                "query": BASE_QUERY,
                "metadata": metadata,
            },
            "micro": micro_results,
            "stages": stage_results,
            "gate": gate,
        }
        atomic_write_json(output / "result.json", result)
        _write_report(output / "result.md", result)
    update_manifest(output)
    return 0 if gate["gate_pass"] else 1


def _write_report(path: Path, result: dict[str, Any]) -> None:
    """写出人可读的性能与门禁报告。"""

    gate = result["gate"]
    lines = [
        "# P1-06 分位收益、换手和自相关 C++ 核",
        "",
        f"- Schema：`{SCHEMA_VERSION}`",
        f"- 门禁：`{'通过' if gate['gate_pass'] else '未通过'}`",
        f"- `bins` + `turnover` 合计加速：`{gate['aggregate_speedup']:.3f}x`",
        f"- 合计改善：`{gate['stage_improvement']:.2%}`",
        "",
        "## 核加速",
        "",
        "| 核 | Python | native | 加速 |",
        "|---|---:|---:|---:|",
    ]
    python_micro = result["micro"]["python"]["kernels"]
    native_micro = result["micro"]["native"]["kernels"]
    for name, speedup in gate["kernel_speedups"].items():
        lines.append(
            f"| `{name}` | {python_micro[name]['median_s']:.6f}s | "
            f"{native_micro[name]['median_s']:.6f}s | {speedup:.3f}x |"
        )
    lines.extend([
        "",
        "## Stage",
        "",
        "| Stage | Python | native | 加速 |",
        "|---|---:|---:|---:|",
    ])
    for stage, speedup in gate["stage_speedups"].items():
        lines.append(
            f"| `{stage}` | "
            f"{result['stages'][stage]['python']['stage_elapsed_ms'] / 1000:.3f}s | "
            f"{result['stages'][stage]['native']['stage_elapsed_ms'] / 1000:.3f}s | "
            f"{speedup:.3f}x |"
        )
    lines.extend([
        "",
        "## 结论",
        "",
        f"- 微基准对照：`{gate['micro_matches']}`",
        f"- 核门槛：`{gate['kernel_pass']}`",
        f"- Stage 门槛：`{gate['stage_pass']}`",
        f"- V105 金标准：`{gate['golden_matches']}`",
        f"- native 未劣于 Python 基线：`{gate['native_not_worse_than_python_gold']}`",
        f"- RSS 门槛：`{gate['rss_pass']}`",
        f"- auto 选择 native：`{gate['auto_selects_native']}`",
        "",
    ])
    atomic_write_text(path, "\n".join(lines))


def _parse_args() -> argparse.Namespace:
    """解析正式运行和子进程 worker 参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--worker-backend", choices=BACKENDS)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker-stage", choices=STAGES)
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--worker-context", type=Path)
    return parser.parse_args()


def main() -> int:
    """运行基准套件或一个 worker。"""

    args = _parse_args()
    if args.worker_backend is None:
        return run_suite(args.output)
    if args.worker_output is None:
        raise ValueError("worker 缺少 --worker-output")
    if args.worker_stage is not None:
        if args.worker_context is None:
            raise ValueError("stage worker 缺少 --worker-context")
        _run_stage_worker(
            args.worker_stage,
            args.worker_context,
            args.worker_output,
        )
    else:
        if args.worker_input is None:
            raise ValueError("微基准 worker 缺少 --worker-input")
        _run_micro_worker(
            args.worker_backend,
            args.worker_input,
            args.worker_output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
