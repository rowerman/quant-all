"""验证逐日 MAD、z-score 与融合标准化 C++ 核的收益。"""

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

_BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "monitor"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from backend import service
from calculate.evaluate import evaluate_context
from calculate.panel import build_panel
from common.paths import PROJECT_ROOT
from kernels import get_backend
from tools.monitor.baseline_contract import (
    atomic_write_json,
    atomic_write_text,
    compare_golden,
    load_golden,
    project_relative,
    strip_volatile,
    update_manifest,
    prepare_common,
)

SCHEMA_VERSION = "standardize_native_baseline_v1"
EVIDENCE_ROOT = PROJECT_ROOT / "AiRecord/benchmarks/V116/p1-05-20261027"
V105_INPUTS = PROJECT_ROOT / "AiRecord/benchmarks/V105/p0-01-20261027/inputs.json"
V115_INPUTS = PROJECT_ROOT / "AiRecord/benchmarks/V115/p1-04-20261027/inputs"
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
STAGES = ("overview", "exposure")
BACKENDS = ("python", "native", "auto")
KERNELS = (
    "daily_winsorize_mad",
    "daily_zscore",
    "daily_winsorize_zscore",
)
MICRO_REPEATS = 5
STAGE_REPEATS = 3
MIN_KERNEL_SPEEDUP = 1.5
MAX_RSS_RATIO = 1.2


def _current_rss_mb() -> float:
    """从 /proc 读取当前进程常驻内存。"""

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


def _run_micro_worker(
    backend: str,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """在独立进程测量三个标准化核。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.load(input_dir / "left.npy")
    offsets = np.load(input_dir / "offsets.npy")
    import_started = time.perf_counter()
    selected = get_backend(backend).backend
    import_s = time.perf_counter() - import_started
    measurements: dict[str, dict[str, Any]] = {}

    with RssSampler() as sampler:
        for name in KERNELS:
            call = getattr(selected, name)
            cold_started = time.perf_counter()
            result = call(values, offsets)
            cold_s = time.perf_counter() - cold_started
            runs = []
            for _ in range(MICRO_REPEATS):
                started = time.perf_counter()
                result = call(values, offsets)
                runs.append(time.perf_counter() - started)
            np.save(output_dir / f"{name}.npy", result)
            measurements[name] = {
                "cold_s": round(cold_s, 6),
                "runs_s": [round(value, 6) for value in runs],
                "median_s": float(np.median(runs)),
            }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "rows": int(values.size),
        "groups": int(offsets.size - 1),
        "import_s": round(import_s, 6),
        "kernels": measurements,
        "baseline_rss_mb": round(sampler.baseline_mb, 3),
        "peak_rss_mb": round(sampler.peak_mb, 3),
    }
    atomic_write_json(output_dir / "timing.json", payload)
    return payload


def _run_stage_worker(
    backend: str,
    stage: str,
    context_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """在独立进程运行一个真实评估 stage。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    with context_path.open("rb") as stream:
        context = pickle.load(stream)
    actual_backend = get_backend(backend).name
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
        "backend": backend,
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
    """构造 worker 命令。"""

    command = [
        sys.executable,
        "-m",
        "tools.monitor.standardize_native_baseline",
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
        raise RuntimeError(
            f"worker 失败：{completed.stderr or completed.stdout}"
        )


def _compare_micro(python_dir: Path, native_dir: Path) -> dict[str, bool]:
    """比较三个标准化核的数组结果。"""

    matches: dict[str, bool] = {}
    for name in KERNELS:
        try:
            np.testing.assert_allclose(
                np.load(native_dir / f"{name}.npy"),
                np.load(python_dir / f"{name}.npy"),
                atol=1e-10,
                rtol=1e-10,
                equal_nan=True,
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


def _sha256_array(values: np.ndarray) -> str:
    """返回数组字节的 SHA-256。"""

    return hashlib.sha256(values.tobytes()).hexdigest()


def run_suite(output: Path) -> int:
    """运行核微基准和两条真实 stage，并写出机器可读门禁。"""

    output.mkdir(parents=True, exist_ok=True)
    common = prepare_common(output, refresh=True)
    expected = load_golden(V105_INPUTS)
    input_hash = common["inputs"]["input_tree_sha256"]
    if input_hash != expected["input_tree_sha256"]:
        raise RuntimeError("输入数据相对 V105 已漂移，停止性能对照")
    if not (V115_INPUTS / "left.npy").is_file():
        raise RuntimeError("缺少 V115 固定标准化输入")

    workers = output / "workers"
    micro_results: dict[str, dict[str, Any]] = {}
    for backend in ("python", "native"):
        target = workers / "micro" / backend
        _run_child(
            _subprocess_command(
                backend=backend,
                output_dir=target,
                input_dir=V115_INPUTS,
            ),
            backend,
        )
        micro_results[backend] = load_golden(target / "timing.json")
    micro_match = _compare_micro(
        workers / "micro" / "python",
        workers / "micro" / "native",
    )
    micro_speedups = {
        name: (
            micro_results["python"]["kernels"][name]["median_s"]
            / micro_results["native"]["kernels"][name]["median_s"]
        )
        for name in KERNELS
    }
    micro_pass = (
        all(micro_match.values())
        and micro_speedups["daily_winsorize_mad"] >= MIN_KERNEL_SPEEDUP
        and micro_speedups["daily_zscore"] >= MIN_KERNEL_SPEEDUP
        and micro_speedups["daily_winsorize_zscore"] >= 1.0
    )

    with TemporaryDirectory(prefix="quant-standardize-context-") as directory:
        context_path = Path(directory) / "context.pkl"
        context = build_panel(service.prepare(BASE_QUERY).params)
        with context_path.open("wb") as stream:
            pickle.dump(context, stream, protocol=pickle.HIGHEST_PROTOCOL)

        stage_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for stage in STAGES:
            stage_records[stage] = {}
            for backend in BACKENDS:
                stage_records[stage][backend] = []
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
                    stage_records[stage][backend].append(
                        load_golden(target / "timing.json")
                    )
            for backend in BACKENDS:
                target = (
                    workers
                    / "stages"
                    / stage
                    / backend
                    / "run-1"
                    / "payload.json"
                )
                if backend == "python":
                    continue
                compare_golden(
                    load_golden(
                        workers
                        / "stages"
                        / stage
                        / "python"
                        / "run-1"
                        / "payload.json"
                    ),
                    load_golden(target),
                    abs_tol=1e-10,
                    rel_tol=1e-10,
                )

    stage_results = {
        stage: {
            backend: _median_record(records)
            for backend, records in backends.items()
        }
        for stage, backends in stage_records.items()
    }
    stage_speedups = {
        stage: (
            stage_results[stage]["python"]["stage_elapsed_ms"]
            / stage_results[stage]["native"]["stage_elapsed_ms"]
        )
        for stage in STAGES
    }
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
    auto_selects_native = all(
        stage_results[stage]["auto"]["actual_backend"] == "native"
        for stage in STAGES
    )
    stage_pass = auto_selects_native and any(
        speedup > 1.0 for speedup in stage_speedups.values()
    )
    rss_pass = all(
        ratio <= MAX_RSS_RATIO
        for backend_ratios in rss_ratios.values()
        for ratio in backend_ratios.values()
    )
    passed = micro_pass and stage_pass and rss_pass
    values = np.load(V115_INPUTS / "left.npy")
    offsets = np.load(V115_INPUTS / "offsets.npy")
    result = {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "input_tree_sha256": input_hash,
            "query": BASE_QUERY,
            "rows": int(values.size),
            "groups": int(offsets.size - 1),
            "left_sha256": _sha256_array(values),
            "offsets_sha256": _sha256_array(offsets),
            "threads": 1,
            "micro_repeats": MICRO_REPEATS,
            "stage_repeats": STAGE_REPEATS,
        },
        "micro": {
            "python": micro_results["python"],
            "native": micro_results["native"],
            "match": micro_match,
            "speedups": micro_speedups,
            "pass": micro_pass,
        },
        "stages": stage_results,
        "stage_speedups": stage_speedups,
        "rss_ratios": rss_ratios,
        "auto_selects_native": auto_selects_native,
        "gate": {
            "micro_pass": micro_pass,
            "stage_pass": stage_pass,
            "rss_pass": rss_pass,
            "pass": passed,
            "cpp_modules_replaced": 2 if passed else 0,
        },
    }
    atomic_write_json(output / "result.json", result)

    lines = [
        "# P1-05 逐日 MAD 与 z-score C++ 核基准",
        "",
        "## 固定口径",
        "",
        f"- 输入树 SHA-256：`{input_hash}`。",
        f"- 微基准：{values.size:,} 行、{offsets.size - 1} 组，单线程，热运行 "
        f"{MICRO_REPEATS} 次取中位数。",
        f"- `overview` 和 `exposure` 在独立进程中分别重复 {STAGE_REPEATS} 次并取中位数。",
        "",
        "## 核函数",
        "",
        "| 核 | Python 中位 | C++ 中位 | 加速 | 正确 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in KERNELS:
        lines.append(
            f"| {name} | "
            f"{micro_results['python']['kernels'][name]['median_s']:.6f}s | "
            f"{micro_results['native']['kernels'][name]['median_s']:.6f}s | "
            f"{micro_speedups[name]:.3f}x | "
            f"{'是' if micro_match[name] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 真实 stage",
            "",
            "| stage | Python | native | auto | native 加速 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for stage in STAGES:
        record = stage_results[stage]
        lines.append(
            f"| {stage} | "
            f"{record['python']['stage_elapsed_ms'] / 1000:.3f}s | "
            f"{record['native']['stage_elapsed_ms'] / 1000:.3f}s | "
            f"{record['auto']['stage_elapsed_ms'] / 1000:.3f}s | "
            f"{stage_speedups[stage]:.3f}x |"
        )
    lines.extend(
        [
            "",
            f"- 门禁：{'通过' if passed else '未通过'}。",
            f"- 微基准：{'通过' if micro_pass else '未通过'}；"
            f"stage：{'通过' if stage_pass else '未通过'}；"
            f"RSS：{'通过' if rss_pass else '未通过'}。",
            f"- 实际新增 C++ 替换模块数：{result['gate']['cpp_modules_replaced']}。",
            "",
        ]
    )
    atomic_write_text(output / "result.md", "\n".join(lines))
    update_manifest(output)
    print(
        json.dumps(
            {
                "output": project_relative(output),
                "micro_speedups": micro_speedups,
                "stage_speedups": stage_speedups,
                "rss_ratios": rss_ratios,
                "gate_pass": passed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    """解析命令；默认运行完整套件。"""

    parser = argparse.ArgumentParser(
        description="验证逐日 MAD、z-score 与融合标准化 C++ 核"
    )
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
        if args.worker_stage:
            if args.worker_context is None:
                parser.error("stage worker 需要 --worker-context")
            _run_stage_worker(
                args.worker_backend,
                args.worker_stage,
                args.worker_context.resolve(),
                args.worker_output.resolve(),
            )
        else:
            if args.worker_input is None:
                parser.error("微基准 worker 需要 --worker-input")
            _run_micro_worker(
                args.worker_backend,
                args.worker_input.resolve(),
                args.worker_output.resolve(),
            )
        return 0
    return run_suite(output)


if __name__ == "__main__":
    raise SystemExit(main())
