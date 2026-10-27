"""验证 C++ Spearman 核的速度、内存和数值一致性。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from typing import Any

import numpy as np

from backend import service
from calculate.correlation import daily_spearman_matrix
from calculate.panel import build_panel
from common.paths import PROJECT_ROOT
from tools.monitor.baseline_contract import (
    atomic_write_json,
    atomic_write_text,
    load_golden,
    prepare_common,
    project_relative,
    update_manifest,
)
from tools.monitor.kernel_baseline import (
    CORRELATION_QUERY,
    MICRO_DATE,
    _load_spearman_matrix,
)

SCHEMA_VERSION = "native_kernel_baseline_v1"
EVIDENCE_ROOT = PROJECT_ROOT / "AiRecord/benchmarks/V113/p1-02-20261027"
V105_INPUTS = (
    PROJECT_ROOT / "AiRecord/benchmarks/V105/p0-01-20261027/inputs.json"
)
REPEATS = 5
MIN_SPEEDUP = 1.5
MAX_RSS_RATIO = 1.2


def _current_rss_mb() -> float:
    """从 /proc 读取当前进程常驻内存。"""

    page_size = os.sysconf("SC_PAGE_SIZE")
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * page_size / (1024 * 1024)


class RssSampler:
    """后台采样当前进程 RSS，避免继承 fork 前的 ru_maxrss。"""

    def __init__(self, interval_s: float = 0.001):
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


def _backend_command(
    backend: str,
    matrix_path: Path,
    output_dir: Path,
) -> list[str]:
    """构造独立测量进程命令；参数 backend 为 python 或 native。"""

    return [
        sys.executable,
        "-m",
        "tools.monitor.native_kernel_baseline",
        "--worker-backend",
        backend,
        "--matrix",
        str(matrix_path),
        "--worker-output",
        str(output_dir),
    ]


def run_worker(backend: str, matrix_path: Path, output_dir: Path) -> None:
    """在独立进程中测量指定后端并保存相关矩阵。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.load(matrix_path)
    if not values.flags.c_contiguous:
        raise RuntimeError("固定 Spearman 矩阵不是 C 连续数组")
    offsets = np.array([0, values.shape[0]], dtype=np.int64)
    import_started = time.perf_counter()
    if backend == "python":
        call = lambda: daily_spearman_matrix(values)
        correlation_name = "python_correlation.npy"
        count_name = "python_count.npy"
    elif backend == "native":
        import quant_native

        call = lambda: quant_native.pairwise_spearman(values, offsets, 5)
        correlation_name = "native_correlation.npy"
        count_name = "native_count.npy"
    else:
        raise ValueError(f"未知 worker 后端：{backend}")
    import_elapsed = time.perf_counter() - import_started

    with RssSampler() as sampler:
        first_started = time.perf_counter()
        first_result = call()
        first_elapsed = time.perf_counter() - first_started
        runs = []
        result = first_result
        for _ in range(REPEATS):
            started = time.perf_counter()
            result = call()
            runs.append(time.perf_counter() - started)

    if backend == "python":
        correlation = np.asarray(result)
        count = None
    else:
        correlation = np.asarray(result[0])[0]
        count = np.asarray(result[1])
        np.save(output_dir / count_name, count)
    np.save(output_dir / correlation_name, correlation)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "import_s": import_elapsed,
        "first_s": first_elapsed,
        "runs_s": runs,
        "median_s": float(np.median(np.asarray(runs, dtype=np.float64))),
        "baseline_rss_mb": round(sampler.baseline_mb, 3),
        "peak_rss_mb": round(sampler.peak_mb, 3),
        "delta_rss_mb": round(sampler.peak_mb - sampler.baseline_mb, 3),
    }
    atomic_write_json(output_dir / f"{backend}.json", payload)


def summarize_gate(
    python: dict[str, Any],
    native: dict[str, Any],
    *,
    correct: bool,
    counts_exact: bool,
) -> dict[str, Any]:
    """按正确性、加速和内存门槛生成机器可读结论。"""

    speedup = float(python["median_s"]) / float(native["median_s"])
    rss_ratio = float(native["peak_rss_mb"]) / float(python["peak_rss_mb"])
    if not correct:
        reason = "incorrect_correlation"
    elif not counts_exact:
        reason = "incorrect_count"
    elif speedup < MIN_SPEEDUP:
        reason = "speedup_below_1.5x"
    elif rss_ratio > MAX_RSS_RATIO:
        reason = "rss_above_1.2x"
    else:
        reason = None
    return {
        "schema_version": SCHEMA_VERSION,
        "correct": bool(correct),
        "counts_exact": bool(counts_exact),
        "speedup": speedup,
        "rss_ratio": rss_ratio,
        "speedup_pass": speedup >= MIN_SPEEDUP,
        "rss_pass": rss_ratio <= MAX_RSS_RATIO,
        "gate_pass": reason is None,
        "reason": reason,
        "cpp_modules_replaced": 0,
        "business_path": False,
    }


def run_suite(output: Path) -> int:
    """读取固定真实截面，分别运行 Python/C++ 并生成 V113 证据。"""

    output.mkdir(parents=True, exist_ok=True)
    common = prepare_common(output, refresh=True)
    expected_inputs = load_golden(V105_INPUTS)
    current_hash = common["inputs"]["input_tree_sha256"]
    expected_hash = expected_inputs["input_tree_sha256"]
    if current_hash != expected_hash:
        raise RuntimeError(
            "输入数据相对 V105 已漂移，停止性能对照："
            f"{current_hash} != {expected_hash}"
        )

    spec_context = build_panel(service.prepare(CORRELATION_QUERY).params)
    matrix = _load_spearman_matrix(spec_context)
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)

    with TemporaryDirectory(prefix="quant-native-spearman-") as temp:
        temp_path = Path(temp)
        matrix_path = temp_path / "matrix.npy"
        np.save(matrix_path, matrix)
        worker_results = {}
        for backend in ("python", "native"):
            worker_dir = temp_path / backend
            completed = subprocess.run(
                _backend_command(backend, matrix_path, worker_dir),
                cwd=PROJECT_ROOT,
                env={**os.environ, "QUANT_NATIVE_THREADS": "1"},
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"{backend} 测量失败：{completed.stderr or completed.stdout}"
                )
            worker_results[backend] = load_golden(worker_dir / f"{backend}.json")

        python_correlation = np.load(temp_path / "python/python_correlation.npy")
        native_correlation = np.load(temp_path / "native/native_correlation.npy")
        native_count = np.load(temp_path / "native/native_count.npy")
        try:
            np.testing.assert_allclose(
                native_correlation,
                python_correlation,
                atol=1e-10,
                rtol=1e-10,
                equal_nan=True,
            )
            correct = True
        except AssertionError:
            correct = False
        finite = np.isfinite(matrix)
        expected_count = finite.T.astype(np.int64) @ finite.astype(np.int64)
        counts_exact = bool(
            np.array_equal(native_count[0], expected_count)
        )

    gate = summarize_gate(
        worker_results["python"],
        worker_results["native"],
        correct=correct,
        counts_exact=counts_exact,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "case": "native_spearman_micro",
        "input": {
            "date": MICRO_DATE,
            "rows": int(matrix.shape[0]),
            "columns": int(matrix.shape[1]),
            "groups": 1,
            "threads": 1,
            "matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
            "input_tree_sha256": current_hash,
        },
        "python": worker_results["python"],
        "native": worker_results["native"],
        "gate": gate,
    }
    atomic_write_json(output / "micro.json", result)
    lines = [
        "# P1-02 C++ Spearman 单核 POC",
        "",
        "## 口径",
        "",
        f"- 固定日期：{MICRO_DATE}；截面 {matrix.shape[0]}×{matrix.shape[1]}；单组、单线程。",
        "- Python/C++ 在独立子进程完成首次调用和 5 次热运行；峰值 RSS 按进程分别采样。",
        f"- 输入树 SHA-256：`{current_hash}`。",
        "",
        "## 结果",
        "",
        "| 后端 | 热运行中位 | 峰值 RSS |",
        "|---|---:|---:|",
        (
            f"| Python | {worker_results['python']['median_s']:.6f}s | "
            f"{worker_results['python']['peak_rss_mb']:.3f}MB |"
        ),
        (
            f"| C++ | {worker_results['native']['median_s']:.6f}s | "
            f"{worker_results['native']['peak_rss_mb']:.3f}MB |"
        ),
        "",
        f"- 加速：{gate['speedup']:.3f}x；RSS 比例：{gate['rss_ratio']:.3f}x。",
        f"- RSS 增量：Python {worker_results['python']['delta_rss_mb']:.3f}MB；"
        f"C++ {worker_results['native']['delta_rss_mb']:.3f}MB。",
        f"- 相关系数正确：{'是' if gate['correct'] else '否'}；计数一致："
        f"{'是' if gate['counts_exact'] else '否'}。",
        f"- 门禁结论：{'通过' if gate['gate_pass'] else '未通过'}；"
        f"原因：{gate['reason'] or '满足全部门槛'}。",
        "- 本任务未接入业务路径，实际 C++ 替换模块数为 0。",
        "",
    ]
    atomic_write_text(output / "result.md", "\n".join(lines))
    update_manifest(output)
    print(
        json.dumps(
            {
                "output": project_relative(output),
                "speedup": gate["speedup"],
                "rss_ratio": gate["rss_ratio"],
                "gate_pass": gate["gate_pass"],
                "reason": gate["reason"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if gate["gate_pass"] else 1


def main(argv: list[str] | None = None) -> int:
    """解析并执行 C++ Spearman 微基准。"""

    parser = argparse.ArgumentParser(description="验证 C++ Spearman 单核")
    parser.add_argument("--output", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--worker-backend", choices=("python", "native"))
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    if args.worker_backend:
        if args.matrix is None or args.worker_output is None:
            parser.error("worker 模式必须提供 --matrix 和 --worker-output")
        run_worker(args.worker_backend, args.matrix, args.worker_output)
        return 0
    return run_suite(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
