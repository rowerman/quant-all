"""验证模型预处理复用批量核后的正确性、耗时和峰值内存。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from alpha.dataset import load_dataset
from alpha.preprocess import PreprocessConfig, transform
from common.paths import PROJECT_ROOT
from tools.monitor.baseline_contract import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    update_manifest,
)
from tools.monitor.standardize_native_baseline import RssSampler

SCHEMA_VERSION = "modeling_preprocess_baseline_v1"
DATASET_ID = "6a59680fd71c50a8"
EXPECTED_ROWS = 2_138_239
EXPECTED_COLUMNS = 50
BACKENDS = ("python", "native", "auto")
REPEATS = 3
EVIDENCE_ROOT = PROJECT_ROOT / "AiRecord/benchmarks/V120/p2-01-20261027"
DATASET_PATH = PROJECT_ROOT / "data/models/datasets" / DATASET_ID
CASES = {
    "rank": {"method": "rank", "winsorize": False},
    "zscore": {"method": "zscore", "winsorize": False},
    "rank_winsorized": {"method": "rank", "winsorize": True},
    "zscore_winsorized": {"method": "zscore", "winsorize": True},
}

def _worker_environment(backend: str) -> dict[str, str]:
    """返回固定后端和单线程的独立进程环境。"""

    return {
        **os.environ,
        "QUANT_KERNEL_BACKEND": backend,
        "QUANT_NATIVE_THREADS": "1",
    }


def _output_summary(result, manifest: dict[str, Any]) -> dict[str, Any]:
    """只保存列哈希和诊断，避免复制整份预处理结果。"""

    digest = hashlib.sha256()
    columns: dict[str, str] = {}
    features = tuple(manifest["features"]) + tuple(manifest["context_features"])
    for name in features:
        values = result.data[name].to_numpy(dtype=np.float32, copy=False)
        column_digest = hashlib.sha256(values.tobytes()).hexdigest()
        columns[name] = column_digest
        digest.update(name.encode("utf-8"))
        digest.update(values.tobytes())
    missing_counts = {
        name: int(result.data[name].isna().sum()) for name in features
    }
    return {
        "feature_names": list(result.feature_names),
        "missing_columns": list(result.missing_columns),
        "report": result.report,
        "column_sha256": columns,
        "combined_sha256": digest.hexdigest(),
        "missing_counts": missing_counts,
    }


def run_worker(
    *,
    backend: str,
    case: str,
    output: Path,
) -> int:
    """在独立进程运行一种预处理模式。"""

    selected = CASES[case]
    data, manifest = load_dataset(DATASET_ID)
    if int(manifest["row_count"]) != EXPECTED_ROWS:
        raise RuntimeError("固定数据集行数发生变化")
    features = tuple(manifest["features"]) + tuple(manifest["context_features"])
    if len(features) != EXPECTED_COLUMNS:
        raise RuntimeError("固定数据集列数发生变化")
    started = time.perf_counter()
    with RssSampler() as sampler:
        result = transform(
            data,
            PreprocessConfig(
                method=str(selected["method"]),
                winsorize=bool(selected["winsorize"]),
            ),
            feature_columns=features,
        )
        elapsed_s = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    summary = _output_summary(result, manifest)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "dataset_id": DATASET_ID,
        "case": case,
        "elapsed_s": round(elapsed_s, 6),
        "baseline_rss_mb": round(sampler.baseline_mb, 3),
        "peak_rss_mb": round(sampler.peak_mb, 3),
        **summary,
    }
    atomic_write_json(output / "result.json", payload)
    return 0


def _worker_command(
    *,
    backend: str,
    case: str,
    output: Path,
) -> list[str]:
    """构造独立 worker 命令。"""

    return [
        sys.executable,
        "-m",
        "tools.monitor.modeling_preprocess_baseline",
        "--worker-backend",
        backend,
        "--worker-case",
        case,
        "--worker-output",
        str(output),
    ]


def _run_child(command: list[str], backend: str) -> None:
    """运行 worker；失败时带出 stderr。"""

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


def _median_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总同一后端三次独立运行。"""

    return {
        "elapsed_s": float(np.median([item["elapsed_s"] for item in records])),
        "peak_rss_mb": max(float(item["peak_rss_mb"]) for item in records),
        "combined_sha256": records[0]["combined_sha256"],
        "column_sha256": records[0]["column_sha256"],
        "runs": records,
    }


def _correction_pass(cases: dict[str, dict[str, dict[str, Any]]]) -> bool:
    """确认三后端列哈希完全一致。"""

    return all(
        result["native"]["combined_sha256"]
        == result["python"]["combined_sha256"]
        == result["auto"]["combined_sha256"]
        and result["native"]["column_sha256"]
        == result["python"]["column_sha256"]
        == result["auto"]["column_sha256"]
        for result in cases.values()
    )


def _performance_pass(cases: dict[str, dict[str, dict[str, Any]]]) -> bool:
    """要求 native 和 auto 的中位耗时不慢于 Python。"""

    return all(
        result["native"]["elapsed_s"] <= result["python"]["elapsed_s"]
        and result["auto"]["elapsed_s"] <= result["python"]["elapsed_s"]
        for result in cases.values()
    )


def run_suite(output: Path) -> int:
    """运行四类预处理模式并写出证据。"""

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = DATASET_PATH / "manifest.json"
    data_path = DATASET_PATH / "data.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != DATASET_ID:
        raise RuntimeError("固定数据集 manifest ID 不匹配")
    input_record = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "data_sha256": sha256_file(data_path),
        "manifest_sha256": sha256_file(manifest_path),
        "rows": int(manifest["row_count"]),
        "dates": int(manifest["date_count"]),
        "stocks": int(manifest["stock_count"]),
        "features": len(manifest["features"]),
        "context_features": len(manifest["context_features"]),
        "versions": manifest["versions"],
    }
    atomic_write_json(output / "inputs.json", input_record)

    raw_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for case in CASES:
        raw_results[case] = {}
        for backend in BACKENDS:
            raw_results[case][backend] = []
            for repeat in range(REPEATS):
                target = output / "workers" / case / backend / f"run-{repeat + 1}"
                _run_child(
                    _worker_command(
                        backend=backend,
                        case=case,
                        output=target,
                    ),
                    backend,
                )
                raw_results[case][backend].append(
                    json.loads((target / "result.json").read_text(encoding="utf-8"))
                )

    cases = {
        case: {
            backend: _median_record(records)
            for backend, records in backend_records.items()
        }
        for case, backend_records in raw_results.items()
    }
    correctness = _correction_pass(cases)
    performance = _performance_pass(cases)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "repeats": REPEATS,
        "threads": 1,
        "cases": {
            case: {
                backend: {
                    "elapsed_s": result["elapsed_s"],
                    "peak_rss_mb": result["peak_rss_mb"],
                    "combined_sha256": result["combined_sha256"],
                }
                for backend, result in backend_results.items()
            }
            for case, backend_results in cases.items()
        },
        "gate": {
            "correctness_pass": correctness,
            "performance_pass": performance,
            "passed": correctness and performance,
        },
    }
    atomic_write_json(output / "result.json", payload)
    lines = [
        "# P2-01 模型预处理基准",
        "",
        f"- 数据集：`{DATASET_ID}`，{EXPECTED_ROWS:,} 行、{EXPECTED_COLUMNS} 列。",
        f"- 每个后端独立运行 {REPEATS} 次，单线程。",
        "",
        "| 模式 | Python | native | auto | native 加速 |",
        "|---|---:|---:|---:|---:|",
    ]
    for case, backend_results in cases.items():
        python_s = backend_results["python"]["elapsed_s"]
        native_s = backend_results["native"]["elapsed_s"]
        automatic_s = backend_results["auto"]["elapsed_s"]
        lines.append(
            f"| `{case}` | {python_s:.3f}s | {native_s:.3f}s | "
            f"{automatic_s:.3f}s | {python_s / native_s:.3f}x |"
        )
    lines.extend(
        [
            "",
            f"正确性：{'通过' if correctness else '未通过'}。",
            f"不退化：{'通过' if performance else '未通过'}。",
        ]
    )
    atomic_write_text(output / "result.md", "\n".join(lines) + "\n")
    update_manifest(output)
    return 0 if correctness and performance else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--worker-backend", choices=BACKENDS)
    parser.add_argument("--worker-case", choices=tuple(CASES))
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    worker_values = (args.worker_backend, args.worker_case, args.worker_output)
    if any(value is not None for value in worker_values):
        if not all(value is not None for value in worker_values):
            parser.error("worker 参数必须同时提供")
        return run_worker(
            backend=args.worker_backend,
            case=args.worker_case,
            output=args.worker_output,
        )
    return run_suite(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
