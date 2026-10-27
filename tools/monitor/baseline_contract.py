"""算子优化基准的固定契约、输入快照和稳定比较工具。"""

from __future__ import annotations

import datetime as dt
import fnmatch
import hashlib
import json
import math
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calculate.panel import open_source
from common.paths import (
    DB_FILE,
    FACTOR_DB_FILE,
    FACTOR_DIR,
    MINING_CATALOG_FILE,
    MINING_EVALUATION_FILE,
    MINING_ORCHESTRATOR_FILE,
    PROJECT_ROOT,
)

SCHEMA_VERSION = "operator_baseline_v1"
EVIDENCE_ROOT = PROJECT_ROOT / "AiRecord/benchmarks/V105/p0-01-20261027"

CASE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "evaluation": {
        "factor": "MOM_20",
        "start": "2023-10-15",
        "end": "2026-10-15",
        "universe": "all_a",
        "horizon": 5,
        "price": "close",
        "quantiles": 10,
        "winsorize": True,
        "stages": None,
    },
    "correlation": {
        "factor": "MOM_20",
        "start": "2023-10-15",
        "end": "2026-10-15",
        "universe": "all_a",
        "horizon": 5,
        "price": "close",
        "quantiles": 10,
        "winsorize": True,
        "top_n": 10,
        "stages": ("correlation",),
    },
    "correlation_target": {
        "factor": "MOM_20",
        "start": "2023-10-15",
        "end": "2026-10-15",
        "universe": "all_a",
        "horizon": 5,
        "price": "close",
        "quantiles": 10,
        "winsorize": True,
        "top_n": 10,
        "formal": False,
        "stages": ("correlation",),
    },
    "history": {
        "factor": "MOM_20",
        "start": "2000-01-04",
        "end": "2026-10-15",
        "universe": "all_a",
        "horizon": 5,
        "price": "close",
        "quantiles": 10,
        "winsorize": True,
    },
    "factor_build": {
        "families": ("momentum",),
        "start": "2024-01-01",
        "end": "2024-12-31",
    },
    "factor_build_yearly": {
        "families": ("momentum",),
        "start": "2023-01-01",
        "end": "2024-12-31",
        "build_mode": "yearly",
    },
    "factor_build_full_history": {
        "families": ("momentum",),
        "start": "2023-01-01",
        "end": "2024-12-31",
        "build_mode": "full_history",
    },
    "factor_build_all_history": {
        "families": ("momentum",),
        "start": "2000-01-01",
        "end": "2026-10-15",
        "build_mode": "full_history",
    },
    "mining": {
        "expression": "mean(close,5)",
        "direction": 1,
        "research_start": "2024-03-01",
        "research_end": "2024-12-31",
        "full_eval_start": "2024-09-01",
        "full_eval_end": "2024-12-31",
        "holdout_start": "2025-01-01",
        "holdout_end": "2025-03-31",
        "quick_start": "2024-05-06",
        "quick_end": "2024-06-28",
        "run_full": False,
    },
    "mining_batch_serial": {
        "mode": "serial",
        "expressions": (
            "mean(close, 5) + rank(open)",
            "mean(close, 5) - open",
            "rank(mean(close, 5))",
            "mean(close, 5) + close",
        ),
        "start": "2024-03-01",
        "end": "2024-12-31",
    },
    "mining_batch_dag": {
        "mode": "batch",
        "expressions": (
            "mean(close, 5) + rank(open)",
            "mean(close, 5) - open",
            "rank(mean(close, 5))",
            "mean(close, 5) + close",
        ),
        "start": "2024-03-01",
        "end": "2024-12-31",
    },
    "edge": {
        "seed": 20261027,
        "min_cross_section": 5,
    },
}

CASE_ORDER = tuple(CASE_DEFINITIONS)

_VOLATILE_KEYS = frozenset(
    {
        "artifact_path",
        "created_at",
        "elapsed_ms",
        "finished_at",
        "generated_at",
        "ic_load_elapsed_ms",
        "run_id",
        "started_at",
        "total_elapsed_ms",
    }
)

_SOURCE_PATHS = (
    Path("tools/monitor/operator_baseline.py"),
    Path("tools/monitor/baseline_contract.py"),
    Path("tools/monitor/baseline_cases.py"),
    Path("tools/monitor/kernel_baseline.py"),
    Path("tools/monitor/native_kernel_baseline.py"),
    Path("tools/monitor/correlation_native_baseline.py"),
    Path("tools/monitor/daily_native_baseline.py"),
    Path("core/kernels/__init__.py"),
    Path("core/kernels/backend.py"),
    Path("core/kernels/errors.py"),
    Path("core/kernels/fallback.py"),
    Path("core/kernels/native.py"),
    Path("core/kernels/prototype.py"),
    Path("native/CMakeLists.txt"),
    Path("native/include/quant_native/daily.hpp"),
    Path("native/include/quant_native/spearman.hpp"),
    Path("native/src/bindings.cpp"),
    Path("native/src/rank.cpp"),
    Path("native/src/pearson.cpp"),
    Path("native/src/spearman.cpp"),
    Path("tests/test_cpp_native.py"),
    Path("tests/test_cpp_daily_kernels.py"),
    Path("tests/test_daily_native_routing.py"),
    Path("tests/test_daily_native_baseline.py"),
    Path("tests/test_native_kernel_baseline.py"),
    Path("research/evaluation/calculate/correlation.py"),
    Path("research/evaluation/calculate/daily.py"),
    Path("research/evaluation/calculate/normalize.py"),
    Path("research/evaluation/calculate/exposure.py"),
    Path("research/evaluation/calculate/history.py"),
    Path("research/evaluation/calculate/panel.py"),
    Path("core/data/scripts/build_factors/build.py"),
    Path("core/data/scripts/build_factors/sql.py"),
    Path("research/mining/candidate_evaluator.py"),
    Path("research/mining/candidate_batch.py"),
    Path("research/mining/candidate_materialization.py"),
    Path("research/mining/evaluation_gates.py"),
    Path("research/mining/expression_batch.py"),
    Path("research/mining/expression_compiler.py"),
    Path("research/mining/expression_dag.py"),
)


class GoldenMismatch(AssertionError):
    """当前结果与金标准不一致。"""


def sha256_file(path: Path | str) -> str:
    """计算文件 SHA-256；参数 path 为待读取文件。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """生成稳定、不允许 NaN 的 JSON；参数 value 为待编码对象。"""

    serializable = canonical_value(value)
    return json.dumps(
        serializable,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_value(value: Any) -> Any:
    """转换 NumPy/pandas 值，并用显式字符串保留非有限语义。"""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "__NaN__"
        if math.isinf(value):
            return "__+Inf__" if value > 0 else "__-Inf__"
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return canonical_value(float(value))
    if isinstance(value, dict):
        return {str(key): canonical_value(item) for key, item in value.items()}
    if isinstance(value, set):
        return [
            canonical_value(item)
            for item in sorted(value, key=lambda item: str(item))
        ]
    if isinstance(value, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        return [canonical_value(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [canonical_value(row) for row in value.to_dict(orient="records")]
    if isinstance(value, (pd.Timestamp, np.datetime64, dt.datetime, dt.date)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return project_relative(value)
    return str(value)


def strip_volatile(value: Any) -> Any:
    """递归移除计时、时间和临时路径字段，同时保留非有限明确语义。"""

    value = canonical_value(value)
    if isinstance(value, dict):
        return {
            key: strip_volatile(item)
            for key, item in value.items()
            if key not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [strip_volatile(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    """原子写入稳定 JSON；参数 path 为目标文件，payload 为内容。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(payload), encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    """原子写入 UTF-8 文本；参数 path 为目标文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_golden(path: Path | str) -> Any:
    """读取基准 JSON；参数 path 为证据文件。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_golden(
    expected: Any,
    actual: Any,
    *,
    abs_tol: float = 1e-10,
    rel_tol: float = 1e-10,
    exact_paths: tuple[str, ...] = (),
) -> None:
    """比较金标准；参数 exact_paths 中的浮点路径必须完全相等。"""

    _compare_value(expected, actual, "$", abs_tol, rel_tol, exact_paths)


def _compare_value(
    expected: Any,
    actual: Any,
    path: str,
    abs_tol: float,
    rel_tol: float,
    exact_paths: tuple[str, ...],
) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise GoldenMismatch(f"{path}: 字段不同")
        for key in expected:
            _compare_value(
                expected[key],
                actual[key],
                f"{path}.{key}",
                abs_tol,
                rel_tol,
                exact_paths,
            )
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise GoldenMismatch(f"{path}: 长度 {len(expected)} != {len(actual)}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare_value(
                left,
                right,
                f"{path}[{index}]",
                abs_tol,
                rel_tol,
                exact_paths,
            )
        return
    if (
        isinstance(expected, float)
        and isinstance(actual, float)
        and not isinstance(expected, bool)
        and not isinstance(actual, bool)
    ):
        exact = any(fnmatch.fnmatchcase(path, pattern) for pattern in exact_paths)
        if exact:
            if expected != actual:
                raise GoldenMismatch(f"{path}: {expected!r} != {actual!r}")
        elif not math.isclose(expected, actual, abs_tol=abs_tol, rel_tol=rel_tol):
            raise GoldenMismatch(
                f"{path}: {expected!r} != {actual!r} "
                f"(abs_tol={abs_tol:g}, rel_tol={rel_tol:g})"
            )
        return
    if expected != actual:
        raise GoldenMismatch(f"{path}: {expected!r} != {actual!r}")


def project_relative(path: Path | str) -> str:
    """尽量返回项目相对路径；参数 path 为本地文件。"""

    target = Path(path).resolve()
    try:
        return target.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return target.as_posix()


def file_record(path: Path | str) -> dict[str, Any]:
    """返回文件路径、大小和 SHA-256；参数 path 为输入文件。"""

    target = Path(path)
    return {
        "path": project_relative(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def _records_digest(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.get("sha256", "")).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _cpu_model() -> str | None:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _memory_bytes() -> int | None:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^MemTotal:\s+(\d+) kB$", text, re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def _git(*args: str) -> str:
    import subprocess

    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip()


def environment_snapshot() -> dict[str, Any]:
    """返回可复现 Python、依赖、机器和源码快照。"""

    import duckdb
    from importlib.metadata import version
    import scipy

    sources = [
        file_record(PROJECT_ROOT / path)
        for path in _SOURCE_PATHS
        if (PROJECT_ROOT / path).is_file()
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "status_porcelain": _git("status", "--porcelain").splitlines(),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": {
            "model": _cpu_model(),
            "logical_count": os.cpu_count(),
        },
        "memory_bytes": _memory_bytes(),
        "dependencies": {
            "duckdb": duckdb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "cmake": version("cmake"),
            "ninja": version("ninja"),
            "pybind11": version("pybind11"),
            "scikit-build-core": version("scikit-build-core"),
        },
        "source_files": sources,
        "source_tree_sha256": _records_digest(sources),
    }


def _data_snapshot() -> dict[str, Any]:
    factor_files = sorted(FACTOR_DIR.glob("*/year=*/part.parquet"))
    with open_source() as conn:
        daily_rows = int(conn.execute("SELECT count(*) FROM src.md.daily_hfq").fetchone()[0])
        first_day, last_day, calendar_days = conn.execute(
            "SELECT min(trade_date), max(trade_date), count(*) FROM src.md.calendar"
        ).fetchone()
    return {
        "daily_hfq_rows": daily_rows,
        "calendar_start": first_day,
        "calendar_end": last_day,
        "calendar_days": int(calendar_days),
        "factor_files": len(factor_files),
        "factor_bytes": sum(path.stat().st_size for path in factor_files),
    }


def input_snapshot() -> dict[str, Any]:
    """返回基准依赖的数据和配置文件哈希。"""

    catalog = json.loads(MINING_CATALOG_FILE.read_text(encoding="utf-8"))
    factor_files = sorted(FACTOR_DIR.glob("*/year=*/part.parquet"))
    fixed_files = [
        DB_FILE,
        FACTOR_DB_FILE,
        MINING_CATALOG_FILE,
        MINING_EVALUATION_FILE,
        MINING_ORCHESTRATOR_FILE,
    ]
    records = [file_record(path) for path in [*fixed_files, *factor_files]]
    return {
        "schema_version": SCHEMA_VERSION,
        "data_version": catalog.get("data_version"),
        "catalog_schema_version": catalog.get("schema_version"),
        "snapshot": _data_snapshot(),
        "files": records,
        "input_tree_sha256": _records_digest(records),
    }


def prepare_common(output: Path, *, refresh: bool = False) -> dict[str, Any]:
    """生成环境、输入和固定案例定义；参数 output 为证据目录。"""

    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "environment": output / "environment.json",
        "inputs": output / "inputs.json",
        "definitions": output / "case_definitions.json",
    }
    if refresh or not all(path.is_file() for path in paths.values()):
        environment = environment_snapshot()
        inputs = input_snapshot()
        atomic_write_json(paths["environment"], environment)
        atomic_write_json(paths["inputs"], inputs)
        atomic_write_json(
            paths["definitions"],
            {
                "schema_version": SCHEMA_VERSION,
                "comparison": {
                    "abs_tol": 1e-10,
                    "rel_tol": 1e-10,
                    "exact": "日期、名称、顺序、有效计数和排名类输出必须完全一致",
                    "public_payload_nonfinite": "沿用接口约定保存为 null",
                    "raw_nonfinite": "__NaN__、__+Inf__、__-Inf__",
                    "numpy": "数组规范化后保存在 JSON 金标准中",
                    "parquet": "通过文件 SHA-256、行集规模和覆盖统计对照",
                    "kernel_microbench": "至少 3 次取中位数",
                },
                "cases": CASE_DEFINITIONS,
            },
        )
    return {
        "environment": load_golden(paths["environment"]),
        "inputs": load_golden(paths["inputs"]),
        "definitions": load_golden(paths["definitions"]),
    }


def update_manifest(output: Path) -> dict[str, Any]:
    """更新证据目录 manifest；参数 output 为证据目录。"""

    excluded = {"manifest.json", "report.md"}
    files = [
        file_record(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": files,
        "tree_sha256": _records_digest(files),
    }
    atomic_write_json(output / "manifest.json", manifest)
    return manifest
