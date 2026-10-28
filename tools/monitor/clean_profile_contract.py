"""清洗逐层剖析的固定案例、环境、输入和证据契约。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common.paths import PROJECT_ROOT
from tools.monitor.baseline_contract import (
    atomic_write_json,
    file_record,
)

SCHEMA_VERSION = "clean_cpp_profile_v1"
EVIDENCE_ROOT = PROJECT_ROOT / "AiRecord/benchmarks/V124/p3-02-20261028"
THRESHOLD = 0.20
PRIMARY_CASE = "2025"
CASE_ORDER = ("2015", "2025")
CASE_DEFINITIONS = {
    "2015": {
        "year": 2015,
        "seed_year": 2014,
        "role": "historical_cross_check",
        "start": "2015-01-01",
        "end": "2015-12-31",
    },
    "2025": {
        "year": 2025,
        "seed_year": 2024,
        "role": "primary_decision",
        "start": "2025-01-01",
        "end": "2025-12-31",
    },
}
RAW_ROOT = PROJECT_ROOT / "data/source/tushare/raw"
REFERENCE_ROOT = PROJECT_ROOT / "data/source/tushare"
RAW_APIS = ("daily", "adj_factor", "stk_limit", "daily_basic")
REFERENCE_FILES = (
    "trade_cal.parquet",
    "stock_basic_all.parquet",
    "namechange_all.parquet",
    "suspend_d.parquet",
)
SOURCE_FILES = (
    "core/data/scripts/clean/pipeline.py",
    "core/data/scripts/clean/io.py",
    "core/data/scripts/clean/adjust.py",
    "core/data/scripts/clean/lifecycle.py",
    "core/data/scripts/clean/limits.py",
    "core/data/scripts/clean/sources.py",
    "core/data/scripts/build_db/build.py",
    "core/data/scripts/build_db/ddl.py",
    "tools/monitor/clean_profile.py",
    "tools/monitor/clean_profile_contract.py",
    "tools/monitor/clean_profile_probe.py",
    "tools/monitor/clean_profile_report.py",
    "tools/monitor/clean_profile_runtime.py",
    "tools/monitor/clean_profile_worker.py",
)


def median(values: list[float]) -> float:
    """返回非空数值列表的中位数。"""

    if not values:
        raise ValueError("中位数至少需要一个观测值")
    return round(float(np.median(np.asarray(values, dtype=np.float64))), 6)


def records_digest(records: list[dict[str, Any]]) -> str:
    """对文件记录生成与路径、摘要绑定的稳定总指纹。"""

    digest = hashlib.sha256()
    for item in sorted(records, key=lambda value: value["path"]):
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _git(*args: str) -> str:
    """执行只读 Git 查询。"""

    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.rstrip() if completed.returncode == 0 else ""


def environment_snapshot() -> dict[str, Any]:
    """返回环境、依赖和实现源码快照。"""

    source_records = [
        file_record(PROJECT_ROOT / path)
        for path in SOURCE_FILES
        if (PROJECT_ROOT / path).is_file()
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "status_porcelain": _git("status", "--porcelain").splitlines(),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "source_files": source_records,
        "source_tree_sha256": records_digest(source_records),
    }


def _relevant_input_paths() -> list[Path]:
    """列出代表案例及其种子年份实际读取的全部输入文件。"""

    years = {
        year
        for definition in CASE_DEFINITIONS.values()
        for year in (int(definition["seed_year"]), int(definition["year"]))
    }
    paths: dict[str, Path] = {}
    for api in RAW_APIS:
        for year in years:
            directory = RAW_ROOT / api / f"year={year}"
            parts = sorted(directory.glob("month=*.parquet"))
            if not parts:
                raise FileNotFoundError(f"输入分片不存在：{directory}")
            for path in parts:
                paths[path.as_posix()] = path
    for name in REFERENCE_FILES:
        path = REFERENCE_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在：{path}")
        paths[path.as_posix()] = path
    return [paths[key] for key in sorted(paths)]


def input_snapshot() -> dict[str, Any]:
    """返回真实输入文件清单和总指纹。"""

    records = [file_record(path) for path in _relevant_input_paths()]
    return {
        "schema_version": SCHEMA_VERSION,
        "files": records,
        "file_count": len(records),
        "bytes": sum(int(item["bytes"]) for item in records),
        "input_tree_sha256": records_digest(records),
    }


def update_manifest(output: Path) -> dict[str, Any]:
    """更新证据目录文件哈希，排除报告和 manifest 自身。"""

    excluded = {"manifest.json", "report.md"}
    records = [
        file_record(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": records,
        "tree_sha256": records_digest(records),
    }
    atomic_write_json(output / "manifest.json", manifest)
    return manifest
