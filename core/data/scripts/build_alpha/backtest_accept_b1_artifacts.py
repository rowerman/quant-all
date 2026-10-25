"""P4-4B1 逐候选产物的原子写入和哈希校验。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha.artifacts import file_sha256
from scripts.build_alpha.backtest_accept_artifacts import (
    atomic_write_frame,
    atomic_write_json,
)

SCHEMA_VERSION = "p4_4b1_candidate_v1"
COMPLETE_FILE = "complete.json"
CANDIDATE_FILES = (
    "metrics.json",
    "strategy.json",
    "yearly.csv",
    "daily.parquet",
    "orders.parquet",
    "holdings.parquet",
    "run_manifest.json",
)


def _indexed_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """把日结果索引恢复成显式交易日列。"""

    payload = daily.copy()
    if not isinstance(payload.index, pd.RangeIndex):
        payload.index.name = "trade_date"
        payload = payload.reset_index()
    return payload


def write_candidate_artifacts(
    folder: Path,
    *,
    analysis_id: str,
    candidate: str,
    metrics: dict,
    strategy: dict,
    yearly: pd.DataFrame,
    daily: pd.DataFrame,
    orders: pd.DataFrame,
    holdings: pd.DataFrame,
    run_manifest: dict,
    schema_version: str = SCHEMA_VERSION,
) -> dict:
    """写入单候选全部文件，最后写完成标记。"""

    folder.mkdir(parents=True, exist_ok=True)
    atomic_write_json(folder / "metrics.json", metrics)
    atomic_write_json(folder / "strategy.json", strategy)
    atomic_write_frame(folder / "yearly.csv", yearly)
    atomic_write_frame(folder / "daily.parquet", _indexed_daily(daily))
    atomic_write_frame(folder / "orders.parquet", orders.reset_index(drop=True))
    atomic_write_frame(folder / "holdings.parquet", holdings.reset_index(drop=True))
    atomic_write_json(folder / "run_manifest.json", run_manifest)
    completion = {
        "schema_version": schema_version,
        "analysis_id": analysis_id,
        "candidate": candidate,
        "files": {
            name: file_sha256(folder / name) for name in CANDIDATE_FILES
        },
    }
    atomic_write_json(folder / COMPLETE_FILE, completion)
    return completion


def read_candidate_completion(
    folder: Path,
    *,
    schema_version: str = SCHEMA_VERSION,
) -> dict | None:
    """读取并校验完成标记；缺失或哈希不符时返回 None。"""

    marker = folder / COMPLETE_FILE
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("schema_version") != schema_version:
            return None
        files = payload.get("files")
        if not isinstance(files, dict) or set(files) != set(CANDIDATE_FILES):
            return None
        for name, expected in files.items():
            path = folder / name
            if not path.is_file() or file_sha256(path) != expected:
                return None
        return payload
    except (OSError, ValueError, TypeError):
        return None


def candidate_is_complete(
    folder: Path,
    analysis_id: str,
    candidate: str,
    *,
    schema_version: str = SCHEMA_VERSION,
) -> bool:
    """返回候选是否已按当前分析 ID 完整落盘。"""

    completion = read_candidate_completion(folder, schema_version=schema_version)
    return bool(
        completion
        and completion.get("analysis_id") == analysis_id
        and completion.get("candidate") == candidate
    )
