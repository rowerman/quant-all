"""P4-4A 逐候选产物的原子写入、哈希和恢复校验。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid
from typing import Any

import numpy as np
import pandas as pd

from alpha.artifacts import file_sha256
from alpha.errors import AlphaArtifactError

SCHEMA_VERSION = "p4_4a_candidate_v1"
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


def _json_default(value: Any) -> Any:
    """把时间、路径和 numpy 标量转换成稳定 JSON 值。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _temporary_path(path: Path) -> Path:
    """在目标文件旁边生成同后缀临时路径。"""

    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")


def atomic_write_json(path: Path, payload: Any) -> None:
    """原子写入 JSON，失败时保留原文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, content: str) -> None:
    """原子写入文本文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    """原子写入 CSV 或 Parquet，失败时保留原文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        if path.suffix == ".csv":
            frame.to_csv(temporary, index=False)
        elif path.suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        else:
            raise AlphaArtifactError(f"不支持的表格后缀：{path.suffix}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    """写入一个候选的全部产物，最后写完成标记。"""

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
            name: file_sha256(folder / name)
            for name in CANDIDATE_FILES
        },
    }
    atomic_write_json(folder / COMPLETE_FILE, completion)
    return completion


def read_candidate_completion(
    folder: Path,
    *,
    schema_version: str = SCHEMA_VERSION,
) -> dict | None:
    """读取并校验完成标记；缺失、损坏或哈希不符时返回 None。"""

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

    completion = read_candidate_completion(
        folder,
        schema_version=schema_version,
    )
    return bool(
        completion
        and completion.get("analysis_id") == analysis_id
        and completion.get("candidate") == candidate
    )


def read_json(path: Path) -> dict:
    """读取 JSON 对象。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AlphaArtifactError(f"JSON 产物必须是对象：{path}")
    return payload
