"""模型产物路径、哈希与 JSON 写入工具。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha.config import (
    ARCHIVE_MODELS_DIR,
    DEFAULT_CONTRACT,
    MODELS_DIR,
)
from alpha.errors import AlphaArtifactError, AlphaConfigError

MODEL_SCHEMA_VERSION = "alpha_model_v1"
MODEL_SCHEMA_VERSION_V2 = "alpha_model_v2"
PREDICTION_COLUMNS = ("trade_date", "stock_code", "score")


def json_safe(value: Any) -> Any:
    """把 numpy、路径和路径对象转换为稳定 JSON 值。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    """原子写入稳定格式 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    """计算文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_sha256(frame: pd.DataFrame) -> str:
    """按预测表键和取值计算与 Parquet 元数据无关的 SHA256。"""

    missing = sorted(set(PREDICTION_COLUMNS) - set(frame.columns))
    if missing:
        raise AlphaArtifactError(f"预测表缺少列：{missing}")
    work = frame.loc[:, list(PREDICTION_COLUMNS)].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work = work.sort_values(["trade_date", "stock_code"], kind="mergesort")
    if work.duplicated(["trade_date", "stock_code"]).any():
        raise AlphaArtifactError("预测表存在重复日期和股票")
    scores = pd.to_numeric(work["score"], errors="coerce").to_numpy(dtype="<f8")
    if not np.isfinite(scores).all():
        raise AlphaArtifactError("预测分数存在缺失或非有限值")
    digest = hashlib.sha256()
    digest.update(str(len(work)).encode("ascii"))
    digest.update(work["trade_date"].dt.strftime("%Y-%m-%d").to_numpy(dtype="U10").tobytes())
    digest.update(work["stock_code"].astype(str).to_numpy(dtype="U16").tobytes())
    digest.update(scores.tobytes())
    return digest.hexdigest()


def model_dir(
    model_id: str,
    root: Path | None = None,
    *,
    include_archived: bool = True,
) -> Path:
    """返回模型目录；优先活跃目录，必要时查找归档目录。"""

    if not model_id or Path(model_id).name != model_id:
        raise AlphaArtifactError(f"非法模型 ID：{model_id!r}")
    if root is None:
        active = MODELS_DIR / model_id
        archived = ARCHIVE_MODELS_DIR / model_id
    else:
        base = Path(root)
        candidates = [base / "models" / model_id]
        if include_archived:
            candidates.append(base / "archive" / "models" / model_id)
        candidates.append(base / model_id)
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return candidates[0]
    if active.is_dir() or not include_archived:
        return active
    if archived.is_dir():
        return archived
    return active


def validate_dev_range(
    start: str,
    end: str,
    *,
    allow_holdout: bool = False,
    holdout_window: object | None = None,
) -> None:
    """校验日期区间；默认拒绝跨入最终测试集。"""

    try:
        start_stamp = pd.Timestamp(start)
        end_stamp = pd.Timestamp(end)
    except (TypeError, ValueError) as error:
        raise AlphaConfigError(f"非法日期区间：{start!r} ~ {end!r}") from error
    if start_stamp > end_stamp:
        raise AlphaConfigError(f"起始日 {start} 晚于截止日 {end}")
    if allow_holdout:
        from alpha.holdout import LEGACY_HOLDOUT_WINDOW_ID, get_holdout_window

        window = holdout_window or get_holdout_window(LEGACY_HOLDOUT_WINDOW_ID)
        window_start = pd.Timestamp(window.start)
        window_end = pd.Timestamp(window.end)
        if start_stamp < window_start:
            raise AlphaConfigError(
                f"留出集起始日 {start} 早于窗口 {window.window_id} 的 "
                f"{window.start}"
            )
        if end_stamp > window_end:
            raise AlphaConfigError(
                f"留出集截止日 {end} 超过窗口 {window.window_id} 的 "
                f"{window.end}"
            )
        return
    if start_stamp < pd.Timestamp(DEFAULT_CONTRACT.dev_start):
        raise AlphaConfigError(
            f"起始日 {start} 早于研究契约 {DEFAULT_CONTRACT.dev_start}"
        )
    if end_stamp > pd.Timestamp(DEFAULT_CONTRACT.dev_end):
        raise AlphaConfigError(
            f"截止日 {end} 超过开发期 {DEFAULT_CONTRACT.dev_end}；最终测试集未开放"
        )
