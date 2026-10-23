"""标准预测和模型产物写入工具。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.artifacts import prediction_sha256
from alpha.errors import AlphaModelError


def prediction_from_scores(rows: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """把一组模型分数整理为标准预测表。"""

    result = rows.loc[:, ["trade_date", "stock_code"]].copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result["stock_code"] = result["stock_code"].astype(str)
    result["score"] = np.asarray(scores, dtype="float64")
    result = result.sort_values(
        ["trade_date", "stock_code"], kind="mergesort"
    ).reset_index(drop=True)
    if result.duplicated(["trade_date", "stock_code"]).any():
        raise AlphaModelError("预测表存在重复日期和股票")
    if not np.isfinite(result["score"].to_numpy(dtype="float64")).all():
        raise AlphaModelError("预测表包含非有限分数")
    return result


def write_prediction_parts(
    frame: pd.DataFrame,
    folder: Path,
) -> dict[str, str]:
    """按年原子写入预测分片。"""

    hashes: dict[str, str] = {}
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    for year, group in work.groupby(work["trade_date"].dt.year, sort=True):
        target = folder / f"year={int(year)}" / "part.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.tmp")
        group.loc[:, ["trade_date", "stock_code", "score"]].to_parquet(
            temporary, index=False
        )
        os.replace(temporary, target)
        hashes[str(int(year))] = prediction_sha256(group)
    return hashes


def prediction_metadata(frame: pd.DataFrame, files: dict[str, str]) -> dict:
    """汇总预测分片元数据。"""

    return {
        "sha256": prediction_sha256(frame),
        "files": files,
        "rows": int(len(frame)),
        "dates": int(frame["trade_date"].nunique()),
        "start": str(frame["trade_date"].min().date()),
        "end": str(frame["trade_date"].max().date()),
    }
