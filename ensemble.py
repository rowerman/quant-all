"""日截面 rank 组合与成员预测校验。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alpha.errors import AlphaModelError

PREDICTION_COLUMNS = ("trade_date", "stock_code", "score")


def _validate_predictions(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """校验并整理一组标准预测表。"""

    missing = sorted(set(PREDICTION_COLUMNS) - set(frame.columns))
    if missing:
        raise AlphaModelError(f"{name} 预测缺少列：{missing}")
    work = frame.loc[:, list(PREDICTION_COLUMNS)].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["stock_code"] = work["stock_code"].astype(str)
    work = work.sort_values(
        ["trade_date", "stock_code"], kind="mergesort"
    ).reset_index(drop=True)
    if work.empty:
        raise AlphaModelError(f"{name} 预测不能为空")
    if work.duplicated(["trade_date", "stock_code"]).any():
        raise AlphaModelError(f"{name} 预测存在重复日期和股票")
    scores = pd.to_numeric(work["score"], errors="coerce").to_numpy(
        dtype="float64"
    )
    if not np.isfinite(scores).all():
        raise AlphaModelError(f"{name} 预测包含缺失或非有限分数")
    work["score"] = scores
    return work


def rank_by_date(frame: pd.DataFrame, *, name: str = "成员") -> pd.DataFrame:
    """把成员分数转换为每个日期截面内的百分比 rank。"""

    work = _validate_predictions(frame, name)
    work["score"] = work.groupby("trade_date", sort=False)["score"].rank(
        method="average",
        pct=True,
    )
    return work


def combine_rank_predictions(
    frames: Sequence[pd.DataFrame],
    *,
    weights: Sequence[float] | None = None,
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """按日截面 rank 加权组合成员预测。"""

    if len(frames) < 2:
        raise AlphaModelError("预测组合至少需要两个成员")
    labels = tuple(names or (f"成员{index + 1}" for index in range(len(frames))))
    if len(labels) != len(frames):
        raise AlphaModelError("成员名称数量与预测数量不一致")
    if weights is None:
        values = np.full(len(frames), 1.0 / len(frames), dtype="float64")
    else:
        values = np.asarray(weights, dtype="float64")
        if values.shape != (len(frames),):
            raise AlphaModelError("成员权重数量与预测数量不一致")
        if not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any():
            raise AlphaModelError("成员权重必须有限、非负且至少一个为正")
        values = values / values.sum()

    ranked = [
        rank_by_date(frame, name=label)
        for frame, label in zip(frames, labels)
    ]
    base = ranked[0]
    dates = base["trade_date"].to_numpy()
    stocks = base["stock_code"].to_numpy(dtype=str)
    for label, member in zip(labels[1:], ranked[1:]):
        if not np.array_equal(dates, member["trade_date"].to_numpy()):
            raise AlphaModelError(f"{label} 预测日期与基准成员不一致")
        if not np.array_equal(stocks, member["stock_code"].to_numpy(dtype=str)):
            raise AlphaModelError(f"{label} 预测股票与基准成员不一致")

    score_matrix = np.column_stack(
        [member["score"].to_numpy(dtype="float64") for member in ranked]
    )
    result = base.loc[:, ["trade_date", "stock_code"]].copy()
    result["score"] = score_matrix @ values
    if not np.isfinite(result["score"].to_numpy(dtype="float64")).all():
        raise AlphaModelError("组合预测包含非有限分数")
    return result
