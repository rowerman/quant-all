"""模型分数的历史平滑与 Top-N 持仓规则。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from alpha.errors import AlphaEvaluationError


@dataclass(frozen=True)
class HoldingPolicy:
    """组合持仓规则；默认只保留当日前 N 名。"""

    kind: Literal["top_n", "rank_buffer"] = "top_n"
    buffer_rank: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("top_n", "rank_buffer"):
            raise AlphaEvaluationError(f"未知持仓规则：{self.kind!r}")
        if self.kind == "top_n" and self.buffer_rank is not None:
            raise AlphaEvaluationError("top_n 规则不能设置 buffer_rank")
        if self.kind == "rank_buffer":
            if type(self.buffer_rank) is not int or self.buffer_rank <= 0:
                raise AlphaEvaluationError(
                    "rank_buffer 规则要求 buffer_rank 为正整数"
                )


def smooth_rank_scores(
    scores: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    window: int = 4,
    min_periods: int = 1,
) -> pd.DataFrame:
    """按日截面 rank 后做历史滚动平均，不使用未来信号。"""

    if type(window) is not int or window <= 0:
        raise AlphaEvaluationError("平滑窗口必须是正整数")
    if type(min_periods) is not int or not 1 <= min_periods <= window:
        raise AlphaEvaluationError("min_periods 必须在 1 和窗口长度之间")
    if scores.index.has_duplicates or scores.columns.has_duplicates:
        raise AlphaEvaluationError("分数面板的日期或股票存在重复")
    try:
        numeric = scores.astype("float64")
    except (TypeError, ValueError) as error:
        raise AlphaEvaluationError("分数面板必须是数值") from error
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    eligible_frame = (
        eligible.reindex(index=numeric.index, columns=numeric.columns)
        .fillna(False)
        .astype(bool)
    )
    original_index = numeric.index
    ordered = numeric.sort_index()
    ordered_eligible = eligible_frame.reindex(index=ordered.index)
    ranked = ordered.where(ordered_eligible).rank(
        axis=1,
        method="average",
        pct=True,
    )
    smoothed = ranked.rolling(window=window, min_periods=min_periods).mean()
    return smoothed.reindex(index=original_index).astype("float64")


def select_holdings(
    score: pd.Series,
    eligible: pd.Series,
    previous: Sequence[object] | None,
    *,
    top_n: int,
    policy: HoldingPolicy | None = None,
) -> pd.Index:
    """按分数和持仓规则选出目标持仓，并补齐到 top_n。"""

    if type(top_n) is not int or top_n <= 0:
        raise AlphaEvaluationError("top_n 必须是正整数")
    rule = policy or HoldingPolicy()
    try:
        numeric = pd.to_numeric(score, errors="coerce").astype("float64")
    except (TypeError, ValueError) as error:
        raise AlphaEvaluationError("分数序列必须是数值") from error
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    eligible_series = (
        eligible.reindex(numeric.index).fillna(False).astype(bool)
    )
    candidates = numeric.where(eligible_series).dropna().sort_index()
    ordered = candidates.sort_values(ascending=False, kind="mergesort").index
    if rule.kind == "top_n":
        return pd.Index(ordered[:top_n])
    if rule.buffer_rank is None or rule.buffer_rank < top_n:
        raise AlphaEvaluationError("buffer_rank 不能小于 top_n")

    cutoff = min(len(ordered), rule.buffer_rank)
    rank = {stock: index + 1 for index, stock in enumerate(ordered)}
    previous_values = pd.Index(previous) if previous is not None else pd.Index([])
    kept: list[object] = []
    seen: set[object] = set()
    for stock in previous_values:
        if stock in seen or stock not in rank or rank[stock] > cutoff:
            continue
        kept.append(stock)
        seen.add(stock)
    kept.sort(key=rank.__getitem__)
    for stock in ordered:
        if len(kept) >= top_n:
            break
        if stock not in seen:
            kept.append(stock)
            seen.add(stock)
    return pd.Index(kept[:top_n])


__all__ = ["HoldingPolicy", "select_holdings", "smooth_rank_scores"]
