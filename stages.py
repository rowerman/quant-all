"""评估 stage 顺序：前后端的公共契约。

单独成模块，是为了让指标注册表也能按它排序，避免出现
「注册表 ↔ 评估入口」的循环依赖。后续阶段只往空位里填内容，不改顺序。
"""

from __future__ import annotations

STAGE_ORDER: tuple[str, ...] = (
    "meta",
    "overview",
    "ic_series",
    "ic_dist",
    "bins",
    "bin_curves",
    "spread",
    "bin_dist",
    "turnover",
    "decay",
    "exposure",
    "incremental_ic",
    "correlation",
    "done",
)
