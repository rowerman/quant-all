"""P4-4B2 分资金档位账本产物的原子写入和哈希校验。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_alpha.backtest_accept_b1_artifacts import (
    candidate_is_complete as b1_candidate_is_complete,
    read_candidate_completion as read_b1_candidate_completion,
    write_candidate_artifacts as write_b1_candidate_artifacts,
)

SCHEMA_VERSION = "p4_4b2_candidate_v1"


def write_tier_artifacts(
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
) -> dict:
    """写入单资金档位全部产物，最后写完成标记。"""

    return write_b1_candidate_artifacts(
        folder,
        analysis_id=analysis_id,
        candidate=candidate,
        metrics=metrics,
        strategy=strategy,
        yearly=yearly,
        daily=daily,
        orders=orders,
        holdings=holdings,
        run_manifest=run_manifest,
        schema_version=SCHEMA_VERSION,
    )


def read_tier_completion(folder: Path) -> dict | None:
    """读取并校验 P4-4B2 档位完成标记。"""

    return read_b1_candidate_completion(
        folder,
        schema_version=SCHEMA_VERSION,
    )


def tier_is_complete(
    folder: Path,
    analysis_id: str,
    candidate: str,
) -> bool:
    """返回资金档位是否已按当前分析 ID 完整落盘。"""

    return b1_candidate_is_complete(
        folder,
        analysis_id,
        candidate,
        schema_version=SCHEMA_VERSION,
    )
