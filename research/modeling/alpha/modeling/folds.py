"""训练折读取和开发期校验。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from alpha.artifacts import validate_dev_range
from alpha.config import MODEL_DATA_DIR
from alpha.errors import AlphaModelError
from alpha.split import Fold


def load_folds(path: Path | None = None) -> tuple[Fold, ...]:
    """读取 walk-forward 切分表。"""

    target = path or MODEL_DATA_DIR / "splits" / "walk_forward_v1.csv"
    if not target.is_file():
        raise AlphaModelError(f"切分表不存在：{target}")
    frame = pd.read_csv(target)
    required = {
        "fold_id",
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
        "test_start",
        "test_end",
        "purge_days",
        "embargo_days",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AlphaModelError(f"切分表缺少列：{missing}")
    folds = tuple(
        Fold(
            fold_id=str(row.fold_id),
            train_start=str(row.train_start),
            train_end=str(row.train_end),
            valid_start=str(row.valid_start),
            valid_end=str(row.valid_end),
            test_start=str(row.test_start),
            test_end=str(row.test_end),
            purge_days=int(row.purge_days),
            embargo_days=int(row.embargo_days),
        )
        for row in frame.itertuples(index=False)
    )
    if not folds:
        raise AlphaModelError("切分表没有可用折")
    return folds


def validate_folds(folds: Sequence[Fold]) -> None:
    """拒绝跨入未开放最终测试集的折。"""

    if not folds:
        raise AlphaModelError("没有可训练的折")
    for fold in folds:
        validate_dev_range(fold.test_start, fold.test_end)
