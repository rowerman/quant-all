"""Purged walk-forward 切分的单元测试。"""

import pandas as pd
import pytest

from alpha.split import (
    SplitConfig,
    build_folds,
    fold_masks,
    validate_label_gap,
)
from alpha.errors import AlphaSplitError


def _calendar() -> pd.DatetimeIndex:
    """构造 2012-2023 工作日日历。"""

    return pd.bdate_range("2012-01-02", "2023-12-29")


def test_build_folds_has_six_expanding_years():
    """默认得到 2018-2023 六折，训练期从 2012 开始。"""

    folds = build_folds(_calendar(), SplitConfig())
    assert [fold.fold_id for fold in folds] == [
        f"wf_{year}" for year in range(2018, 2024)
    ]
    assert folds[0].train_start == "2012-01-02"
    assert folds[-1].test_end < "2024-01-01"


def test_purge_and_embargo_leave_label_gap():
    """训练/验证和验证/测试之间都留出 h+1 个交易日。"""

    calendar = _calendar()
    folds = build_folds(calendar, SplitConfig())
    for fold in folds:
        validate_label_gap(calendar, fold)


def test_holdout_is_never_a_test_year():
    """切分器拒绝把 2024 年作为测试年。"""

    with pytest.raises(AlphaSplitError, match="最终测试集"):
        build_folds(
            pd.bdate_range("2012-01-02", "2024-12-31"),
            SplitConfig(last_test_year=2024),
        )


def test_fold_masks_are_disjoint():
    """训练、验证、测试日期掩码互不重叠。"""

    fold = build_folds(_calendar(), SplitConfig())[0]
    dates = pd.date_range("2012-01-02", "2023-12-29", freq="B")
    data = pd.DataFrame({"trade_date": dates})
    masks = fold_masks(data, fold)
    assert not (masks["train"] & masks["valid"]).any()
    assert not (masks["valid"] & masks["test"]).any()
    assert not (masks["train"] & masks["test"]).any()


def test_rolling_mode_uses_requested_window():
    """rolling 模式训练起点按窗口年份前移。"""

    folds = build_folds(
        _calendar(),
        SplitConfig(mode="rolling", train_window_years=2),
    )
    first = folds[0]
    assert first.train_start > "2012-01-02"
    assert first.train_start <= first.train_end
