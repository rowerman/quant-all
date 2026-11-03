"""按年面板、周频日期与基线分数的单元测试。"""

import numpy as np
import pandas as pd

from backtest.signals import rebalance_mask
from scripts.build_alpha.panel import (
    eligible_flags,
    linear_zscore_score,
    rank_score,
    weekly_dates,
)


def _calendar() -> pd.DatetimeIndex:
    """构造三周工作日日历。"""

    return pd.bdate_range("2024-01-01", "2024-01-19")


def test_weekly_dates_match_backtest_mask():
    """周最后交易日与回测 rebalance_mask 完全一致。"""

    calendar = _calendar()
    expected = calendar[rebalance_mask(calendar, "weekly").to_numpy()]
    assert weekly_dates(calendar).equals(expected)


def test_eligible_flags_require_trading_st_and_listing_days():
    """股票池口径与回测一致。"""

    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "list_date": pd.to_datetime(["2023-01-01", "2023-12-01", "2023-01-01"]),
            "can_buy_close": [True, True, True],
            "is_suspended": [False, False, False],
            "is_st": [False, False, True],
            "is_delisting": [False, False, False],
        }
    )
    assert eligible_flags(frame).tolist() == [True, False, False]


def test_linear_zscore_score_matches_manual_formula():
    """线性基线分数与手工截面 z-score 一致。"""

    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02"] * 3),
            "stock_code": ["A", "B", "C"],
            "eligible": [True, True, True],
            "X": [1.0, 2.0, 3.0],
            "Y": [3.0, 2.0, 1.0],
        }
    )
    score = linear_zscore_score(frame, {"X": 1.0, "Y": -1.0})
    expected = pd.Series(
        [-2.449489742783178, 0.0, 2.449489742783178],
        index=pd.Index(["A", "B", "C"], name="stock_code"),
        name="2024-01-02",
    )
    assert np.allclose(score.iloc[0].to_numpy(), expected.to_numpy())


def test_rank_score_uses_direction_and_eligible_cross_section():
    """rank 合成分数只使用可交易股票，并按因子方向调整。"""

    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02"] * 3),
            "stock_code": ["A", "B", "C"],
            "eligible": [True, True, False],
            "MOM_20": [1.0, 3.0, 2.0],
            "VOL_20": [3.0, 1.0, 2.0],
        }
    )
    score = rank_score(frame, ("MOM_20", "VOL_20"))
    assert pd.isna(score.loc["2024-01-02", "C"])
    assert score.loc["2024-01-02", "B"] > score.loc["2024-01-02", "A"]
