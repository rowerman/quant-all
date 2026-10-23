"""最终留出集隐藏 provider 的日期和权重测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.errors import SpecError
from backtest.strategies._final_holdout import FinalHoldoutRankProvider

STOCKS = ["SH600000", "SZ000001", "SZ000002", "SZ000003"]
DATES = pd.DatetimeIndex(["2026-09-17", "2026-09-24", "2026-09-30"])


def _data():
    """构造 provider 所需的最小日期和股票面板。"""

    return type(
        "Data",
        (),
        {
            "index": DATES,
            "columns": pd.Index(STOCKS, name="stock_code"),
        },
    )()


def _predictions() -> pd.DataFrame:
    """构造三日四股票的确定性预测。"""

    rows = []
    for day_index, day in enumerate(DATES):
        for stock_index, stock in enumerate(STOCKS):
            rows.append(
                {
                    "trade_date": day,
                    "stock_code": stock,
                    "score": float(day_index * 10 + stock_index),
                }
            )
    return pd.DataFrame(rows)


def test_params_require_explicit_dates_and_known_candidate():
    """参数必须显式给出留出 ID、候选和日期。"""

    provider = FinalHoldoutRankProvider()
    params = provider.validate_params(
        {
            "holdout_id": "final_holdout_123456789abc",
            "candidate": "lightgbm",
            "top_n": 2,
            "smoothing_weeks": 1,
            "rebalance_dates": ["2026-09-30", "2026-09-24"],
        }
    )
    assert params["rebalance_dates"] == ["2026-09-24", "2026-09-30"]
    with pytest.raises(SpecError, match="candidate"):
        provider.validate_params(
            {
                "holdout_id": "final_holdout_123456789abc",
                "candidate": "unknown",
                "rebalance_dates": ["2026-09-24"],
            }
        )
    with pytest.raises(SpecError, match="rebalance_dates"):
        provider.validate_params(
            {
                "holdout_id": "final_holdout_123456789abc",
                "candidate": "lightgbm",
            }
        )


def test_provider_smooths_and_only_allocates_to_eligible(monkeypatch):
    """选股使用因果关系平滑且不给不可交易股票正权重。"""

    provider = FinalHoldoutRankProvider()
    monkeypatch.setattr(
        provider,
        "_load_predictions",
        staticmethod(
            lambda holdout_id, name: (
                _predictions(),
                {
                    "schema_version": "alpha_final_holdout_predictions_v1",
                    "predictions": {},
                },
            )
        ),
    )
    eligible = pd.DataFrame(True, index=DATES, columns=STOCKS)
    eligible.loc["2026-09-30", "SZ000002"] = False
    weights = provider.build(
        _data(),
        eligible,
        {
            "holdout_id": "final_holdout_123456789abc",
            "candidate": "lightgbm",
            "top_n": 2,
            "smoothing_weeks": 2,
            "rebalance_dates": [str(day.date()) for day in DATES],
        },
    )
    assert weights.loc["2026-09-17"].notna().all()
    assert weights.loc["2026-09-24"].notna().all()
    assert weights.loc["2026-09-30"].notna().all()
    assert weights.loc["2026-09-30", "SZ000002"] == 0.0
    for day in DATES:
        active = weights.loc[day]
        assert np.isclose(active.sum(), 1.0)
        assert (active >= 0).all()
