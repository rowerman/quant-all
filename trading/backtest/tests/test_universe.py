"""指数成分池口径：上月月末快照、无前视、缺失月份。"""

import pandas as pd
import pytest

from backtest import universe as universe_module
from backtest.errors import ResourceLimitError, SpecError
from backtest.universe import (
    check_limits,
    membership_frame,
    resolve_universe,
    universe_stocks,
)


@pytest.fixture
def fake_index(monkeypatch):
    """把日历与权重快照换成可控的小样本。"""
    calendar = pd.DatetimeIndex(
        [
            "2024-01-31", "2024-02-29", "2024-03-29",
            "2024-04-02", "2024-04-30", "2024-05-06", "2024-05-31",
        ]
    )
    weights = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"]
            ),
            "stock_code": ["SH600000", "SH600000", "SZ000001", "SZ000001"],
            "weight": [10.0, 9.0, 8.0, 0.0],
        }
    )
    monkeypatch.setattr(
        universe_module, "_calendar", lambda end, db_path: calendar[calendar <= pd.Timestamp(end)]
    )
    monkeypatch.setattr(
        universe_module,
        "_weights",
        lambda code, end, db_path: weights.loc[weights["trade_date"] <= pd.Timestamp(end)],
    )
    return weights


def test_membership_uses_previous_month_end_snapshot(fake_index):
    spec = resolve_universe("index:000300.SH")
    index = pd.DatetimeIndex(["2024-04-02", "2024-04-30", "2024-05-06"])
    columns = pd.Index(["SH600000", "SZ000001"], name="stock_code")
    membership = membership_frame(spec, index, columns, "2024-05-31")
    # 4 月用 3 月末快照（只有 SZ000001），5 月用 4 月末快照（权重为 0，视为无成分）
    assert membership.loc["2024-04-02", "SZ000001"]
    assert not membership.loc["2024-04-02", "SH600000"]
    assert not membership.loc["2024-05-06"].any()


def test_universe_stocks_returns_member_union(fake_index):
    spec = resolve_universe("index:000300.SH")
    assert universe_stocks(spec, "2024-04-01", "2024-05-31") == ("SZ000001",)
    assert universe_stocks(resolve_universe("all_a"), "2024-04-01", "2024-05-31") is None


def test_membership_raises_for_all_a_pool(fake_index):
    spec = resolve_universe("all_a")
    with pytest.raises(SpecError, match="不是指数成分池"):
        membership_frame(spec, pd.DatetimeIndex(["2024-04-02"]), pd.Index(["SH600000"]), "2024-05-31")


def test_resolve_universe_reports_unknown_key():
    with pytest.raises(SpecError, match="未知股票池"):
        resolve_universe("index:999999.XX")


def test_check_limits_blocks_long_and_wide_panels():
    check_limits(2500, 5000, 2520, 16_000_000)
    with pytest.raises(ResourceLimitError, match="超过上限"):
        check_limits(2600, 100, 2520, 16_000_000)
    with pytest.raises(ResourceLimitError, match="单元格"):
        check_limits(2500, 7000, 2520, 16_000_000)

