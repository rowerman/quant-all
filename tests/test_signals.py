"""因子信号、股票池过滤与目标权重。"""

import numpy as np
import pandas as pd

from backtest.signals import build_scores, build_target_weights, eligible_mask, rebalance_mask
from backtest.specs import PortfolioSpec, RebalanceSpec, SignalSpec
from backtest.trading_rules import BSE_TRADING_START

INDEX = pd.date_range("2024-01-29", periods=4, freq="B")
COLUMNS = pd.Index(["SH600000", "SZ000001", "SZ000002"], name="stock_code")


def _frame(values) -> pd.DataFrame:
    """构造与默认面板同形的宽表。"""
    return pd.DataFrame([values] * len(INDEX), index=INDEX, columns=COLUMNS)


def test_monthly_rebalance_marks_period_end():
    mask = rebalance_mask(pd.date_range("2024-01-29", periods=4, freq="B"), "monthly")
    assert mask.iloc[2]
    assert mask.iloc[-1]
    assert mask.sum() == 2


def test_top_n_equal_weight_keeps_nan_on_non_signal_day(make_data):
    data = make_data(factors={"MOM_20": _frame([3.0, 2.0, 1.0]).astype("float32")})
    weights, ranks = build_target_weights(
        data, SignalSpec(standardize=False), PortfolioSpec(top_n=2), RebalanceSpec(frequency="daily")
    )
    day = data.index[0]
    assert weights.loc[day, "SH600000"] == 0.5
    assert weights.loc[day, "SZ000001"] == 0.5
    assert weights.loc[day, "SZ000002"] == 0.0
    assert ranks.loc[day, "SZ000001"] == 2.0


def test_market_cap_weighting_uses_total_mv(make_data):
    data = make_data(
        factors={"MOM_20": _frame([3.0, 2.0, 1.0]).astype("float32")},
        total_mv=_frame([100.0, 300.0, 600.0]).astype("float32"),
    )
    weights, _ = build_target_weights(
        data,
        SignalSpec(standardize=False),
        PortfolioSpec(top_n=3, weighting="market_cap"),
        RebalanceSpec(frequency="daily"),
    )
    assert np.allclose(weights.loc[data.index[0]].to_numpy(), [0.1, 0.3, 0.6])


def test_market_cap_falls_back_to_equal_when_missing(make_data):
    data = make_data(total_mv=_frame([np.nan, np.nan, np.nan]).astype("float32"))
    weights, _ = build_target_weights(
        data,
        SignalSpec(standardize=False),
        PortfolioSpec(top_n=2, weighting="market_cap"),
        RebalanceSpec(frequency="daily"),
    )
    assert weights.loc[data.index[0], "SH600000"] == 0.5


def test_multi_factor_linear_combination_respects_negative_weight(make_data):
    factors = {
        "MOM_20": _frame([1.0, 2.0, 3.0]).astype("float32"),
        "VOL_20": _frame([1.0, 2.0, 3.0]).astype("float32"),
    }
    data = make_data(factors=factors)
    scores = build_scores(data, SignalSpec(factors=("MOM_20", "VOL_20"), weights=(1.0, -1.0)))
    assert np.allclose(scores.iloc[0].to_numpy(), 0.0)
    weights, _ = build_target_weights(
        data,
        SignalSpec(factors=("MOM_20", "VOL_20"), weights=(1.0, -1.0)),
        PortfolioSpec(top_n=2),
        RebalanceSpec(frequency="daily"),
    )
    # 两个因子完全同向、权重相反，分数恒为 0，按并列名次取前两只
    assert np.allclose(weights.loc[data.index[0]].to_numpy(), [0.5, 0.5, 0.0])


def test_flat_factor_gives_no_selection(make_data):
    data = make_data()
    weights, _ = build_target_weights(
        data, SignalSpec(standardize=True), PortfolioSpec(top_n=3), RebalanceSpec(frequency="daily")
    )
    # 截面标准差为 0，标准化后全是 NaN，当天不选股
    assert weights.isna().all(axis=None)


def test_membership_filter_limits_the_pool(make_data):
    membership = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    membership["SZ000002"] = True
    data = make_data(membership=membership)
    mask = eligible_mask(data)
    assert mask["SZ000002"].all()
    assert not mask["SH600000"].any()
    weights, _ = build_target_weights(
        data, SignalSpec(standardize=False), PortfolioSpec(top_n=3), RebalanceSpec(frequency="daily")
    )
    assert weights.loc[INDEX[0], "SZ000002"] == 1.0
    assert weights.loc[INDEX[0], "SH600000"] == 0.0


def test_delisting_stock_is_excluded_and_top_n_backfills(make_data):
    delisting = pd.DataFrame(False, index=INDEX, columns=COLUMNS)
    delisting["SH600000"] = True
    data = make_data(
        is_delisting=delisting,
        factors={"MOM_20": _frame([3.0, 2.0, 1.0]).astype("float32")},
    )
    mask = eligible_mask(data)
    assert not mask["SH600000"].any()
    weights, _ = build_target_weights(
        data,
        SignalSpec(standardize=False),
        PortfolioSpec(top_n=2),
        RebalanceSpec(frequency="daily"),
    )
    assert weights.loc[INDEX[0], "SH600000"] == 0.0
    assert weights.loc[INDEX[0], "SZ000001"] == 0.5
    assert weights.loc[INDEX[0], "SZ000002"] == 0.5


def test_newly_listed_stock_is_excluded(make_data):
    data = make_data(list_date=pd.Series(pd.Timestamp("2024-01-29"), index=COLUMNS))
    mask = eligible_mask(data)
    assert not mask.any(axis=None)
    assert pd.Timestamp("2024-01-29") + pd.Timedelta(days=60) > data.index[-1]


def test_bse_is_not_eligible_before_trading_start(make_data):
    """北交所代码在 2021-11-15 前不进入可买股票池。"""

    dates = pd.DatetimeIndex(["2021-11-12", "2021-11-15"])
    stocks = ["BJ920000", "SH600000"]
    data = make_data(
        dates=dates,
        stocks=stocks,
        list_date=pd.Series(pd.Timestamp("2020-01-01"), index=stocks),
    )
    mask = eligible_mask(data)
    assert not mask.loc[pd.Timestamp("2021-11-12"), "BJ920000"]
    assert mask.loc[pd.Timestamp(BSE_TRADING_START), "BJ920000"]
    assert mask.loc[pd.Timestamp("2021-11-12"), "SH600000"]

