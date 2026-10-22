"""研究用 factor_rank 提供者的单元测试。"""

import numpy as np
import pandas as pd

from backtest.data import MarketData
from backtest.providers import get_provider, list_providers
from backtest.signals import eligible_mask, build_target_weights
from backtest.specs import (
    PortfolioSpec,
    ProviderSpec,
    RebalanceSpec,
    SignalSpec,
)
from backtest.strategies._factor_rank import (
    ensure_registered,
    rank_composite,
    weekly_last_dates,
)
from backtest.strategies._factor_rank_v2 import ensure_registered as ensure_v2_registered
from backtest.strategies._factor_linear import (
    ensure_registered as ensure_linear_registered,
)
from backtest.trading_rules import board_from_stock_code


def _calendar() -> pd.DataFrame:
    """构造三周工作日日历。"""

    return pd.DataFrame({"trade_date": pd.bdate_range("2024-01-01", "2024-01-19")})


def _data() -> MarketData:
    """构造六日、四只股票的最小行情面板。"""

    index = pd.bdate_range("2024-01-01", periods=6)
    columns = pd.Index(["SH600000", "SZ000001", "SZ000002", "SH600004"], name="stock_code")

    def panel(values, dtype="float64"):
        return pd.DataFrame(values, index=index, columns=columns).astype(dtype)

    mom = panel(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 4.0, 3.0],
            [1.0, 3.0, 2.0, 4.0],
            [3.0, 1.0, 2.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        "float32",
    )
    vol = panel(
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 4.0, 3.0],
            [4.0, 2.0, 3.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        "float32",
    )
    return MarketData(
        open=panel(10.0, "float32"),
        close=panel(10.5, "float32"),
        can_buy_close=panel(True, bool),
        can_buy_open_next=panel(True, bool),
        can_sell_open_next=panel(True, bool),
        is_suspended=panel(False, bool),
        is_st=panel(False, bool),
        is_delisting=panel(False, bool),
        is_limit_up=panel(False, bool),
        is_limit_down=panel(False, bool),
        total_mv=panel(1000.0, "float32"),
        amount_cny=panel(1e9, "float32"),
        factors={"MOM_20": mom, "VOL_20": vol},
        list_date=pd.Series(pd.Timestamp("2023-01-01"), index=columns),
        board=pd.Series([board_from_stock_code(code) for code in columns], index=columns),
    )


def test_weekly_last_dates_are_chunk_independent():
    """分块只给局部日期时，仍按完整日历识别周最后交易日。"""

    calendar = _calendar()
    full = weekly_last_dates(calendar["trade_date"], calendar)
    partial = weekly_last_dates(
        pd.DatetimeIndex(calendar["trade_date"]).tolist()[:8], calendar
    )
    assert pd.Timestamp("2024-01-05") in full
    assert pd.Timestamp("2024-01-05") in partial
    assert pd.Timestamp("2024-01-12") not in partial


def test_rank_composite_excludes_ineligible_stock():
    """不可交易股票不进入当日分位。"""

    data = _data()
    eligible = pd.DataFrame(
        True, index=data.index, columns=data.columns
    )
    eligible.iloc[:, 3] = False
    score = rank_composite(data.factors, eligible, ("MOM_20", "VOL_20"))
    assert score.iloc[:, 3].isna().all()
    assert score.iloc[0, :3].notna().all()


def test_provider_builds_weekly_top_n_equal_weights():
    """提供者在周最后交易日给出 Top-N 等权，其他日期保持 NaN。"""

    ensure_registered()
    provider = get_provider(ProviderSpec("factor_rank", "1", {"top_n": 2, "factor_names": ["MOM_20", "VOL_20"]}))
    data = _data()
    eligible = pd.DataFrame(True, index=data.index, columns=data.columns)
    weights = provider.build(data, eligible, {"top_n": 2, "factor_names": ["MOM_20", "VOL_20"]})
    friday = pd.Timestamp("2024-01-05")
    assert weights.loc[friday].notna().all()
    positive = weights.loc[friday][weights.loc[friday] > 0]
    assert len(positive) == 2
    assert np.allclose(positive.to_numpy(), 0.5)
    assert weights.loc[pd.Timestamp("2024-01-03")].isna().all()


def test_v2_provider_uses_explicit_rebalance_dates():
    """v2 只在显式日期生成目标，v1 仍可通过注册表取得。"""

    ensure_v2_registered()
    assert get_provider(ProviderSpec("factor_rank", "1")).version == "1"
    provider = get_provider(
        ProviderSpec(
            "factor_rank",
            "2",
            {
                "top_n": 2,
                "factor_names": ["MOM_20", "VOL_20"],
                "rebalance_dates": ["2024-01-03", "2024-01-04"],
            },
        )
    )
    data = _data()
    eligible = pd.DataFrame(True, index=data.index, columns=data.columns)
    weights = provider.build(
        data,
        eligible,
        {
            "top_n": 2,
            "factor_names": ["MOM_20", "VOL_20"],
            "rebalance_dates": ["2024-01-03", "2024-01-04"],
        },
    )
    assert weights.loc["2024-01-03"].notna().all()
    assert weights.loc["2024-01-04"].notna().all()
    assert weights.loc["2024-01-05"].isna().all()


def test_factor_linear_matches_direct_baseline_on_explicit_dates():
    """隐藏线性 provider 与现有直接因子策略在相同日期选股一致。"""

    ensure_linear_registered()
    data = _data()
    day = pd.Timestamp("2024-01-03")
    params = {
        "top_n": 2,
        "factor_names": ["MOM_20", "VOL_20"],
        "factor_weights": [1.0, -1.0],
        "standardize": True,
        "winsorize": False,
        "rebalance_dates": [str(day.date())],
    }
    provider = get_provider(ProviderSpec("factor_linear", "1", params))
    actual = provider.build(data, eligible_mask(data), params)
    mask = pd.Series(False, index=data.index)
    mask.loc[day] = True
    expected, _ = build_target_weights(
        data,
        SignalSpec(
            factors=("MOM_20", "VOL_20"),
            weights=(1.0, -1.0),
            standardize=True,
            winsorize=False,
        ),
        PortfolioSpec(selection="top_n", top_n=2, weighting="equal"),
        RebalanceSpec(frequency="weekly"),
        rebalance_days=mask,
    )
    pd.testing.assert_series_equal(
        actual.loc[day],
        expected.loc[day],
        check_names=False,
    )
    assert actual.drop(index=day).isna().all().all()
    assert "factor_linear" not in {item["name"] for item in list_providers()}
