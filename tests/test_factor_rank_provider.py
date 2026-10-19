"""研究用 factor_rank 提供者的单元测试。"""

import numpy as np
import pandas as pd

from backtest.data import MarketData
from backtest.providers import get_provider
from backtest.specs import ProviderSpec
from backtest.strategies._factor_rank import (
    ensure_registered,
    rank_composite,
    weekly_last_dates,
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
