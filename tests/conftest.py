"""回测测试共用的输入面板构造器。"""

import pandas as pd
import pytest

from backtest.data import MarketData

DEFAULT_DATES = pd.date_range("2024-01-29", periods=4, freq="B")
DEFAULT_STOCKS = ["SH600000", "SZ000001", "SZ000002"]


@pytest.fixture
def make_data():
    """返回构造 ``MarketData`` 的工厂，默认 4 个交易日、3 只股票、恒定价格。"""

    def factory(*, dates=None, stocks=None, factors=None, **overrides):
        index = pd.DatetimeIndex(dates) if dates is not None else DEFAULT_DATES
        columns = pd.Index(list(stocks) if stocks is not None else DEFAULT_STOCKS, name="stock_code")

        def panel(value, dtype="float64"):
            if not isinstance(value, pd.DataFrame):
                value = pd.DataFrame(value, index=index, columns=columns)
            return value.astype(dtype)

        fields = {
            "open": panel(10.0, "float32"),
            "close": panel(10.5, "float32"),
            "can_buy_close": panel(True, bool),
            "can_buy_open_next": panel(True, bool),
            "can_sell_open_next": panel(True, bool),
            "is_suspended": panel(False, bool),
            "is_st": panel(False, bool),
            "is_limit_up": panel(False, bool),
            "is_limit_down": panel(False, bool),
            "total_mv": panel(1000.0, "float32"),
            "list_date": pd.Series(pd.Timestamp("2023-01-01"), index=columns),
            "factors": factors if factors is not None else {"MOM_20": panel(1.0, "float32")},
        }
        fields.update(overrides)
        return MarketData(**fields)

    return factory


@pytest.fixture
def probe_factory():
    """返回构造探针成交明细的工厂。"""

    def factory(rows):
        if not rows:
            return pd.DataFrame(columns=["trade_date", "stock_code", "side", "value"])
        return pd.DataFrame(rows, columns=["trade_date", "stock_code", "side", "value"])

    return factory

