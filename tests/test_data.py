"""回测输入面板的契约与真实取数。"""

import numpy as np
import pandas as pd
import pytest

from backtest.data import PANEL_FIELDS, load_market_data
from backtest.errors import DataAlignmentError, ResourceLimitError, SpecError


def test_market_data_rejects_misaligned_panels(make_data):
    index = pd.date_range("2024-01-29", periods=3, freq="B")
    stocks = pd.Index(["SH600000", "SZ000001"], name="stock_code")
    short = pd.DataFrame(True, index=index[:2], columns=stocks)
    with pytest.raises(DataAlignmentError, match="形状不一致"):
        make_data(dates=index, stocks=stocks, can_buy_close=short)


def test_market_data_rejects_factor_with_wrong_columns(make_data):
    index = pd.date_range("2024-01-29", periods=3, freq="B")
    stocks = pd.Index(["SH600000", "SZ000001"], name="stock_code")
    wrong = pd.DataFrame(1.0, index=index, columns=["SH600000", "SZ000002"])
    with pytest.raises(DataAlignmentError, match="因子"):
        make_data(dates=index, stocks=stocks, factors={"MOM_20": wrong.astype("float32")})


def test_market_data_crop_keeps_interval(make_data):
    index = pd.date_range("2024-01-29", periods=5, freq="B")
    data = make_data(dates=index)
    cropped = data.crop(str(index[1].date()), str(index[3].date()))
    assert list(cropped.index) == [index[1], index[2], index[3]]
    assert cropped.list_date.equals(data.list_date)
    with pytest.raises(DataAlignmentError):
        data.crop("2024-06-01", "2024-06-30")


def test_market_data_crop_is_a_noop_inside_range(make_data):
    index = pd.date_range("2024-01-29", periods=3, freq="B")
    data = make_data(dates=index)
    assert data.crop(str(index[0].date()), str(index[-1].date())) is data


def test_load_market_data_reads_real_panel():
    """真实库短区间冒烟：形状、类型与基准对齐。"""
    data = load_market_data("2026-08-03", "2026-08-14", ["MOM_20"])
    assert data.membership is None
    assert len(data.index) == 10
    assert len(data.columns) > 4000
    assert data.open.dtypes.iloc[0] == np.dtype("float32")
    assert data.can_buy_close.dtypes.iloc[0] == np.dtype("bool")
    assert data.factors["MOM_20"].shape == data.open.shape
    assert data.benchmark is not None
    assert data.benchmark.notna().all()
    for name in PANEL_FIELDS:
        assert getattr(data, name).shape == data.open.shape


def test_load_market_data_applies_index_membership():
    """指数成分池只取成分并集，并给出成分布尔矩阵。"""
    data = load_market_data("2026-06-01", "2026-08-31", ["MOM_20"], universe="index:000300.SH")
    assert data.membership is not None
    assert len(data.columns) < 1200
    assert data.membership.to_numpy().any()
    # 成分布尔矩阵逐日有成分，且成分列的并集等于面板股票列
    assert data.membership.any(axis=1).all()
    assert set(data.columns) == set(data.membership.columns)


def test_load_market_data_rejects_unknown_universe():
    with pytest.raises(SpecError, match="未知股票池"):
        load_market_data("2026-08-03", "2026-08-14", ["MOM_20"], universe="index:999999.XX")


def test_load_market_data_blocks_too_long_range():
    with pytest.raises(ResourceLimitError, match="超过上限"):
        load_market_data("2000-01-04", "2026-09-28", ["MOM_20"])

