"""回测输入面板的契约与真实取数。"""

import numpy as np
import pandas as pd
import pytest

import quant
from backtest.data import (
    PANEL_FIELDS,
    PRICE_ADJUST,
    STATE_ADJUST,
    _load_field,
    _load_versions,
    load_market_data,
)
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
    assert data.versions["dataset"]["data_version"].startswith("tushare_fullA_")
    assert data.delist_date is not None
    for name in PANEL_FIELDS:
        assert getattr(data, name).shape == data.open.shape


def test_price_fields_use_adjusted_prices(monkeypatch):
    """价格字段走前复权口径，交易标记与市值保持原始口径。"""
    seen: dict[str, str] = {}

    def fake_load_panel(start=None, end=None, fields=None, adjust="raw", **kwargs):
        seen[fields[0]] = adjust
        return pd.DataFrame(
            {
                "trade_date": [pd.Timestamp("2024-01-29")],
                "stock_code": ["SH600000"],
                fields[0]: [1.0],
            }
        )

    monkeypatch.setattr(quant, "load_panel", fake_load_panel)
    for field in PANEL_FIELDS:
        _load_field(field, "2024-01-01", "2024-01-31", None, None)

    assert PRICE_ADJUST == "qfq" and STATE_ADJUST == "raw"
    assert seen["open"] == "qfq"
    assert seen["close"] == "qfq"
    assert seen["total_mv"] == "raw"
    assert seen["is_st"] == "raw"
    assert seen["can_buy_open_next"] == "raw"


def test_close_has_no_fake_gap_on_ex_dividend():
    """除权日不做假跳空：SZ002594 在 2025-07-29 每 10 股送转 20 股。

    未复权收盘从 337.00 跌到 111.42（-66.9%），后复权口径当日只有 +0.37%。
    """
    data = load_market_data(
        "2025-07-21", "2025-08-08", ["MOM_20"], universe="index:000300.SH"
    )
    assert "SZ002594" in set(data.columns)
    day = pd.Timestamp("2025-07-29")
    close = data.close["SZ002594"].dropna()
    hfq_return = close.loc[day] / close.loc[:day].iloc[-2] - 1.0

    raw = quant.load_panel(
        start="2025-07-25", end="2025-07-29", fields=["close"], adjust="raw",
        stocks=["SZ002594"],
    ).set_index("trade_date")["close"]
    raw_return = raw.loc[day] / raw.iloc[-2] - 1.0

    assert raw_return < -0.5
    assert abs(hfq_return) < 0.11
    assert hfq_return == pytest.approx(0.0037, abs=0.02)


def test_versions_are_read_from_manifests():
    """运行元数据里的数据版本与因子版本来自两个库的清单表。"""
    versions = _load_versions(None)
    assert versions["dataset"]["data_version"] == "tushare_fullA_20000104_20260928"
    assert versions["dataset"]["rows"] == 17955297
    assert versions["dataset"]["generated_at"]
    assert versions["factors"]["factor_version"] == "factors_v1"
    assert versions["factors"]["factor_count"] == 29


def test_versions_degrade_without_factor_db(tmp_path):
    """因子库缺失时只降级不报错，回测仍能跑。"""
    versions = _load_versions(None, tmp_path / "missing.duckdb")
    assert versions["dataset"]["data_version"]
    assert versions["factors"] is None


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

