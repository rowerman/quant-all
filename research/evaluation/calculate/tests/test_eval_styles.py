"""风格因子的单元测试：7 档取值与 pandas 手算逐行对照（需要真实主库）。"""

import duckdb
import numpy as np
import pandas as pd
import pytest

import quant
from calculate.config import DB_FILE
from calculate.errors import IndustryError, StyleFactorError
from calculate.panel import open_source
from calculate.params import EvalParams
from calculate.styles import (
    BETA_BENCHMARK,
    RAW_COLUMNS,
    StockBasicIndustry,
    build_styles,
    lookback_start,
)

pytestmark = pytest.mark.skipif(not DB_FILE.exists(), reason="需要真实主库 data/databases/quant.duckdb")

START, END = "2024-03-01", "2024-03-29"
LOOKBACK_START = "2022-12-01"
STOCKS = ["SH600000", "SZ000001", "SH600519"]


def rolling_beta(values: np.ndarray, market: np.ndarray, window: int = 60,
                 min_pairs: int = 40) -> np.ndarray:
    """手算滚动贝塔：窗口内对指数收益做一元回归取斜率。

    参数:
        values: 个股日收益。
        market: 指数日收益。
        window: 窗口的交易日数（含当日）。
        min_pairs: 窗口内最少有效配对数。
    """
    result = np.full(values.size, np.nan, dtype="float64")
    for index in range(values.size):
        start = max(0, index - window + 1)
        left = values[start:index + 1]
        right = market[start:index + 1]
        mask = np.isfinite(left) & np.isfinite(right)
        if mask.sum() < min_pairs:
            continue
        result[index] = np.polyfit(right[mask], left[mask], 1)[0]
    return result


def index_return() -> pd.Series:
    """读沪深300 的日收益（按交易日索引）。"""
    frame = quant.load_index_daily(BETA_BENCHMARK, start=LOOKBACK_START, end=END)
    return pd.Series(
        frame["pct_chg"].to_numpy(dtype="float64") / 100.0,
        index=frame["trade_date"],
    )


def hand_style_frame(code: str, market_return: pd.Series) -> pd.DataFrame:
    """按定义用 pandas 手算一只股票的风格因子。

    参数:
        code: 股票代码。
        market_return: 指数日收益（按交易日索引）。
    """
    frame = quant.load_panel(
        start=LOOKBACK_START, end=END,
        fields=["close", "ret", "total_mv", "turnover_rate", "pe_ttm", "pb"],
        adjust="hfq", stocks=[code], drop_suspended=True,
    ).sort_values("trade_date").reset_index(drop=True)
    price = frame["close"]
    price_20 = price.shift(20)
    price_250 = price.shift(250)
    market = frame["trade_date"].map(market_return).to_numpy(dtype="float64")
    turnover = frame["turnover_rate"]
    result = pd.DataFrame({
        "stock_code": code,
        "trade_date": frame["trade_date"],
        "SIZE": np.log(frame["total_mv"].where(frame["total_mv"] > 0)),
        "BP": (1.0 / frame["pb"]).where(frame["pb"] > 0),
        "EP": (1.0 / frame["pe_ttm"]).where(frame["pe_ttm"] > 0),
        "MOMENTUM": (price_20 / price_250 - 1.0).where(price_20.notna() & price_250.notna()),
        "SHORT_REV": (1.0 - price / price_20).where(price_20.notna() & (price_20 > 0)),
        "VOL": frame["ret"].rolling(60).std(ddof=1).where(
            frame["ret"].rolling(60).count() >= 40
        ),
        "LIQUIDITY": turnover.rolling(20).mean().where(turnover.rolling(20).count() >= 15),
        "BETA_60": rolling_beta(frame["ret"].to_numpy(dtype="float64"), market),
    })
    return result[result["trade_date"] >= pd.Timestamp(START)]


def test_style_values_match_pandas_hand_computation():
    """7 个风格列与 pandas 手算逐行一致（含最小观测阈值）。"""
    market = index_return()
    expected = pd.concat(
        [hand_style_frame(code, market) for code in STOCKS], ignore_index=True
    )
    # 手算的贝塔与动量至少要有值，否则这个用例什么都没验到
    assert expected["BETA_60"].notna().sum() > 0
    assert expected["MOMENTUM"].notna().sum() > 0

    params = EvalParams(factor="MOM_20", start=START, end=END)
    mine = build_styles(params)
    mine = mine[mine["stock_code"].isin(STOCKS)]
    assert set(mine.columns) >= set(RAW_COLUMNS)
    assert mine["stock_code"].dtype == object

    joined = mine.merge(
        expected, on=["stock_code", "trade_date"], suffixes=("_mine", "_hand")
    )
    assert len(joined) == len(mine) == len(expected)
    for column in RAW_COLUMNS:
        left = joined[f"{column}_mine"].to_numpy(dtype="float64")
        right = joined[f"{column}_hand"].to_numpy(dtype="float64")
        assert np.allclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True), column


def test_compact_styles_preserve_values_as_categories():
    """内部紧凑模式只改变股票列类型，不改变任何风格值。"""

    params = EvalParams(factor="MOM_20", start=START, end=END)
    public = build_styles(params)
    compact = build_styles(params, compact=True)

    assert isinstance(compact["stock_code"].dtype, pd.CategoricalDtype)
    joined = public.merge(
        compact,
        on=["trade_date", "stock_code"],
        suffixes=("_public", "_compact"),
    )
    assert len(joined) == len(public) == len(compact)
    for column in RAW_COLUMNS:
        np.testing.assert_allclose(
            joined[f"{column}_public"],
            joined[f"{column}_compact"],
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )


def test_short_lookback_leaves_long_window_null():
    """回看不足 250 个交易日时 MOMENTUM 全空，同一批样本上 SIZE 仍有值。"""
    params = EvalParams(factor="MOM_20", start="2000-01-04", end="2000-06-30")
    styles = build_styles(params)
    assert styles["MOMENTUM"].notna().sum() == 0
    assert styles["SIZE"].notna().mean() > 0.9
    # 沪深300 从 2002 年起才有数据，2000 年算不出 BETA_60
    assert styles["BETA_60"].notna().sum() == 0
    assert styles["LIQUIDITY"].notna().mean() > 0.5


def test_lookback_start_counts_trade_days():
    """回看起点是「起始日往前第 260 个交易日」，不是自然日。"""
    conn = open_source(DB_FILE)
    try:
        day = lookback_start(conn, START, days=260)
        calendar = conn.execute(
            "SELECT trade_date FROM src.md.calendar"
            f" WHERE trade_date BETWEEN DATE '{day}' AND DATE '{START}' ORDER BY trade_date"
        ).df()
    finally:
        conn.close()
    assert pd.Timestamp(day) < pd.Timestamp(START)
    assert len(calendar) == 261  # 起点当天 + 起始日之前的 260 个交易日


def test_style_coverage_on_real_data():
    """真实数据上的覆盖率下限：短窗口风格几乎全覆盖，长窗口允许次新股缺失。"""
    params = EvalParams(factor="MOM_20", start="2024-01-02", end="2024-01-31")
    styles = build_styles(params)
    coverage = styles[list(RAW_COLUMNS)].notna().mean()
    for column in ("SIZE", "SHORT_REV", "VOL", "LIQUIDITY", "BETA_60", "BP"):
        assert coverage[column] > 0.9, (column, coverage[column])
    assert coverage["MOMENTUM"] > 0.85
    assert coverage["EP"] > 0.5  # 亏损股没有正的 pe_ttm


def test_industry_source_loads_and_reports_coverage():
    """静态行业源能读出来，且绝大多数股票有行业。"""
    conn = open_source(DB_FILE)
    try:
        frame = StockBasicIndustry().load(conn)
    finally:
        conn.close()
    assert {"stock_code", "industry"} <= set(frame.columns)
    assert frame["industry"].notna().all()
    assert (frame["industry"] != "未知").mean() > 0.9
    assert frame["stock_code"].is_unique


def test_industry_source_missing_table_raises(tmp_path):
    """库里没有 ref.stock_basic 时报 EVAL-303。"""
    path = tmp_path / "empty_industry.duckdb"
    duckdb.connect(str(path)).execute("CREATE SCHEMA ref")
    conn = duckdb.connect()
    conn.execute(f"ATTACH '{path}' AS src (READ_ONLY)")
    try:
        with pytest.raises(IndustryError) as error:
            StockBasicIndustry().load(conn)
    finally:
        conn.close()
    assert error.value.code == "EVAL-303"


def test_missing_index_table_raises(tmp_path):
    """库里没有 md.index_daily 时报 EVAL-302，并说明缺的是哪张表。"""
    path = tmp_path / "no_index.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE SCHEMA md")
    conn.execute("CREATE TABLE md.calendar (trade_date DATE)")
    conn.execute(
        "CREATE TABLE md.daily_hfq (stock_code VARCHAR, trade_date DATE, status VARCHAR)"
    )
    conn.close()
    params = EvalParams(factor="MOM_20", start="2024-01-02", end="2024-01-31")
    with pytest.raises(StyleFactorError) as error:
        build_styles(params, db_file=path)
    assert error.value.code == "EVAL-302"
    assert "index_daily" in error.value.message
