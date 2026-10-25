"""未来收益标签的单元测试。

用内存 DuckDB 造一份含停牌行与尾部不足的小面板，核对两套口径的取值、
缺失规则与自检逻辑，不依赖真实数据库。
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

from quant.schema import LABEL_HORIZONS, LABEL_PRICES
from scripts.build_db import labels

DATES = [f"2024-01-{day:02d}" for day in (2, 3, 4, 5, 8, 9, 10, 11)]
# A 全程交易，用来验证正常口径
CLOSES_A = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
# B 第 2 天停牌，后面仍有两个交易日，用来验证填平与尾部规则
CLOSES_B = [20.0, None, 22.0, 23.0]
SUSPENDED_DAY = DATES[1]


def _panel() -> pd.DataFrame:
    """构造两只股票的测试面板。"""
    rows = []
    for index, (day, close) in enumerate(zip(DATES, CLOSES_A)):
        rows.append({
            "stock_code": "SH600000",
            "trade_date": day,
            "close": close,
            "open": close - 0.5,
            "status": "trading",
            "ret": np.nan if index == 0 else close / CLOSES_A[index - 1] - 1,
        })
    previous = None
    for day, close in zip(DATES, CLOSES_B):
        if close is None:
            rows.append({
                "stock_code": "SZ000001",
                "trade_date": day,
                "close": np.nan,
                "open": np.nan,
                "status": "suspended",
                "ret": np.nan,
            })
            continue
        rows.append({
            "stock_code": "SZ000001",
            "trade_date": day,
            "close": close,
            "open": close + 0.5,
            "status": "trading",
            "ret": np.nan if previous is None else close / previous - 1,
        })
        previous = close
    return pd.DataFrame(rows)


@pytest.fixture()
def conn():
    """建一张内存日线表并生成标签。"""
    connection = duckdb.connect()
    connection.execute("CREATE SCHEMA md")
    frame = _panel()
    connection.register("src", frame)
    connection.execute("CREATE TABLE md.daily_hfq AS SELECT * FROM src")
    connection.unregister("src")
    labels.build_labels(connection)
    yield connection
    connection.close()


def _row(conn, stock_code: str, trade_date: str) -> pd.Series:
    """取某只股票某天的整行标签。"""
    return conn.execute(
        "SELECT * FROM md.fwd_return WHERE stock_code = ? AND trade_date = ?",
        [stock_code, trade_date],
    ).df().iloc[0]


def test_label_table_has_both_channels(conn):
    """两套通道各六档，列数固定。"""
    columns = conn.execute("SELECT * FROM md.fwd_return LIMIT 0").df().columns.tolist()
    for price in LABEL_PRICES:
        for horizon in LABEL_HORIZONS:
            assert f"fwd_ret_{price}_{horizon}" in columns
    assert len(columns) == 2 + 2 * len(LABEL_HORIZONS)


def test_row_count_matches_daily_table(conn):
    """标签表与日线表行数一致，方便直接 join。"""
    assert conn.execute("SELECT count(*) FROM md.fwd_return").fetchone()[0] == len(_panel())


def test_close_channel_is_close_to_close(conn):
    """收盘通道等于 t 收盘到 t+N 收盘的涨幅。"""
    row = _row(conn, "SH600000", DATES[0])
    assert row["fwd_ret_close_1"] == pytest.approx(11.0 / 10.0 - 1)
    assert row["fwd_ret_close_3"] == pytest.approx(13.0 / 10.0 - 1)
    assert row["fwd_ret_close_5"] == pytest.approx(15.0 / 10.0 - 1)


def test_open_channel_buys_next_open_and_keeps_horizon(conn):
    """开盘通道从 t+1 开盘买入，卖出价取 t+1+N，持有长度与收盘通道一致。"""
    row = _row(conn, "SH600000", DATES[0])
    assert row["fwd_ret_open_1"] == pytest.approx(11.5 / 10.5 - 1)
    assert row["fwd_ret_open_3"] == pytest.approx(13.5 / 10.5 - 1)


def test_open_channel_needs_one_more_trading_day(conn):
    """开盘通道比收盘通道多消耗一天，倒数第二个交易日只有收盘口径有值。"""
    row = _row(conn, "SH600000", DATES[6])
    assert not np.isnan(row["fwd_ret_close_1"])
    assert np.isnan(row["fwd_ret_open_1"])


def test_suspended_gap_is_filled_by_last_valid_price(conn):
    """t+1 停牌时按上一有效价填平，收益为零而不是缺失。"""
    row = _row(conn, "SZ000001", DATES[0])
    assert row["fwd_ret_close_1"] == pytest.approx(0.0)
    assert row["fwd_ret_close_3"] == pytest.approx(23.0 / 20.0 - 1)


def test_suspended_rows_have_no_label(conn):
    """停牌当天不是可建仓的决策点，整行标签为空。"""
    row = _row(conn, "SZ000001", SUSPENDED_DAY)
    for price in LABEL_PRICES:
        for horizon in LABEL_HORIZONS:
            assert np.isnan(row[f"fwd_ret_{price}_{horizon}"])


def test_tail_rows_are_null(conn):
    """尾部不足 N 个交易日的行保持缺失。"""
    last = _row(conn, "SH600000", DATES[7])
    assert np.isnan(last["fwd_ret_close_1"])
    assert np.isnan(_row(conn, "SH600000", DATES[5])["fwd_ret_close_3"])


def test_verify_labels_passes_on_clean_panel(conn):
    """自检要能发现口径错误，正常面板必须通过。"""
    result = labels.verify_labels(conn)
    assert result["passed"], result["failures"]
    assert result["ret_mismatch_rows"] == 0
