"""公司行动归并、配股条款与股本兜底。"""

from __future__ import annotations

import duckdb
import pandas as pd

from scripts.build_db import corporate_actions
from scripts.sources import gbbq

DIVIDEND_COLUMNS = [
    "ts_code", "end_date", "ann_date", "div_proc", "stk_div", "stk_bo_rate",
    "stk_co_rate", "cash_div", "cash_div_tax", "div_listdate", "record_date",
    "ex_date", "pay_date",
]
GBBQ_COLUMNS = [
    "market", "code", "datetime", "category", "hongli_panqianliutong",
    "peigujia_qianzongguben", "songgu_qianzongguben", "peigu_houzongguben",
]


def _connection(dates: list[str], bars: list[tuple]) -> duckdb.DuckDBPyConnection:
    """构造最小行情库。"""
    conn = duckdb.connect()
    conn.execute("CREATE SCHEMA md")
    conn.execute(
        "CREATE TABLE md.calendar AS SELECT * FROM (VALUES "
        + ", ".join(f"(DATE '{day}', 1)" for day in dates)
        + ") t(trade_date, is_open)"
    )
    conn.execute(
        "CREATE TABLE md.daily_bar AS SELECT * FROM (VALUES "
        + ", ".join(
            f"('{code}', DATE '{day}', {close}, {factor}, {str(is_ex).upper()})"
            for code, day, close, factor, is_ex in bars
        )
        + ") t(stock_code, trade_date, close, factor, is_ex_div)"
    )
    return conn


def _write_sources(
    tmp_path,
    monkeypatch,
    *,
    gbbq_rows: list[dict],
    dividend_rows: list[dict] | None = None,
    basic_rows: list[tuple] | None = None,
):
    """写入临时 gbbq、分红和股本文件。"""
    gbbq_path = tmp_path / "gbbq.parquet"
    pd.DataFrame(gbbq_rows, columns=GBBQ_COLUMNS).to_parquet(gbbq_path, index=False)
    dividends = tmp_path / "dividend.parquet"
    pd.DataFrame(dividend_rows or [], columns=DIVIDEND_COLUMNS).to_parquet(
        dividends, index=False
    )
    basic = tmp_path / "daily_basic.parquet"
    pd.DataFrame(
        basic_rows or [],
        columns=["ts_code", "trade_date", "total_share"],
    ).to_parquet(basic, index=False)
    monkeypatch.setattr(corporate_actions, "GBBQ_FILE", gbbq_path)
    monkeypatch.setattr(corporate_actions, "DIVIDEND_GLOB", str(dividends))
    monkeypatch.setattr(corporate_actions, "DAILY_BASIC_GLOB", str(basic))
    monkeypatch.setattr(gbbq, "verify_file", lambda path: "test-digest")


def test_gbbq_primary_deduplicates_and_keeps_cash_and_stock(tmp_path, monkeypatch):
    """gbbq 为主，Tushare 同事件去重并保留现金与送转。"""
    _write_sources(
        tmp_path,
        monkeypatch,
        gbbq_rows=[
            [0, "002594", 20250729, 1, 39.74, 0.0, 20.0, 0.0],
            [0, "000001", 20240614, 1, 7.19, 0.0, 0.0, 0.0],
            [0, "002594", 20250729, 1, 39.74, 0.0, 20.0, 0.0],
        ],
        dividend_rows=[
            {
                "ts_code": "002594.SZ", "end_date": "20241231", "ann_date": "20250728",
                "div_proc": "实施", "stk_div": 2.0, "stk_bo_rate": 0.8,
                "stk_co_rate": 1.2, "cash_div": 3.974, "cash_div_tax": 3.974,
                "div_listdate": None, "record_date": "20250728",
                "ex_date": "20250729", "pay_date": "20250729",
            },
            {
                "ts_code": "000001.SZ", "end_date": "20231231", "ann_date": "20240613",
                "div_proc": "实施", "stk_div": 0.0, "stk_bo_rate": None,
                "stk_co_rate": None, "cash_div": 0.719, "cash_div_tax": 0.719,
                "div_listdate": None, "record_date": "20240613",
                "ex_date": "20240614", "pay_date": "20240614",
            },
        ],
    )
    conn = _connection(
        ["2024-06-13", "2024-06-14", "2025-07-28", "2025-07-29"],
        [
            ("SZ002594", "2025-07-29", 111.42, 3.0, True),
            ("SZ000001", "2024-06-14", 10.0, 1.0, False),
        ],
    )
    frame, stats = corporate_actions.build_frame(conn)
    conn.close()

    assert stats["gbbq_events"] == 2
    assert stats["dividend_events"] == 2
    assert stats["tushare_overlap_events"] == 2
    assert stats["tushare_conflict_events"] == 0
    assert not frame.duplicated(["stock_code", "ex_date"]).any()
    row = frame.loc[frame["stock_code"] == "SZ002594"].iloc[0]
    assert row["share_multiplier"] == 3.0
    assert row["cash_div_per_share"] == 3.974
    assert row["source"] == "gbbq"


def test_gbbq_rights_terms_are_normalized(tmp_path, monkeypatch):
    """gbbq 每 10 股配 3 股、配股价 8.19 转成每股口径。"""
    _write_sources(
        tmp_path,
        monkeypatch,
        gbbq_rows=[[1, "601377", 20160107, 1, 0.0, 8.19, 0.0, 3.0]],
    )
    conn = _connection(
        ["2016-01-06", "2016-01-07"],
        [("SH601377", "2016-01-07", 7.0, 2.0, True)],
    )
    frame, stats = corporate_actions.build_frame(conn)
    conn.close()

    row = frame.iloc[0]
    assert stats["rights_events"] == 1
    assert row["action_type"] == "rights"
    assert row["rights_ratio"] == 0.3
    assert row["rights_price"] == 8.19
    assert row["record_date"] == pd.Timestamp("2016-01-06")


def test_share_count_and_factor_crosscheck_fill_missing_split(tmp_path, monkeypatch):
    """gbbq 与分红都缺失时，总股本和复权因子同日一致变化才兜底。"""
    _write_sources(
        tmp_path,
        monkeypatch,
        gbbq_rows=[],
        basic_rows=[
            ("300104.SZ", "20110517", 10000.0),
            ("300104.SZ", "20110518", 22000.0),
        ],
    )
    conn = _connection(
        ["2011-05-17", "2011-05-18"],
        [
            ("SZ300104", "2011-05-17", 60.5, 1.0, False),
            ("SZ300104", "2011-05-18", 26.83, 2.206, True),
        ],
    )
    frame, stats = corporate_actions.build_frame(conn)
    conn.close()

    assert stats["share_count_events"] == 1
    row = frame.iloc[0]
    assert row["source"] == "share_count"
    assert row["share_multiplier"] == 2.2
