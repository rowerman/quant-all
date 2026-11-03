"""财务因子 SQL 与公告日时点测试。"""

import duckdb
import pandas as pd
import pytest

import factors
from scripts.build_db.financial import FINANCIAL_TABLES
from scripts.build_factors.sql import family_query


def _empty_columns(columns, values):
    """构造只填部分列的测试行。"""
    row = {name: None for name in columns}
    row.update(values)
    return row


def _create_tables(conn):
    """建立最小日线与财务表。"""
    conn.execute("CREATE SCHEMA md")
    conn.execute("CREATE SCHEMA fin")
    conn.execute(
        "CREATE TABLE md.daily_hfq AS SELECT * FROM (VALUES "
        "('SZ000001',DATE '2024-04-19','trading',100000.0),"
        "('SZ000001',DATE '2024-04-22','trading',100000.0),"
        "('SZ000001',DATE '2024-08-19','trading',100000.0),"
        "('SZ000001',DATE '2024-08-21','trading',100000.0)"
        ") AS t(stock_code,trade_date,status,circ_mv)"
    )
    key = [
        "stock_code", "ts_code", "report_date", "ann_date", "f_ann_date",
        "available_date", "report_type", "update_flag", "comp_type",
    ]
    indicator_metrics = list(FINANCIAL_TABLES[3].metrics)
    indicator = pd.DataFrame([
        _empty_columns(key + indicator_metrics, {
            "stock_code": "SZ000001", "ts_code": "000001.SZ",
            "report_date": pd.Timestamp("2023-12-31"), "ann_date": pd.Timestamp("2024-04-20"),
            "f_ann_date": None, "available_date": pd.Timestamp("2024-04-20"),
            "report_type": "1", "update_flag": "1",
            "roe_yearly": 10.0,
        }),
        _empty_columns(key + indicator_metrics, {
            "stock_code": "SZ000001", "ts_code": "000001.SZ",
            "report_date": pd.Timestamp("2023-12-31"), "ann_date": pd.Timestamp("2024-08-20"),
            "f_ann_date": None, "available_date": pd.Timestamp("2024-08-20"),
            "report_type": "1", "update_flag": "1",
            "roe_yearly": 8.0,
        }),
    ])
    conn.register("indicator_frame", indicator)
    conn.execute("CREATE TABLE fin.fina_indicator AS SELECT * FROM indicator_frame")
    conn.unregister("indicator_frame")

    income_metrics = list(FINANCIAL_TABLES[0].metrics)
    income = pd.DataFrame([_empty_columns(key + income_metrics, {
        "stock_code": "SZ000001", "ts_code": "000001.SZ",
        "report_date": pd.Timestamp("2023-12-31"), "ann_date": pd.Timestamp("2024-04-20"),
        "f_ann_date": pd.Timestamp("2024-04-20"), "available_date": pd.Timestamp("2024-04-20"),
        "report_type": "1", "update_flag": "1",
        "n_income": 110.0,
    })])
    conn.register("income_frame", income)
    conn.execute("CREATE TABLE fin.income_statement AS SELECT * FROM income_frame")
    conn.unregister("income_frame")

    balance_metrics = list(FINANCIAL_TABLES[1].metrics)
    balance = pd.DataFrame([_empty_columns(key + balance_metrics, {
        "stock_code": "SZ000001", "ts_code": "000001.SZ",
        "report_date": pd.Timestamp("2023-12-31"), "ann_date": pd.Timestamp("2024-04-20"),
        "f_ann_date": pd.Timestamp("2024-04-20"), "available_date": pd.Timestamp("2024-04-20"),
        "report_type": "1", "update_flag": "1",
        "total_assets": 1000.0,
    })])
    conn.register("balance_frame", balance)
    conn.execute("CREATE TABLE fin.balance_sheet AS SELECT * FROM balance_frame")
    conn.unregister("balance_frame")

    cash_metrics = list(FINANCIAL_TABLES[2].metrics)
    cash = pd.DataFrame([_empty_columns(key + cash_metrics, {
        "stock_code": "SZ000001", "ts_code": "000001.SZ",
        "report_date": pd.Timestamp("2023-12-31"), "ann_date": pd.Timestamp("2024-04-20"),
        "f_ann_date": pd.Timestamp("2024-04-20"), "available_date": pd.Timestamp("2024-04-20"),
        "report_type": "1", "update_flag": "1",
        "n_cashflow_act": 80.0,
        "c_pay_acq_const_fiolta": 5.0,
    })])
    conn.register("cash_frame", cash)
    conn.execute("CREATE TABLE fin.cash_flow AS SELECT * FROM cash_frame")
    conn.unregister("cash_frame")


def test_financial_factor_uses_announcement_next_day(tmp_path):
    """公告日和修订日均从次一交易日起生效。"""
    path = tmp_path / "quant.duckdb"
    conn = duckdb.connect(str(path))
    _create_tables(conn)
    conn.close()

    conn = duckdb.connect()
    conn.execute(f"ATTACH '{path}' AS src (READ_ONLY)")
    query = family_query(
        [factors.get("ROE_YEARLY")],
        spine_start="2024-04-01",
        spine_end="2024-09-01",
    )
    frame = conn.execute(query).df()
    conn.close()
    values = dict(zip(frame["trade_date"].astype(str), frame["ROE_YEARLY"]))
    assert pd.isna(values["2024-04-19"])
    assert values["2024-04-22"] == pytest.approx(0.1)
    assert values["2024-08-19"] == pytest.approx(0.1)
    assert values["2024-08-21"] == pytest.approx(0.08)


def test_accrual_joins_three_statements(tmp_path):
    """应计因子同时使用利润表、资产负债表和现金流量表。"""
    path = tmp_path / "quant.duckdb"
    conn = duckdb.connect(str(path))
    _create_tables(conn)
    conn.close()

    conn = duckdb.connect()
    conn.execute(f"ATTACH '{path}' AS src (READ_ONLY)")
    query = family_query(
        [factors.get("ACCRUAL")],
        spine_start="2024-04-01",
        spine_end="2024-09-01",
    )
    frame = conn.execute(query).df()
    conn.close()
    values = dict(zip(frame["trade_date"].astype(str), frame["ACCRUAL"]))
    assert values["2024-04-22"] == pytest.approx(0.03)
