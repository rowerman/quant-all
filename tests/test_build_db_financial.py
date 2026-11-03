"""财务表入库测试。"""

import duckdb
import pandas as pd

from scripts.build_db import financial


def _frame(columns, rows):
    """按列名构造测试表。"""
    return pd.DataFrame(rows, columns=columns)


def _write_inputs(root):
    """写入四张最小财务分片。"""
    income_columns = [
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type",
        "update_flag", "comp_type", *financial.FINANCIAL_TABLES[0].metrics,
    ]
    income = _frame(income_columns, [
        ["000001.SZ", "20240420", "20240420", "20231231", "1", "0", "1",
         1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 1.0],
        ["000001.SZ", "20240420", "20240420", "20231231", "1", "1", "1",
         2.0, 2.0, 2.0, 2.0, 110.0, 110.0, 2.0],
        ["000002.SZ", "20240101", "20240101", "20240630", "1", "1", "1",
         3.0, 3.0, 3.0, 3.0, 200.0, 200.0, 3.0],
        ["T00001.SH", "20240420", "20240420", "20231231", "1", "1", "1",
         4.0, 4.0, 4.0, 4.0, 300.0, 300.0, 4.0],
    ])
    balance_columns = [
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type",
        "update_flag", "comp_type", *financial.FINANCIAL_TABLES[1].metrics,
    ]
    balance = _frame(balance_columns, [
        ["000001.SZ", "20240420", "20240420", "20231231", "1", "1", "1",
         1000.0, 500.0, 500.0, 300.0, 200.0, 100.0, 50.0, 20.0, 1.0, 100.0],
    ])
    cash_columns = [
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type",
        "update_flag", "comp_type", *financial.FINANCIAL_TABLES[2].metrics,
    ]
    cash = _frame(cash_columns, [
        ["000001.SZ", "20240420", "20240420", "20231231", "1", "1", "1",
         80.0, 10.0, 20.0, 5.0, 75.0, 3.0, 2.0],
    ])
    indicator_columns = [
        "ts_code", "ann_date", "end_date", "update_flag",
        *financial.FINANCIAL_TABLES[3].metrics,
    ]
    indicator = _frame(indicator_columns, [
        ["000001.SZ", "20240420", "20231231", "1", *([10.0] * len(financial.FINANCIAL_TABLES[3].metrics))],
        ["000001.SZ", "20240820", "20231231", "1", *([8.0] * len(financial.FINANCIAL_TABLES[3].metrics))],
    ])
    for name, frame in (
        ("income", income),
        ("balancesheet", balance),
        ("cashflow", cash),
        ("fina_indicator", indicator),
    ):
        directory = root / "raw_financial" / name
        directory.mkdir(parents=True)
        frame.to_parquet(directory / "period=20231231.parquet", index=False)


def test_financial_build_deduplicates_revisions_and_drops_invalid_keys(tmp_path):
    """优先保留 update_flag=1，并丢弃代码非法和日期倒置的行。"""
    _write_inputs(tmp_path)
    financial.validate_inputs(tmp_path)
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA fin")
    conn.execute("CREATE SCHEMA meta")
    financial.build_financial(conn, root=tmp_path)

    rows = conn.execute(
        "SELECT stock_code, n_income, available_date, f_ann_date "
        "FROM fin.income_statement ORDER BY stock_code"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SZ000001"
    assert rows[0][1] == 110.0
    assert str(rows[0][2]) == "2024-04-20"
    assert str(rows[0][3]) == "2024-04-20"

    manifest = conn.execute(
        "SELECT table_name, source_rows, duplicates_dropped, invalid_dates_dropped, "
        "invalid_codes_dropped FROM meta.financial_manifest ORDER BY table_name"
    ).fetchall()
    income = [row for row in manifest if row[0] == "income_statement"][0]
    assert income[1] == 4
    assert income[2] == 1
    assert income[3] == 1
    assert income[4] == 1
    assert conn.execute("SELECT count(*) FROM fin.fina_indicator").fetchone()[0] == 2


def test_financial_panel_uses_next_trading_day(tmp_path):
    """公告日次一交易日起才可见，修订公告同样从新公告日次日起生效。"""
    from quant.loaders import load_financial_panel

    db_path = tmp_path / "quant.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE SCHEMA fin")
    conn.execute("CREATE SCHEMA md")
    conn.execute(
        "CREATE TABLE fin.fina_indicator AS SELECT * FROM (VALUES "
        "('SZ000001','000001.SZ',DATE '2023-12-31',DATE '2024-04-20',NULL,DATE '2024-04-20','1','1',NULL,10.0),"
        "('SZ000001','000001.SZ',DATE '2023-12-31',DATE '2024-08-20',NULL,DATE '2024-08-20','1','1',NULL,8.0)"
        ") AS t(stock_code,ts_code,report_date,ann_date,f_ann_date,available_date,report_type,update_flag,comp_type,roe_yearly)"
    )
    conn.execute(
        "CREATE TABLE md.daily_bar AS SELECT * FROM (VALUES "
        "('SZ000001',DATE '2024-04-19','trading'),"
        "('SZ000001',DATE '2024-04-22','trading'),"
        "('SZ000001',DATE '2024-08-19','trading'),"
        "('SZ000001',DATE '2024-08-21','trading')"
        ") AS t(stock_code,trade_date,status)"
    )
    conn.close()

    panel = load_financial_panel(
        "fina_indicator", "2024-04-01", "2024-09-01",
        fields=["roe_yearly"], db_path=db_path,
    )
    values = dict(zip(panel["trade_date"].astype(str), panel["roe_yearly"]))
    assert pd.isna(values["2024-04-19"])
    assert values["2024-04-22"] == 10.0
    assert values["2024-08-19"] == 10.0
    assert values["2024-08-21"] == 8.0
