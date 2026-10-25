"""申万行业覆盖审计脚本测试。"""

import duckdb

from tools.v5 import audit_sw_industry


def test_audit_reports_missing_and_carry_forward(tmp_path):
    """审计必须同时识别缺失格和允许的 carry-forward 格。"""
    path = tmp_path / "quant.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE SCHEMA ref")
    conn.execute("CREATE SCHEMA meta")
    conn.execute("CREATE SCHEMA md")
    conn.execute(
        "CREATE TABLE ref.sw_industry_member AS SELECT * FROM (VALUES "
        "('SZ000001', 'SW2014', DATE '2015-01-01', DATE '2021-12-12', FALSE, 'index_member'), "
        "('SZ000002', 'SW2014', DATE '2021-12-13', NULL, TRUE, 'carry_forward_sw2014')"
        ") AS t(stock_code, classification, in_date, out_date, carry_forward, source_api)"
    )
    conn.execute(
        "CREATE TABLE md.daily_bar AS SELECT * FROM (VALUES "
        "('SZ000001', DATE '2022-01-04'), ('SZ000002', DATE '2022-01-04'), "
        "('SZ000003', DATE '2022-01-04')"
        ") AS t(stock_code, trade_date)"
    )
    conn.execute(
        "CREATE TABLE meta.sw_industry_manifest AS SELECT "
        "'SW2014+SW2021' AS source, DATE '2021-12-13' AS switch_date"
    )
    conn.close()

    result = audit_sw_industry.audit(
        path, "2022-01-01", "2022-12-31"
    )
    assert result["passed"] is False
    assert result["missing_cells"] == 2
    assert result["missing_classified_cells"] == 1
    assert result["unclassified_cells"] == 1
    assert result["carry_forward_cells"] == 1
    assert result["coverage"][0]["year"] == 2022
