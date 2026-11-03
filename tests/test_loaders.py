"""读取层参数映射与错误提示的单元测试。"""

import duckdb
import pandas as pd
import pytest

from quant import loaders


def test_status_code_accepts_documented_vocabulary():
    """文档里的 listed/delisted 与 Tushare 的 L/D/P 都要能用。"""
    assert loaders._status_code("listed") == "L"
    assert loaders._status_code("delisted") == "D"
    assert loaders._status_code("L") == "L"
    assert loaders._status_code("d") == "D"


def test_status_code_rejects_unknown_value():
    """非法状态要立刻报错，不能悄悄退化成返回全部数据。"""
    with pytest.raises(ValueError):
        loaders._status_code("退市")


def test_require_table_reports_available_tables():
    """缺表时报出可用表清单，而不是 DuckDB 的 Catalog Error。"""
    conn = duckdb.connect()
    with pytest.raises(ValueError, match="fin"):
        loaders._require_table(conn, "fin", "income_statement")
    conn.close()


def test_load_label_rejects_unknown_horizon():
    """预测周期只允许文档里那六档，非法值立刻报错并列出可选值。"""
    with pytest.raises(ValueError, match="未知的预测周期"):
        loaders.load_label(horizons=[2])


def test_load_label_rejects_unknown_price_channel():
    """成交通道只有 close / open 两种。"""
    with pytest.raises(ValueError, match="price 只能是"):
        loaders.load_label(price="vwap")


def test_load_sw_index_daily_reports_missing_database():
    """库文件不存在时要抛 FileNotFoundError，而不是悄悄返回空表。"""
    with pytest.raises(FileNotFoundError):
        loaders.load_sw_index_daily(db_path="/tmp/does-not-exist-quant.duckdb")


def test_load_index_basic_reports_missing_database():
    """指数目录读取同样不能静默降级。"""
    with pytest.raises(FileNotFoundError):
        loaders.load_index_basic(db_path="/tmp/does-not-exist-quant.duckdb")


def test_load_sw_industry_member_filters_intersecting_intervals(tmp_path):
    """行业读取按区间相交过滤，不能只看调入日。"""
    path = tmp_path / "quant.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE SCHEMA ref")
    frame = pd.DataFrame(
        {
            "stock_code": ["SZ000001", "SZ000002"], "ts_code": ["000001.SZ", "000002.SZ"],
            "classification": ["SW2014", "SW2021"],
            "l1_code": ["A", "B"], "l1_name": ["甲", "乙"],
            "l2_code": ["A1", "B1"], "l2_name": ["甲1", "乙1"],
            "l3_code": ["A11", "B11"], "l3_name": ["甲11", "乙11"],
            "in_date": pd.to_datetime(["2020-01-01", "2022-01-01"]),
            "out_date": pd.to_datetime(["2020-12-31", None]), "is_new": ["N", "Y"],
            "source_l3_code": ["A11", "B11"], "carry_forward": [False, False],
        }
    )
    conn.register("f", frame)
    conn.execute("CREATE TABLE ref.sw_industry_member AS SELECT * FROM f")
    conn.close()
    loaded = loaders.load_sw_industry_member(
        start="2020-06-01", end="2021-01-01", stocks=["SZ000001"], db_path=path
    )
    assert loaded["stock_code"].tolist() == ["SZ000001"]
    loaded = loaders.load_sw_industry_member(
        start="2021-01-01", end="2022-12-31", classification="SW2021", db_path=path
    )
    assert loaded["stock_code"].tolist() == ["SZ000002"]
