"""因子自检的单元测试，重点验证前视检查器真的能抓出未来数据。

模块里有两套产物：
- ``clean_dir``：正常的动量族数据，用来验证行集、取值、方向与金标准一致性；
- ``leaking_dir``：额外注册了一个使用未来一日收盘价的因子，
  用来验证前视检查器能把它揪出来。
"""

import duckdb
import pytest

import factors
from scripts.build_factors import build as build_module
from scripts.build_factors import sql as build_sql
from scripts.build_factors import validate
from scripts.config import DB_FILE

pytestmark = pytest.mark.skipif(not DB_FILE.exists(), reason="需要真实主库 data/databases/quant.duckdb")

YEAR = 2024
LEAK_NAME = "LEAK_TEST"


@pytest.fixture(scope="module")
def clean_dir(tmp_path_factory):
    """正常的动量族 2024 年产物。"""
    root = tmp_path_factory.mktemp("factor_clean")
    build_module.build(
        ["momentum"], f"{YEAR}-01-01", f"{YEAR}-12-31",
        factor_dir=root, catalog_file=root / "factors.duckdb", write_catalog=False,
    )
    return root


@pytest.fixture()
def leaking_dir(tmp_path):
    """额外注册一个故意使用未来数据的因子后构建的产物。"""
    factors.define(
        name=LEAK_NAME, family="momentum", formula="lead(close, 1) over w",
        description="故意使用未来一日收盘价，用来验证前视检查器",
        steps=("取未来一日的收盘价，故意用来验证自检能抓出未来数据",),
        layers="lead(close, 1) over w",
        lookback=1,
    )
    try:
        build_module.build(
            ["momentum"], f"{YEAR}-01-01", f"{YEAR}-12-31",
            factor_dir=tmp_path, catalog_file=tmp_path / "factors.duckdb", write_catalog=False,
        )
        yield tmp_path
    finally:
        factors.unregister(LEAK_NAME)


def _connection():
    conn = duckdb.connect()
    conn.execute(f"ATTACH '{DB_FILE}' AS src (READ_ONLY)")
    return conn


def _gold_standard_query(specs, day: str) -> str:
    """用「全部历史」当输入拼一条查询，作为因子值的金标准。

    参数:
        specs: 因子定义序列。
        day: 计算日期。
    """
    ctes = [
        "spine AS (SELECT stock_code, trade_date FROM src.md.daily_hfq "
        f"WHERE trade_date = DATE '{day}')",
        "base AS (SELECT * FROM src.md.daily_hfq "
        f"WHERE status = 'trading' AND trade_date <= DATE '{day}')",
    ]
    finals = []
    for spec in specs:
        spec_ctes, final = build_sql.factor_ctes(spec, "base")
        ctes.extend(spec_ctes)
        finals.append((final, spec))
    columns = ["spine.stock_code", "spine.trade_date"]
    columns.extend(f"{alias}.{spec.name}" for alias, spec in finals)
    joins = " ".join(f"LEFT JOIN {alias} USING (stock_code, trade_date)" for alias, _ in finals)
    return (
        "WITH " + ",\n".join(ctes) + "\n"
        f"SELECT {', '.join(columns)} FROM spine {joins}"
    )


def test_forward_look_flags_future_leak(leaking_dir):
    """用了未来数据的因子必须被标出来：截断重算时它算不出值。"""
    conn = _connection()
    checks = validate.check_forward_look(conn, "momentum", [f"{YEAR}-06-03"], leaking_dir)
    conn.close()
    leak = next(row for row in checks if row["name"] == LEAK_NAME and row["kind"] == "forward_look_null")
    clean = next(row for row in checks if row["name"] == "MOM_20" and row["kind"] == "forward_look_null")
    assert leak["value"] > 0
    assert clean["value"] == 0


def test_forward_look_passes_for_honest_factors(clean_dir):
    """只用历史数据的因子，截断重算应当与落盘值一致。"""
    conn = _connection()
    checks = validate.check_forward_look(conn, "momentum", [f"{YEAR}-01-08", f"{YEAR}-06-03"], clean_dir)
    conn.close()
    deviations = {
        row["name"]: row["value"] for row in checks if row["kind"] == "forward_look"
    }
    nulls = {
        row["name"]: row["value"] for row in checks if row["kind"] == "forward_look_null"
    }
    assert deviations
    assert max(deviations.values()) < validate.REL_TOLERANCE
    assert sum(nulls.values()) == 0


def test_row_set_check_passes_on_built_data(clean_dir):
    """行集检查在正常产物上必须为 0。"""
    conn = _connection()
    checks = validate.check_row_sets(conn, "momentum", clean_dir)
    conn.close()
    assert checks
    assert all(row["value"] == 0 for row in checks)


def test_value_check_finds_no_non_finite(clean_dir):
    """取值检查要确认没有 inf / nan。"""
    conn = _connection()
    checks = validate.check_values(conn, "momentum", clean_dir)
    conn.close()
    assert checks
    assert all(row["value"] == 0 for row in checks)


def test_direction_check_reports_ic(clean_dir):
    """方向检查给出 IC 均值，并写明与定义方向是否一致。"""
    conn = _connection()
    checks = validate.check_direction(conn, "momentum", f"{YEAR}-01-01", f"{YEAR}-12-31", clean_dir)
    conn.close()
    assert len(checks) >= 5
    for row in checks:
        assert row["kind"] == "direction"
        assert "Rank IC" in row["detail"]


def test_values_match_all_history_gold_standard(clean_dir):
    """落盘值必须与「喂入全部历史」算出来的结果一致。

    这是最强的一条检查：按年切块、跨年补窗口、回看段的截面完整性，
    任何一处错都会在这里暴露。
    """
    conn = _connection()
    specs = factors.family_factors("momentum")
    for day in (f"{YEAR}-01-05", f"{YEAR}-12-16"):
        gold = conn.execute(_gold_standard_query(specs, day)).df()
        names = ", ".join(spec.name for spec in specs)
        built = conn.execute(
            f"SELECT stock_code, trade_date, {names} "
            f"FROM read_parquet('{clean_dir}/momentum/year={YEAR}/part.parquet') "
            f"WHERE trade_date = DATE '{day}'"
        ).df()
        merged = built.merge(gold, on=["stock_code", "trade_date"], suffixes=("_built", "_gold"))
        assert len(merged) > 1000
        for spec in specs:
            left = merged[f"{spec.name}_built"].astype("float64")
            right = merged[f"{spec.name}_gold"].astype("float64")
            assert (left.isna() == right.isna()).all(), f"{spec.name} 在 {day} 的空值状态与金标准不一致"
            both = left.notna() & right.notna()
            assert (left[both] - right[both]).abs().max() < 1e-6, f"{spec.name} 在 {day} 与金标准不一致"
    conn.close()


def test_sample_dates_include_year_start():
    """抽样日期要含每年年初，专门覆盖跨年窗口。"""
    conn = _connection()
    dates = validate.sample_dates(conn, [YEAR], count=3)
    conn.close()
    assert dates
    assert any(day.startswith(f"{YEAR}-01") for day in dates)
