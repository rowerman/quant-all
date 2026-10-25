"""因子构建的单元测试：单族单年端到端跑真实主库。

需要 data/databases/quant.duckdb 存在；主库缺失时整体跳过。
"""

import duckdb
import pandas as pd
import pytest

from scripts.build_factors import build as build_module
from scripts.config import DB_FILE


pytestmark = pytest.mark.skipif(not DB_FILE.exists(), reason="需要真实主库 data/databases/quant.duckdb")

YEAR = 2024


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """把动量族 2024 年构建到一个临时目录，供本模块复用。"""
    factor_dir = tmp_path_factory.mktemp("factors")
    catalog_file = factor_dir / "factors.duckdb"
    result = build_module.build(
        ["momentum"], f"{YEAR}-01-01", f"{YEAR}-12-31",
        factor_dir=factor_dir, catalog_file=catalog_file,
    )
    return result, factor_dir, catalog_file


def _source_connection():
    conn = duckdb.connect()
    conn.execute(f"ATTACH '{DB_FILE}' AS src (READ_ONLY)")
    return conn


def test_build_writes_one_file_per_year(built):
    """每个族每年一个 parquet，列是主键 + 该族因子。"""
    _, factor_dir, _ = built
    path = factor_dir / "momentum" / f"year={YEAR}" / "part.parquet"
    assert path.exists()
    conn = _source_connection()
    columns = conn.execute(f"describe select * from read_parquet('{path}')").df()["column_name"].tolist()
    conn.close()
    assert columns[:2] == ["stock_code", "trade_date"]
    for name in ("MOM_20", "MOM_60", "MOM_60_20", "REV_5", "REV_20"):
        assert name in columns


def test_row_set_matches_main_database(built):
    """因子文件的行集必须与主库同区间完全一致，不多不少。"""
    _, factor_dir, _ = built
    path = factor_dir / "momentum" / f"year={YEAR}" / "part.parquet"
    conn = _source_connection()
    frame = conn.execute(f"""
    SELECT
      (SELECT count(*) FROM (
          SELECT stock_code, trade_date FROM read_parquet('{path}')
          EXCEPT SELECT stock_code, trade_date FROM src.md.daily_hfq
          WHERE trade_date BETWEEN DATE '{YEAR}-01-01' AND DATE '{YEAR}-12-31')) AS extra_rows,
      (SELECT count(*) FROM (
          SELECT stock_code, trade_date FROM src.md.daily_hfq
          WHERE trade_date BETWEEN DATE '{YEAR}-01-01' AND DATE '{YEAR}-12-31'
          EXCEPT SELECT stock_code, trade_date FROM read_parquet('{path}'))) AS missing_rows
    """).df()
    conn.close()
    assert int(frame.at[0, "extra_rows"]) == 0
    assert int(frame.at[0, "missing_rows"]) == 0


def test_suspended_rows_have_null_factors(built):
    """停牌行的因子必须是空值，与标签口径一致。"""
    _, factor_dir, _ = built
    path = factor_dir / "momentum" / f"year={YEAR}" / "part.parquet"
    conn = _source_connection()
    frame = conn.execute(f"""
    SELECT b.status, count(*) AS rows, count(f.MOM_20) AS non_null
    FROM read_parquet('{path}') f
    JOIN src.md.daily_hfq b USING (stock_code, trade_date)
    GROUP BY 1
    """).df().set_index("status")
    conn.close()
    assert int(frame.loc["suspended", "non_null"]) == 0
    assert int(frame.loc["trading", "non_null"]) > 0


def test_momentum_values_match_manual_calculation(built):
    """抽一只股票核对 MOM_20，确认复权价与窗口口径没走样。"""
    _, factor_dir, _ = built
    path = factor_dir / "momentum" / f"year={YEAR}" / "part.parquet"
    conn = _source_connection()
    frame = conn.execute(f"""
    WITH src AS (
        SELECT stock_code, trade_date, close
        FROM src.md.daily_hfq
        WHERE stock_code = 'SH600000' AND status = 'trading'
          AND trade_date BETWEEN DATE '2023-11-01' AND DATE '{YEAR}-12-31'
    ), manual AS (
        SELECT stock_code, trade_date,
               close / lag(close, 20) OVER (PARTITION BY stock_code ORDER BY trade_date) - 1 AS manual
        FROM src
    )
    SELECT m.trade_date, m.manual, f.MOM_20
    FROM manual m
    JOIN read_parquet('{path}') f USING (stock_code, trade_date)
    WHERE f.MOM_20 IS NOT NULL
    """).df()
    conn.close()
    assert len(frame) > 200
    assert frame["manual"].notna().all()
    assert (frame["manual"] - frame["MOM_20"]).abs().max() < 1e-6


def test_cross_year_window_is_filled(built):
    """1 月初的 20 日动量必须有值，说明跨年补窗口生效。"""
    _, factor_dir, _ = built
    path = factor_dir / "momentum" / f"year={YEAR}" / "part.parquet"
    conn = _source_connection()
    frame = conn.execute(f"""
    SELECT count(*) AS rows, count(MOM_60) AS non_null
    FROM read_parquet('{path}')
    WHERE trade_date <= DATE '{YEAR}-01-10'
    """).df()
    conn.close()
    rows, non_null = int(frame.at[0, "rows"]), int(frame.at[0, "non_null"])
    assert rows > 0
    assert non_null > rows * 0.9, "1 月初就缺 60 日动量，说明没有往前补窗口"


def test_build_writes_catalog_and_manifest(built):
    """目录库里要有定义、覆盖率、构建记录与版本信息。"""
    _, _, catalog_file = built
    conn = duckdb.connect(str(catalog_file), read_only=True)
    definitions = conn.execute(
        "SELECT count(*) FROM meta.factor_definition WHERE status = 'implemented'"
    ).fetchone()[0]
    coverage = conn.execute(
        "SELECT count(*) FROM meta.factor_coverage WHERE family = 'momentum' AND year = ?", [YEAR]
    ).fetchone()[0]
    build_rows = conn.execute("SELECT status FROM meta.factor_build").df()["status"].tolist()
    manifest = conn.execute("SELECT factor_count, rows FROM meta.factor_manifest").df()
    conn.close()
    assert definitions >= 5
    assert coverage == 5
    assert build_rows == ["ok"]
    assert int(manifest.at[0, "factor_count"]) == 5
    assert int(manifest.at[0, "rows"]) > 1_000_000


def test_full_factor_set_is_registered():
    """样板因子要覆盖十个族，便于用户照着扩展。"""
    import factors
    assert len(factors.all_factors()) >= 25
    assert len(factors.used_families()) == len(factors.FAMILIES)


def test_non_finite_values_are_null(built):
    """落盘不能出现 inf / nan，常收益个股的 corr 会算出 NaN，必须置空。"""
    _, factor_dir, _ = built
    path = factor_dir / "momentum" / f"year={YEAR}" / "part.parquet"
    conn = _source_connection()
    frame = conn.execute(f"""
    SELECT count(*) AS bad FROM read_parquet('{path}')
    WHERE isinf(MOM_20) OR isnan(MOM_20) OR isinf(REV_5) OR isnan(REV_5)
    """).df()
    conn.close()
    assert int(frame.at[0, "bad"]) == 0
