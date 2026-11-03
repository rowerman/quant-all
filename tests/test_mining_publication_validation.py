"""正式因子全量树键集校验测试。"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from factors.registry import FactorSpec
from mining.errors import PublicationValidationError
from mining.publication_validation import validate_full_tree


def _fixture(tmp_path: Path, stock_codes: tuple[str, ...]):
    source = tmp_path / "source.duckdb"
    with duckdb.connect(str(source)) as conn:
        conn.execute("CREATE SCHEMA md")
        conn.execute(
            "CREATE TABLE md.daily_hfq(stock_code VARCHAR, trade_date DATE)"
        )
        conn.executemany(
            "INSERT INTO md.daily_hfq VALUES (?, DATE '2024-01-02')",
            [(code,) for code in stock_codes],
        )
    family = tmp_path / "mined"
    target = family / "year=2024" / "part.parquet"
    target.parent.mkdir(parents=True)
    with duckdb.connect() as conn:
        conn.execute(
            "COPY (SELECT * FROM (VALUES "
            "('SH600000', DATE '2024-01-02', CAST(1 AS FLOAT)), "
            "('SZ000001', DATE '2024-01-02', CAST(2 AS FLOAT))"
            ") AS rows(stock_code, trade_date, TEST_FACTOR)) "
            f"TO '{target}' (FORMAT PARQUET)"
        )
        conn.execute(f"ATTACH '{source}' AS src (READ_ONLY)")
        spec = FactorSpec(
            name="TEST_FACTOR",
            family="mined",
            formula="close",
            description="键集校验测试",
            steps=("读取收盘价",),
            layers=("close",),
            lookback=0,
        )
        yield conn, family, spec


def test_full_tree_validation_accepts_matching_keys(tmp_path: Path):
    fixture = _fixture(tmp_path, ("SH600000", "SZ000001"))
    conn, family, spec = next(fixture)
    try:
        result = validate_full_tree(
            conn=conn,
            family_root=family,
            specs=(spec,),
            start="2024-01-02",
            end="2024-01-02",
            min_coverage=1.0,
        )
    finally:
        conn.close()
        fixture.close()
    assert result["rows"] == 2


def test_full_tree_validation_reports_missing_and_extra_keys(tmp_path: Path):
    fixture = _fixture(tmp_path, ("SH600000", "SH600999"))
    conn, family, spec = next(fixture)
    try:
        with pytest.raises(PublicationValidationError, match="键集差异"):
            validate_full_tree(
                conn=conn,
                family_root=family,
                specs=(spec,),
                start="2024-01-02",
                end="2024-01-02",
                min_coverage=1.0,
            )
    finally:
        conn.close()
        fixture.close()
