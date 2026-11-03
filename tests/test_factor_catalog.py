"""公式文档、单因子讲解与手工复现 SQL 的测试。

最要紧的一条是 ``test_reproduction_sql_matches_stored_values``：
文档里给出的手工 SQL 必须算得出库里的数，否则这份文档就是骗人的。
"""

import duckdb
import pytest

import factors
from scripts.build_factors import catalog
from scripts.config import DB_FILE, FACTOR_DIR


def _has_factor_data() -> bool:
    """磁盘上是否有因子产物。"""
    return bool(list(FACTOR_DIR.glob("*/year=*/part.parquet")))


def test_formula_doc_contains_formula_steps_and_sql(tmp_path):
    """公式文档要同时给出公式、白话步骤与真实实现表达式。"""
    path = catalog.write_formula_doc(tmp_path / "formulas.md")
    text = path.read_text(encoding="utf-8")
    assert "怎么读这份文档" in text

    spec = factors.get("MOM_20")
    assert spec.formula in text
    for step in spec.steps:
        assert step in text
    assert spec.layers[0] in text
    # 待实现的规划因子也要列在同一个文档里
    assert "`MOM_250`" in text


def test_formula_doc_covers_every_family(tmp_path):
    """十个族都要出现在公式文档里。"""
    path = catalog.write_formula_doc(tmp_path / "formulas.md")
    text = path.read_text(encoding="utf-8")
    for family in factors.FAMILIES:
        assert f"（`{family}`）" in text


def test_explanation_includes_reproduction_and_comparison():
    """单因子讲解要能直接照做：有 SQL，也有和库值对照的代码。"""
    text = catalog.explanation(factors.get("BOLL_POS_20"), stock="SZ000001")
    assert "**公式**" in text
    assert "**计算步骤**" in text
    assert "```sql" in text
    assert "SZ000001" in text
    assert "load_factor_panel" in text


def test_reproduction_sql_computes_window_before_filtering():
    """窗口必须在过滤之前算完，否则月初会凭空变空值。"""
    sql = catalog.reproduction_sql(factors.get("MOM_60"), stock="SH600000")
    assert sql.index("lag(close, 60)") < sql.index("trade_date >= DATE")


def test_reproduction_sql_expands_named_window():
    """命名窗口 w 要展开成显式窗口，单独一条 SQL 才能跑。"""
    sql = catalog.reproduction_sql(factors.get("MOM_20"), stock="SH600000")
    assert "over w" not in sql
    assert "PARTITION BY stock_code ORDER BY trade_date" in sql


def test_reproduction_sql_joins_extra_sources():
    """用到指数数据的因子，复现 SQL 里必须带上数据源的 CTE 与 join。"""
    sql = catalog.reproduction_sql(factors.get("RS_20"), stock="SH600000")
    assert "src_index_000300 AS (" in sql
    assert "LEFT JOIN src_index_000300" in sql


def test_explain_reports_unknown_factor_without_traceback(capsys):
    """名字写错时给人话，不要甩一段报错栈。"""
    code = catalog.main(["--explain", "NOT_A_FACTOR"])
    out = capsys.readouterr().out
    assert code == 1
    assert "没有叫" in out
    assert "MOM_20" in out


def test_explain_shows_formula_for_planned_factor(capsys):
    """还没实现的因子也要能查到公式与实现要点。"""
    code = catalog.main(["--explain", "MOM_250"])
    out = capsys.readouterr().out
    assert code == 0
    assert "还没实现" in out
    assert "close_t / close_(t-250) - 1" in out


@pytest.mark.skipif(not DB_FILE.exists() or not _has_factor_data(),
                    reason="需要真实主库与已落盘的因子数据")
def test_reproduction_sql_matches_stored_values():
    """文档里 29 条手工复现 SQL 的取值，必须与库里的落盘值一致。"""
    conn = duckdb.connect(str(DB_FILE), read_only=True)
    problems = []
    for spec in factors.all_factors():
        sql = catalog.reproduction_sql(
            spec, stock="SH600000", start="2024-01-01", end="2024-01-31"
        )
        manual = conn.execute(sql).df().rename(columns={spec.name: "manual"})
        stored = conn.execute(
            f"SELECT trade_date, {spec.name} AS stored "
            f"FROM read_parquet('{FACTOR_DIR}/{spec.family}/year=2024/part.parquet') "
            "WHERE stock_code = 'SH600000' "
            "AND trade_date BETWEEN DATE '2024-01-01' AND DATE '2024-01-31' ORDER BY trade_date"
        ).df()
        merged = manual[["trade_date", "manual"]].merge(stored, on="trade_date", how="outer")
        if len(merged) < 20:
            problems.append(f"{spec.name}: 行数不足（{len(merged)}）")
            continue
        if (merged["manual"].isna() != merged["stored"].isna()).any():
            problems.append(f"{spec.name}: 空值状态与库不一致")
            continue
        left = merged["manual"].astype("float64")
        right = merged["stored"].astype("float64")
        scale = max(1.0, float(right.abs().max()))
        if (left - right).abs().max() / scale > 1e-5:
            problems.append(f"{spec.name}: 取值与库不一致")
    conn.close()
    assert not problems, problems
