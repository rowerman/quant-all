"""正式 `mined` 因子全量构建后的独立校验。"""

from __future__ import annotations

from pathlib import Path

import duckdb

from factors.published import MINED_FAMILY
from factors.registry import FactorSpec
from mining.errors import PublicationValidationError
from scripts.build_factors.sql import family_query


def _quoted(path: Path | str) -> str:
    """转义 DuckDB 单引号路径。"""

    return str(path).replace("'", "''")


def _tree_files(root: Path) -> tuple[Path, ...]:
    """返回排序后的年分区文件。"""

    files = tuple(sorted(root.glob("year=*/part.parquet")))
    if not files:
        raise PublicationValidationError("正式 mined 构建没有生成年分区")
    return files


def validate_full_tree(
    *,
    conn: duckdb.DuckDBPyConnection,
    family_root: Path,
    specs: tuple[FactorSpec, ...],
    start: str,
    end: str,
    min_coverage: float,
) -> dict[str, float | int]:
    """校验行集、主键、类型、覆盖率与有限值。"""

    files = _tree_files(family_root)
    glob = _quoted(family_root / "year=*" / "part.parquet")
    expected_rows = int(
        conn.execute(
            "SELECT count(*) FROM src.md.daily_hfq "
            "WHERE trade_date BETWEEN ? AND ?",
            [start, end],
        ).fetchone()[0]
    )
    actual_rows = sum(
        int(conn.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[0])
        for path in files
    )
    if actual_rows != expected_rows:
        raise PublicationValidationError(
            f"正式因子行数为 {actual_rows}，行情基准为 {expected_rows}"
        )
    key_count = int(
        conn.execute(
            f"SELECT count(*) FROM (SELECT stock_code, trade_date "
            f"FROM read_parquet('{glob}', hive_partitioning=1) "
            "GROUP BY 1, 2 HAVING count(*) > 1)"
        ).fetchone()[0]
    )
    if key_count:
        raise PublicationValidationError(f"正式因子主键重复 {key_count} 组")
    source_key_count = int(
        conn.execute(
            "SELECT count(*) FROM ("
            "SELECT stock_code, trade_date FROM src.md.daily_hfq "
            "WHERE trade_date BETWEEN ? AND ? "
            "GROUP BY 1, 2 HAVING count(*) > 1)",
            [start, end],
        ).fetchone()[0]
    )
    if source_key_count:
        raise PublicationValidationError(
            f"行情基准主键重复 {source_key_count} 组"
        )
    source_signature = conn.execute(
        "SELECT count(*), bit_xor(hash(stock_code, trade_date))::VARCHAR "
        "FROM src.md.daily_hfq WHERE trade_date BETWEEN ? AND ?",
        [start, end],
    ).fetchone()
    output_signature = conn.execute(
        "SELECT count(*), bit_xor(hash(stock_code, trade_date))::VARCHAR "
        f"FROM read_parquet('{glob}', hive_partitioning=1)"
    ).fetchone()
    if tuple(source_signature) != tuple(output_signature):
        raise PublicationValidationError(
            "正式因子与行情键集差异："
            f"行情 {tuple(source_signature)}，正式因子 {tuple(output_signature)}"
        )

    descriptions = conn.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{_quoted(files[0])}')"
    ).fetchall()
    types = {row[0]: row[1] for row in descriptions}
    for spec in specs:
        if types.get(spec.name) != "FLOAT":
            raise PublicationValidationError(
                f"因子 {spec.name} 的落盘类型必须是 FLOAT"
            )
    expressions = ", ".join(
        f"sum(CASE WHEN {spec.name} IS NOT NULL "
        f"AND NOT isfinite({spec.name}) THEN 1 ELSE 0 END)"
        for spec in specs
    )
    nonfinite = conn.execute(
        f"SELECT {expressions} FROM read_parquet('{glob}', hive_partitioning=1)"
    ).fetchone()
    if any(int(value or 0) for value in nonfinite):
        raise PublicationValidationError("正式因子包含 NaN 或无穷值")

    coverages: dict[str, float] = {}
    for spec in specs:
        non_null = int(
            conn.execute(
                f"SELECT count({spec.name}) FROM "
                f"read_parquet('{glob}', hive_partitioning=1)"
            ).fetchone()[0]
        )
        coverage = non_null / expected_rows if expected_rows else 0.0
        coverages[spec.name] = coverage
        if coverage < min_coverage:
            raise PublicationValidationError(
                f"因子 {spec.name} 全历史覆盖率 {coverage:.6f} 低于 {min_coverage:.6f}"
            )
    return {
        "rows": actual_rows,
        "files": len(files),
        "min_coverage": min(coverages.values()) if coverages else 0.0,
    }


def validate_no_lookahead(
    *,
    conn: duckdb.DuckDBPyConnection,
    family_root: Path,
    specs: tuple[FactorSpec, ...],
    start: str,
    end: str,
    tolerance: float,
) -> dict[str, float]:
    """用首、中、末三个截止日重算并与全量落盘值比较。"""

    dates = conn.execute(
        "SELECT trade_date FROM src.md.calendar "
        "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
        [start, end],
    ).fetchall()
    if not dates:
        raise PublicationValidationError("全历史区间没有交易日")
    selected = [dates[0][0], dates[len(dates) // 2][0], dates[-1][0]]
    glob = _quoted(family_root / "year=*" / "part.parquet")
    errors: dict[str, float] = {}
    for cutoff in selected:
        query = family_query(specs, spine_start=str(cutoff), spine_end=str(cutoff))
        selected_columns = ", ".join(f"f.{spec.name}" for spec in specs)
        for spec in specs:
            row = conn.execute(
                f"""
                SELECT max(abs(CAST(r.{spec.name} AS DOUBLE) -
                                     CAST(f.{spec.name} AS DOUBLE))),
                       sum(CASE WHEN r.{spec.name} IS NULL
                                 AND f.{spec.name} IS NOT NULL THEN 1 ELSE 0 END),
                       sum(CASE WHEN r.{spec.name} IS NOT NULL
                                 AND f.{spec.name} IS NULL THEN 1 ELSE 0 END)
                FROM ({query}) r
                LEFT JOIN read_parquet('{glob}', hive_partitioning=1) f
                  USING(stock_code, trade_date)
                WHERE f.trade_date = DATE '{cutoff}'
                """
            ).fetchone()
            difference = float(row[0] or 0.0)
            if row[1] or row[2] or difference > tolerance:
                raise PublicationValidationError(
                    f"因子 {spec.name} 在 {cutoff} 的前视复算不一致"
                )
            errors[spec.name] = max(errors.get(spec.name, 0.0), difference)
    return errors
