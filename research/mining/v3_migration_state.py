"""V3 迁移记录、业务写入统计和回滚资格判断。"""

from __future__ import annotations

from typing import Any

import duckdb

from mining.storage_v2_schema import schema_exists
from mining.storage_v3_publication import V3_PUBLICATION_SOURCE_KIND

V3_BUSINESS_TABLES = (
    "research_input",
    "source_reference",
    "candidate_source",
    "evaluation_evidence",
    "run_context",
    "candidate_approval",
    "candidate_state",
    "event",
)


def migration_rows_conn(
    conn: duckdb.DuckDBPyConnection,
) -> dict[str, dict[str, Any]]:
    """按来源读取 V3 迁移记录。"""

    rows = conn.execute(
        "SELECT * FROM mining_v3.migration_source ORDER BY source_schema"
    ).fetchall()
    columns = [item[0] for item in conn.description]
    return {
        str(row[0]): dict(zip(columns, row, strict=True))
        for row in rows
    }


def table_exists_conn(
    conn: duckdb.DuckDBPyConnection,
    schema: str,
    table: str,
) -> bool:
    """判断指定 DuckDB 表是否存在。"""

    row = conn.execute(
        "SELECT count(*) FROM duckdb_tables() "
        "WHERE schema_name=? AND table_name=?",
        [schema, table],
    ).fetchone()
    return bool(row and row[0])


def v3_business_counts_conn(
    conn: duckdb.DuckDBPyConnection,
) -> dict[str, int]:
    """统计会阻止迁移回滚的 V3 业务事实。"""

    return {
        table: int(
            conn.execute(f"SELECT count(*) FROM mining_v3.{table}").fetchone()[0]
        )
        for table in V3_BUSINESS_TABLES
    }


def v3_publication_count_conn(conn: duckdb.DuckDBPyConnection) -> int:
    """统计共享正式发布表中的 V3 批次。"""

    if not schema_exists(conn, "mining"):
        return 0
    row = conn.execute(
        "SELECT count(*) FROM mining.publication_batch WHERE source_kind=?",
        [V3_PUBLICATION_SOURCE_KIND],
    ).fetchone()
    return int(row[0])


__all__ = [
    "V3_BUSINESS_TABLES",
    "migration_rows_conn",
    "table_exists_conn",
    "v3_business_counts_conn",
    "v3_publication_count_conn",
]
