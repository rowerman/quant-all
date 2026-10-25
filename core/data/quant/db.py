"""DuckDB 连接与基础查询。"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from common.paths import DB_FILE

DEFAULT_DB = DB_FILE


def get_conn(path: Path | str | None = None, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """取得 DuckDB 连接。

    参数:
        path: 数据库文件路径，默认 data/databases/quant.duckdb。
        read_only: 只读打开，便于与写入进程并存。

    返回:
        DuckDB 连接对象。
    """
    target = Path(path) if path is not None else DEFAULT_DB
    if not target.exists():
        raise FileNotFoundError(f"数据库不存在: {target}")
    return duckdb.connect(str(target), read_only=read_only)


def query(sql: str, params: list | None = None, path: Path | str | None = None) -> pd.DataFrame:
    """执行查询并返回 DataFrame。

    参数:
        sql: SQL 语句。
        params: 绑定参数。
        path: 数据库文件路径。
    """
    with get_conn(path) as conn:
        return conn.execute(sql, params or []).df()
