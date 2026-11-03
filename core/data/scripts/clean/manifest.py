"""清洗产物的 manifest 记录。"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from scripts.config import CLEAN_DIR
from scripts.logging_conf import get_logger

logger = get_logger("clean.manifest")

MANIFEST_FILE = CLEAN_DIR / "manifest.json"
SCRIPT_VERSION = "1"


def build_manifest(
    *,
    start: str,
    end: str,
    n_stocks: int,
    n_rows: int,
    checks: dict,
    source_stats: dict,
    notes: list[str],
    year_stats: list[dict],
) -> dict:
    """组装 manifest 字典。

    参数:
        start: 数据起始日。
        end: 数据截止日。
        n_stocks: 覆盖的股票数。
        n_rows: 长表总行数。
        checks: 各项校验结果。
        source_stats: 数据源覆盖统计。
        notes: 已知限制说明。
        year_stats: 分年统计。
    """
    return {
        "data_version": f"tushare_fullA_{start.replace('-', '')}_{end.replace('-', '')}",
        "script_version": SCRIPT_VERSION,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "input": {
            "source": "Tushare（经中转服务）",
            "start_date": start,
            "cutoff_date": end,
            "interfaces": [
                "trade_cal", "stock_basic", "namechange",
                "daily", "adj_factor", "daily_basic", "stk_limit", "suspend_d",
            ],
        },
        "shape": {"n_stocks": n_stocks, "n_rows": n_rows, "n_years": len(year_stats)},
        "checks": checks,
        "sources": source_stats,
        "year_stats": year_stats,
        "known_limitations": notes,
    }


def write_manifest(payload: dict) -> Path:
    """把 manifest 写入 data/clean/manifest.json。

    参数:
        payload: manifest 字典。
    """
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("manifest 写入 %s", MANIFEST_FILE)
    return MANIFEST_FILE


def count_rows() -> int:
    """统计 data/clean/daily 的总行数。"""
    import duckdb

    glob = str(CLEAN_DIR / "daily" / "year=*" / "part.parquet")
    conn = duckdb.connect()
    total = int(conn.execute(f"SELECT count(*) FROM read_parquet('{glob}')").fetchone()[0])
    conn.close()
    return total


def count_stocks() -> int:
    """统计 data/clean/daily 覆盖的股票数。"""
    import duckdb

    glob = str(CLEAN_DIR / "daily" / "year=*" / "part.parquet")
    conn = duckdb.connect()
    total = int(
        conn.execute(f"SELECT count(DISTINCT stock_code) FROM read_parquet('{glob}')").fetchone()[0]
    )
    conn.close()
    return total
