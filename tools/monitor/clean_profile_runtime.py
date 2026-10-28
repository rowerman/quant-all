"""清洗年度流程的临时目录运行与稳定结果摘要。"""

from __future__ import annotations

import hashlib
from pathlib import Path
import resource
import tempfile
import time
from typing import Any

import duckdb
import pandas as pd

from scripts.build_db import build as build_db
from scripts.build_db.ddl import SCHEMA_DDL
from scripts.clean import pipeline
from tools.monitor.clean_profile_probe import CleanProfileProbe


def peak_rss_mb() -> float:
    """返回当前进程峰值 RSS，单位 MB。"""

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3)


def stable_frame_digest(frame: pd.DataFrame) -> dict[str, Any]:
    """返回 DataFrame 行序敏感的稳定内容摘要。"""

    normalized = frame.reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(
        normalized,
        index=False,
        categorize=True,
    ).to_numpy(dtype="uint64", copy=False)
    schema = [
        [str(name), str(dtype)]
        for name, dtype in zip(normalized.columns, normalized.dtypes, strict=True)
    ]
    return {
        "rows": int(len(normalized)),
        "columns": schema,
        "sha256": hashlib.sha256(row_hashes.tobytes()).hexdigest(),
    }


def _year_bounds(year: int) -> tuple[str, str]:
    """返回完整自然年首尾日期。"""

    return f"{year}-01-01", f"{year}-12-31"


def _build_duckdb_daily(work_root: Path) -> dict[str, Any]:
    """把临时清洗分区构造成隔离 DuckDB 日线三表。"""

    database = work_root / "profile.duckdb"
    original_glob = build_db.DAILY_GLOB
    build_db.DAILY_GLOB = str(work_root / "daily" / "year=*" / "part.parquet")
    started = time.perf_counter()
    try:
        with duckdb.connect(str(database)) as conn:
            conn.execute("SET threads=1")
            for statement in SCHEMA_DDL:
                conn.execute(statement)
            build_db.build_daily(conn)
            counts = {
                table: int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
                for table in ("md.daily_bar", "md.daily_hfq", "md.daily_qfq")
            }
    finally:
        build_db.DAILY_GLOB = original_glob
    return {
        "elapsed_s": round(time.perf_counter() - started, 6),
        "threads": 1,
        "row_counts": counts,
    }


def profile_year(year: int, *, seed_year: int | None = None) -> dict[str, Any]:
    """在临时目录运行一个目标年度并返回耗时、结果摘要和 DuckDB 证据。"""

    seed = year - 1 if seed_year is None else seed_year
    start, end = _year_bounds(year)
    seed_start, seed_end = _year_bounds(seed)
    worker_started = time.perf_counter()
    original_daily_dir = pipeline.DAILY_DIR
    try:
        with tempfile.TemporaryDirectory(prefix=f"clean-profile-{year}-") as directory:
            work_root = Path(directory)
            seed_daily = work_root / "seed_daily"
            pipeline.DAILY_DIR = seed_daily
            pipeline.DAILY_DIR.mkdir(parents=True, exist_ok=True)

            seed_started = time.perf_counter()
            _seed_stats, carry = pipeline.process_year(
                seed,
                seed_start,
                seed_end,
                {},
            )
            seed_elapsed = time.perf_counter() - seed_started

            pipeline.DAILY_DIR = work_root / "daily"
            pipeline.DAILY_DIR.mkdir(parents=True, exist_ok=True)
            with CleanProfileProbe() as probe:
                started = time.perf_counter()
                stats, next_carry = pipeline.process_year(year, start, end, carry)
                elapsed = time.perf_counter() - started
            summary = probe.summary(elapsed)

            parquet_path = work_root / "daily" / f"year={year}" / "part.parquet"
            frame = pd.read_parquet(parquet_path)
            digest = stable_frame_digest(frame)
            del frame
            duckdb_result = _build_duckdb_daily(work_root)
    finally:
        pipeline.DAILY_DIR = original_daily_dir

    return {
        "schema_version": "clean_cpp_profile_worker_v1",
        "year": year,
        "seed_year": seed,
        "date_range": [start, end],
        "seed_elapsed_s": round(seed_elapsed, 6),
        "elapsed_s": summary["elapsed_s"],
        "peak_rss_mb": peak_rss_mb(),
        "worker_elapsed_s": round(time.perf_counter() - worker_started, 6),
        "stats": stats,
        "carry_rows": int(len(next_carry)),
        "content_digest": digest,
        "profile": summary,
        "duckdb_daily": duckdb_result,
    }
