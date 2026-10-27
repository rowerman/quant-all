"""因子构建：按族计算全区间并分年落盘。

用法：
    python -m scripts.build_factors.build                        # 全部已注册因子、全部年份
    python -m scripts.build_factors.build --families momentum    # 只建动量族
    python -m scripts.build_factors.build --families momentum --start 2024-01-01 --end 2024-12-31

默认把每个族的选定区间一次算完，再按年分区。查询仍然从库开头取到区间末尾，
因此跨年窗口、停牌和截面算子保持完整；`yearly` 模式保留为回退和性能对照。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import duckdb
import pandas as pd

import factors
from factors.registry import family_factors
from scripts.build_factors import catalog
from scripts.build_factors.sql import coverage_query, family_file, family_query
from scripts.config import DB_FILE, FACTOR_DB_FILE, FACTOR_DIR
from scripts.errors import FactorBuildError
from scripts.logging_conf import get_logger

logger = get_logger("build_factors.build")

BUILD_MODES = ("full_history", "yearly")
DEFAULT_MEMORY_LIMIT = "3GB"
_MEMORY_LIMIT_PATTERN = re.compile(r"^\d+(?:\.\d+)?(?:MB|GB)$", re.IGNORECASE)


def open_source(db_file=DB_FILE) -> tuple[duckdb.DuckDBPyConnection, str]:
    """打开内存连接并以只读方式挂载日线主库。

    参数:
        db_file: 主库路径。

    返回:
        ``(连接, 主库别名)``。
    """
    target = Path(db_file)
    if not target.exists():
        raise FileNotFoundError(f"主库不存在: {target}")
    conn = duckdb.connect()
    conn.execute(f"ATTACH '{target}' AS src (READ_ONLY)")
    return conn, "src"


def year_windows(conn: duckdb.DuckDBPyConnection, start=None, end=None) -> list[tuple[int, str, str]]:
    """列出要构建的年份与每年首末交易日。

    参数:
        conn: 已挂载主库的连接。
        start: 起始日，闭区间。
        end: 截止日，闭区间。
    """
    conditions = []
    if start:
        conditions.append(f"trade_date >= DATE '{start}'")
    if end:
        conditions.append(f"trade_date <= DATE '{end}'")
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    frame = conn.execute(
        "SELECT CAST(year(trade_date) AS INTEGER) AS y, min(trade_date) AS first_day, "
        f"max(trade_date) AS last_day FROM src.md.calendar{where} GROUP BY 1 ORDER BY 1"
    ).df()
    return [(int(row.y), str(row.first_day), str(row.last_day)) for row in frame.itertuples()]


def build_family_year(
    conn: duckdb.DuckDBPyConnection,
    family: str,
    year: int,
    first_day: str,
    last_day: str,
    factor_dir=FACTOR_DIR,
) -> int:
    """构建一个族的一年因子并落盘。

    参数:
        conn: 已挂载主库的连接。
        family: 因子族。
        year: 年份。
        first_day: 当年首个交易日。
        last_day: 当年最后一个交易日。
        factor_dir: 因子数据根目录。

    返回:
        写入行数。
    """
    specs = family_factors(family)
    query = family_query(specs, spine_start=first_day, spine_end=last_day)
    path = family_file(factor_dir, family, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(
        f"COPY ({query}) TO '{path}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 200000)"
    )
    rows = conn.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    logger.info("%s %d 落盘 %d 行 -> %s", family, year, rows, path)
    return int(rows)


def _range_memory_limit() -> str:
    """读取并校验全历史构建的 DuckDB 缓冲上限。"""

    value = os.environ.get("QUANT_FACTOR_BUILD_MEMORY_LIMIT", DEFAULT_MEMORY_LIMIT).strip()
    if not _MEMORY_LIMIT_PATTERN.fullmatch(value):
        raise FactorBuildError(
            "QUANT_FACTOR_BUILD_MEMORY_LIMIT 必须是 MB 或 GB 数值，"
            f"收到 {value!r}"
        )
    return value.upper()


def _validate_partition(
    conn: duckdb.DuckDBPyConnection,
    path: Path,
    *,
    family: str,
    year: int,
    first_day: str,
    last_day: str,
    factor_names: list[str],
) -> int:
    """校验年度分区的主键、行数和列类型。"""

    expected_columns = ["stock_code", "trade_date", *factor_names]
    described = conn.execute(
        "DESCRIBE SELECT * FROM read_parquet(?, hive_partitioning = false)",
        [str(path)],
    ).fetchall()
    actual_columns = [row[0] for row in described]
    if actual_columns != expected_columns:
        raise FactorBuildError(
            f"{family} {year} 年分区列不符合契约："
            f"期望 {expected_columns}，实际 {actual_columns}"
        )
    factor_types = {row[0]: str(row[1]).upper() for row in described}
    wrong_types = {
        name: factor_types[name]
        for name in factor_names
        if factor_types[name] != "FLOAT"
    }
    if wrong_types:
        raise FactorBuildError(
            f"{family} {year} 年分区因子列不是 FLOAT：{wrong_types}"
        )

    rows, distinct_keys = conn.execute(
        "SELECT count(*), count(DISTINCT (stock_code, trade_date)) "
        "FROM read_parquet(?)",
        [str(path)],
    ).fetchone()
    if int(rows) != int(distinct_keys):
        raise FactorBuildError(f"{family} {year} 年分区存在重复主键")

    expected_rows = conn.execute(
        "SELECT count(*) FROM src.md.daily_hfq "
        "WHERE trade_date BETWEEN ? AND ?",
        [first_day, last_day],
    ).fetchone()[0]
    if int(rows) != int(expected_rows):
        raise FactorBuildError(
            f"{family} {year} 年分区行数不一致："
            f"输出 {rows}，日线主表 {expected_rows}"
        )

    extra, missing = conn.execute(
        """
        SELECT
          (SELECT count(*) FROM (
              SELECT stock_code, trade_date FROM read_parquet(?)
              EXCEPT
              SELECT stock_code, trade_date FROM src.md.daily_hfq
              WHERE trade_date BETWEEN ? AND ?
          )),
          (SELECT count(*) FROM (
              SELECT stock_code, trade_date FROM src.md.daily_hfq
              WHERE trade_date BETWEEN ? AND ?
              EXCEPT
              SELECT stock_code, trade_date FROM read_parquet(?)
          ))
        """,
        [str(path), first_day, last_day, first_day, last_day, str(path)],
    ).fetchone()
    if int(extra) or int(missing):
        raise FactorBuildError(
            f"{family} {year} 年分区主键不一致："
            f"多余 {extra} 行，缺失 {missing} 行"
        )
    return int(rows)


def _publish_partitions(
    staged: dict[int, Path],
    *,
    family: str,
    factor_dir: Path | str,
) -> None:
    """备份旧年度文件后原子发布；任一步失败时恢复旧文件。"""

    backup_root = next(iter(staged.values())).parents[2] / "_backup"
    published: list[tuple[Path, Path | None]] = []
    try:
        for year, source in sorted(staged.items()):
            target = family_file(factor_dir, family, year)
            target.parent.mkdir(parents=True, exist_ok=True)
            backup = None
            if target.exists():
                backup = backup_root / f"year={year}" / "part.parquet"
                backup.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(target, backup)
                except OSError:
                    shutil.copy2(target, backup)
            os.replace(source, target)
            published.append((target, backup))
    except Exception:
        for target, backup in reversed(published):
            if backup is None:
                target.unlink(missing_ok=True)
                try:
                    target.parent.rmdir()
                except OSError:
                    pass
            else:
                os.replace(backup, target)
        raise


def build_family_range(
    conn: duckdb.DuckDBPyConnection,
    family: str,
    windows: list[tuple[int, str, str]],
    factor_dir=FACTOR_DIR,
) -> int:
    """一次计算一个族的完整区间，按年分区校验后发布。"""

    if not windows:
        raise FactorBuildError(f"因子族 {family} 没有可构建的年份")
    specs = family_factors(family)
    factor_names = [spec.name for spec in specs]
    first_day = windows[0][1]
    last_day = windows[-1][2]
    staging_root = Path(factor_dir) / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    previous_memory_limit = conn.execute(
        "SELECT current_setting('memory_limit')"
    ).fetchone()[0]
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(
        prefix=f"{family}-",
        dir=staging_root,
    ) as directory:
        try:
            staging = Path(directory)
            partitions = staging / "_partitions"
            memory_limit = _range_memory_limit()
            conn.execute("SET memory_limit = ?", [memory_limit])

            query = family_query(specs, spine_start=first_day, spine_end=last_day)
            partitions_sql = str(partitions).replace("'", "''")
            conn.execute(
                f"COPY (SELECT *, CAST(year(trade_date) AS INTEGER) AS year FROM ({query}) q) "
                f"TO '{partitions_sql}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 200000, "
                "PARTITION_BY (year), FILENAME_PATTERN 'part')"
            )

            expected_years = {year for year, _, _ in windows}
            actual_years = {
                int(path.name.split("=")[1])
                for path in partitions.glob("year=*")
                if path.is_dir()
            }
            if actual_years != expected_years:
                raise FactorBuildError(
                    f"{family} 年分区不完整：期望 {sorted(expected_years)}，"
                    f"实际 {sorted(actual_years)}"
                )

            staged: dict[int, Path] = {}
            rows_written = 0
            for year, first, last in windows:
                candidates = sorted((partitions / f"year={year}").glob("*.parquet"))
                if len(candidates) != 1:
                    raise FactorBuildError(
                        f"{family} {year} 年分区应生成一个 Parquet，"
                        f"实际 {len(candidates)} 个"
                    )
                raw = candidates[0]
                normalized = raw.with_name("part.parquet")
                write_target = (
                    raw.with_name("part.normalized.parquet")
                    if raw == normalized
                    else normalized
                )
                columns = ", ".join(
                    f'"{name}"' for name in ["stock_code", "trade_date", *factor_names]
                )
                raw_sql = str(raw).replace("'", "''")
                write_target_sql = str(write_target).replace("'", "''")
                conn.execute(
                    f"COPY (SELECT {columns} FROM read_parquet('{raw_sql}')) "
                    f"TO '{write_target_sql}' "
                    "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 200000)"
                )
                if raw != normalized:
                    raw.unlink()
                if write_target != normalized:
                    os.replace(write_target, normalized)
                rows_written += _validate_partition(
                    conn,
                    normalized,
                    family=family,
                    year=year,
                    first_day=first,
                    last_day=last,
                    factor_names=factor_names,
                )
                staged[year] = normalized

            output_bytes = sum(path.stat().st_size for path in staged.values())
            _publish_partitions(staged, family=family, factor_dir=factor_dir)
        finally:
            conn.execute("SET memory_limit = ?", [previous_memory_limit])
        logger.info(
            "族 %s 全区间发布完成：年=%s 行=%d 输出字节=%d "
            "耗时=%.3fs 内存上限=%s",
            family,
            sorted(expected_years),
            rows_written,
            output_bytes,
            time.perf_counter() - started,
            memory_limit,
        )
        return rows_written


def build(
    families=None,
    start=None,
    end=None,
    *,
    db_file=DB_FILE,
    factor_dir=FACTOR_DIR,
    catalog_file=FACTOR_DB_FILE,
    write_catalog: bool = True,
    build_mode: str = "full_history",
) -> dict:
    """构建因子库。

    参数:
        families: 要构建的族；None 表示全部已注册的族。
        start: 起始日，闭区间；None 表示从数据开头。
        end: 截止日，闭区间；None 表示到数据末尾。
        db_file: 日线主库路径。
        factor_dir: 因子数据根目录。
        catalog_file: 因子目录库路径。
        write_catalog: 是否更新目录库与视图。
        build_mode: ``full_history`` 一次计算全区间并按年发布；``yearly`` 保留旧路径。

    返回:
        构建结果字典，含 run_id、构建模式、族、年份、因子数、行数与覆盖统计。
    """
    if build_mode not in BUILD_MODES:
        raise FactorBuildError(f"未知构建模式 {build_mode!r}；可选：{list(BUILD_MODES)}")
    conn, _ = open_source(db_file)
    windows = year_windows(conn, start, end)
    if not windows:
        raise FactorBuildError(f"区间 {start} ~ {end} 内没有交易日，无法构建因子")

    selected = list(families) if families else list(factors.used_families())
    unknown = [family for family in selected if family not in factors.FAMILIES]
    if unknown:
        raise FactorBuildError(f"未知因子族 {unknown}；可选：{list(factors.FAMILIES)}")

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    started = datetime.now()
    logger.info(
        "开始构建因子：模式=%s 族=%s 年=%s 起=%s",
        build_mode,
        selected,
        [w[0] for w in windows],
        start or "开头",
    )

    rows_written = 0
    coverage_rows: list[dict] = []
    factor_count = 0
    try:
        for family in selected:
            specs = family_factors(family)
            if not specs:
                logger.warning("因子族 %s 没有已注册因子，跳过", family)
                continue
            factor_count += len(specs)
            if build_mode == "full_history":
                rows_written += build_family_range(conn, family, windows, factor_dir)
            else:
                for year, first_day, last_day in windows:
                    rows_written += build_family_year(
                        conn, family, year, first_day, last_day, factor_dir
                    )
            frame: pd.DataFrame = conn.execute(
                coverage_query(specs, factor_dir, family)
            ).df()
            for row in frame.itertuples():
                coverage_rows.append({
                    "name": row.name, "family": family, "year": int(row.year),
                    "rows": int(row.rows), "non_null": int(row.non_null),
                    "min_value": row.min_value, "max_value": row.max_value,
                    "mean_value": row.mean_value, "std_value": row.std_value,
                    "p01": row.p01, "p99": row.p99,
                })
    except Exception as error:  # 构建失败也要留下运行记录
        if write_catalog:
            with catalog.connect(catalog_file) as cat:
                catalog.ensure_catalog(cat)
                catalog.record_build(cat, {
                    "run_id": run_id, "started_at": started, "finished_at": datetime.now(),
                    "families": ",".join(selected),
                    "years": f"{windows[0][0]}-{windows[-1][0]}", "factor_count": factor_count,
                    "rows_written": rows_written, "status": "failed", "message": str(error)[:500],
                })
        logger.exception("因子构建失败: %s", error)
        raise

    result = {
        "run_id": run_id,
        "build_mode": build_mode,
        "families": selected,
        "years": [window[0] for window in windows],
        "factor_count": factor_count,
        "rows_written": rows_written,
        "coverage": coverage_rows,
    }

    if write_catalog:
        with catalog.connect(catalog_file) as cat:
            catalog.ensure_catalog(cat)
            catalog.refresh_definitions(cat)
            catalog.refresh_coverage(cat, coverage_rows)
            catalog.refresh_views(cat, factor_dir)
            catalog.record_build(cat, {
                "run_id": run_id, "started_at": started, "finished_at": datetime.now(),
                "families": ",".join(selected),
                "years": f"{windows[0][0]}-{windows[-1][0]}",
                "factor_count": factor_count, "rows_written": rows_written,
                "status": "ok", "message": "",
            })
            catalog.refresh_manifest(cat)
        logger.info("目录库已更新: %s", catalog_file)

    logger.info(
        "构建完成：%d 个因子 × %d 年，写入 %d 行",
        factor_count, len(windows), rows_written,
    )
    return result


def main(argv=None) -> int:
    """构建命令行入口。

    参数:
        argv: 命令行参数，默认取 sys.argv。
    """
    parser = argparse.ArgumentParser(description="构建因子库：按族计算并分年落盘")
    parser.add_argument("--families", default="", help="逗号分隔的因子族；留空表示全部")
    parser.add_argument("--start", default=None, help="起始日 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="截止日 YYYY-MM-DD")
    parser.add_argument(
        "--build-mode",
        choices=BUILD_MODES,
        default="full_history",
        help="full_history 为默认一次构建；yearly 为旧路径",
    )
    parser.add_argument("--no-catalog", action="store_true", help="只落盘，不更新目录库")
    args = parser.parse_args(argv)

    families = [item.strip() for item in args.families.split(",") if item.strip()]
    result = build(
        families or None,
        args.start,
        args.end,
        write_catalog=not args.no_catalog,
        build_mode=args.build_mode,
    )
    print(
        f"完成：{result['factor_count']} 个因子 × {len(result['years'])} 年，"
        f"写入 {result['rows_written']:,} 行"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
