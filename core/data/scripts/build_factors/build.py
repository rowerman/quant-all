"""因子构建：逐族逐年计算并落盘。

用法：
    python -m scripts.build_factors.build                        # 全部已注册因子、全部年份
    python -m scripts.build_factors.build --families momentum    # 只建动量族
    python -m scripts.build_factors.build --families momentum --start 2024-01-01 --end 2024-12-31

跨年窗口的处理：每个族按最长回看窗口往前多取一段交易日，算完只写当年的行，
所以 250 日动量在 1 月初也有值，不必整段历史一次算完。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

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


def build(
    families=None,
    start=None,
    end=None,
    *,
    db_file=DB_FILE,
    factor_dir=FACTOR_DIR,
    catalog_file=FACTOR_DB_FILE,
    write_catalog: bool = True,
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

    返回:
        构建结果字典，含 run_id、族、年份、因子数、行数与各因子覆盖统计。
    """
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
    logger.info("开始构建因子：族=%s 年=%s 起=%s", selected, [w[0] for w in windows], start or "开头")

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
    parser = argparse.ArgumentParser(description="构建因子库：逐族逐年计算并落盘")
    parser.add_argument("--families", default="", help="逗号分隔的因子族；留空表示全部")
    parser.add_argument("--start", default=None, help="起始日 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="截止日 YYYY-MM-DD")
    parser.add_argument("--no-catalog", action="store_true", help="只落盘，不更新目录库")
    args = parser.parse_args(argv)

    families = [item.strip() for item in args.families.split(",") if item.strip()]
    result = build(
        families or None, args.start, args.end, write_catalog=not args.no_catalog
    )
    print(
        f"完成：{result['factor_count']} 个因子 × {len(result['years'])} 年，"
        f"写入 {result['rows_written']:,} 行"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
