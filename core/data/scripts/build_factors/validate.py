"""因子自检：行集、取值、前视偏差与方向。

四类检查：

1. **行集**：每个族每年的因子文件，键集必须与主库同区间的日线键集完全一致（双向为空）；
2. **取值**：非空值不能出现 inf / nan；同一交易日的截面不能整体是常数；
3. **前视**：抽样若干交易日（含每年年初，专门覆盖跨年窗口），只用该日及以前的数据重算因子，
   与落盘值比对；任何用未来数据、或跨年窗口取错的因子都会在这里暴露；
4. **方向**：在标准股票池上算因子与 ``fwd_ret_close_5`` 的 Rank IC 与 ICIR，
   核对符号是否与因子定义里的 ``direction`` 一致。

结论写进 ``data/databases/factors.duckdb`` 的 ``meta.factor_check`` 表，并打印到终端。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import factors
from factors.registry import family_factors
from scripts.build_factors import catalog
from scripts.build_factors.build import open_source
from scripts.build_factors.sql import family_file, family_glob, family_query
from scripts.config import DB_FILE, FACTOR_DB_FILE, FACTOR_DIR
from scripts.errors import FactorCheckError
from scripts.logging_conf import get_logger

logger = get_logger("build_factors.validate")

# 算 IC 时的标准股票池：可买入、非 ST、上市满 60 天
POOL_FILTER = (
    "b.can_buy_close AND NOT b.is_st "
    "AND datediff('day', b.list_date, b.trade_date) >= 60"
)

# 抽样重算的容差：落盘是 float32，重算是 double，量级上按相对误差比较
REL_TOLERANCE = 1e-5


def built_years(factor_dir=FACTOR_DIR, family: str = "") -> list[int]:
    """列出某个族已经落盘的年份。

    参数:
        factor_dir: 因子数据根目录。
        family: 因子族。
    """
    root = Path(factor_dir) / family
    return sorted(int(path.parent.name.split("=")[1]) for path in root.glob("year=*/part.parquet"))


def sample_dates(conn, years: list[int], count: int = 12) -> list[str]:
    """挑抽样日期：每年第三个交易日 + 全区间均匀取点。

    参数:
        conn: 已挂载主库的连接。
        years: 已构建的年份。
        count: 期望的抽样日期数。
    """
    frame = conn.execute(
        "SELECT trade_date, CAST(year(trade_date) AS INTEGER) AS y FROM src.md.calendar "
        f"WHERE CAST(year(trade_date) AS INTEGER) IN ({', '.join(str(y) for y in years)}) "
        "ORDER BY trade_date"
    ).df()
    if frame.empty:
        raise FactorCheckError("已构建的年份里没有交易日，无法抽样")

    picked: list[str] = []
    for year in years:
        days = frame.loc[frame["y"] == year, "trade_date"].tolist()
        if len(days) >= 3:
            picked.append(str(days[2]))
    step = max(1, len(frame) // max(1, count))
    picked.extend(str(day) for day in frame["trade_date"].tolist()[::step][:count])
    return sorted(set(picked))


def check_row_sets(conn, family: str, factor_dir=FACTOR_DIR) -> list[dict]:
    """检查每个年文件的键集与主库一致。

    参数:
        conn: 已挂载主库的连接。
        family: 因子族。
        factor_dir: 因子数据根目录。
    """
    results = []
    for year in built_years(factor_dir, family):
        path = family_file(factor_dir, family, year)
        frame = conn.execute(f"""
        SELECT
          (SELECT count(*) FROM (
              SELECT stock_code, trade_date FROM read_parquet('{path}')
              EXCEPT
              SELECT stock_code, trade_date FROM src.md.daily_hfq
              WHERE trade_date BETWEEN DATE '{year}-01-01' AND DATE '{year}-12-31')) AS extra_rows,
          (SELECT count(*) FROM (
              SELECT stock_code, trade_date FROM src.md.daily_hfq
              WHERE trade_date BETWEEN DATE '{year}-01-01' AND DATE '{year}-12-31'
              EXCEPT
              SELECT stock_code, trade_date FROM read_parquet('{path}'))) AS missing_rows
        """).df()
        extra = int(frame.at[0, "extra_rows"])
        missing = int(frame.at[0, "missing_rows"])
        results.append({
            "name": f"{family}:{year}",
            "kind": "row_set",
            "scope": family,
            "value": float(extra + missing),
            "detail": f"{year} 年多余 {extra} 行、缺失 {missing} 行",
        })
    return results


def check_values(conn, family: str, factor_dir=FACTOR_DIR) -> list[dict]:
    """检查非有限值与截面常数。

    参数:
        conn: 已挂载主库的连接。
        family: 因子族。
        factor_dir: 因子数据根目录。
    """
    glob = family_glob(factor_dir, family)
    results = []
    for spec in family_factors(family):
        column = spec.name
        frame = conn.execute(f"""
        SELECT
          (SELECT count(*) FROM read_parquet('{glob}', hive_partitioning = 1)
             WHERE isinf({column}) OR isnan({column})) AS non_finite,
          (SELECT count(*) FROM (
              SELECT trade_date FROM read_parquet('{glob}', hive_partitioning = 1)
              WHERE {column} IS NOT NULL GROUP BY 1 HAVING stddev_pop({column}) = 0)) AS flat_days
        """).df()
        non_finite = int(frame.at[0, "non_finite"])
        flat_days = int(frame.at[0, "flat_days"])
        results.append({
            "name": column,
            "kind": "value",
            "scope": family,
            "value": float(non_finite),
            "detail": f"非有限值 {non_finite} 个；截面为常数的交易日 {flat_days} 天",
        })
    return results


def check_forward_look(conn, family: str, dates, factor_dir=FACTOR_DIR) -> list[dict]:
    """抽样日期上只用历史数据重算，与落盘值比对。

    参数:
        conn: 已挂载主库的连接。
        family: 因子族。
        dates: 抽样交易日序列。
        factor_dir: 因子数据根目录。
    """
    specs = family_factors(family)
    worst: dict[str, float] = {spec.name: 0.0 for spec in specs}
    null_mismatch: dict[str, int] = {spec.name: 0 for spec in specs}
    max_abs: dict[str, float] = {spec.name: 0.0 for spec in specs}

    for day in dates:
        year = int(day[:4])
        path = family_file(factor_dir, family, year)
        if not Path(path).exists():
            continue
        query = family_query(specs, spine_start=day, spine_end=day)
        computed = conn.execute(query).df()
        names = ", ".join(spec.name for spec in specs)
        stored = conn.execute(
            f"SELECT stock_code, trade_date, {names} FROM read_parquet('{path}') "
            f"WHERE trade_date = DATE '{day}'"
        ).df()
        merged = stored.merge(computed, on=["stock_code", "trade_date"], suffixes=("_s", "_r"))
        for spec in specs:
            left = merged[f"{spec.name}_s"].astype("float64")
            right = merged[f"{spec.name}_r"].astype("float64")
            both = left.notna() & right.notna()
            null_mismatch[spec.name] += int((left.isna() != right.isna()).sum())
            if both.any():
                diff = (left[both] - right[both]).abs()
                scale = np.maximum(1.0, right[both].abs())
                worst[spec.name] = max(worst[spec.name], float((diff / scale).max()))
                max_abs[spec.name] = max(max_abs[spec.name], float(diff.max()))

    results = []
    for spec in specs:
        results.append({
            "name": spec.name,
            "kind": "forward_look",
            "scope": family,
            "value": worst[spec.name],
            "detail": (
                f"抽样 {len(dates)} 个交易日重算，最大相对偏差 {worst[spec.name]:.3e}，"
                f"最大绝对偏差 {max_abs[spec.name]:.3e}，容差 {REL_TOLERANCE:.0e}"
            ),
        })
        # 空值不一致单独记一条：用了未来数据的因子会在这里暴露（截断后算不出来）
        results.append({
            "name": spec.name,
            "kind": "forward_look_null",
            "scope": family,
            "value": float(null_mismatch[spec.name]),
            "detail": f"抽样 {len(dates)} 个交易日里，落盘值与截断重算值的空值状态不一致 {null_mismatch[spec.name]} 处",
        })
    return results


def check_direction(conn, family: str, start: str, end: str, factor_dir=FACTOR_DIR,
                    horizon: int = 5) -> list[dict]:
    """算因子与未来收益的 Rank IC 与 ICIR。

    参数:
        conn: 已挂载主库的连接。
        family: 因子族。
        start: IC 统计区间起始日。
        end: IC 统计区间截止日。
        factor_dir: 因子数据根目录。
        horizon: 标签周期，默认 5 日。
    """
    glob = family_glob(factor_dir, family)
    label = f"fwd_ret_close_{horizon}"
    results = []
    for spec in family_factors(family):
        frame = conn.execute(f"""
        WITH joined AS (
            SELECT b.trade_date AS trade_date, r.{spec.name} AS factor, y.{label} AS label,
                   rank() OVER (PARTITION BY b.trade_date ORDER BY r.{spec.name})
                     + (count(*) OVER (PARTITION BY b.trade_date, r.{spec.name}) - 1) / 2.0 AS rx,
                   rank() OVER (PARTITION BY b.trade_date ORDER BY y.{label})
                     + (count(*) OVER (PARTITION BY b.trade_date, y.{label}) - 1) / 2.0 AS ry
            FROM read_parquet('{glob}', hive_partitioning = 1) r
            JOIN src.md.fwd_return y
              ON y.stock_code = r.stock_code AND y.trade_date = r.trade_date
            JOIN src.md.daily_hfq b
              ON b.stock_code = r.stock_code AND b.trade_date = r.trade_date
            WHERE b.trade_date BETWEEN DATE '{start}' AND DATE '{end}'
              AND {POOL_FILTER}
              AND isfinite(r.{spec.name}) AND isfinite(y.{label})
        ), daily AS (
            SELECT trade_date, corr(rx, ry) AS ic FROM joined GROUP BY 1 HAVING count(*) >= 5
        )
        SELECT avg(ic) AS ic_mean, stddev_samp(ic) AS ic_std, count(*) AS days
        FROM daily WHERE isfinite(ic)
        """).df()
        ic_mean = frame.at[0, "ic_mean"]
        ic_std = frame.at[0, "ic_std"]
        days = int(frame.at[0, "days"])
        if ic_mean is None or pd.isna(ic_mean) or days < 20:
            results.append({
                "name": spec.name, "kind": "direction", "scope": family, "value": float("nan"),
                "detail": f"{start}~{end} 有效交易日 {days} 天，IC 不足，方向未判定",
            })
            continue
        icir = float(ic_mean) / float(ic_std) if ic_std else float("nan")
        aligned = (float(ic_mean) > 0) == (spec.direction > 0)
        results.append({
            "name": spec.name,
            "kind": "direction",
            "scope": family,
            "value": float(ic_mean),
            "detail": (
                f"{start}~{end} Rank IC 均值 {ic_mean:+.4f}，ICIR {icir:+.3f}，"
                f"{days} 个交易日；因子定义方向 {'越大越看多' if spec.direction > 0 else '越大越看空'}，"
                f"与实测{'一致' if aligned else '相反'}"
            ),
        })
    return results


def default_ic_period(conn, years: list[int]) -> tuple[str, str]:
    """选默认的 IC 统计区间：最近一个完整年份。

    参数:
        conn: 已挂载主库的连接。
        years: 已构建的年份。
    """
    last_data_year = int(conn.execute(
        "SELECT max(CAST(year(trade_date) AS INTEGER)) FROM src.md.calendar"
    ).fetchone()[0])
    candidates = [year for year in years if year < last_data_year]
    year = max(candidates) if candidates else max(years)
    return f"{year}-01-01", f"{year}-12-31"


def run_checks(families=None, *, factor_dir=FACTOR_DIR, db_file=DB_FILE,
               catalog_file=FACTOR_DB_FILE, samples: int = 12,
               ic_start: str | None = None, ic_end: str | None = None,
               write_catalog: bool = True) -> dict:
    """跑完全部自检。

    参数:
        families: 要检查的族；None 表示全部已落盘的族。
        factor_dir: 因子数据根目录。
        db_file: 日线主库路径。
        catalog_file: 因子目录库路径。
        samples: 前视检查抽样日期数。
        ic_start: IC 统计起始日；None 用默认年份。
        ic_end: IC 统计截止日；None 用默认年份。
        write_catalog: 是否把结论写进目录库。
    """
    conn, _ = open_source(db_file)
    available = [family for family in factors.FAMILIES if built_years(factor_dir, family)]
    selected = [family for family in (families or available) if family in available]
    if not selected:
        raise FactorCheckError(f"没有可检查的因子产物，先跑 scripts.build_factors.build；已有：{available}")

    checks: list[dict] = []
    summary: list[dict] = []
    for family in selected:
        years = built_years(factor_dir, family)
        dates = sample_dates(conn, years, samples)
        if ic_start and ic_end:
            period = (ic_start, ic_end)
        else:
            period = default_ic_period(conn, years)

        row_checks = check_row_sets(conn, family, factor_dir)
        value_checks = check_values(conn, family, factor_dir)
        forward_checks = check_forward_look(conn, family, dates, factor_dir)
        direction_checks = check_direction(conn, family, period[0], period[1], factor_dir)
        checks.extend(row_checks + value_checks + forward_checks + direction_checks)

        worst_forward = max(
            (row["value"] for row in forward_checks if row["kind"] == "forward_look"), default=0.0
        )
        null_mismatch = sum(
            row["value"] for row in forward_checks if row["kind"] == "forward_look_null"
        )
        ic_frame = {row["name"]: row for row in direction_checks}
        aligned = sum(
            1 for row in direction_checks
            if "一致" in row["detail"] and "相反" not in row["detail"]
        )
        summary.append({
            "family": family,
            "years": f"{years[0]}~{years[-1]}",
            "samples": len(dates),
            "worst_forward": worst_forward,
            "null_mismatch": null_mismatch,
            "row_diff": sum(row["value"] for row in row_checks),
            "non_finite": sum(row["value"] for row in value_checks),
            "aligned": aligned,
            "count": len(ic_frame),
            "ic_period": f"{period[0]}~{period[1]}",
        })
        logger.info(
            "%s 自检完成：前视最大偏差 %.2e，行集差异 %d，方向一致 %d/%d",
            family, worst_forward, sum(row["value"] for row in row_checks), aligned, len(ic_frame),
        )

    if write_catalog:
        with catalog.connect(catalog_file) as cat:
            catalog.ensure_catalog(cat)
            catalog.refresh_checks(cat, checks)

    return {"checks": checks, "summary": summary}


def print_report(result: dict) -> None:
    """把自检结果打印成表。

    参数:
        result: ``run_checks`` 的返回值。
    """
    print("\n=== 因子自检汇总 ===")
    print(
        f"{'族':<14}{'年份':<12}{'抽点':>4}{'行集差异':>10}{'非有限':>8}"
        f"{'前视偏差':>12}{'空值不一致':>12}{'方向一致':>10}"
    )
    for row in result["summary"]:
        print(
            f"{row['family']:<14}{row['years']:<12}{row['samples']:>4}{row['row_diff']:>10.0f}"
            f"{row['non_finite']:>8.0f}{row['worst_forward']:>12.2e}{row['null_mismatch']:>12.0f}"
            f"{row['aligned']:>7}/{row['count']:<3}"
        )
    print("\n=== 方向与 IC 明细 ===")
    for row in result["checks"]:
        if row["kind"] != "direction":
            continue
        print(f"  {row['name']:<18}{row['detail']}")
    issues = [row for row in result["checks"] if row["kind"] == "row_set" and row["value"] > 0]
    if issues:
        print("\n行集不一致:", [row["name"] for row in issues])


def main(argv=None) -> int:
    """自检命令行入口。

    参数:
        argv: 命令行参数，默认取 sys.argv。
    """
    parser = argparse.ArgumentParser(description="因子自检：行集、取值、前视偏差与方向")
    parser.add_argument("--families", default="", help="逗号分隔的因子族；留空表示全部")
    parser.add_argument("--samples", type=int, default=12, help="前视检查抽样日期数")
    parser.add_argument("--ic-start", default=None, help="IC 统计起始日 YYYY-MM-DD")
    parser.add_argument("--ic-end", default=None, help="IC 统计截止日 YYYY-MM-DD")
    parser.add_argument("--no-catalog", action="store_true", help="只打印，不写目录库")
    args = parser.parse_args(argv)

    families = [item.strip() for item in args.families.split(",") if item.strip()]
    result = run_checks(
        families or None, samples=args.samples, ic_start=args.ic_start, ic_end=args.ic_end,
        write_catalog=not args.no_catalog,
    )
    print_report(result)

    failed = [row for row in result["checks"] if row["kind"] == "row_set" and row["value"] > 0]
    if failed:
        raise FactorCheckError(f"行集检查未通过：{[row['name'] for row in failed]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
