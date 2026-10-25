"""只读审计申万历史行业覆盖和分类版本切换。

用法：
    .venv/bin/python tools/v5/audit_sw_industry.py
    .venv/bin/python tools/v5/audit_sw_industry.py --start 2016-01-01 --end 2026-10-15
    .venv/bin/python tools/v5/audit_sw_industry.py --json

2016~2026 的硬验收区间内，只要存在有行情但无有效行业的股票日，退出码就是 1。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "db" / "quant.duckdb"


def _parser() -> argparse.ArgumentParser:
    """构造参数解析器。"""
    parser = argparse.ArgumentParser(description="审计申万历史行业覆盖")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-10-15")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    return parser


def audit(db_path: Path, start: str, end: str) -> dict:
    """返回覆盖统计和硬验收结论。"""
    conn = duckdb.connect(str(db_path), read_only=True)
    conn.execute("SET memory_limit='2GB'")
    conn.execute("SET threads=2")
    columns = {
        row[0] for row in conn.execute("DESCRIBE ref.sw_industry_member").fetchall()
    }
    required = {
        "classification", "carry_forward", "in_date", "out_date", "stock_code", "source_api"
    }
    missing_columns = required - columns
    if missing_columns:
        raise SystemExit(f"ref.sw_industry_member 缺少字段：{sorted(missing_columns)}")

    manifest = conn.execute("SELECT * FROM meta.sw_industry_manifest LIMIT 1").df()
    manifest_row = manifest.iloc[0].to_dict() if not manifest.empty else {}
    segments = conn.execute(
        "SELECT classification, count(*) AS segments, count(DISTINCT stock_code) AS stocks, "
        "sum(CASE WHEN carry_forward THEN 1 ELSE 0 END) AS carry_segments, "
        "sum(CASE WHEN source_api LIKE 'backfill_earliest_%' THEN 1 ELSE 0 END) AS backfill_segments "
        "FROM ref.sw_industry_member GROUP BY 1 ORDER BY 1"
    ).fetchall()
    coverage = conn.execute(
        f"""
        WITH bars AS (
            SELECT trade_date, stock_code
            FROM md.daily_bar
            WHERE trade_date BETWEEN DATE '{start}' AND DATE '{end}'
        ),
        missing AS (
            SELECT b.trade_date, b.stock_code
            FROM bars b
            WHERE NOT EXISTS (
                SELECT 1 FROM ref.sw_industry_member i
                WHERE i.stock_code = b.stock_code
                  AND i.in_date <= b.trade_date
                  AND (i.out_date IS NULL OR i.out_date >= b.trade_date)
            )
        ),
        classified_stocks AS (
            SELECT DISTINCT stock_code FROM ref.sw_industry_member
        ),
        missing_tagged AS (
            SELECT m.trade_date, m.stock_code,
                   c.stock_code IS NOT NULL AS classified
            FROM missing m
            LEFT JOIN classified_stocks c USING (stock_code)
        ),
        carry AS (
            SELECT b.trade_date, b.stock_code
            FROM bars b
            JOIN ref.sw_industry_member i
              ON i.stock_code = b.stock_code
             AND i.in_date <= b.trade_date
             AND (i.out_date IS NULL OR i.out_date >= b.trade_date)
             AND i.carry_forward
        ),
        bar_counts AS (
            SELECT year(trade_date) AS year, count(*) AS bar_cells
            FROM bars GROUP BY 1
        ),
        missing_counts AS (
            SELECT year(trade_date) AS year,
                   count(*) AS missing_cells,
                   count(DISTINCT stock_code) AS missing_stocks,
                   count(*) FILTER (WHERE classified) AS missing_classified_cells,
                   count(DISTINCT stock_code) FILTER (WHERE classified)
                       AS missing_classified_stocks,
                   count(*) FILTER (WHERE NOT classified) AS unclassified_cells,
                   count(DISTINCT stock_code) FILTER (WHERE NOT classified)
                       AS unclassified_stocks
            FROM missing_tagged GROUP BY 1
        ),
        carry_counts AS (
            SELECT year(trade_date) AS year, count(*) AS carry_forward_cells,
                   count(DISTINCT stock_code) AS carry_forward_stocks
            FROM carry GROUP BY 1
        )
        SELECT bar_counts.year, bar_counts.bar_cells,
               coalesce(missing_counts.missing_cells, 0) AS missing_cells,
               coalesce(missing_counts.missing_stocks, 0) AS missing_stocks,
               coalesce(missing_counts.missing_classified_cells, 0)
                   AS missing_classified_cells,
               coalesce(missing_counts.missing_classified_stocks, 0)
                   AS missing_classified_stocks,
               coalesce(missing_counts.unclassified_cells, 0) AS unclassified_cells,
               coalesce(missing_counts.unclassified_stocks, 0) AS unclassified_stocks,
               coalesce(carry_counts.carry_forward_cells, 0) AS carry_forward_cells,
               coalesce(carry_counts.carry_forward_stocks, 0) AS carry_forward_stocks
        FROM bar_counts
        LEFT JOIN missing_counts USING (year)
        LEFT JOIN carry_counts USING (year)
        ORDER BY 1
        """
    ).fetchall()
    conn.close()

    rows = [
        {
            "year": int(row[0]),
            "bar_cells": int(row[1]),
            "missing_cells": int(row[2]),
            "missing_stocks": int(row[3]),
            "missing_classified_cells": int(row[4]),
            "missing_classified_stocks": int(row[5]),
            "unclassified_cells": int(row[6]),
            "unclassified_stocks": int(row[7]),
            "carry_forward_cells": int(row[8]),
            "carry_forward_stocks": int(row[9]),
        }
        for row in coverage
    ]
    missing_total = sum(row["missing_cells"] for row in rows)
    missing_classified = sum(row["missing_classified_cells"] for row in rows)
    unclassified = sum(row["unclassified_cells"] for row in rows)
    carry_total = sum(row["carry_forward_cells"] for row in rows)
    return {
        "db": str(db_path),
        "start": start,
        "end": end,
        "manifest": {
            key: (None if value is None else str(value))
            for key, value in manifest_row.items()
        },
        "segments": [
            {
                "classification": row[0],
                "segments": int(row[1]),
                "stocks": int(row[2]),
                "carry_forward_segments": int(row[3] or 0),
                "backfill_segments": int(row[4] or 0),
            }
            for row in segments
        ],
        "coverage": rows,
        "missing_cells": missing_total,
        "missing_classified_cells": missing_classified,
        "unclassified_cells": unclassified,
        "carry_forward_cells": carry_total,
        "passed": missing_classified == 0,
    }


def main() -> int:
    """命令行入口。"""
    args = _parser().parse_args()
    result = audit(args.db, args.start, args.end)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"库：{result['db']}")
        print(f"区间：{result['start']} ~ {result['end']}")
        print("分类版本：")
        for row in result["segments"]:
            print(
                f"  {row['classification']}: {row['segments']} 段 / "
                f"{row['stocks']} 只，carry-forward {row['carry_forward_segments']} 段，"
                f"上市日前向补全 {row['backfill_segments']} 段"
            )
        print("年度覆盖：")
        print("  年份  行情格数  缺失格数  已分类缺失  未分类新股  carry格数")
        for row in result["coverage"]:
            print(
                f"  {row['year']}  {row['bar_cells']:>10}  {row['missing_cells']:>8}  "
                f"{row['missing_classified_cells']:>10}  {row['unclassified_cells']:>10}  "
                f"{row['carry_forward_cells']:>9}"
            )
        print(
            f"结论：已分类缺失={result['missing_classified_cells']}，"
            f"未分类新股={result['unclassified_cells']}，"
            f"carry_forward_cells={result['carry_forward_cells']}，"
            f"{'通过' if result['passed'] else '未通过'}"
        )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
