"""只读检查公司行动账本与原始分红记录的覆盖情况。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "db" / "quant.duckdb"


def audit(db_path: Path) -> dict:
    """返回账本规模、来源分布和关键样例。"""
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        summary = conn.execute(
            """
            SELECT count(*) AS events,
                   count(DISTINCT stock_code) AS stocks,
                   count(DISTINCT ex_date) AS ex_dates,
                   count(*) FILTER (WHERE source = 'gbbq') AS gbbq_events,
                   count(*) FILTER (WHERE source = 'dividend') AS dividend_events,
                   count(*) FILTER (WHERE source = 'share_count') AS share_count_events,
                   count(*) FILTER (WHERE rights_ratio > 0) AS rights_events,
                   count(DISTINCT stock_code) FILTER (WHERE rights_ratio > 0) AS rights_stocks,
                   count(*) FILTER (WHERE action_type = 'consolidation') AS consolidations,
                   min(ex_date) AS min_ex_date,
                   max(ex_date) AS max_ex_date
            FROM ref.corporate_action
            """
        ).df().to_dict("records")[0]
        duplicates = conn.execute(
            """
            SELECT count(*) FROM (
              SELECT stock_code, ex_date, count(*) AS n
              FROM ref.corporate_action
              GROUP BY 1, 2 HAVING count(*) > 1
            )
            """
        ).fetchone()[0]
        invalid_rights = conn.execute(
            """
            SELECT count(*) FROM ref.corporate_action
            WHERE rights_ratio < 0 OR (rights_ratio > 0 AND (rights_price IS NULL OR rights_price <= 0))
            """
        ).fetchone()[0]
        samples = {}
        for code, day in (
            ("SZ002594", "2025-07-29"),
            ("SZ000001", "2024-06-14"),
            ("SZ300487", "2017-05-10"),
            ("SH601377", "2016-01-07"),
            ("SH600381", "2014-09-30"),
        ):
            rows = conn.execute(
                "SELECT * FROM ref.corporate_action "
                "WHERE stock_code = ? AND ex_date = CAST(? AS DATE)",
                [code, day],
            ).df().to_dict("records")
            samples[f"{code}:{day}"] = rows
        manifest = conn.execute("SELECT * FROM meta.corporate_action_manifest").df()
        return {
            "summary": summary,
            "duplicate_keys": int(duplicates),
            "invalid_rights": int(invalid_rights),
            "samples": samples,
            "manifest": manifest.to_dict("records"),
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="检查公司行动账本")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(Path(args.db))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2, default=str))
        print(f"重复键：{result['duplicate_keys']}")
        print(f"非法配股条款：{result['invalid_rights']}")
        for key, rows in result["samples"].items():
            print(f"{key}: {rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
