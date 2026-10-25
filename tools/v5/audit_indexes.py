"""指数相关数据的只读自检。

用法：
    .venv/bin/python tools/v5/audit_indexes.py

检查项：指数日线与交易日历的对齐情况、停编指数清单、成分权重的期数与归一命中率。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb  # noqa: E402

from scripts.config import DB_FILE  # noqa: E402

# 指数日线的有效区间与交易日历一致
DATA_START = "2002-01-04"
DATA_END = "2026-09-28"


def run() -> int:
    """跑完整套自检，返回进程退出码。"""
    conn = duckdb.connect(str(DB_FILE), read_only=True)
    failures: list[str] = []
    try:
        print("=== 指数日线 ===")
        print(conn.execute(
            "SELECT count(DISTINCT index_code) AS indexes, count(*) AS n,"
            " min(trade_date) AS d0, max(trade_date) AS d1 FROM md.index_daily"
        ).df().to_string(index=False))

        # 每个指数按自己的首末日期比对交易日历：历史短的（上市晚）不算问题，
        # 中间有空洞才是问题
        gaps = conn.execute("""
            WITH span AS (
              SELECT index_code, count(*) AS n, min(trade_date) AS d0, max(trade_date) AS d1
              FROM md.index_daily GROUP BY 1
            )
            SELECT s.index_code, s.n AS rows, s.d0, s.d1,
                   (SELECT count(*) FROM md.calendar c
                     WHERE c.is_open = 1 AND c.trade_date BETWEEN s.d0 AND s.d1) AS expected
            FROM span s
            WHERE s.n < 0.99 * (SELECT count(*) FROM md.calendar c
                                 WHERE c.is_open = 1 AND c.trade_date BETWEEN s.d0 AND s.d1)
            ORDER BY 1
        """).df()
        print(f"自身区间内有空洞的指数：{len(gaps)} 个")
        if not gaps.empty:
            print(gaps.to_string(index=False))
            failures.append("有指数的日线在自身区间内存在空洞")

        stopped = conn.execute(
            "SELECT index_code, min(trade_date) AS d0, max(trade_date) AS d1 FROM md.index_daily"
            " GROUP BY 1 HAVING max(trade_date) < ? ORDER BY 3", [DATA_END]
        ).df()
        print(f"已停编的指数（{len(stopped)} 个）：")
        print(stopped.to_string(index=False) if not stopped.empty else "  无")

        print("\n=== 申万行业日线 ===")
        print(conn.execute(
            "SELECT count(DISTINCT index_code) AS indexes, count(*) AS n,"
            " min(trade_date) AS d0, max(trade_date) AS d1 FROM md.sw_index_daily"
        ).df().to_string(index=False))

        print("\n=== 成分权重 ===")
        weights = conn.execute("""
            SELECT index_code, any_value(index_name) AS idx_name,
                   count(DISTINCT trade_date) AS periods,
                   min(trade_date) AS d0, max(trade_date) AS d1
            FROM ref.index_weight GROUP BY 1 ORDER BY 1
        """).df()
        print(weights.to_string(index=False))

        bad = []
        for row in weights.to_dict("records"):
            first, last = row["d0"], row["d1"]
            span = (last.year - first.year) * 12 + (last.month - first.month) + 1
            if row["periods"] / span < 0.9:
                bad.append(f"{row['index_code']} 覆盖 {row['periods']}/{span} 个月")
        if bad:
            failures.append("成分权重覆盖不足：" + "；".join(bad))

        hit, total = conn.execute("""
            SELECT count(*) FILTER (WHERE EXISTS (
                       SELECT 1 FROM md.daily_bar b WHERE b.stock_code = w.stock_code)), count(*)
            FROM ref.index_weight w
        """).fetchone()
        ratio = hit / max(total, 1)
        print(f"成分代码归一后 join 日线表命中 {hit:,}/{total:,} = {ratio:.2%}")
        if ratio < 0.999:
            failures.append(f"成分代码命中率只有 {ratio:.2%}")

        print("\n=== 指数目录 ===")
        print(conn.execute(
            "SELECT market, count(*) AS n FROM ref.index_basic GROUP BY 1 ORDER BY 2 DESC"
        ).df().to_string(index=False))
    finally:
        conn.close()

    if failures:
        print("\n自检未通过：")
        for item in failures:
            print("  -", item)
        return 1
    print("\n自检通过")
    return 0


if __name__ == "__main__":
    sys.exit(run())
