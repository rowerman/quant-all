"""只读诊断：把 manifest 里非零的校验指标拆开看，判断是口径问题还是数据问题。

用法：
    .venv/bin/python tools/v5/audit_checks.py
"""

import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAILY_GLOB = str(ROOT / "clean" / "daily" / "year=*" / "part.parquet")

conn = duckdb.connect()
conn.execute(f"CREATE VIEW daily AS SELECT * FROM read_parquet('{DAILY_GLOB}')")

print("== 异常表按类型与年份 ==")
print(
    conn.execute(
        f"""
        SELECT kind, strftime(CAST(trade_date AS DATE), '%Y') AS year, count(*) AS rows
        FROM read_parquet('{ROOT / "clean" / "anomalies.parquet"}')
        GROUP BY 1, 2 ORDER BY rows DESC LIMIT 20
        """
    ).df().to_string(index=False)
)

print("\n== A3 收益口径不一致：按年份 ==")
print(
    conn.execute(
        """
        WITH t AS (
          SELECT stock_code, trade_date, ret, close * factor AS adj,
                 lag(close * factor) OVER (PARTITION BY stock_code ORDER BY trade_date) AS prev_adj
          FROM daily WHERE factor IS NOT NULL AND close IS NOT NULL
        )
        SELECT strftime(CAST(trade_date AS DATE), '%Y') AS year, count(*) AS mismatch
        FROM t WHERE prev_adj IS NOT NULL AND ret IS NOT NULL
          AND abs(ret - (adj / prev_adj - 1)) > 1e-4
        GROUP BY 1 ORDER BY 1 LIMIT 30
        """
    ).df().to_string(index=False)
)

print("\n== A3 样例（差异最大的 5 行） ==")
print(
    conn.execute(
        """
        WITH t AS (
          SELECT stock_code, trade_date, ret, change, close, factor, is_ex_div,
                 close * factor AS adj,
                 lag(close * factor) OVER (PARTITION BY stock_code ORDER BY trade_date) AS prev_adj
          FROM daily WHERE factor IS NOT NULL AND close IS NOT NULL
        )
        SELECT stock_code, trade_date, ret, change, close, factor, is_ex_div,
               round(adj / prev_adj - 1, 6) AS factor_ret,
               round(abs(ret - (adj / prev_adj - 1)), 6) AS diff
        FROM t WHERE prev_adj IS NOT NULL AND ret IS NOT NULL
          AND abs(ret - (adj / prev_adj - 1)) > 1e-4
        ORDER BY diff DESC LIMIT 5
        """
    ).df().to_string(index=False)
)

print("\n== A4 涨跌停越界：按年份 ==")
print(
    conn.execute(
        """
        SELECT strftime(CAST(trade_date AS DATE), '%Y') AS year, count(*) AS rows
        FROM daily
        WHERE is_price_limit_applicable AND close IS NOT NULL
          AND (close > limit_up + 0.011 OR close < limit_down - 0.011)
        GROUP BY 1 ORDER BY 1 LIMIT 30
        """
    ).df().to_string(index=False)
)

print("\n== A4 越界样例（前 5 行） ==")
print(
    conn.execute(
        """
        SELECT stock_code, trade_date, close, limit_up, limit_down, limit_ratio,
               is_st, is_delisting, pre_close_source
        FROM daily
        WHERE is_price_limit_applicable AND close IS NOT NULL
          AND (close > limit_up + 0.011 OR close < limit_down - 0.011)
        ORDER BY trade_date LIMIT 5
        """
    ).df().to_string(index=False)
)

print("\n== 涨停家数占比最高 / 最低的交易日 ==")
print(
    conn.execute(
        """
        WITH s AS (
          SELECT trade_date,
                 count(*) FILTER (WHERE status = 'trading') AS n,
                 count(*) FILTER (WHERE is_limit_up) AS up
          FROM daily GROUP BY trade_date
        )
        SELECT trade_date, n, up, round(up * 1.0 / n, 4) AS ratio FROM (
          (SELECT * FROM s ORDER BY up * 1.0 / n DESC LIMIT 3)
          UNION ALL
          (SELECT * FROM s ORDER BY up * 1.0 / n ASC LIMIT 3)
        ) ORDER BY ratio DESC
        """
    ).df().to_string(index=False)
)

print("\n== 因子跳变但未识别为除权日：按年份 ==")
print(
    conn.execute(
        f"""
        SELECT strftime(CAST(trade_date AS DATE), '%Y') AS year, count(*) AS rows
        FROM read_parquet('{ROOT / "clean" / "anomalies.parquet"}')
        WHERE kind = 'factor_jump_without_ex_div'
        GROUP BY 1 ORDER BY 1 LIMIT 30
        """
    ).df().to_string(index=False)
)

conn.close()

# 规则/官方涨停价一致率由清洗阶段在分年统计里给出（data/clean/daily 只保留优先口径，无法反算）
print("\n== 各年规则/官方涨停价一致率（取 data/clean/manifest.json 的 year_stats） ==")
manifest = json.loads((ROOT / "clean" / "manifest.json").read_text(encoding="utf-8"))
rows = [
    {
        "year": item["year"],
        "compared": item["rule_consistency"]["compared"],
        "match_ratio": item["rule_consistency"]["match_ratio"],
    }
    for item in manifest["year_stats"]
]
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 校验值 vs 异常表条数（应一一相等） ==")
anomaly = pd.read_parquet(ROOT / "clean" / "anomalies.parquet")
kind_counts = anomaly["kind"].value_counts().to_dict()
pairs = [
    ("A3_mismatch_rows", "return_mismatch"),
    ("A2_ex_without_jump", "ex_div_without_factor_jump"),
    ("A2_jump_without_ex_div", "factor_jump_without_ex_div"),
    ("A4_over_limit_rows", "limit_out_of_range"),
]
for check_key, kind in pairs:
    check_value = manifest["checks"].get(check_key)
    kind_value = kind_counts.get(kind)
    flag = "一致" if check_value == kind_value else "**不一致**"
    print(f"{check_key}: {check_value} | {kind}: {kind_value} -> {flag}")
