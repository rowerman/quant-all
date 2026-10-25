"""只读诊断：查清 T 前缀历史遗留代码（如 T00018.SH / T600018.SH）的归属。

用法：
    .venv/bin/python tools/v5/probe_legacy_code.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TUSHARE = ROOT / "data" / "source" / "tushare"
RAW = TUSHARE / "raw"

adj = pd.concat(
    [pd.read_parquet(p) for p in sorted((RAW / "adj_factor").glob("year=*/month=*.parquet"))],
    ignore_index=True,
)
adj["ts_code"] = adj["ts_code"].astype(str)
adj["trade_date"] = adj["trade_date"].astype(str)

legacy = adj[adj["ts_code"] == "T00018.SH"]
print(f"T00018.SH 行数 {len(legacy)}，日期 {legacy['trade_date'].min()} ~ {legacy['trade_date'].max()}")
print(legacy.assign(year=legacy["trade_date"].str[:4]).groupby("year")["trade_date"].count().to_string())

current = adj[adj["ts_code"] == "600018.SH"]
print(f"\n600018.SH 行数 {len(current)}，日期 {current['trade_date'].min()} ~ {current['trade_date'].max()}")
print("两者重叠交易日数", len(set(legacy["trade_date"]) & set(current["trade_date"])))

basic = pd.read_parquet(TUSHARE / "stock_basic_all.parquet")
hit = basic[basic["ts_code"].astype(str).str.contains("00018", na=False)]
print("\nstock_basic 里含 00018 的记录：")
print(hit[["ts_code", "name", "list_date", "delist_date", "list_status"]].to_string())

for year in (2000, 2006):
    day = pd.concat(
        [pd.read_parquet(p) for p in sorted((RAW / "daily").glob(f"year={year}/month=*.parquet"))],
        ignore_index=True,
    )
    day["ts_code"] = day["ts_code"].astype(str)
    hit = day[day["ts_code"].str.contains("00018", na=False)]
    print(f"\n== daily {year}：含 00018 的行 {len(hit)}")
    if not hit.empty:
        print(hit.groupby("ts_code")["trade_date"].agg(["min", "max", "count"]).to_string())
