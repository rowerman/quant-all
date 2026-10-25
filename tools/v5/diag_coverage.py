"""只读诊断：各接口分片的交易日覆盖与起止日期。

用法：
    .venv/bin/python tools/v5/diag_coverage.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "source" / "tushare" / "raw"

for api in ("daily", "adj_factor", "stk_limit", "daily_basic"):
    per_year: dict[str, int] = {}
    shards = sorted((RAW / api).glob("year=*/month=*.parquet"))
    days: set[str] = set()
    for path in shards:
        frame = pd.read_parquet(path, columns=["trade_date"])
        if frame.empty:
            continue
        values = frame["trade_date"].astype(str)
        days.update(values.unique().tolist())
        year = path.parent.name.split("=")[1]
        per_year[year] = per_year.get(year, 0) + int(values.nunique())
    ordered = sorted(days)
    print(f"== {api}：分片 {len(shards)} 个，非空交易日 {len(ordered)} 天")
    if ordered:
        print(f"   最早 {ordered[0]}，最新 {ordered[-1]}")
        head = {y: per_year[y] for y in sorted(per_year)[:8]}
        tail = {y: per_year[y] for y in sorted(per_year)[-4:]}
        print(f"   前 8 年天数 {head}")
        print(f"   后 4 年天数 {tail}")
