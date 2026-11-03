"""只读诊断：扫描原始分片与参考表里无法归一化的股票代码。

用法：
    .venv/bin/python tools/data_audit/scan_codes.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quant.codes import normalize_code

RAW = ROOT / "data" / "source" / "tushare" / "raw"
REFERENCE = ROOT / "data" / "source" / "tushare"


def scan_reference() -> None:
    """扫描参考表里所有代码列。"""
    for path in sorted(REFERENCE.glob("*.parquet")):
        frame = pd.read_parquet(path)
        columns = [name for name in frame.columns if "code" in name.lower()]
        print(f"== {path.name}：{len(frame)} 行，代码列 {columns}", flush=True)
        for column in columns:
            bad: dict[str, int] = defaultdict(int)
            for value in frame[column].astype(str):
                try:
                    normalize_code(value)
                except ValueError:
                    bad[value] += 1
            if bad:
                print(f"   列 {column} 无法归一化：{dict(bad)}", flush=True)


def scan_raw(api: str) -> None:
    """扫描某个行情接口分片里的代码。"""
    bad_rows: dict[str, int] = defaultdict(int)
    bad_days: dict[str, list[str]] = defaultdict(list)
    total = 0
    for path in sorted((RAW / api).glob("year=*/month=*.parquet")):
        frame = pd.read_parquet(path, columns=["ts_code", "trade_date"])
        if frame.empty:
            continue
        total += len(frame)
        for raw, day in zip(frame["ts_code"].astype(str), frame["trade_date"].astype(str)):
            try:
                normalize_code(raw)
            except ValueError:
                bad_rows[raw] += 1
                bad_days[raw].append(day)
    print(f"== {api}：总行数 {total}，无法归一化的代码 {len(bad_rows)} 个", flush=True)
    for raw in sorted(bad_rows, key=lambda key: -bad_rows[key])[:15]:
        days = sorted(bad_days[raw])
        print(f"   {raw}: {bad_rows[raw]} 行，{days[0]} ~ {days[-1]}", flush=True)


for name in ("daily", "adj_factor", "stk_limit", "daily_basic"):
    scan_raw(name)
scan_reference()
