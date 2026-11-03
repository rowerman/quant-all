"""清洗输入层：从 Tushare 落地分片读取原始数据。

分片布局：``data/source/tushare/raw/<接口>/year=YYYY/month=MM.parquet``。
所有接口都按交易日批量拉取，因此天然是「股票 × 交易日」的长表。
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

from quant.codes import normalize_code, normalize_codes as shared_normalize_codes
from scripts.config import EXTRA_DIR
from scripts.errors import DataReadError
from scripts.logging_conf import get_logger

logger = get_logger("clean.sources")

RAW_DIR = EXTRA_DIR / "tushare" / "raw"
REFERENCE_DIR = EXTRA_DIR / "tushare"

# Tushare daily 的单位换算：vol 是手、amount 是千元
VOLUME_TO_SHARE = 100.0
AMOUNT_TO_CNY = 1000.0


def _to_iso(value: str) -> str:
    """把 20240102 形式的日期转成 2024-01-02。

    参数:
        value: 原始日期字符串。
    """
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text[:10]


def _is_date_like(value: str) -> bool:
    """判断字符串是否是一个有效日期（排除 nan / None / 空串）。

    参数:
        value: 待判断的字符串。
    """
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "nat"):
        return False
    return len(text) >= 8 and text[:8].isdigit()


def normalize_codes(values, where: str) -> list[str | None]:
    """批量把原始代码归一成项目口径，认不出的历史遗留代码记 None。

    参数:
        values: 原始代码列。
        where: 调用来源，用于告警定位。

    返回:
        与输入等长的代码列表，无法识别的元素为 None。
    """
    return shared_normalize_codes(values, where, logger)


def _shards(api_name: str, year: int | None = None) -> list[Path]:
    """列出某个接口的分片文件。

    参数:
        api_name: 接口名。
        year: 只取某一年；None 表示全部。
    """
    directory = RAW_DIR / api_name
    if not directory.exists():
        return []
    if year is None:
        return sorted(directory.glob("year=*/month=*.parquet"))
    return sorted((directory / f"year={year}").glob("month=*.parquet"))


def load_api(api_name: str, year: int | None = None) -> pd.DataFrame:
    """读取某个接口的分片数据。

    参数:
        api_name: 接口名。
        year: 只读某一年；None 表示全部。
    """
    parts = _shards(api_name, year)
    if not parts:
        raise DataReadError(f"接口 {api_name} 在 year={year} 没有分片数据")
    frame = pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True)
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def available_years(api_name: str = "daily") -> list[int]:
    """列出某个接口已落盘的年份。

    参数:
        api_name: 接口名。
    """
    directory = RAW_DIR / api_name
    if not directory.exists():
        return []
    return sorted(int(path.name.split("=")[1]) for path in directory.glob("year=*"))


def latest_data_date() -> str:
    """扫描全部 daily 分片，取最新的交易日作为有效数据的截止日。"""
    parts = _shards("daily")
    if not parts:
        raise DataReadError("daily 没有任何分片，无法确定数据边界")
    newest = ""
    for part in parts:
        frame = pd.read_parquet(part, columns=["trade_date"])
        if frame.empty:
            continue
        value = str(frame["trade_date"].astype(str).max())
        if value > newest:
            newest = value
    if not newest:
        raise DataReadError("daily 分片都是空的")
    return _to_iso(newest)


@functools.lru_cache(maxsize=1)
def trade_days() -> list[str]:
    """从 trade_cal 取全部交易日。"""
    path = REFERENCE_DIR / "trade_cal.parquet"
    if not path.exists():
        raise DataReadError("缺少 trade_cal.parquet")
    calendar = pd.read_parquet(path)
    opened = calendar[calendar["is_open"].astype(int) == 1]
    return sorted(_to_iso(value) for value in opened["cal_date"].astype(str).tolist())


@functools.lru_cache(maxsize=1)
def stock_reference() -> pd.DataFrame:
    """读取股票基础表，附带归一后的代码与板块。"""
    path = REFERENCE_DIR / "stock_basic_all.parquet"
    if not path.exists():
        raise DataReadError("缺少 stock_basic_all.parquet")
    frame = pd.read_parquet(path)
    frame = frame.drop_duplicates(subset=["ts_code"], keep="first").reset_index(drop=True)
    frame["stock_code"] = [normalize_code(value) for value in frame["ts_code"]]
    # 退市清单里存在 T600018.SH 这类历史代码，归一后会与在市股撞车，按「在市优先」去重
    frame["_rank"] = (frame["list_status"] != "L").astype(int)
    frame = (
        frame.sort_values(["stock_code", "_rank"])
        .drop_duplicates(subset=["stock_code"], keep="first")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )
    frame["list_date"] = [_to_iso(value) for value in frame["list_date"].astype(str)]
    if "delist_date" in frame.columns:
        frame["delist_date"] = [
            _to_iso(value) if _is_date_like(value) else None
            for value in frame["delist_date"].astype(str)
        ]
    else:
        frame["delist_date"] = None
    return frame


@functools.lru_cache(maxsize=1)
def namechange_intervals() -> pd.DataFrame:
    """把 namechange 规整成代码-名称区间表。"""
    path = REFERENCE_DIR / "namechange_all.parquet"
    if not path.exists():
        raise DataReadError("缺少 namechange_all.parquet")
    frame = pd.read_parquet(path).copy()
    frame["stock_code"] = [normalize_code(value) for value in frame["ts_code"]]
    frame["start_date"] = pd.to_datetime(frame["start_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    end = pd.to_datetime(frame["end_date"], format="%Y%m%d", errors="coerce")
    frame["end_date"] = end.dt.strftime("%Y-%m-%d").fillna(trade_days()[-1])
    return frame[["stock_code", "start_date", "end_date", "name", "change_reason"]]


@functools.lru_cache(maxsize=1)
def suspend_records() -> pd.DataFrame:
    """读取官方停牌记录。"""
    path = REFERENCE_DIR / "suspend_d.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["stock_code", "trade_date"])
    frame = pd.read_parquet(path)
    frame["stock_code"] = normalize_codes(frame["ts_code"], "停牌记录 suspend_d")
    frame = frame[frame["stock_code"].notna()]
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame[["stock_code", "trade_date"]].drop_duplicates()


def year_trade_days(year: int, first: str, last: str) -> list[str]:
    """取某一年落在有效区间内的交易日。

    参数:
        year: 年份。
        first: 有效区间起始日。
        last: 有效区间结束日。
    """
    prefix = f"{year}-"
    return [day for day in trade_days() if day.startswith(prefix) and first <= day <= last]


def codes_alive(year: int, first: str, last: str) -> list[str]:
    """取某一年内处于上市状态（或当年将上市）的股票代码。

    参数:
        year: 年份。
        first: 有效区间起始日。
        last: 有效区间结束日。
    """
    reference = stock_reference()
    year_start = f"{year}-01-01"
    year_end = f"{year}-12-31"
    window_start = max(year_start, first)
    window_end = min(year_end, last)
    listed_before = reference["list_date"] <= window_end
    not_delisted = reference["delist_date"].isna() | (reference["delist_date"] >= window_start)
    selected = reference[listed_before & not_delisted]
    return sorted(selected["stock_code"].tolist())


def index_map(codes: list[str], dates: list[str]) -> tuple[dict, dict]:
    """构造代码与日期到下标的映射。

    参数:
        codes: 股票代码列表。
        dates: 交易日列表。
    """
    return {code: position for position, code in enumerate(codes)}, {
        day: position for position, day in enumerate(dates)
    }
