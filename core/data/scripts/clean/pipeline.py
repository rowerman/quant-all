"""清洗流水线：按年构建面板、派生指标并落盘。

之所以按年切分而不是一次性载入全部历史：全 A 从 2000 年起约 3600 万个
「股票 × 交易日」单元格，整块载入会吃掉十几 GB 内存，按年后单年只有百万级。
跨年只传一个状态：上一年的最后收盘价。
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from scripts.clean import adjust, io, lifecycle, limits, manifest, sources, validate
from scripts.config import CLEAN_DIR, DAILY_DIR, EARLIEST_START, ensure_dirs
from scripts.logging_conf import get_logger

logger = get_logger("clean.pipeline")

FLOAT_COLUMNS = (
    "open", "high", "low", "close", "volume_share", "amount_cny", "vwap",
    "factor", "change", "pre_close", "ret",
    "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv", "turnover_rate",
    "limit_ratio", "limit_up", "limit_down",
)
BOOLEAN_COLUMNS = (
    "is_suspended", "is_resume", "is_ex_div", "is_st", "is_delisting",
    "is_limit_up", "is_limit_down", "touched_limit_up", "touched_limit_down",
    "is_one_word_board", "is_price_limit_applicable",
    "can_buy_close", "can_sell_close", "can_buy_open_next", "can_sell_open_next",
)
COLUMN_ORDER = [
    "stock_code", "trade_date",
    "open", "high", "low", "close", "volume_share", "amount_cny", "vwap",
    "factor", "change", "pre_close", "ret",
    "status", "is_suspended", "is_resume", "is_ex_div", "is_st", "is_delisting",
    "list_date", "delist_date",
    "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv", "turnover_rate",
    "limit_ratio", "limit_up", "limit_down",
    "is_limit_up", "is_limit_down", "touched_limit_up", "touched_limit_down",
    "is_one_word_board", "is_price_limit_applicable",
    "can_buy_close", "can_sell_close", "can_buy_open_next", "can_sell_open_next",
    "pre_close_source", "st_source",
]
STATS_FILE = CLEAN_DIR / "build_stats.json"


def _flat(array: np.ndarray) -> np.ndarray:
    """把二维面板摊平成「按交易日优先」的一维数组。

    参数:
        array: 形状 (股票数, 交易日数) 的面板。
    """
    return np.ascontiguousarray(array.T).ravel()


def process_year(year: int, start: str, last: str, carry_close: dict[str, float]) -> tuple[dict, dict]:
    """处理一个年份并写出分区文件。

    参数:
        year: 年份。
        start: 整体数据起始日。
        last: 整体数据截止日。
        carry_close: 上一年最后交易日的收盘价，按股票代码索引。

    返回:
        (该年统计信息, 新的 carry_close)。
    """
    dates = sources.year_trade_days(year, start, last)
    if not dates:
        return {}, carry_close
    codes = sources.codes_alive(year, start, last)
    panel = io.build_year_panel(year, dates, codes)
    reference = lifecycle.build_reference(codes)

    close = panel.arrays["close"]
    open_price = panel.arrays["open"]
    high = panel.arrays["high"]
    low = panel.arrays["low"]
    change = panel.arrays["change"]

    status = lifecycle.build_status(dates, reference, close)
    panel.put("status_code", status)

    is_st, is_delisting, st_sources = io.load_st_matrix(codes, dates)
    panel.put("is_st", is_st)
    panel.put("is_delisting", is_delisting)
    panel.meta["st_sources"] = st_sources

    official_suspend = io.load_suspend_matrix(codes, dates)
    is_suspended = (status == lifecycle.STATUS_SUSPENDED) | official_suspend
    panel.put("is_suspended", is_suspended)
    panel.put("is_resume", lifecycle.build_resume_flags(status))
    panel.put("status_code", np.where(is_suspended & (status == lifecycle.STATUS_TRADING),
                                      lifecycle.STATUS_SUSPENDED, status).astype(np.int8))

    # 昨收取「上一有效交易日收盘价」：停牌日没有价格，除权判定必须跨过它，
    # 否则停牌期间发生除权的个股在复牌首日永远标不出除权
    seeded = close.copy()
    if carry_close:
        for row, code in enumerate(codes):
            value = carry_close.get(code)
            if value is not None and not np.isnan(value) and np.isnan(seeded[row, 0]):
                seeded[row, 0] = value
    prev_close = adjust.shift_down(adjust.forward_fill_close(seeded))
    del seeded
    panel.put("pre_close", adjust.compute_pre_close(close, change))
    panel.put("ret", adjust.compute_return(panel.arrays["pre_close"], change))
    panel.put("is_ex_div", adjust.mark_ex_dividend(close, change, prev_close))
    del prev_close

    with np.errstate(invalid="ignore", divide="ignore"):
        panel.put("vwap", panel.arrays["amount_cny"] / panel.arrays["volume_share"])

    up_ratio, down_ratio, rule_applicable = limits.compute_limit_ratios(
        codes, dates, reference, is_st, is_delisting
    )
    rule_up, rule_down = limits.compute_limit_prices(
        panel.arrays["pre_close"], up_ratio, down_ratio, rule_applicable
    )
    official_up = panel.arrays["limit_up_official"]
    official_down = panel.arrays["limit_down_official"]
    limit_up, limit_down, has_official = limits.prefer_official(
        official_up, official_down, rule_up, rule_down
    )
    applicable = (has_official | rule_applicable) & ~is_suspended

    limit_up = np.where(applicable, limit_up, np.nan)
    limit_down = np.where(applicable, limit_down, np.nan)
    panel.put("limit_ratio", np.where(applicable, up_ratio, np.nan))
    panel.put("limit_up", limit_up)
    panel.put("limit_down", limit_down)
    panel.put("is_price_limit_applicable", applicable)
    panel.meta["rule_consistency"] = limits.official_consistency(official_up, rule_up)

    for name, values in limits.mark_limits(
        close, high, low, open_price, limit_up, limit_down, applicable
    ).items():
        panel.put(name, values)
    for name, values in limits.mark_tradeable(
        panel.arrays["status_code"], close, open_price, limit_up, limit_down, applicable
    ).items():
        panel.put(name, values)

    frame = _to_frame(panel, codes, dates, reference)
    directory = DAILY_DIR / f"year={year}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "part.parquet", index=False)

    # 跨年传递的也是最后一个有效交易日的收盘价（当天可能停牌）
    last_valid = adjust.last_valid_close(close)
    new_carry = {code: last_valid[row] for row, code in enumerate(codes)}
    stats = {
        "year": year,
        "codes": len(codes),
        "days": len(dates),
        "rows": int(len(frame)),
        "rule_consistency": panel.meta["rule_consistency"],
    }
    logger.info(
        "完成 %d：股票 %d 只，交易日 %d 天，落盘 %d 行，规则/官方涨停价一致率 %s",
        year, len(codes), len(dates), len(frame),
        panel.meta["rule_consistency"].get("match_ratio"),
    )
    return stats, new_carry


def _to_frame(panel: io.Panel, codes: list[str], dates: list[str], reference: pd.DataFrame) -> pd.DataFrame:
    """把面板转成落盘长表，只保留上市期内的行。

    参数:
        panel: 面板。
        codes: 股票代码列表。
        dates: 交易日列表。
        reference: 股票基础表，行序与 codes 一致。
    """
    status = panel.arrays["status_code"]
    keep = (status == lifecycle.STATUS_TRADING) | (status == lifecycle.STATUS_SUSPENDED)
    keep_flat = np.ascontiguousarray(keep.T).ravel()

    data = {
        "stock_code": np.tile(np.array(codes, dtype=object), len(dates)),
        "trade_date": np.repeat(np.array(dates), len(codes)),
    }
    for name in FLOAT_COLUMNS:
        data[name] = _flat(panel.arrays[name])
    data["status"] = lifecycle.STATUS_TEXT[_flat(status).astype(np.int8)]
    for name in BOOLEAN_COLUMNS:
        data[name] = _flat(panel.arrays[name])
    list_dates = reference["list_date"].tolist()
    delist_dates = reference["delist_date"].tolist()
    data["list_date"] = np.tile(np.array(list_dates, dtype=object), len(dates))
    data["delist_date"] = np.tile(np.array(delist_dates, dtype=object), len(dates))
    official_pre = _flat(panel.arrays["pre_close_tushare"])
    close_flat = data["close"]
    data["pre_close_source"] = np.where(
        np.isnan(close_flat),
        "none",
        np.where(np.isnan(official_pre), "derived", "tushare"),
    )
    data["st_source"] = np.tile(np.array(panel.meta["st_sources"], dtype=object), len(dates))

    frame = pd.DataFrame({name: values[keep_flat] for name, values in data.items()})
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype("category")
    for name in BOOLEAN_COLUMNS:
        frame[name] = frame[name].astype(bool)
    return frame[COLUMN_ORDER]


def run(start: str = EARLIEST_START, end: str | None = None) -> dict:
    """执行整条清洗流水线。

    参数:
        start: 起始交易日。
        end: 截止交易日，None 表示取数据里最新的交易日。
    """
    ensure_dirs()
    last = end or sources.latest_data_date()
    years = io.years_in_range(start, last)
    logger.info("清洗区间 %s ~ %s，共 %d 个年份", start, last, len(years))
    started = time.time()
    stats = []
    carry: dict[str, float] = {}
    for year in years:
        year_stats, carry = process_year(year, start, last, carry)
        if year_stats:
            stats.append(year_stats)
    payload = {
        "start": start,
        "end": last,
        "years": stats,
        "total_rows": int(sum(item["rows"] for item in stats)),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    STATS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("清洗完成，总行数 %d，用时 %.1f 秒", payload["total_rows"], payload["elapsed_seconds"])
    payload["manifest"] = finalize(start, last, stats)
    return payload


def finalize(start: str, end: str, year_stats: list[dict]) -> dict:
    """跑校验、落盘衍生表并写 manifest。

    参数:
        start: 数据起始日。
        end: 数据截止日。
        year_stats: 分年统计。
    """
    checks, anomalies, segments = validate.run_all()
    anomalies.to_parquet(CLEAN_DIR / "anomalies.parquet", index=False)
    segments.to_parquet(CLEAN_DIR / "suspend_segments.parquet", index=False)

    calendar = [day for day in sources.trade_days() if start <= day <= end]
    pd.DataFrame({"trade_date": calendar, "is_open": 1}).to_parquet(
        CLEAN_DIR / "calendar.parquet", index=False
    )

    conn = validate.connect()
    st_sources = conn.execute(
        "SELECT st_source, count(*) AS rows FROM daily GROUP BY st_source ORDER BY rows DESC"
    ).df()
    basic = sources.stock_reference()
    conn.close()

    reference_stats = {
        "reference_stocks": int(len(basic)),
        "stocks_with_delist_date": int(basic["delist_date"].notna().sum()),
        "listed": int((basic["list_status"] == "L").sum()),
        "delisted": int((basic["list_status"] == "D").sum()),
    }
    notes = [
        "全 A 日线自 2000-01-04 起；上市前与退市后的交易日不铺行，区间由 ref.stock_basic 的 list_date/delist_date 表达。",
        "涨跌停价优先取 Tushare stk_limit 官方值，缺失时按板块+日期+ST+退市整理期规则推算。",
        "退市整理期区间由 namechange 的证券简称推出，首日不设涨跌幅、其余 10%。",
        "停牌标记来自 Tushare suspend_d 官方记录，并补上「上市期内无行情」的交易日。",
        "Tushare daily 的 vol 单位是手、amount 单位是千元，落盘已换算为股与元。",
        "总市值/流通市值单位是万元，换手率与股息率单位是百分比，与 Tushare 原始口径一致。",
    ]
    payload = manifest.build_manifest(
        start=start,
        end=end,
        n_stocks=manifest.count_stocks(),
        n_rows=manifest.count_rows(),
        checks=checks,
        source_stats={"reference": reference_stats, "st_source_counts": st_sources.to_dict("records")},
        notes=notes,
        year_stats=year_stats,
    )
    manifest.write_manifest(payload)
    return payload


def main() -> int:
    """命令行入口。"""
    payload = run()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
