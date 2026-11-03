"""数据库物理结构定义。

清洗产物、建库脚本与读取层共用这里的列定义，避免两边各写一份导致漂移。
"""

from __future__ import annotations

DB_SCHEMAS = ("md", "ref", "fin", "evt", "meta")

KEY_COLUMNS = ["stock_code", "trade_date"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
MARKET_COLUMNS = ["volume_share", "amount_cny", "vwap"]
ADJUST_COLUMNS = ["factor", "change", "pre_close", "ret"]
LIFECYCLE_COLUMNS = [
    "status",
    "is_suspended",
    "is_resume",
    "is_ex_div",
    "is_st",
    "is_delisting",
    "list_date",
    "delist_date",
]
VALUATION_COLUMNS = [
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dv_ttm",
    "total_mv",
    "circ_mv",
    "turnover_rate",
]
LIMIT_COLUMNS = [
    "limit_ratio",
    "limit_up",
    "limit_down",
    "is_limit_up",
    "is_limit_down",
    "touched_limit_up",
    "touched_limit_down",
    "is_one_word_board",
    "is_price_limit_applicable",
]
TRADEABLE_COLUMNS = [
    "can_buy_close",
    "can_sell_close",
    "can_buy_open_next",
    "can_sell_open_next",
]
AUDIT_COLUMNS = ["pre_close_source", "st_source"]

DAILY_COLUMNS = (
    KEY_COLUMNS
    + PRICE_COLUMNS
    + MARKET_COLUMNS
    + ADJUST_COLUMNS
    + LIFECYCLE_COLUMNS
    + VALUATION_COLUMNS
    + LIMIT_COLUMNS
    + TRADEABLE_COLUMNS
    + AUDIT_COLUMNS
)

# 三张同构日线表：只有四价与 vwap 的口径不同
DAILY_TABLES = {
    "raw": "md.daily_bar",
    "hfq": "md.daily_hfq",
    "qfq": "md.daily_qfq",
}

ADJUST_MODES = tuple(DAILY_TABLES)

# 复权口径只改这几列，其余标记列三张表完全一致
ADJUSTED_VALUE_COLUMNS = PRICE_COLUMNS + ["vwap"]

# 未来收益标签（md.fwd_return）的预测周期，单位是交易日
LABEL_HORIZONS = (1, 3, 5, 10, 15, 20)
# 标签的成交通道：close = t 日收盘买入、t+N 日收盘卖出；
# open = t+1 日开盘买入、t+1+N 日开盘卖出（两者持有交易日数一致）
LABEL_PRICES = ("close", "open")


def label_columns(price: str, horizons=LABEL_HORIZONS) -> list[str]:
    """返回某个成交通道下标签列的名字。

    参数:
        price: close / open。
        horizons: 预测周期列表。
    """
    return [f"fwd_ret_{price}_{n}" for n in horizons]
