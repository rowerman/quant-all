"""涨跌停价推算、封板标记与可交易性列。

因为本账号拿不到官方涨跌停价（Tushare ``stk_limit`` 无权限），这里完全按规则推算：
基准价用交易所给的 ``close - change``，限幅比例按「板块 + 日期 + ST + 上市阶段」确定，
最后按 A 股惯例四舍五入到 0.01 元。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.config import LIMIT_EPS
from scripts.errors import LimitRuleError
from scripts.logging_conf import get_logger

logger = get_logger("clean.limits")

# 注册制新股「上市前 5 个交易日不设涨跌幅」的生效日期
REGIME_FIVE_DAY = {
    "star": 20190722,
    "chinext": 20200824,
    "main_sh": 20230410,
    "main_sz": 20230410,
    "bse": 0,
}
# 主板/创业板新股上市首日 ±44% / -36% 的生效区间起点
REGIME_FIRST_DAY_44 = 20140101
# 老规则里上市首日不设涨跌幅的板块（2014 年之前）
OLD_NO_LIMIT_BOARDS = ("main_sh", "main_sz", "chinext")

BASE_RATIO = {
    "main_sh": 0.10,
    "main_sz": 0.10,
    "b_sh": 0.10,
    "b_sz": 0.10,
    "other": 0.10,
    "chinext": 0.10,
    "star": 0.20,
    "bse": 0.30,
}
ST_RATIO = 0.05
CHINEXT_20PCT_DATE = 20200824
DELISTING_PERIOD_DAYS = 30


def round_half_up(values: np.ndarray) -> np.ndarray:
    """按 A 股惯例四舍五入到 0.01 元。

    numpy 的 round 是「四舍六入五取偶」，4.045 会变成 4.04，而交易所是 4.05，
    因此在缩放后加一个极小量再向下取整。

    参数:
        values: 待取整的价格数组。

    返回:
        取整到 0.01 的数组，NaN 原样保留。
    """
    scaled = values * 100.0
    rounded = np.floor(scaled + 0.5 + 1e-9)
    result = rounded / 100.0
    return np.where(np.isnan(values), np.nan, result)


def date_to_int(values: list[str]) -> np.ndarray:
    """把 YYYY-MM-DD 日期列表转成 YYYYMMDD 整数。

    参数:
        values: 日期字符串列表。
    """
    return np.array([int(date.replace("-", "")) for date in values], dtype=np.int64)


def _as_date(value) -> str | None:
    """把可能为 NaN 的日期值规整成字符串或 None。

    参数:
        value: 原始日期值。
    """
    if value is None:
        return None
    if isinstance(value, float):
        return None if np.isnan(value) else str(value)[:10]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text[:10]


def board_rows(codes: list[str], reference: pd.DataFrame) -> list[np.ndarray]:
    """返回各板块在股票维度上的行掩码。

    参数:
        codes: 股票代码列表。
        reference: 股票基础信息表。
    """
    mapping = dict(zip(reference["stock_code"], reference["board"]))
    boards = np.array([mapping.get(code, "other") for code in codes])
    return {name: boards == name for name in BASE_RATIO}


def days_since_listing(
    codes: list[str], dates: list[str], reference: pd.DataFrame
) -> np.ndarray:
    """求每个股票-交易日相对上市日的交易日序号。

    参数:
        codes: 股票代码列表。
        dates: 交易日列表。
        reference: 股票基础信息表。

    返回:
        形状 (股票数, 交易日数) 的整数数组；窗口开始前已上市或上市日缺失的记为 -1。
    """
    first_date = dates[0]
    listing = {code: _as_date(value) for code, value in zip(reference["stock_code"], reference["list_date"])}
    n_days = len(dates)
    result = np.full((len(codes), n_days), -1, dtype=np.int32)
    for row, code in enumerate(codes):
        value = listing.get(code)
        if value is None or value < first_date:
            continue
        column = int(np.searchsorted(dates, value))
        if column >= n_days:
            continue
        result[row, column:] = np.arange(n_days - column, dtype=np.int32)
    return result


def compute_limit_ratios(
    codes: list[str],
    dates: list[str],
    reference: pd.DataFrame,
    is_st: np.ndarray,
    is_delisting: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """推算每个股票-交易日的涨停比例、跌停比例与是否适用涨跌停。

    参数:
        codes: 股票代码列表。
        dates: 交易日列表。
        reference: 股票基础信息表。
        is_st: ST 布尔面板。
        is_delisting: 退市整理期布尔面板。

    返回:
        (涨停比例, 跌停比例, 是否适用) 三个同形数组。
    """
    rows = board_rows(codes, reference)
    date_int = date_to_int(dates)[None, :]
    since = days_since_listing(codes, dates, reference)
    delisting = is_delisting

    up = np.zeros((len(codes), len(dates)), dtype=np.float64)
    for name, mask in rows.items():
        up[mask, :] = BASE_RATIO[name]
    chinext = rows["chinext"]
    if chinext.any():
        up[chinext, :] = np.where(date_int >= CHINEXT_20PCT_DATE, 0.20, 0.10)
    main = rows["main_sh"] | rows["main_sz"]
    up[np.ix_(main, np.arange(len(dates)))] = np.where(
        is_st[main], ST_RATIO, up[main]
    )
    down = up.copy()

    # 注册制新股上市前 5 个交易日不设涨跌幅
    no_limit = np.zeros(up.shape, dtype=bool)
    for name, regime in REGIME_FIVE_DAY.items():
        mask = rows[name]
        if not mask.any():
            continue
        active = date_int >= regime
        window = (since >= 0) & (since < 5)
        no_limit[mask, :] |= (active & window)[mask, :]

    # 2014 年以前的老规则：主板与创业板上市首日不设涨跌幅
    first_day = since == 0
    legacy = rows["main_sh"] | rows["main_sz"] | rows["chinext"]
    no_limit |= first_day & legacy[:, None] & (date_int < REGIME_FIRST_DAY_44)

    # 2014-01-01 至注册制前：主板与创业板上市首日 ±44% / -36%
    for name in ("main_sh", "main_sz", "chinext"):
        mask = rows[name]
        if not mask.any():
            continue
        regime = REGIME_FIVE_DAY[name]
        special = first_day & (date_int >= REGIME_FIRST_DAY_44) & (date_int < regime)
        mask_special = mask[:, None] & special
        up[mask_special] = 0.44
        down[mask_special] = 0.36

    # 退市整理期：首日不设限，其余按 10%
    if delisting.any():
        up[delisting] = 0.10
        down[delisting] = 0.10
        first_delist = delisting & ~np.concatenate(
            [np.zeros((delisting.shape[0], 1), dtype=bool), delisting[:, :-1]], axis=1
        )
        no_limit |= first_delist

    applicable = ~no_limit
    return up, down, applicable


def prefer_official(
    official_up: np.ndarray,
    official_down: np.ndarray,
    rule_up: np.ndarray,
    rule_down: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """优先采用交易所官方涨跌停价，缺失时回退到规则推算。

    参数:
        official_up: 官方涨停价面板。
        official_down: 官方跌停价面板。
        rule_up: 规则推算的涨停价面板。
        rule_down: 规则推算的跌停价面板。

    返回:
        (最终涨停价, 最终跌停价, 是否使用官方值) 三个同形数组。
    """
    has_official = ~np.isnan(official_up) & ~np.isnan(official_down)
    limit_up = np.where(has_official, official_up, rule_up)
    limit_down = np.where(has_official, official_down, rule_down)
    return limit_up, limit_down, has_official


def official_consistency(official_up: np.ndarray, rule_up: np.ndarray) -> dict:
    """比较官方涨停价与规则推算涨停价的一致率。

    规则推算没有考虑新股、退市整理期等特殊安排，因此只在两者都有值时比较，
    且允许 0.01 元的取整差。

    参数:
        official_up: 官方涨停价面板。
        rule_up: 规则推算涨停价面板。
    """
    both = ~np.isnan(official_up) & ~np.isnan(rule_up)
    total = int(both.sum())
    if total == 0:
        return {"compared": 0, "match": 0, "match_ratio": 0.0}
    diff = np.abs(official_up - rule_up)
    match = int(((diff <= 0.011) & both).sum())
    return {
        "compared": total,
        "match": match,
        "match_ratio": round(match / total, 6),
    }


def compute_limit_prices(
    pre_close: np.ndarray, up_ratio: np.ndarray, down_ratio: np.ndarray, applicable: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """计算涨停价与跌停价。

    参数:
        pre_close: 前收盘价面板。
        up_ratio: 涨停比例面板。
        down_ratio: 跌停比例面板。
        applicable: 是否适用涨跌停。

    返回:
        (涨停价, 跌停价)，不适用或前收缺失处为 NaN。
    """
    limit_up = round_half_up(pre_close * (1.0 + up_ratio))
    limit_down = round_half_up(pre_close * (1.0 - down_ratio))
    limit_up = np.where(applicable, limit_up, np.nan)
    limit_down = np.where(applicable, limit_down, np.nan)
    return limit_up, limit_down


def mark_limits(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_price: np.ndarray,
    limit_up: np.ndarray,
    limit_down: np.ndarray,
    applicable: np.ndarray,
) -> dict[str, np.ndarray]:
    """生成封板、触及、一字板标记。

    参数:
        close: 收盘价面板。
        high: 最高价面板。
        low: 最低价面板。
        open_price: 开盘价面板。
        limit_up: 涨停价面板。
        limit_down: 跌停价面板。
        applicable: 是否适用涨跌停。

    返回:
        各项标记的字典。
    """
    is_limit_up = applicable & (close >= limit_up - LIMIT_EPS)
    is_limit_down = applicable & (close <= limit_down + LIMIT_EPS)
    touched_up = applicable & (high >= limit_up - LIMIT_EPS)
    touched_down = applicable & (low <= limit_down + LIMIT_EPS)
    one_word = is_limit_up & (np.abs(high - low) < 1e-9) & (np.abs(open_price - limit_up) < LIMIT_EPS)
    return {
        "is_limit_up": is_limit_up,
        "is_limit_down": is_limit_down,
        "touched_limit_up": touched_up,
        "touched_limit_down": touched_down,
        "is_one_word_board": one_word,
    }


def mark_tradeable(
    status: np.ndarray,
    close: np.ndarray,
    open_price: np.ndarray,
    limit_up: np.ndarray,
    limit_down: np.ndarray,
    applicable: np.ndarray,
) -> dict[str, np.ndarray]:
    """生成可交易性列。

    口径是保守近似：一字板一律视为买不到；盘中开板也按封板处理。

    参数:
        status: 状态码面板。
        close: 收盘价面板。
        open_price: 开盘价面板。
        limit_up: 涨停价面板。
        limit_down: 跌停价面板。
        applicable: 是否适用涨跌停。

    返回:
        四个可交易性列。
    """
    trading = status == 1
    next_trading = np.full_like(trading, False)
    next_trading[:, :-1] = trading[:, 1:]
    next_open = np.full_like(open_price, np.nan)
    next_open[:, :-1] = open_price[:, 1:]
    next_up = np.full_like(limit_up, np.nan)
    next_up[:, :-1] = limit_up[:, 1:]
    next_down = np.full_like(limit_down, np.nan)
    next_down[:, :-1] = limit_down[:, 1:]
    next_applicable = np.full_like(applicable, False)
    next_applicable[:, :-1] = applicable[:, 1:]

    buy_close = trading & (np.isnan(limit_up) | (close < limit_up - LIMIT_EPS))
    sell_close = trading & (np.isnan(limit_down) | (close > limit_down + LIMIT_EPS))
    buy_next = next_trading & (
        ~next_applicable | np.isnan(next_up) | (next_open < next_up - LIMIT_EPS)
    )
    sell_next = next_trading & (
        ~next_applicable | np.isnan(next_down) | (next_open > next_down + LIMIT_EPS)
    )
    return {
        "can_buy_close": buy_close,
        "can_sell_close": sell_close,
        "can_buy_open_next": buy_next,
        "can_sell_open_next": sell_next,
    }


def check_limit_bounds(
    close: np.ndarray,
    pre_close: np.ndarray,
    up_ratio: np.ndarray,
    down_ratio: np.ndarray,
    applicable: np.ndarray,
) -> np.ndarray:
    """检查是否存在超出限幅的越界样本。

    参数:
        close: 收盘价面板。
        pre_close: 前收盘价面板。
        up_ratio: 涨停比例面板。
        down_ratio: 跌停比例面板。
        applicable: 是否适用涨跌停。

    返回:
        越界样本的布尔面板。
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        move = close / pre_close - 1.0
    upper = up_ratio + LIMIT_EPS / np.maximum(pre_close, 0.01)
    lower = -(down_ratio + LIMIT_EPS / np.maximum(pre_close, 0.01))
    beyond = (move > upper) | (move < lower)
    return applicable & ~np.isnan(move) & beyond
