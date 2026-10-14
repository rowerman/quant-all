"""回测输入数据和现有读取层之间的适配。

行情字段逐列读取并转成 ``float32`` 宽表，避免一次性把全字段长表读进来：
十年全 A 面板实测峰值从约 2.0GB 降到约 1.3GB。指数成分池还会先把成分并集
传给读取层，只取可能用到的股票。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import quant
from backtest.config import MAX_PANEL_CELLS, MAX_TRADE_DAYS
from backtest.errors import DataAlignmentError, ResourceLimitError
from backtest.logging_conf import get_logger
from backtest.universe import (
    check_limits,
    membership_frame,
    resolve_universe,
    universe_stocks,
)

logger = get_logger("backtest.data")

# 价格与市值用浮点宽表，其余都是可交易性/状态标记，用布尔宽表
PRICE_FIELDS = ("open", "close")
VALUE_FIELDS = ("total_mv",)
FLAG_FIELDS = (
    "can_buy_close",
    "can_buy_open_next",
    "can_sell_open_next",
    "is_suspended",
    "is_st",
    "is_limit_up",
    "is_limit_down",
)
PANEL_FIELDS = PRICE_FIELDS + FLAG_FIELDS + VALUE_FIELDS


@dataclass(frozen=True)
class MarketData:
    """VectorBT 所需的行情与交易规则宽表。

    所有价格与标记面板共用同一个交易日索引和同一套股票列；
    ``factors`` 是「因子名 → 宽表」，``list_date`` 是每股上市日；
    ``membership`` 只在指数成分池下存在，为 True 表示当日属于成分。
    """

    open: pd.DataFrame
    close: pd.DataFrame
    can_buy_close: pd.DataFrame
    can_buy_open_next: pd.DataFrame
    can_sell_open_next: pd.DataFrame
    is_suspended: pd.DataFrame
    is_st: pd.DataFrame
    is_limit_up: pd.DataFrame
    is_limit_down: pd.DataFrame
    total_mv: pd.DataFrame
    factors: dict[str, pd.DataFrame]
    list_date: pd.Series
    membership: pd.DataFrame | None = None
    benchmark: pd.Series | None = None

    def __post_init__(self) -> None:
        panels = [getattr(self, name) for name in PANEL_FIELDS]
        shapes = {panel.shape for panel in panels}
        if len(shapes) != 1:
            raise DataAlignmentError(f"行情与交易规则面板形状不一致：{shapes}")
        index, columns = panels[0].index, panels[0].columns
        for panel in panels[1:]:
            if not panel.index.equals(index) or not panel.columns.equals(columns):
                raise DataAlignmentError("行情与交易规则面板的日期或股票列不一致")
        if self.membership is not None and (
            not self.membership.index.equals(index) or not self.membership.columns.equals(columns)
        ):
            raise DataAlignmentError("成分矩阵与行情面板的日期或股票列不一致")
        for name, frame in self.factors.items():
            if not frame.index.equals(index) or not frame.columns.equals(columns):
                raise DataAlignmentError(f"因子 {name} 与行情面板的日期或股票列不一致")
        if not self.list_date.index.equals(columns):
            raise DataAlignmentError("上市日序列与行情面板的股票列不一致")

    @property
    def index(self) -> pd.DatetimeIndex:
        """交易日索引。"""
        return pd.DatetimeIndex(self.open.index)

    @property
    def columns(self) -> pd.Index:
        """股票列索引。"""
        return self.open.columns

    def crop(self, start: str, end: str) -> "MarketData":
        """按区间截取面板，区间外的行直接丢弃。

        参数:
            start: 起始交易日。
            end: 截止交易日。
        """
        mask = (self.index >= pd.Timestamp(start)) & (self.index <= pd.Timestamp(end))
        if mask.all():
            return self
        if not mask.any():
            raise DataAlignmentError(f"回测区间 {start} ~ {end} 内没有交易日")
        frames = {
            name: getattr(self, name).loc[mask] for name in PANEL_FIELDS
        }
        return MarketData(
            factors={name: frame.loc[mask] for name, frame in self.factors.items()},
            list_date=self.list_date,
            membership=self.membership.loc[mask] if self.membership is not None else None,
            benchmark=None if self.benchmark is None else self.benchmark.reindex(self.index[mask]),
            **frames,
        )


def _load_field(
    field: str,
    start: str,
    end: str,
    stocks: tuple[str, ...] | None,
    db_path: Path | str | None,
) -> pd.DataFrame:
    """读取单个行情字段并透视成宽表。

    参数:
        field: 字段名。
        start: 起始交易日。
        end: 截止交易日。
        stocks: 股票池；None 表示全部。
        db_path: 行情库路径。
    """
    frame = quant.load_panel(
        start=start,
        end=end,
        fields=[field],
        adjust="raw",
        stocks=list(stocks) if stocks else None,
        db_path=db_path,
    )
    wide = frame.pivot(index="trade_date", columns="stock_code", values=field)
    if field in PRICE_FIELDS or field in VALUE_FIELDS:
        return wide.astype("float32")
    # 缺失的交易日（未上市/已退市）按「不可交易」处理
    return wide.astype("boolean").fillna(False).astype(bool)


def _load_factor(
    name: str,
    start: str,
    end: str,
    index: pd.DatetimeIndex,
    columns: pd.Index,
    stocks: tuple[str, ...] | None,
    factor_dir: Path | str | None,
) -> pd.DataFrame:
    """读取单个因子并对齐到统一的日期与股票列。

    参数:
        name: 因子名。
        start: 起始交易日。
        end: 截止交易日。
        index: 统一交易日索引。
        columns: 统一股票列。
        stocks: 股票池；None 表示全部。
        factor_dir: 因子数据根目录。
    """
    frame = quant.load_factor_panel(
        start=start,
        end=end,
        names=[name],
        stocks=list(stocks) if stocks else None,
        factor_dir=factor_dir,
    )
    wide = frame.pivot(index="trade_date", columns="stock_code", values=name)
    wide.index = pd.DatetimeIndex(wide.index)
    return wide.reindex(index=index, columns=columns).astype("float32")


def _trade_days(start: str, end: str, db_path: Path | str | None) -> int:
    """从交易日历取区间内的交易日数。"""
    calendar = quant.load_calendar(start=start, end=end, db_path=db_path)
    return int(calendar["trade_date"].nunique())


def _stock_upper_bound(start: str, end: str, db_path: Path | str | None) -> int:
    """用股票基础表估计区间内可能出现的股票数，用于取数前的规模预检。"""
    basic = quant.load_stock_basic(db_path=db_path)
    listed = pd.to_datetime(basic["list_date"])
    delisted = pd.to_datetime(basic["delist_date"])
    alive = (listed <= pd.Timestamp(end)) & (delisted.isna() | (delisted >= pd.Timestamp(start)))
    return int(alive.sum())


def _load_benchmark(
    code: str | None,
    start: str,
    end: str,
    index: pd.DatetimeIndex,
    db_path: Path | str | None,
) -> pd.Series | None:
    """读取基准指数收盘价并对齐到回测交易日。"""
    if not code:
        return None
    daily = quant.load_index_daily(code, start=start, end=end, db_path=db_path)
    if daily.empty:
        logger.warning("基准指数 %s 在 %s ~ %s 没有行情，本次回测不计算超额", code, start, end)
        return None
    series = daily.assign(trade_date=pd.to_datetime(daily["trade_date"])).set_index("trade_date")
    return series["close"].sort_index().reindex(index).astype("float64")


def load_market_data(
    start: str,
    end: str,
    factor_names: list[str],
    *,
    universe: str = "all_a",
    db_path: Path | str | None = None,
    factor_dir: Path | str | None = None,
    benchmark_code: str | None = "000300.SH",
) -> MarketData:
    """从主库和因子目录读取回测输入。

    参数:
        start: 起始交易日。
        end: 截止交易日。
        factor_names: 需要的因子名。
        universe: 股票池标识，``all_a`` 或 ``index:<指数代码>``。
        db_path: 行情库路径。
        factor_dir: 因子数据根目录。
        benchmark_code: 基准指数代码；None 表示不取基准。
    """
    spec = resolve_universe(universe)
    stocks = universe_stocks(spec, start, end, db_path=db_path)
    days = _trade_days(start, end, db_path)
    expected = len(stocks) if stocks else _stock_upper_bound(start, end, db_path)
    check_limits(days, expected, MAX_TRADE_DAYS, MAX_PANEL_CELLS)

    frames = {name: _load_field(name, start, end, stocks, db_path) for name in PANEL_FIELDS}
    index, columns = frames["open"].index, frames["open"].columns
    if len(index) * len(columns) > MAX_PANEL_CELLS:
        raise ResourceLimitError(
            f"面板规模 {len(index)}×{len(columns)} 超过上限 {MAX_PANEL_CELLS} 个单元格；请缩短区间"
        )
    frames = {
        name: frame.reindex(index=index, columns=columns) for name, frame in frames.items()
    }
    factors = {
        name: _load_factor(name, start, end, index, columns, stocks, factor_dir)
        for name in factor_names
    }
    membership = (
        membership_frame(spec, index, columns, end, db_path=db_path) if spec.is_index else None
    )
    basic = quant.load_stock_basic(db_path=db_path)
    list_date = pd.to_datetime(basic.set_index("stock_code")["list_date"]).reindex(columns)
    logger.info(
        "回测输入就绪：池=%s 交易日=%d 股票=%d 因子=%s",
        spec.key,
        len(index),
        len(columns),
        list(factors),
    )
    return MarketData(
        factors=factors,
        list_date=list_date,
        membership=membership,
        benchmark=_load_benchmark(benchmark_code, start, end, index, db_path),
        **frames,
    )

