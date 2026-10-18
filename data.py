"""回测输入数据和现有读取层之间的适配。

行情字段逐列读取并转成 ``float32`` 宽表，避免一次性把全字段长表读进来：
十年全 A 面板实测峰值从约 2.0GB 降到约 1.3GB。指数成分池还会先把成分并集
传给读取层，只取可能用到的股票。

价格字段使用**原始价**，交易标记、市值和成交额同样使用原始口径。现金分红、送转
和拆并由 `corporate_actions` 交给执行账本逐日处理，避免把前复权合成股数当成真
实交易股数。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

import quant
from quant.db import query as db_query
from backtest.config import FACTOR_DB, MAX_PANEL_CELLS, MAX_TRADE_DAYS
from backtest.errors import DataAlignmentError, ResourceLimitError
from backtest.logging_conf import get_logger
from backtest.specs import TOTAL_RETURN_BENCHMARKS
from backtest.trading_rules import VALID_BOARDS
from backtest.universe import (
    check_limits,
    membership_frame,
    resolve_universe,
    universe_stocks,
)

logger = get_logger("backtest.data")

# 价格、成交额与市值用浮点宽表，其余都是可交易性/状态标记，用布尔宽表
PRICE_FIELDS = ("open", "close")
VALUE_FIELDS = ("total_mv", "amount_cny")
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

# 执行账本使用真实成交价；公司行动在 ledger 中单独入账
PRICE_ADJUST = "raw"
STATE_ADJUST = "raw"

# 退市强制清算候选的列，与 backtest.orders.LIQUIDATION_COLUMNS 一致
LIQUIDATION_COLUMNS = ("trade_date", "stock_code", "price", "delist_date")


@dataclass(frozen=True)
class MarketData:
    """执行账本所需的行情、规则与公司行动宽表。

    所有价格与标记面板共用同一个交易日索引和同一套股票列；
    ``factors`` 是「因子名 → 宽表」，``list_date`` 是每股上市日；
    ``board`` 是每股所属板块，交易单位按它解析；
    ``membership`` 只在指数成分池下存在，为 True 表示当日属于成分。
    ``delist_date`` 为 None 时不做退市强制清算；``versions`` 是数据与因子版本。
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
    board: pd.Series
    delist_date: pd.Series | None = None
    membership: pd.DataFrame | None = None
    industry_l1: pd.DataFrame | None = None
    industry_labels: dict[int, str] | None = None
    benchmark: pd.Series | None = None
    benchmark_info: dict | None = None
    versions: dict | None = None
    amount_cny: pd.DataFrame | None = None
    corporate_actions: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.amount_cny is None:
            # 旧的合成夹具未提供成交额；关闭参与率时保持旧行为，启用时按成交额不可用处理。
            object.__setattr__(
                self,
                "amount_cny",
                pd.DataFrame(np.nan, index=self.open.index, columns=self.open.columns),
            )
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
        if self.industry_l1 is not None and (
            not self.industry_l1.index.equals(index) or not self.industry_l1.columns.equals(columns)
        ):
            raise DataAlignmentError("行业矩阵与行情面板的日期或股票列不一致")
        for name, frame in self.factors.items():
            if not frame.index.equals(index) or not frame.columns.equals(columns):
                raise DataAlignmentError(f"因子 {name} 与行情面板的日期或股票列不一致")
        if not self.list_date.index.equals(columns):
            raise DataAlignmentError("上市日序列与行情面板的股票列不一致")
        if not self.board.index.equals(columns):
            raise DataAlignmentError("板块序列与行情面板的股票列不一致")
        missing_board = self.board.isna() | (self.board.astype(str).str.strip() == "")
        if missing_board.any():
            sample = [str(item) for item in self.board.index[missing_board][:5]]
            raise DataAlignmentError(f"股票缺少板块信息：{sample}")
        unknown_board = ~self.board.astype(str).isin(VALID_BOARDS)
        if unknown_board.any():
            pairs = [
                f"{code}:{self.board.loc[code]}"
                for code in self.board.index[unknown_board][:5]
            ]
            raise DataAlignmentError(f"股票板块不在支持范围内：{pairs}")
        if self.delist_date is not None and not self.delist_date.index.equals(columns):
            raise DataAlignmentError("退市日序列与行情面板的股票列不一致")

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
        benchmark = None if self.benchmark is None else self.benchmark.reindex(self.index[mask])
        benchmark_info = None if self.benchmark_info is None else dict(self.benchmark_info)
        corporate_actions = (
            None
            if self.corporate_actions is None
            else self.corporate_actions.copy()
        )
        if corporate_actions is not None and not corporate_actions.empty:
            stamps = pd.to_datetime(corporate_actions["ex_date"])
            corporate_actions = corporate_actions.loc[
                (stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end))
            ].reset_index(drop=True)
        if benchmark_info is not None and benchmark is not None and len(benchmark):
            benchmark_info["start"] = str(pd.Timestamp(benchmark.index[0]).date())
            benchmark_info["end"] = str(pd.Timestamp(benchmark.index[-1]).date())
        return MarketData(
            factors={name: frame.loc[mask] for name, frame in self.factors.items()},
            list_date=self.list_date,
            board=self.board,
            delist_date=self.delist_date,
            membership=self.membership.loc[mask] if self.membership is not None else None,
            industry_l1=self.industry_l1.loc[mask] if self.industry_l1 is not None else None,
            industry_labels=self.industry_labels,
            benchmark=benchmark,
            benchmark_info=benchmark_info,
            versions=self.versions,
            corporate_actions=corporate_actions,
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
    adjust = PRICE_ADJUST if field in PRICE_FIELDS else STATE_ADJUST
    frame = quant.load_panel(
        start=start,
        end=end,
        fields=[field],
        adjust=adjust,
        stocks=list(stocks) if stocks else None,
        db_path=db_path,
    )
    wide = frame.pivot(index="trade_date", columns="stock_code", values=field)
    if field in PRICE_FIELDS or field in VALUE_FIELDS:
        return wide.astype("float32")
    # 缺失的交易日（未上市/已退市）按「不可交易」处理
    return wide.astype("boolean").fillna(False).astype(bool)


def _dataset_version(db_path: Path | str | None) -> dict | None:
    """读取行情库的构建清单（``meta.manifest``）。

    参数:
        db_path: 行情库路径。
    """
    try:
        frame = db_query(
            "SELECT data_version, generated_at, payload FROM meta.manifest LIMIT 1",
            path=db_path,
        )
    except Exception as error:  # 老库可能没有清单表，降级为 None 并留痕
        logger.warning("读取数据版本失败：%s", error)
        return None
    if frame.empty:
        logger.warning("行情库没有 meta.manifest 记录，本次运行不记数据版本")
        return None
    row = frame.iloc[0]
    shape: dict = {}
    try:
        shape = json.loads(row["payload"]).get("shape", {})
    except (TypeError, ValueError) as error:
        logger.warning("数据清单 payload 无法解析：%s", error)
    return {
        "data_version": None if pd.isna(row["data_version"]) else str(row["data_version"]),
        "generated_at": None if pd.isna(row["generated_at"]) else str(row["generated_at"]),
        "stocks": shape.get("n_stocks"),
        "rows": shape.get("n_rows"),
    }


def _factor_version(factor_db: Path | str | None) -> dict | None:
    """读取因子目录库的构建清单（``meta.factor_manifest``）。

    参数:
        factor_db: 因子目录库路径。
    """
    try:
        frame = db_query(
            "SELECT factor_version, factor_count, rows, built_at FROM meta.factor_manifest LIMIT 1",
            path=factor_db,
        )
    except Exception as error:  # 因子库缺失或未建清单时不影响回测
        logger.warning("读取因子版本失败：%s", error)
        return None
    if frame.empty:
        logger.warning("因子库没有 meta.factor_manifest 记录，本次运行不记因子版本")
        return None
    row = frame.iloc[0]
    return {
        "factor_version": None if pd.isna(row["factor_version"]) else str(row["factor_version"]),
        "factor_count": None if pd.isna(row["factor_count"]) else int(row["factor_count"]),
        "rows": None if pd.isna(row["rows"]) else int(row["rows"]),
        "built_at": None if pd.isna(row["built_at"]) else str(row["built_at"]),
    }


def _industry_version(db_path: Path | str | None) -> dict | None:
    """读取申万历史行业内容摘要，老库缺表时降级为 None。"""
    try:
        frame = db_query(
            "SELECT * FROM meta.sw_industry_manifest LIMIT 1",
            path=db_path,
        )
    except Exception as error:
        logger.warning("读取申万行业版本失败：%s", error)
        return None
    if frame.empty:
        return None
    row = frame.iloc[0]
    text_fields = (
        "source", "api", "digest", "min_date", "max_date",
        "source_updated_at", "built_at", "classifications", "switch_date",
    )
    result = {
        name: (None if name not in row or pd.isna(row[name]) else str(row[name]))
        for name in text_fields
    }
    for name in (
        "rows", "source_rows", "stocks", "history_rows", "current_snapshot_rows",
        "overlap_segments_resolved", "gap_segments_filled", "gap_days_filled",
        "repair_rows", "carry_forward_rows", "carry_forward_stocks",
        "empty_source_rows", "backfill_rows", "backfill_stocks",
    ):
        result[name] = (
            int(row[name]) if name in row and not pd.isna(row[name]) else 0
        )
    return result


def _load_versions(db_path: Path | str | None, factor_db: Path | str | None = None) -> dict:
    """读取本次运行用到的数据版本与因子版本。

    参数:
        db_path: 行情库路径。
        factor_db: 因子目录库路径；None 表示用默认路径。
    """
    result = {
        "dataset": _dataset_version(db_path),
        "factors": _factor_version(FACTOR_DB if factor_db is None else factor_db),
        "industry": _industry_version(db_path),
    }
    try:
        actions = db_query(
            "SELECT * FROM meta.corporate_action_manifest LIMIT 1",
            path=db_path,
        )
        result["corporate_actions"] = (
            None if actions.empty else {
                str(key): (None if pd.isna(value) else value)
                for key, value in actions.iloc[0].items()
            }
        )
    except Exception as error:
        logger.warning("读取公司行动版本失败：%s", error)
        result["corporate_actions"] = None
    return result


def _load_industry_l1(
    start: str,
    end: str,
    index: pd.DatetimeIndex,
    columns: pd.Index,
    db_path: Path | str | None,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """把申万一级历史区间铺成紧凑整数宽表，0 表示无行业。"""
    frame = quant.load_sw_industry_member(
        start=start, end=end, stocks=list(columns), db_path=db_path
    )
    if frame.empty:
        return (
            pd.DataFrame(np.zeros((len(index), len(columns)), dtype="int16"),
                         index=index, columns=columns),
            {},
        )
    frame = frame.copy()
    if "classification" in frame.columns:
        frame["group_code"] = (
            frame["classification"].fillna("").astype(str)
            + ":"
            + frame["l1_code"].fillna("").astype(str)
        )
    else:
        frame["group_code"] = frame["l1_code"].astype(str)
    frame["group_label"] = frame.apply(
        lambda row: _industry_label(row), axis=1
    )
    codes = sorted(frame["group_code"].dropna().astype(str).unique())
    encoded = {code: position + 1 for position, code in enumerate(codes)}
    labels = (
        frame.drop_duplicates("group_code")
        .set_index("group_code")["group_label"]
        .astype(str)
        .to_dict()
    )
    labels = {encoded[code]: labels[code] for code in codes}
    values = np.zeros((len(index), len(columns)), dtype="int16")
    column_positions = {str(code): position for position, code in enumerate(columns)}
    stamps = index.to_numpy(dtype="datetime64[ns]")
    for row in frame.itertuples(index=False):
        column = column_positions.get(str(row.stock_code))
        if column is None:
            continue
        start_at = int(np.searchsorted(stamps, np.datetime64(row.in_date), side="left"))
        end_at = len(index) if pd.isna(row.out_date) else int(
            np.searchsorted(stamps, np.datetime64(row.out_date), side="right")
        )
        if start_at >= end_at:
            continue
        code = encoded[str(row.group_code)]
        occupied = values[start_at:end_at, column]
        conflict = (occupied != 0) & (occupied != code)
        if conflict.any():
            first = start_at + int(np.flatnonzero(conflict)[0])
            raise DataAlignmentError(
                f"申万一级行业区间冲突：{columns[column]} 在 {index[first].date()} "
                f"同时命中 {labels[int(values[first, column])]} 与 {labels[code]}"
            )
        occupied[occupied == 0] = code
    return pd.DataFrame(values, index=index, columns=columns), labels


def _industry_label(row) -> str:
    """生成行业组标签；旧库没有分类和 carry_forward 时保持旧格式。"""
    code = str(row["l1_code"])
    name = "" if pd.isna(row.get("l1_name")) else str(row["l1_name"])
    classification = row.get("classification")
    if classification is None or pd.isna(classification):
        return f"{code} {name}".strip()
    label = f"{classification} {code} {name}".strip()
    carry = row.get("carry_forward")
    if carry is not None and not pd.isna(carry) and bool(carry):
        label += "（沿用）"
    return label


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


def estimate_panel(
    start: str,
    end: str,
    universe: str = "all_a",
    *,
    db_path: Path | str | None = None,
) -> tuple[int, int]:
    """预估区间内的交易日数与股票数上界。

    分块回测用它决定要不要切块；单块取数前的规模预检也用同一份口径。

    参数:
        start: 起始交易日。
        end: 截止交易日。
        universe: 股票池标识。
        db_path: 行情库路径。
    """
    spec = resolve_universe(universe)
    stocks = universe_stocks(spec, start, end, db_path=db_path)
    days = _trade_days(start, end, db_path)
    return days, len(stocks) if stocks else _stock_upper_bound(start, end, db_path)


def load_liquidation_candidates(
    start: str,
    end: str,
    *,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """读取整段区间的退市强制清算候选：每只退市股在区间内最后一只有价行情日。

    分块回测必须一次算好整段区间的清算候选：按块现算时，持仓股最后一条行情日
    正好落在块边界，清算会被记到下一块的预热行，与原块的日结果打架。

    口径与 :func:`backtest.orders.liquidation_cells` 的面板版一致：退市日不晚于区间末、
    最后一只有价行情日早于区间末（最后一天仍在交易的股票不按清算处理）。

    参数:
        start: 起始交易日。
        end: 截止交易日。
        db_path: 行情库路径。
    """
    frame = db_query(
        """
        WITH delisted AS (
            SELECT DISTINCT stock_code, delist_date
            FROM md.daily_bar
            WHERE delist_date IS NOT NULL AND delist_date <= CAST(? AS DATE)
        ),
        last_priced AS (
            SELECT b.stock_code, MAX(b.trade_date) AS trade_date
            FROM md.daily_bar b
            WHERE b.trade_date >= CAST(? AS DATE) AND b.trade_date < CAST(? AS DATE)
              AND b.open IS NOT NULL
              AND b.stock_code IN (SELECT stock_code FROM delisted)
            GROUP BY b.stock_code
        )
        SELECT b.trade_date, b.stock_code, b.open AS price, d.delist_date
        FROM last_priced l
        JOIN md.daily_bar b
          ON b.stock_code = l.stock_code AND b.trade_date = l.trade_date
        JOIN delisted d ON d.stock_code = l.stock_code
        ORDER BY b.trade_date, b.stock_code
        """,
        [start, start, end],
        path=db_path,
    )
    if frame.empty:
        return pd.DataFrame(columns=list(LIQUIDATION_COLUMNS))
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["delist_date"] = pd.to_datetime(frame["delist_date"])
    return frame[list(LIQUIDATION_COLUMNS)]


def load_benchmark(
    code: str | None,
    start: str,
    end: str,
    index: pd.DatetimeIndex,
    db_path: Path | str | None = None,
) -> tuple[pd.Series | None, dict | None]:
    """读取基准指数收盘价、严格对齐交易日并返回口径说明。"""
    if not code:
        return None, None
    daily = quant.load_index_daily(code, start=start, end=end, db_path=db_path)
    if daily.empty:
        raise DataAlignmentError(f"基准指数 {code} 在 {start} ~ {end} 没有行情")
    series = daily.assign(trade_date=pd.to_datetime(daily["trade_date"])).set_index("trade_date")
    aligned = series["close"].sort_index().reindex(index).astype("float64")
    if aligned.isna().any():
        missing = [str(day.date()) for day in aligned.index[aligned.isna()][:5]]
        raise DataAlignmentError(
            f"基准指数 {code} 缺少 {int(aligned.isna().sum())} 个回测交易日，示例：{missing}"
        )
    basic = quant.load_index_basic(index_code=code, db_path=db_path)
    name = str(basic.iloc[0]["index_name"]) if not basic.empty else code
    return_type = "total_return" if code in TOTAL_RETURN_BENCHMARKS or "收益" in name else "price"
    info = {
        "code": code,
        "name": name,
        "return_type": return_type,
        "start": str(aligned.index.min().date()),
        "end": str(aligned.index.max().date()),
    }
    return aligned, info


def load_market_data(
    start: str,
    end: str,
    factor_names: list[str],
    *,
    universe: str = "all_a",
    db_path: Path | str | None = None,
    factor_dir: Path | str | None = None,
    benchmark_code: str | None = "H00300.CSI",
    membership_end: str | None = None,
    stock_codes: tuple[str, ...] | None = None,
    industry_level: str | None = None,
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
        membership_end: 指数成分的截止日；None 表示用 ``end``。分块回测传整段区间的
            截止日，保证每块的成分口径与不分块一致。
        stock_codes: 外部固定的股票列；None 表示按本区间解析股票池。分块回测传整段
            区间的股票并集，保证每块列集合一致，持仓不会因股票暂时不在本块池内而丢失。
        industry_level: None 表示不加载行业；第一版只支持 ``L1``。
    """
    if industry_level not in (None, "L1"):
        raise ValueError("industry_level 第一版只支持 None/L1")
    spec = resolve_universe(universe)
    stocks = stock_codes if stock_codes is not None else universe_stocks(spec, start, end, db_path=db_path)
    days = _trade_days(start, end, db_path)
    expected = len(stocks) if stocks else _stock_upper_bound(start, end, db_path)
    check_limits(days, expected, MAX_TRADE_DAYS, MAX_PANEL_CELLS)

    frames = {name: _load_field(name, start, end, stocks, db_path) for name in PANEL_FIELDS}
    index = frames["open"].index
    columns = (
        pd.Index(stocks, name=frames["open"].columns.name)
        if stock_codes is not None
        else frames["open"].columns
    )
    if len(index) * len(columns) > MAX_PANEL_CELLS:
        raise ResourceLimitError(
            f"面板规模 {len(index)}×{len(columns)} 超过上限 {MAX_PANEL_CELLS} 个单元格；请缩短区间"
        )
    aligned = {}
    for name, frame in frames.items():
        if name in FLAG_FIELDS:
            value = frame.reindex(index=index, columns=columns, fill_value=False).astype(bool)
        elif name in PRICE_FIELDS or name in VALUE_FIELDS:
            value = frame.reindex(index=index, columns=columns)
            value = value.astype("float32")
        else:
            value = frame.reindex(index=index, columns=columns)
        aligned[name] = value
    frames = aligned
    factors = {
        name: _load_factor(name, start, end, index, columns, stocks, factor_dir)
        for name in factor_names
    }
    membership = (
        membership_frame(
            spec, index, columns, end if membership_end is None else membership_end, db_path=db_path
        )
        if spec.is_index
        else None
    )
    basic = quant.load_stock_basic(db_path=db_path)
    indexed = basic.set_index("stock_code")
    if "market" not in indexed.columns:
        raise DataAlignmentError("股票基础表缺少 market 列，无法解析板块交易单位")
    board = indexed["market"].reindex(columns).astype("string")
    missing_board = board.isna() | (board.str.strip() == "")
    if missing_board.any():
        sample = [str(item) for item in columns[missing_board.to_numpy()][:5]]
        raise DataAlignmentError(f"股票基础表缺少股票板块：{sample}")
    board = board.astype(str).rename("board")
    unknown_board = ~board.isin(VALID_BOARDS)
    if unknown_board.any():
        pairs = [f"{code}:{board.loc[code]}" for code in board.index[unknown_board][:5]]
        raise DataAlignmentError(f"股票基础表包含不支持的板块：{pairs}")
    list_date = pd.to_datetime(indexed["list_date"]).reindex(columns)
    delist_date = (
        pd.to_datetime(indexed["delist_date"]).reindex(columns)
        if "delist_date" in indexed.columns
        else None
    )
    if delist_date is None:
        logger.warning("股票基础表没有 delist_date，本次运行不做退市强制清算")
    logger.info(
        "回测输入就绪：池=%s 交易日=%d 股票=%d 因子=%s",
        spec.key,
        len(index),
        len(columns),
        list(factors),
    )
    benchmark, benchmark_info = load_benchmark(benchmark_code, start, end, index, db_path)
    corporate_actions = quant.load_corporate_actions(
        start=start,
        end=end,
        stocks=list(columns),
        db_path=db_path,
    )
    if not corporate_actions.empty:
        corporate_actions["ex_date"] = pd.to_datetime(corporate_actions["ex_date"])
        corporate_actions["record_date"] = pd.to_datetime(
            corporate_actions["record_date"], errors="coerce"
        )
        corporate_actions["pay_date"] = pd.to_datetime(
            corporate_actions["pay_date"], errors="coerce"
        )
    industry_l1, industry_labels = (
        _load_industry_l1(start, end, index, columns, db_path)
        if industry_level == "L1" else (None, None)
    )
    return MarketData(
        factors=factors,
        list_date=list_date,
        board=board,
        delist_date=delist_date,
        membership=membership,
        industry_l1=industry_l1,
        industry_labels=industry_labels,
        benchmark=benchmark,
        benchmark_info=benchmark_info,
        versions=_load_versions(db_path),
        corporate_actions=corporate_actions,
        **frames,
    )

