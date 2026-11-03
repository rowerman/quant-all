"""执行层持仓状态：目标权重提供者可以读取的只读组合快照。

状态由原始价账本逐日生成，记录真实现金、持仓、成本、浮盈和建仓以来最高
收盘价。提供者只读取这里的结果，不自行模拟成交或持仓。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from backtest.errors import DataAlignmentError

STATE_MODEL_VERSION = "execution_position_state_v1"

POSITION_STATE_COLUMNS = (
    "trade_date",
    "stock_code",
    "shares",
    "avg_cost",
    "price",
    "market_value",
    "weight",
    "unrealized_return",
    "high_water_price",
    "holding_since",
)
EXIT_STATE_COLUMNS = ("trade_date", "stock_code", "price")

_POSITION_FIELDS = POSITION_STATE_COLUMNS[1:]
_WEIGHT_TOLERANCE = 1e-9


def empty_position_state() -> pd.DataFrame:
    """返回带固定列的空持仓状态表。"""
    frame = pd.DataFrame(columns=POSITION_STATE_COLUMNS)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["holding_since"] = pd.to_datetime(frame["holding_since"])
    for name in _POSITION_FIELDS[1:-1]:
        frame[name] = frame[name].astype("float64")
    return frame


def empty_exit_state() -> pd.DataFrame:
    """返回带固定列的空清仓事件表。"""
    frame = pd.DataFrame(columns=EXIT_STATE_COLUMNS)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["price"] = frame["price"].astype("float64")
    return frame


@dataclass(frozen=True)
class PositionSnapshot:
    """某个收盘时点的组合状态。

    参数:
        trade_date: 状态对应的交易日。
        cash: 当日收盘现金。
        portfolio_value: 现金加应收分红和持仓市值。
        positions: 按股票代码索引的持仓表，只包含有仓位的股票。
    """

    trade_date: pd.Timestamp
    cash: float
    portfolio_value: float
    positions: pd.DataFrame

    def weights(self) -> pd.Series:
        """返回按股票代码索引的当前权重。"""
        return self.positions["weight"].astype("float64").copy()


@dataclass(frozen=True)
class ProviderState:
    """提供者可读取的逐日执行状态。

    参数:
        cash: 逐日收盘现金。
        portfolio_value: 逐日收盘组合价值，包含应收分红。
        positions: 长表持仓状态，列见 ``POSITION_STATE_COLUMNS``。
        exits: 清仓事件长表，列见 ``EXIT_STATE_COLUMNS``。
    """

    cash: pd.Series
    portfolio_value: pd.Series
    positions: pd.DataFrame
    exits: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        cash = pd.Series(self.cash, dtype="float64").sort_index()
        value = pd.Series(self.portfolio_value, dtype="float64").sort_index()
        cash.index = pd.DatetimeIndex(cash.index)
        value.index = pd.DatetimeIndex(value.index)
        if not cash.index.equals(value.index):
            raise DataAlignmentError("执行状态的现金与组合价值日期不一致")
        if not cash.index.is_unique:
            raise DataAlignmentError("执行状态日期不能重复")
        if cash.empty:
            raise DataAlignmentError("执行状态不能为空")
        if not np.isfinite(cash.to_numpy()).all():
            raise DataAlignmentError("执行状态现金包含非有限值")
        if not np.isfinite(value.to_numpy()).all():
            raise DataAlignmentError("执行状态组合价值包含非有限值")
        if (value.to_numpy() < -_WEIGHT_TOLERANCE).any():
            raise DataAlignmentError("执行状态组合价值不能为负")
        positions = self._validate_positions(self.positions, cash.index)
        exits = self._validate_exits(self.exits)
        object.__setattr__(self, "cash", cash)
        object.__setattr__(self, "portfolio_value", value)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "exits", exits)

    @staticmethod
    def _validate_positions(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """校验并规范化持仓长表。"""
        if not isinstance(frame, pd.DataFrame):
            raise DataAlignmentError("执行状态持仓必须是 DataFrame")
        missing = [name for name in POSITION_STATE_COLUMNS if name not in frame.columns]
        if missing:
            raise DataAlignmentError(f"执行状态持仓缺少列：{missing}")
        if frame.empty:
            return empty_position_state()
        positions = frame.loc[:, POSITION_STATE_COLUMNS].copy()
        positions["trade_date"] = pd.to_datetime(positions["trade_date"]).dt.normalize()
        positions["holding_since"] = pd.to_datetime(positions["holding_since"])
        positions["stock_code"] = positions["stock_code"].astype(str)
        for name in _POSITION_FIELDS[1:-1]:
            positions[name] = pd.to_numeric(positions[name], errors="coerce").astype(
                "float64"
            )
        if positions.duplicated(["trade_date", "stock_code"]).any():
            raise DataAlignmentError("执行状态同一交易日不能重复记录同一股票")
        unknown_dates = positions["trade_date"].loc[
            ~positions["trade_date"].isin(dates)
        ]
        if len(unknown_dates):
            sample = "; ".join(str(item.date()) for item in unknown_dates.head(3))
            raise DataAlignmentError(f"执行状态持仓日期不在现金序列内：{sample}")
        numeric = positions[
            ["shares", "avg_cost", "price", "market_value", "weight", "high_water_price"]
        ]
        labels = numeric.columns.to_numpy()
        invalid = ~np.isfinite(numeric.to_numpy(dtype="float64"))
        if invalid.any():
            row, column = np.argwhere(invalid)[0]
            raise DataAlignmentError(
                f"执行状态持仓 {positions['stock_code'].iloc[row]} "
                f"{positions['trade_date'].iloc[row].date()} 的 {labels[column]} 非有限"
            )
        if (positions["shares"].to_numpy() <= 0).any():
            raise DataAlignmentError("执行状态持仓股数必须为正")
        if (positions["avg_cost"].to_numpy() < 0).any():
            raise DataAlignmentError("执行状态平均成本不能为负")
        if (positions["price"].to_numpy() < 0).any():
            raise DataAlignmentError("执行状态估值价不能为负")
        if (positions["market_value"].to_numpy() < -_WEIGHT_TOLERANCE).any():
            raise DataAlignmentError("执行状态持仓市值不能为负")
        weights = positions["weight"].to_numpy(dtype="float64")
        if (weights < -_WEIGHT_TOLERANCE).any() or (
            weights > 1.0 + _WEIGHT_TOLERANCE
        ).any():
            raise DataAlignmentError("执行状态持仓权重必须在 [0, 1] 之间")
        if positions["holding_since"].isna().any():
            raise DataAlignmentError("执行状态建仓日不能缺失")
        return positions.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)

    @staticmethod
    def _validate_exits(frame: pd.DataFrame | None) -> pd.DataFrame:
        """校验并规范化清仓事件表。"""
        if frame is None or frame.empty:
            return empty_exit_state()
        if not isinstance(frame, pd.DataFrame):
            raise DataAlignmentError("执行状态清仓事件必须是 DataFrame")
        missing = [name for name in EXIT_STATE_COLUMNS if name not in frame.columns]
        if missing:
            raise DataAlignmentError(f"执行状态清仓事件缺少列：{missing}")
        exits = frame.loc[:, EXIT_STATE_COLUMNS].copy()
        exits["trade_date"] = pd.to_datetime(exits["trade_date"]).dt.normalize()
        exits["stock_code"] = exits["stock_code"].astype(str)
        exits["price"] = pd.to_numeric(exits["price"], errors="coerce").astype(
            "float64"
        )
        if exits.duplicated(["trade_date", "stock_code"]).any():
            raise DataAlignmentError("执行状态同一交易日不能重复记录同一清仓事件")
        if exits["trade_date"].isna().any():
            raise DataAlignmentError("执行状态清仓日期不能缺失")
        if not np.isfinite(exits["price"].to_numpy(dtype="float64")).all():
            raise DataAlignmentError("执行状态清仓价格必须有限")
        if (exits["price"].to_numpy(dtype="float64") < 0).any():
            raise DataAlignmentError("执行状态清仓价格不能为负")
        return exits.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)

    @classmethod
    def merge(cls, parts: Iterable["ProviderState | None"]) -> "ProviderState | None":
        """合并多个连续区间的状态，重复日期以后出现的为准。"""
        frames = [item for item in parts if item is not None]
        if not frames:
            return None
        cash = pd.concat([item.cash for item in frames]).sort_index()
        cash = cash[~cash.index.duplicated(keep="last")]
        value = pd.concat([item.portfolio_value for item in frames]).sort_index()
        value = value[~value.index.duplicated(keep="last")]
        positions = pd.concat(
            [item.positions for item in frames], ignore_index=True
        )
        if not positions.empty:
            positions = positions.drop_duplicates(
                ["trade_date", "stock_code"], keep="last"
            )
        exits = pd.concat([item.exits for item in frames], ignore_index=True)
        if not exits.empty:
            exits = exits.drop_duplicates(
                ["trade_date", "stock_code"], keep="last"
            )
        return cls(
            cash=cash, portfolio_value=value, positions=positions, exits=exits
        )

    @property
    def dates(self) -> pd.DatetimeIndex:
        """返回状态覆盖的交易日。"""
        return pd.DatetimeIndex(self.cash.index)

    def at(self, day: pd.Timestamp) -> PositionSnapshot:
        """返回指定交易日的组合快照。

        参数:
            day: 交易日；必须落在状态覆盖范围内。
        """
        stamp = pd.Timestamp(day).normalize()
        if stamp not in self.cash.index:
            raise DataAlignmentError(f"执行状态没有 {stamp.date()} 的记录")
        rows = self.positions.loc[self.positions["trade_date"].eq(stamp)]
        if rows.empty:
            frame = empty_position_state().iloc[0:0].set_index("stock_code")
            frame = frame.drop(columns=["trade_date"])
        else:
            frame = rows.set_index("stock_code").drop(columns=["trade_date"]).copy()
        return PositionSnapshot(
            trade_date=stamp,
            cash=float(self.cash.loc[stamp]),
            portfolio_value=float(self.portfolio_value.loc[stamp]),
            positions=frame,
        )

    def weights(self, day: pd.Timestamp) -> pd.Series:
        """返回指定交易日的当前权重。"""
        return self.at(day).weights()

    def last_exit(self, day: pd.Timestamp) -> pd.DataFrame:
        """返回指定日期及之前的最近一次清仓，按股票代码索引。"""
        stamp = pd.Timestamp(day).normalize()
        rows = self.exits.loc[self.exits["trade_date"].le(stamp)]
        if rows.empty:
            return empty_exit_state().iloc[0:0].set_index("stock_code")
        latest = rows.sort_values("trade_date").drop_duplicates(
            "stock_code", keep="last"
        )
        return latest.set_index("stock_code")

    def slice(self, index: pd.Index) -> "ProviderState":
        """按交易日索引切出子状态，顺序与索引一致。

        参数:
            index: 目标交易日索引；每个日期都必须存在于当前状态。
        """
        dates = pd.DatetimeIndex(index)
        missing = dates.difference(self.cash.index)
        if len(missing):
            sample = "; ".join(str(item.date()) for item in missing[:3])
            raise DataAlignmentError(f"执行状态缺少交易日：{sample}")
        positions = self.positions.loc[self.positions["trade_date"].isin(dates)]
        exits = self.exits.loc[self.exits["trade_date"].le(dates.max())]
        return ProviderState(
            cash=self.cash.reindex(dates),
            portfolio_value=self.portfolio_value.reindex(dates),
            positions=positions,
            exits=exits,
        )
