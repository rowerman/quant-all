"""示例提供者：MOM_20 Top-N 等权，叠加止损和移动止损。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.data import MarketData
from backtest.errors import DataAlignmentError, SpecError
from backtest.execution_state import ProviderState
from backtest.providers import register_provider


class MomentumTrailingStopProvider:
    """每周调仓；持仓触发止损或移动止损时，下一交易日开盘卖出。"""

    name = "momentum_trailing_stop"
    version = "1"
    description = "MOM_20 Top-N 等权 + 固定止损 + 建仓以来最高收盘价移动止损"
    required_factors = ("MOM_20",)
    uses_position_state = True

    @staticmethod
    def validate_params(params: dict) -> dict:
        """校验并返回规范化参数。"""
        unknown = set(params) - {
            "top_n",
            "stop_loss",
            "trailing_stop",
            "cooldown_days",
        }
        if unknown:
            raise SpecError(
                f"momentum_trailing_stop.params 包含未知字段：{sorted(unknown)}"
            )
        top_n = params.get("top_n", 30)
        if type(top_n) is not int or top_n <= 0:
            raise SpecError("momentum_trailing_stop.params.top_n 必须是正整数")
        settings = {"top_n": top_n}
        for name, default in (("stop_loss", 0.10), ("trailing_stop", 0.15)):
            value = params.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SpecError(f"momentum_trailing_stop.params.{name} 必须是数字")
            value = float(value)
            if not 0.0 < value < 1.0:
                raise SpecError(
                    f"momentum_trailing_stop.params.{name} 必须在 (0, 1) 之间"
                )
            settings[name] = value
        cooldown = params.get("cooldown_days", 10)
        if type(cooldown) is not int or cooldown < 0:
            raise SpecError(
                "momentum_trailing_stop.params.cooldown_days 必须是非负整数"
            )
        settings["cooldown_days"] = cooldown
        return settings

    @staticmethod
    def _rebalance_days(index: pd.DatetimeIndex) -> pd.Series:
        """每周首个交易日调仓；首行作为块内参考日，不调仓。"""
        weeks = pd.Series(
            [stamp.isocalendar()[:2] for stamp in index], index=index, dtype="object"
        )
        flags = weeks.ne(weeks.shift(1)).fillna(False)
        if len(flags):
            flags.iloc[0] = False
        return flags.astype(bool)

    @staticmethod
    def _selected(
        score: pd.Series,
        eligible: pd.Series,
        top_n: int,
        exclude: set[str] | None = None,
    ) -> pd.Index:
        """返回当日可买入股票中分数最高的 Top-N。"""
        ranked = (
            score.where(eligible)
            .dropna()
            .sort_values(ascending=False)
            .index
        )
        if exclude:
            ranked = ranked[~ranked.isin(sorted(exclude))]
        return ranked[:top_n]

    @staticmethod
    def _triggered(
        positions: pd.DataFrame, stop_loss: float, trailing_stop: float
    ) -> set[str]:
        """返回触发固定止损或移动止损的股票。"""
        result: set[str] = set()
        for code, row in positions.iterrows():
            if float(row.get("unrealized_return", np.nan)) <= -stop_loss:
                result.add(str(code))
                continue
            peak = float(row.get("high_water_price", np.nan))
            price = float(row.get("price", np.nan))
            if np.isfinite(peak) and peak > 0 and price / peak - 1.0 <= -trailing_stop:
                result.add(str(code))
        return result

    def build(
        self,
        data: MarketData,
        eligible: pd.DataFrame,
        params: dict,
        state: ProviderState | None = None,
    ) -> pd.DataFrame:
        """返回每日目标权重；未触发止损且非调仓日的行保持 NaN。"""
        settings = self.validate_params(params)
        if "MOM_20" not in data.factors:
            raise DataAlignmentError(
                f"momentum_trailing_stop 需要因子 MOM_20；已加载：{sorted(data.factors)}"
            )
        score = data.factors["MOM_20"].astype("float64")
        rebalance = self._rebalance_days(pd.DatetimeIndex(data.index))
        weights = pd.DataFrame(
            np.nan, index=data.index, columns=data.columns, dtype="float64"
        )
        for row_index, day in enumerate(data.index):
            snapshot = None if state is None else state.at(day)
            current = (
                pd.Series(0.0, index=data.columns, dtype="float64")
                if snapshot is None
                else snapshot.weights().reindex(data.columns).fillna(0.0)
            )
            triggered = (
                set()
                if snapshot is None
                else self._triggered(
                    snapshot.positions,
                    settings["stop_loss"],
                    settings["trailing_stop"],
                )
            )
            if bool(rebalance.iloc[row_index]):
                excluded = set(triggered)
                if snapshot is not None and settings["cooldown_days"] > 0:
                    exits = state.last_exit(day)
                    cutoff = day - pd.Timedelta(days=settings["cooldown_days"])
                    excluded.update(
                        str(code)
                        for code, item in exits.iterrows()
                        if pd.Timestamp(item["trade_date"]) > cutoff
                    )
                chosen = self._selected(
                    score.loc[day],
                    eligible.loc[day],
                    settings["top_n"],
                    exclude=excluded,
                )
                target = pd.Series(0.0, index=data.columns, dtype="float64")
                if len(chosen):
                    target.loc[chosen] = 1.0 / len(chosen)
                weights.loc[day] = target
                continue
            if triggered:
                target = current.where(eligible.loc[day], 0.0).copy()
                codes = [code for code in sorted(triggered) if code in target.index]
                target.loc[codes] = 0.0
                weights.loc[day] = target
        return weights


register_provider(MomentumTrailingStopProvider())
