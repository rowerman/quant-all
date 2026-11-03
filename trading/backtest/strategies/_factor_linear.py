"""使用显式调仓日的线性因子组合提供者。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtest.data import MarketData
from backtest.errors import DataAlignmentError, SpecError
from backtest.providers import register_provider
from backtest.signals import build_scores
from backtest.specs import SignalSpec
from backtest.strategies._factor_rank_v2 import _rebalance_dates

_PROVIDER: "FactorLinearProvider | None" = None


class FactorLinearProvider:
    """在显式日期按线性因子分数选 Top-N 等权持有。"""

    name = "factor_linear"
    version = "1"
    description = "线性因子分数 Top-N；供同日期基线对照使用"
    required_factors: tuple[str, ...] = ()
    hidden = True

    @staticmethod
    def validate_params(params: dict) -> dict:
        """校验因子权重、分数处理、Top-N 和显式调仓日。"""

        allowed = {
            "top_n",
            "factor_names",
            "factor_weights",
            "standardize",
            "winsorize",
            "rebalance_dates",
        }
        unknown = set(params) - allowed
        if unknown:
            raise SpecError(f"factor_linear.params 包含未知字段：{sorted(unknown)}")
        top_n = params.get("top_n", 30)
        if type(top_n) is not int or top_n <= 0:
            raise SpecError("factor_linear.params.top_n 必须是正整数")
        names = params.get("factor_names")
        if not isinstance(names, (list, tuple)) or not names:
            raise SpecError("factor_linear.params.factor_names 必须是非空序列")
        if any(not isinstance(name, str) or not name for name in names):
            raise SpecError("factor_linear.params.factor_names 只能包含非空字符串")
        unique_names = tuple(dict.fromkeys(names))
        values = params.get("factor_weights")
        if not isinstance(values, (list, tuple)) or len(values) != len(names):
            raise SpecError("factor_linear.params.factor_weights 必须与因子一一对应")
        weights: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SpecError("factor_linear.params.factor_weights 必须是数值")
            if not math.isfinite(float(value)):
                raise SpecError("factor_linear.params.factor_weights 必须是有限数值")
            weights.append(float(value))
        if len(unique_names) != len(names):
            raise SpecError("factor_linear.params.factor_names 不能重复")
        standardize = params.get("standardize", True)
        winsorize = params.get("winsorize", False)
        if not isinstance(standardize, bool) or not isinstance(winsorize, bool):
            raise SpecError("factor_linear 的 standardize/winsorize 必须是布尔值")
        dates = _rebalance_dates(params.get("rebalance_dates"))
        if dates is None:
            raise SpecError("factor_linear.params.rebalance_dates 不能为空")
        return {
            "top_n": top_n,
            "factor_names": list(unique_names),
            "factor_weights": weights,
            "standardize": standardize,
            "winsorize": winsorize,
            "rebalance_dates": list(dates),
        }

    def build(
        self,
        data: MarketData,
        eligible: pd.DataFrame,
        params: dict,
    ) -> pd.DataFrame:
        """返回显式日期目标权重；其他日期保持 NaN。"""

        settings = self.validate_params(params)
        missing = sorted(set(settings["factor_names"]) - set(data.factors))
        if missing:
            raise DataAlignmentError(
                f"factor_linear 需要因子 {missing}；已加载：{sorted(data.factors)}"
            )
        signal = SignalSpec(
            factors=tuple(settings["factor_names"]),
            weights=tuple(settings["factor_weights"]),
            standardize=settings["standardize"],
            winsorize=settings["winsorize"],
        )
        score = build_scores(data, signal)
        signal_dates = pd.DatetimeIndex(settings["rebalance_dates"]).intersection(
            data.index
        )
        weights = pd.DataFrame(
            np.nan,
            index=data.index,
            columns=data.columns,
            dtype="float64",
        )
        for day in signal_dates:
            row = score.loc[day].where(eligible.loc[day]).dropna()
            weights.loc[day] = 0.0
            if row.empty:
                continue
            chosen = row.sort_values(ascending=False, kind="mergesort").head(
                settings["top_n"]
            ).index
            weights.loc[day, chosen] = 1.0 / len(chosen)
        return weights


def ensure_registered() -> FactorLinearProvider:
    """注册隐藏的线性因子提供者。"""

    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = register_provider(FactorLinearProvider())
    return _PROVIDER
