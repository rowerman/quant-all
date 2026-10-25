"""支持显式调仓日的 P0 全因子 rank 基线提供者。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

import factors
from backtest.data import MarketData
from backtest.errors import DataAlignmentError, SpecError
from backtest.providers import register_provider
from backtest.strategies._factor_rank import (
    FactorRankProvider,
    ensure_registered as ensure_v1_registered,
    rank_composite,
    weekly_last_dates,
)

_PROVIDER: "FactorRankV2Provider | None" = None


def _rebalance_dates(value) -> tuple[str, ...] | None:
    """校验显式调仓日并返回排序后的 ISO 日期。"""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise SpecError("factor_rank.params.rebalance_dates 必须是非空日期序列")
    normalized: list[str] = []
    for item in value:
        try:
            if not isinstance(item, str) or date.fromisoformat(item).isoformat() != item:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise SpecError(
                f"factor_rank.params.rebalance_dates 日期必须是 YYYY-MM-DD：{item!r}"
            ) from error
        normalized.append(item)
    return tuple(sorted(dict.fromkeys(normalized)))


class FactorRankV2Provider(FactorRankProvider):
    """在显式日期或默认周最后交易日选多因子 rank Top-N。"""

    name = "factor_rank"
    version = "2"
    description = "多因子方向调整截面分位等权；可选显式调仓日"
    required_factors: tuple[str, ...] = ()

    @staticmethod
    def validate_params(params: dict) -> dict:
        """校验 Top-N、因子清单和可选调仓日。"""

        unknown = set(params) - {"top_n", "factor_names", "rebalance_dates"}
        if unknown:
            raise SpecError(f"factor_rank.params 包含未知字段：{sorted(unknown)}")
        top_n = params.get("top_n", 30)
        if type(top_n) is not int or top_n <= 0:
            raise SpecError("factor_rank.params.top_n 必须是正整数")
        names = params.get("factor_names")
        if names is None:
            names = tuple(spec.name for spec in factors.available_specs())
        if not isinstance(names, (list, tuple)) or not names:
            raise SpecError("factor_rank.params.factor_names 必须是非空序列")
        if any(not isinstance(name, str) or not name for name in names):
            raise SpecError("factor_rank.params.factor_names 只能包含非空字符串")
        unique = tuple(dict.fromkeys(str(name) for name in names))
        result = {"top_n": top_n, "factor_names": list(unique)}
        dates = _rebalance_dates(params.get("rebalance_dates"))
        if dates is not None:
            result["rebalance_dates"] = list(dates)
        return result

    def build(
        self,
        data: MarketData,
        eligible: pd.DataFrame,
        params: dict,
    ) -> pd.DataFrame:
        """返回显式信号日的目标权重；其他日期保持 NaN。"""

        settings = self.validate_params(params)
        missing = sorted(set(settings["factor_names"]) - set(data.factors))
        if missing:
            raise DataAlignmentError(
                f"factor_rank 需要因子 {missing}；已加载：{sorted(data.factors)}"
            )
        score = rank_composite(
            data.factors,
            eligible,
            tuple(settings["factor_names"]),
        )
        weights = pd.DataFrame(
            np.nan,
            index=data.index,
            columns=data.columns,
            dtype="float64",
        )
        configured_dates = settings.get("rebalance_dates")
        signal_dates = (
            weekly_last_dates(data.index)
            if configured_dates is None
            else pd.DatetimeIndex(configured_dates).intersection(data.index)
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


def ensure_registered() -> FactorRankV2Provider:
    """显式注册 v1 和 v2；v1 保持原有严格周历行为。"""

    global _PROVIDER
    ensure_v1_registered()
    if _PROVIDER is None:
        _PROVIDER = register_provider(FactorRankV2Provider())
    return _PROVIDER
