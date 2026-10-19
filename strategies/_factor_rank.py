"""研究基线提供者：多因子截面分位等权。

文件名以下划线开头，不会被 ``backtest.providers.discover_providers`` 自动导入，
因此不会出现在常规提供者列表中。P0 基线脚本显式导入本模块完成注册。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

import factors
import quant
from backtest.data import MarketData
from backtest.errors import DataAlignmentError, SpecError
from backtest.providers import register_provider

_PROVIDER: "FactorRankProvider | None" = None


def weekly_last_dates(
    index: Iterable[pd.Timestamp],
    calendar: pd.DataFrame | None = None,
) -> pd.DatetimeIndex:
    """返回给定交易日中每周最后一个交易日。

    日历向区间后多取 21 天，避免区间末落在周中时把局部最后一天误判为周最后一天。
    """

    days = pd.DatetimeIndex(sorted(pd.to_datetime(list(index))))
    if days.empty:
        return days
    source = quant.load_calendar(
        start=str(days[0].date()),
        end=str((days[-1] + pd.Timedelta(days=21)).date()),
    ) if calendar is None else calendar
    stamps = pd.DatetimeIndex(pd.to_datetime(source["trade_date"])).sort_values()
    periods = pd.Series(stamps, index=stamps).dt.to_period("W-FRI")
    weekly = stamps[periods.ne(periods.shift(-1)).fillna(True).to_numpy()]
    return days.intersection(weekly)


def rank_composite(
    factor_panels: dict[str, pd.DataFrame],
    eligible: pd.DataFrame,
    factor_names: tuple[str, ...],
) -> pd.DataFrame:
    """按日计算「方向调整后的截面分位」等权平均。

    参数:
        factor_panels: 因子名到宽表的映射。
        eligible: 与宽表同形的可交易布尔矩阵。
        factor_names: 参与合成的因子名。
    """

    total = pd.DataFrame(0.0, index=eligible.index, columns=eligible.columns)
    count = pd.DataFrame(0.0, index=eligible.index, columns=eligible.columns)
    for name in factor_names:
        if name not in factor_panels:
            raise DataAlignmentError(f"factor_rank 缺少因子 {name}")
        panel = factor_panels[name].astype("float64")
        rank = panel.where(eligible).rank(axis=1, pct=True)
        direction = int(factors.get(name).direction)
        total = total.add(rank * direction, fill_value=0.0)
        count = count.add(rank.notna().astype("float64"), fill_value=0.0)
    return total.div(count.replace(0.0, np.nan))


class FactorRankProvider:
    """在周最后交易日按多因子分位等权合成分数，选 Top-N 等权持有。"""

    name = "factor_rank"
    version = "1"
    description = "多因子方向调整后的截面分位等权合成；周最后交易日 Top-N 等权"
    required_factors: tuple[str, ...] = ()
    param_schema = (
        {
            "name": "top_n",
            "label": "持仓只数",
            "type": "integer",
            "default": 30,
            "min": 1,
            "step": 1,
            "description": "周最后交易日等权持有的股票数。",
        },
    )

    @staticmethod
    def validate_params(params: dict) -> dict:
        """校验并返回规范化参数。"""

        unknown = set(params) - {"top_n", "factor_names"}
        if unknown:
            raise SpecError(f"factor_rank.params 包含未知字段：{sorted(unknown)}")
        top_n = params.get("top_n", 30)
        if type(top_n) is not int or top_n <= 0:
            raise SpecError("factor_rank.params.top_n 必须是正整数")
        names = params.get("factor_names")
        if names is None:
            names = tuple(spec.name for spec in factors.all_factors())
        if not isinstance(names, (list, tuple)) or not names:
            raise SpecError("factor_rank.params.factor_names 必须是非空序列")
        if any(not isinstance(name, str) or not name for name in names):
            raise SpecError("factor_rank.params.factor_names 只能包含非空字符串")
        unique = tuple(dict.fromkeys(str(name) for name in names))
        return {"top_n": top_n, "factor_names": list(unique)}

    def build(
        self,
        data: MarketData,
        eligible: pd.DataFrame,
        params: dict,
    ) -> pd.DataFrame:
        """返回周最后交易日的目标权重；其他日期保持 NaN。"""

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
            np.nan, index=data.index, columns=data.columns, dtype="float64"
        )
        for day in weekly_last_dates(data.index):
            row = score.loc[day].where(eligible.loc[day]).dropna()
            weights.loc[day] = 0.0
            if row.empty:
                continue
            chosen = row.sort_values(ascending=False, kind="mergesort").head(
                settings["top_n"]
            ).index
            weights.loc[day, chosen] = 1.0 / len(chosen)
        return weights


def ensure_registered() -> FactorRankProvider:
    """显式注册研究基线提供者；模块导入本身不注册。"""

    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = register_provider(FactorRankProvider())
    return _PROVIDER
