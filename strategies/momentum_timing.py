"""示例提供者：MOM_20 截面广度择时 + Top-N 等权。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.data import MarketData
from backtest.errors import DataAlignmentError, SpecError
from backtest.providers import register_provider


class MomentumTimingProvider:
    """市场广度低于阈值时清仓，否则持有分数最高的若干只股票。"""

    name = "momentum_timing"
    version = "1"
    description = "MOM_20 截面广度择时；广度达标时持有 Top-N 等权，否则清仓"
    required_factors = ("MOM_20",)

    @staticmethod
    def validate_params(params: dict) -> dict:
        """校验并返回规范化参数。"""
        unknown = set(params) - {"top_n", "min_breadth"}
        if unknown:
            raise SpecError(f"momentum_timing.params 包含未知字段：{sorted(unknown)}")
        top_n = params.get("top_n", 30)
        if type(top_n) is not int or top_n <= 0:
            raise SpecError("momentum_timing.params.top_n 必须是正整数")
        min_breadth = params.get("min_breadth", 0.5)
        if isinstance(min_breadth, bool) or not isinstance(min_breadth, (int, float)):
            raise SpecError("momentum_timing.params.min_breadth 必须是 [0, 1] 之间的数字")
        min_breadth = float(min_breadth)
        if not 0.0 <= min_breadth <= 1.0:
            raise SpecError("momentum_timing.params.min_breadth 必须在 [0, 1] 之间")
        return {"top_n": top_n, "min_breadth": min_breadth}

    def build(
        self,
        data: MarketData,
        eligible: pd.DataFrame,
        params: dict,
    ) -> pd.DataFrame:
        """返回每日目标权重；广度不足的交易日返回全 0 行。"""
        settings = self.validate_params(params)
        if "MOM_20" not in data.factors:
            raise DataAlignmentError(
                f"momentum_timing 需要因子 MOM_20；已加载：{sorted(data.factors)}"
            )
        score = data.factors["MOM_20"].astype("float64")
        valid = eligible & score.notna()
        counts = valid.sum(axis=1)
        breadth = (score.gt(0) & valid).sum(axis=1).div(counts.replace(0, np.nan))
        weights = pd.DataFrame(
            np.nan, index=data.index, columns=data.columns, dtype="float64"
        )
        for position, day in enumerate(data.index):
            weights.loc[day] = 0.0
            value = breadth.iloc[position]
            if not np.isfinite(value) or float(value) < settings["min_breadth"]:
                continue
            chosen = (
                score.loc[day]
                .where(valid.loc[day])
                .dropna()
                .sort_values(ascending=False)
                .head(settings["top_n"])
                .index
            )
            if len(chosen):
                weights.loc[day, chosen] = 1.0 / len(chosen)
        return weights


register_provider(MomentumTimingProvider())
