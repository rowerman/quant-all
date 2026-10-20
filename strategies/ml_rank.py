"""已落盘 ML 预测的周频 Top-N 目标权重提供者。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.errors import AlphaError
from alpha.portfolio import smooth_rank_scores
from alpha.predict import load_predictions
from backtest.data import MarketData
from backtest.errors import DataAlignmentError, SpecError
from backtest.providers import register_provider
from backtest.strategies._factor_rank import weekly_last_dates

SMOOTHING_WEEKS = 4


class MlRankProvider:
    """读取模型预测，按 4 周 rank 平滑选 Top-N 等权持有。"""

    name = "ml_rank"
    version = "1"
    description = "已落盘模型预测的 4 周 rank 平滑；周最后交易日 Top-N 等权"
    required_factors: tuple[str, ...] = ()

    @staticmethod
    def validate_params(params: dict) -> dict:
        """校验并返回规范化参数。"""

        unknown = set(params) - {"model_id", "top_n"}
        if unknown:
            raise SpecError(f"ml_rank.params 包含未知字段：{sorted(unknown)}")
        model_id = params.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise SpecError("ml_rank.params.model_id 必须是非空字符串")
        top_n = params.get("top_n", 30)
        if type(top_n) is not int or top_n <= 0:
            raise SpecError("ml_rank.params.top_n 必须是正整数")
        return {"model_id": model_id.strip(), "top_n": top_n}

    def build(
        self,
        data: MarketData,
        eligible: pd.DataFrame,
        params: dict,
    ) -> pd.DataFrame:
        """返回周最后交易日的目标权重；其他日期保持 NaN。"""

        settings = self.validate_params(params)
        try:
            _, manifest, predictions = load_predictions(settings["model_id"])
        except AlphaError as error:
            raise DataAlignmentError(
                f"ml_rank 无法读取模型预测 {settings['model_id']}：{error}"
            ) from error

        signal_dates = weekly_last_dates(data.index)
        weights = pd.DataFrame(
            np.nan,
            index=data.index,
            columns=data.columns,
            dtype="float64",
        )
        if signal_dates.empty:
            return weights
        prediction_start = pd.Timestamp(manifest["prediction_start"])
        prediction_end = pd.Timestamp(manifest["prediction_end"])
        if (
            signal_dates.min() < prediction_start
            or signal_dates.max() > prediction_end
        ):
            raise DataAlignmentError(
                "ml_rank 回测信号日超出模型预测覆盖："
                f"{signal_dates.min().date()} ~ {signal_dates.max().date()}，"
                f"预测覆盖 {prediction_start.date()} ~ {prediction_end.date()}"
            )
        wide = predictions.pivot(
            index="trade_date",
            columns="stock_code",
            values="score",
        ).astype("float64")
        missing = signal_dates.difference(wide.index)
        if not missing.empty:
            raise DataAlignmentError(
                "ml_rank 缺少回测信号日预测："
                + ", ".join(str(day.date()) for day in missing[:5])
            )
        score = smooth_rank_scores(
            wide,
            wide.notna(),
            window=SMOOTHING_WEEKS,
            min_periods=1,
        )
        for day in signal_dates:
            weights.loc[day] = 0.0
            pool = (
                eligible.loc[day]
                .reindex(score.columns, fill_value=False)
                .astype(bool)
            )
            chosen = (
                score.loc[day]
                .where(pool)
                .dropna()
                .sort_values(ascending=False, kind="mergesort")
                .head(settings["top_n"])
                .index
            )
            if len(chosen):
                weights.loc[day, chosen] = 1.0 / len(chosen)
        return weights


register_provider(MlRankProvider())
