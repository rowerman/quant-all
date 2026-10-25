"""带显式调仓日的已落盘 ML 预测提供者。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from alpha.errors import AlphaError
from alpha.evaluation_plan import current_model_ids
from alpha.portfolio import smooth_rank_scores
from alpha.predict import load_predictions, model_artifact_snapshot
from backtest.data import MarketData
from backtest.errors import DataAlignmentError, SpecError
from backtest.providers import register_provider
from backtest.strategies._factor_rank import weekly_last_dates
from backtest.strategies.ml_rank import MlRankProvider

SMOOTHING_WEEKS = 4
SCHEMA_MODEL_ID, _ = current_model_ids()


def _rebalance_dates(value) -> tuple[str, ...] | None:
    """校验显式调仓日并返回排序后的 ISO 日期。"""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise SpecError("ml_rank.params.rebalance_dates 必须是非空日期序列")
    normalized: list[str] = []
    for item in value:
        try:
            if not isinstance(item, str) or date.fromisoformat(item).isoformat() != item:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise SpecError(
                f"ml_rank.params.rebalance_dates 日期必须是 YYYY-MM-DD：{item!r}"
            ) from error
        normalized.append(item)
    return tuple(sorted(dict.fromkeys(normalized)))


class MlRankV2Provider(MlRankProvider):
    """读取模型预测，在显式调仓日或默认周最后交易日生成 Top-N 目标。"""

    name = "ml_rank"
    version = "2"
    description = "已落盘预测的 4 周 rank 平滑；可选显式调仓日"
    required_factors: tuple[str, ...] = ()
    param_schema = (
        {
            "name": "model_id",
            "label": "模型 ID",
            "type": "string",
            "default": SCHEMA_MODEL_ID,
            "description": "使用哪个已登记模型的样本外预测。",
        },
        {
            "name": "top_n",
            "label": "持仓只数",
            "type": "integer",
            "default": 30,
            "min": 1,
            "step": 1,
            "description": "4 周平滑后等权持有的股票数。",
        },
        {
            "name": "smoothing_weeks",
            "label": "平滑周数",
            "type": "integer",
            "default": SMOOTHING_WEEKS,
            "min": 1,
            "step": 1,
            "description": "使用 1 表示原始分数，使用 4 表示当前选中的四周 rank 平滑。",
        },
    )

    @staticmethod
    def validate_params(params: dict) -> dict:
        """校验模型、Top-N、平滑周数和可选调仓日。"""

        unknown = set(params) - {
            "model_id",
            "top_n",
            "smoothing_weeks",
            "rebalance_dates",
        }
        if unknown:
            raise SpecError(f"ml_rank.params 包含未知字段：{sorted(unknown)}")
        model_id = params.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise SpecError("ml_rank.params.model_id 必须是非空字符串")
        top_n = params.get("top_n", 30)
        if type(top_n) is not int or top_n <= 0:
            raise SpecError("ml_rank.params.top_n 必须是正整数")
        smoothing_weeks = params.get("smoothing_weeks", SMOOTHING_WEEKS)
        if type(smoothing_weeks) is not int or smoothing_weeks <= 0:
            raise SpecError("ml_rank.params.smoothing_weeks 必须是正整数")
        result = {
            "model_id": model_id.strip(),
            "top_n": top_n,
            "smoothing_weeks": smoothing_weeks,
        }
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
        try:
            _, manifest, predictions = load_predictions(settings["model_id"])
        except AlphaError as error:
            raise DataAlignmentError(
                f"ml_rank 无法读取模型预测 {settings['model_id']}：{error}"
            ) from error

        configured_dates = settings.get("rebalance_dates")
        signal_dates = (
            weekly_last_dates(data.index)
            if configured_dates is None
            else pd.DatetimeIndex(configured_dates).intersection(data.index)
        )
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
            window=settings["smoothing_weeks"],
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


register_provider(MlRankV2Provider())
