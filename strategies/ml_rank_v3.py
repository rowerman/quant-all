"""带生命周期状态切换的已落盘 ML 预测提供者。"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from alpha.errors import AlphaError
from alpha.evaluation_plan import current_model_ids
from alpha.lifecycle import (
    DEFAULT_FALLBACK_MODEL_ID,
    DEFAULT_PRIMARY_MODEL_ID,
    load_model_lifecycle,
)
from alpha.portfolio import smooth_rank_scores
from alpha.predict import load_predictions, model_artifact_snapshot
from backtest.data import MarketData
from backtest.errors import DataAlignmentError, SpecError
from backtest.providers import register_provider
from backtest.strategies._factor_rank import weekly_last_dates
from backtest.strategies.ml_rank import MlRankProvider
from backtest.strategies.ml_rank_v2 import _rebalance_dates

SMOOTHING_WEEKS = 4
SCHEMA_MODEL_ID, SCHEMA_FALLBACK_ID = current_model_ids()


class MlRankLifecycleProvider(MlRankProvider):
    """按生命周期状态使用主模型、Ridge 回滚或空仓目标。"""

    name = "ml_rank"
    version = "3"
    description = "已落盘预测的生命周期降权、回滚、暂停与四周 rank 平滑"
    required_factors: tuple[str, ...] = ()
    param_schema = (
        {
            "name": "lifecycle_id",
            "label": "生命周期 ID",
            "type": "string",
            "default": "p5_3_dd7c9b4bf235",
            "description": "P5-3 生成且通过完整哈希校验的生命周期包。",
        },
        {
            "name": "model_id",
            "label": "主模型 ID",
            "type": "string",
            "default": SCHEMA_MODEL_ID,
            "description": "生命周期 normal 和 reduced 状态使用的主模型。",
        },
        {
            "name": "fallback_model_id",
            "label": "回滚模型 ID",
            "type": "string",
            "default": SCHEMA_FALLBACK_ID,
            "description": "生命周期 rollback 状态使用的兜底模型。",
        },
        {
            "name": "top_n",
            "label": "持仓只数",
            "type": "integer",
            "default": 30,
            "min": 1,
            "step": 1,
            "description": "按活动模型四周平滑分数等权持有的股票数。",
        },
        {
            "name": "smoothing_weeks",
            "label": "平滑周数",
            "type": "integer",
            "default": SMOOTHING_WEEKS,
            "min": 1,
            "step": 1,
            "description": "活动模型分数的因果 rank 平滑周数。",
        },
    )

    def __init__(self) -> None:
        self._snapshot_cache: dict[tuple[str, str, str], dict] = {}

    @staticmethod
    def validate_params(params: dict) -> dict:
        """校验生命周期、模型、Top-N、平滑周数和调仓日。"""

        unknown = set(params) - {
            "lifecycle_id",
            "model_id",
            "fallback_model_id",
            "top_n",
            "smoothing_weeks",
            "rebalance_dates",
        }
        if unknown:
            raise SpecError(f"ml_rank.params 包含未知字段：{sorted(unknown)}")
        lifecycle_id = params.get("lifecycle_id")
        if not isinstance(lifecycle_id, str) or not lifecycle_id.strip():
            raise SpecError("ml_rank.params.lifecycle_id 必须是非空字符串")
        model_id = params.get("model_id")
        fallback_model_id = params.get("fallback_model_id")
        for name, value in (
            ("model_id", model_id),
            ("fallback_model_id", fallback_model_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise SpecError(f"ml_rank.params.{name} 必须是非空字符串")
        if model_id.strip() == fallback_model_id.strip():
            raise SpecError("ml_rank.params 的主模型与回滚模型不能相同")
        top_n = params.get("top_n", 30)
        if type(top_n) is not int or top_n <= 0:
            raise SpecError("ml_rank.params.top_n 必须是正整数")
        smoothing_weeks = params.get("smoothing_weeks", SMOOTHING_WEEKS)
        if type(smoothing_weeks) is not int or smoothing_weeks <= 0:
            raise SpecError("ml_rank.params.smoothing_weeks 必须是正整数")
        result = {
            "lifecycle_id": lifecycle_id.strip(),
            "model_id": model_id.strip(),
            "fallback_model_id": fallback_model_id.strip(),
            "top_n": top_n,
            "smoothing_weeks": smoothing_weeks,
        }
        dates = _rebalance_dates(params.get("rebalance_dates"))
        if dates is not None:
            result["rebalance_dates"] = list(dates)
        return result

    @staticmethod
    def _score_frame(
        predictions: pd.DataFrame,
        columns: pd.Index,
        smoothing_weeks: int,
    ) -> pd.DataFrame:
        """构造单个模型的因果 rank 平滑宽表。"""

        try:
            wide = predictions.pivot(
                index="trade_date",
                columns="stock_code",
                values="score",
            ).reindex(columns=columns)
        except (KeyError, ValueError) as error:
            raise DataAlignmentError("ml_rank v3 无法构造模型分数宽表") from error
        return smooth_rank_scores(
            wide.astype("float64"),
            wide.notna(),
            window=smoothing_weeks,
            min_periods=1,
        )

    def build(
        self,
        data: MarketData,
        eligible: pd.DataFrame,
        params: dict,
    ) -> pd.DataFrame:
        """返回生命周期驱动后的调仓日目标权重。"""

        settings = self.validate_params(params)
        try:
            lifecycle = load_model_lifecycle(
                settings["lifecycle_id"],
                model_id=settings["model_id"],
            )
            _, primary_manifest, primary_predictions = load_predictions(
                settings["model_id"]
            )
            _, fallback_manifest, fallback_predictions = load_predictions(
                settings["fallback_model_id"]
            )
        except AlphaError as error:
            raise DataAlignmentError(
                f"ml_rank v3 无法读取模型或生命周期产物：{error}"
            ) from error

        if lifecycle["manifest"]["sources"].get(
            "fallback_model_id"
        ) != settings["fallback_model_id"]:
            raise DataAlignmentError("ml_rank v3 的回滚模型与生命周期包不一致")
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
        lifecycle_frame = lifecycle["timeseries"].set_index("trade_date")
        lifecycle_dates = pd.DatetimeIndex(lifecycle_frame.index)
        if len(signal_dates.difference(lifecycle_dates)):
            raise DataAlignmentError("ml_rank v3 的生命周期包缺少信号日")
        primary_dates = pd.DatetimeIndex(
            sorted(pd.to_datetime(primary_predictions["trade_date"]).unique())
        )
        fallback_dates = pd.DatetimeIndex(
            sorted(pd.to_datetime(fallback_predictions["trade_date"]).unique())
        )
        for label, dates, manifest in (
            ("主模型", primary_dates, primary_manifest),
            ("回滚模型", fallback_dates, fallback_manifest),
        ):
            start = pd.Timestamp(manifest["prediction_start"])
            end = pd.Timestamp(manifest["prediction_end"])
            if signal_dates.min() < start or signal_dates.max() > end:
                raise DataAlignmentError(
                    f"ml_rank v3 {label}信号日超出预测覆盖："
                    f"{signal_dates.min().date()} ~ {signal_dates.max().date()}"
                )
            if len(signal_dates.difference(dates)):
                raise DataAlignmentError(f"ml_rank v3 {label}缺少信号日预测")

        columns = pd.Index(data.columns)
        primary_score = self._score_frame(
            primary_predictions, columns, settings["smoothing_weeks"]
        )
        fallback_score = self._score_frame(
            fallback_predictions, columns, settings["smoothing_weeks"]
        )
        for day in signal_dates:
            row = lifecycle_frame.loc[day]
            state = str(row["state"])
            multiplier = float(row["exposure_multiplier"])
            if state in {"normal", "reduced"}:
                active = settings["model_id"]
                score = primary_score
            elif state == "rollback":
                active = settings["fallback_model_id"]
                score = fallback_score
            elif state == "paused":
                active = None
                score = None
            else:
                raise DataAlignmentError(f"ml_rank v3 遇到未知生命周期状态：{state}")
            if row["active_model_id"] is not None and not pd.isna(
                row["active_model_id"]
            ):
                recorded = str(row["active_model_id"])
                if recorded != active:
                    raise DataAlignmentError("生命周期活动模型与状态不一致")
            weights.loc[day] = 0.0
            if score is None:
                continue
            if not np.isfinite(multiplier) or not 0.0 <= multiplier <= 1.0:
                raise DataAlignmentError("生命周期仓位乘数非法")
            if day not in score.index:
                raise DataAlignmentError(f"ml_rank v3 缺少 {day.date()} 的平滑分数")
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
                weights.loc[day, chosen] = multiplier / len(chosen)
        return weights

    def artifact_snapshot(self, params: dict) -> dict:
        """返回主模型、回滚模型和生命周期包的可复现摘要。"""

        settings = self.validate_params(params)
        key = (
            settings["lifecycle_id"],
            settings["model_id"],
            settings["fallback_model_id"],
        )
        if key in self._snapshot_cache:
            return copy.deepcopy(self._snapshot_cache[key])
        try:
            lifecycle = load_model_lifecycle(
                settings["lifecycle_id"],
                model_id=settings["model_id"],
            )
            primary = model_artifact_snapshot(settings["model_id"])
            fallback = model_artifact_snapshot(settings["fallback_model_id"])
        except AlphaError as error:
            raise DataAlignmentError(
                f"ml_rank v3 无法读取生命周期输入快照：{error}"
            ) from error
        manifest = lifecycle["manifest"]
        payload = {
            "schema_version": "ml_rank_lifecycle_artifacts_v1",
            "lifecycle_id": settings["lifecycle_id"],
            "lifecycle_manifest": manifest,
            "lifecycle_file_hashes": manifest.get("files") or {},
            "primary": primary,
            "fallback": fallback,
        }
        self._snapshot_cache[key] = copy.deepcopy(payload)
        return payload


register_provider(MlRankLifecycleProvider())
