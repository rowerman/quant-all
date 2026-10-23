"""最终留出集冻结预测的隐式回测提供者。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.artifacts import file_sha256, prediction_sha256
from alpha.config import MODEL_DATA_DIR
from alpha.portfolio import smooth_rank_scores
from backtest.data import MarketData
from backtest.errors import DataAlignmentError, SpecError
from backtest.providers import register_provider
from backtest.strategies._factor_rank_v2 import _rebalance_dates

_PROVIDER: "FinalHoldoutRankProvider | None" = None
SCHEMA_VERSION = "alpha_final_holdout_predictions_v1"


def _read_json(path: Path, label: str) -> dict:
    """读取 JSON 对象。"""

    if not path.is_file():
        raise DataAlignmentError(f"{label}不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DataAlignmentError(f"无法读取{label}：{path}") from error
    if not isinstance(payload, dict):
        raise DataAlignmentError(f"{label}必须是 JSON 对象：{path}")
    return payload


class FinalHoldoutRankProvider:
    """读取最终留出集预测并按显式信号日输出 Top-N 目标。"""

    name = "final_holdout_rank"
    version = "1"
    description = "冻结留出预测的显式日期 Top-N；可供静态和生命周期候选使用"
    required_factors: tuple[str, ...] = ()
    hidden = True
    param_schema = ()

    @staticmethod
    def validate_params(params: dict) -> dict:
        """校验留出集、候选、平滑周数和显式调仓日。"""

        allowed = {
            "holdout_id",
            "candidate",
            "top_n",
            "smoothing_weeks",
            "rebalance_dates",
        }
        unknown = set(params) - allowed
        if unknown:
            raise SpecError(f"final_holdout_rank.params 包含未知字段：{sorted(unknown)}")
        holdout_id = params.get("holdout_id")
        if (
            not isinstance(holdout_id, str)
            or not holdout_id.startswith("final_holdout_")
            or Path(holdout_id).name != holdout_id
        ):
            raise SpecError("final_holdout_rank.params.holdout_id 非法")
        candidate = params.get("candidate")
        if candidate not in {"lightgbm", "ridge", "b4", "lifecycle"}:
            raise SpecError(
                "final_holdout_rank.params.candidate 必须是 "
                "lightgbm/ridge/b4/lifecycle"
            )
        top_n = params.get("top_n", 30)
        if type(top_n) is not int or top_n <= 0:
            raise SpecError("final_holdout_rank.params.top_n 必须是正整数")
        smoothing_weeks = params.get("smoothing_weeks", 4)
        if type(smoothing_weeks) is not int or smoothing_weeks <= 0:
            raise SpecError("final_holdout_rank.params.smoothing_weeks 必须是正整数")
        dates = _rebalance_dates(params.get("rebalance_dates"))
        if dates is None:
            raise SpecError("final_holdout_rank.params.rebalance_dates 不能为空")
        return {
            "holdout_id": holdout_id,
            "candidate": candidate,
            "top_n": top_n,
            "smoothing_weeks": smoothing_weeks,
            "rebalance_dates": list(dates),
        }

    @staticmethod
    def _root(holdout_id: str) -> Path:
        """返回留出集目录。"""

        return MODEL_DATA_DIR / "final_holdout" / holdout_id

    @staticmethod
    def _load_predictions(holdout_id: str, name: str) -> tuple[pd.DataFrame, dict]:
        """读取并校验一个留出预测文件。"""

        folder = FinalHoldoutRankProvider._root(holdout_id) / "predictions"
        manifest = _read_json(folder / "manifest.json", "留出预测 manifest")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise DataAlignmentError("留出预测 manifest schema 不匹配")
        record = (manifest.get("predictions") or {}).get(name)
        if not isinstance(record, dict):
            raise DataAlignmentError(f"留出预测 manifest 缺少 {name}")
        path = folder / f"{name}.parquet"
        if not path.is_file():
            raise DataAlignmentError(f"留出预测文件缺失：{path}")
        frame = pd.read_parquet(path)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["stock_code"] = frame["stock_code"].astype(str)
        if prediction_sha256(frame) != str(record.get("sha256")):
            raise DataAlignmentError(f"留出预测哈希不一致：{name}")
        return frame, manifest

    @staticmethod
    def _score(
        frame: pd.DataFrame,
        columns: pd.Index,
        smoothing_weeks: int,
    ) -> pd.DataFrame:
        """生成因果 rank 平滑宽表。"""

        wide = frame.pivot(
            index="trade_date",
            columns="stock_code",
            values="score",
        ).reindex(columns=columns)
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
        """返回显式留出信号日的目标权重。"""

        settings = self.validate_params(params)
        signal_dates = pd.DatetimeIndex(settings["rebalance_dates"]).intersection(
            data.index
        )
        if signal_dates.empty:
            raise DataAlignmentError("final_holdout_rank 没有可用信号日")
        columns = pd.Index(data.columns)
        candidate = settings["candidate"]
        lifecycle = None
        if candidate == "lifecycle":
            lifecycle_path = (
                self._root(settings["holdout_id"])
                / "lifecycle"
                / "timeseries.csv"
            )
            if not lifecycle_path.is_file():
                raise DataAlignmentError(f"留出生命周期文件缺失：{lifecycle_path}")
            lifecycle = pd.read_csv(lifecycle_path)
            lifecycle["trade_date"] = pd.to_datetime(lifecycle["trade_date"])
            lifecycle = lifecycle.set_index("trade_date")
            if len(signal_dates.difference(lifecycle.index)):
                raise DataAlignmentError("留出生命周期缺少信号日")
            primary, _ = self._load_predictions(
                settings["holdout_id"], "lightgbm"
            )
            fallback, _ = self._load_predictions(
                settings["holdout_id"], "ridge"
            )
            primary_score = self._score(
                primary,
                columns,
                settings["smoothing_weeks"],
            )
            fallback_score = self._score(
                fallback,
                columns,
                settings["smoothing_weeks"],
            )
        else:
            prediction, _ = self._load_predictions(
                settings["holdout_id"], candidate
            )
            score = self._score(
                prediction,
                columns,
                settings["smoothing_weeks"],
            )

        weights = pd.DataFrame(
            np.nan,
            index=data.index,
            columns=data.columns,
            dtype="float64",
        )
        for day in signal_dates:
            multiplier = 1.0
            if candidate == "lifecycle":
                row = lifecycle.loc[day]
                state = str(row["state"])
                multiplier = float(row["exposure_multiplier"])
                if state in {"normal", "reduced"}:
                    active_score = primary_score
                elif state == "rollback":
                    active_score = fallback_score
                elif state == "paused":
                    weights.loc[day] = 0.0
                    continue
                else:
                    raise DataAlignmentError(
                        f"留出生命周期未知状态：{state}"
                    )
            else:
                active_score = score
            if day not in active_score.index:
                raise DataAlignmentError(
                    f"留出候选 {candidate} 缺少 {day.date()} 分数"
                )
            if not np.isfinite(multiplier) or not 0.0 <= multiplier <= 1.0:
                raise DataAlignmentError("留出生命周期仓位乘数非法")
            pool = (
                eligible.loc[day]
                .reindex(active_score.columns, fill_value=False)
                .astype(bool)
            )
            chosen = (
                active_score.loc[day]
                .where(pool)
                .dropna()
                .sort_values(ascending=False, kind="mergesort")
                .head(settings["top_n"])
                .index
            )
            weights.loc[day] = 0.0
            if len(chosen):
                weights.loc[day, chosen] = multiplier / len(chosen)
        return weights

    def artifact_snapshot(self, params: dict) -> dict:
        """返回留出预测和生命周期的不可变摘要。"""

        settings = self.validate_params(params)
        root = self._root(settings["holdout_id"])
        prediction_manifest_path = root / "predictions" / "manifest.json"
        prediction_manifest = _read_json(
            prediction_manifest_path,
            "留出预测 manifest",
        )
        payload = {
            "schema_version": "final_holdout_provider_artifacts_v1",
            "holdout_id": settings["holdout_id"],
            "candidate": settings["candidate"],
            "prediction_manifest": prediction_manifest,
            "prediction_manifest_sha256": file_sha256(
                prediction_manifest_path
            ),
        }
        if settings["candidate"] == "lifecycle":
            lifecycle_path = root / "lifecycle" / "timeseries.csv"
            if not lifecycle_path.is_file():
                raise DataAlignmentError("留出生命周期文件缺失")
            payload["lifecycle_timeseries_sha256"] = file_sha256(
                lifecycle_path
            )
        return payload


def ensure_registered() -> FinalHoldoutRankProvider:
    """显式注册隐藏的留出集 provider。"""

    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = register_provider(FinalHoldoutRankProvider())
    return _PROVIDER


__all__ = ["FinalHoldoutRankProvider", "SCHEMA_VERSION", "ensure_registered"]
