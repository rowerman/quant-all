"""最终留出集的数据构建、冻结模型评分与产物读取。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import factors
from alpha.artifacts import prediction_sha256
from alpha.calendar import load_calendar
from alpha.config import DEFAULT_CONTRACT, MODEL_DATA_DIR
from alpha.dataset import DatasetArtifact, DatasetSpec, build_dataset, load_dataset
from alpha.evaluation_plan import EvaluationPlan
from alpha.errors import AlphaFinalHoldoutError
from alpha.holdout import (
    DEFAULT_HOLDOUT_WINDOWS_PATH,
    FUTURE_HOLDOUT_WINDOW_ID,
    HoldoutWindow,
    get_holdout_window,
)
from alpha.modeling.forecast import predict_model_rows
from alpha.predict import load_model_manifest
from alpha.preprocess import PreprocessConfig, transform
from common.paths import FACTOR_DIR

FINAL_HOLDOUT_SCHEMA_VERSION = "alpha_final_holdout_v1"
PREDICTION_NAMES = ("lightgbm", "ridge", "b4")


@dataclass(frozen=True)
class FinalHoldoutSpec:
    """一次性最终留出验证的窗口和运行口径。"""

    window_id: str = FUTURE_HOLDOUT_WINDOW_ID
    window_registry_path: str = str(DEFAULT_HOLDOUT_WINDOWS_PATH)
    plan_path: str | None = None
    top_n: int = DEFAULT_CONTRACT.top_n
    init_cash: float = 1_000_000.0
    risk_free_rate: float = 0.02
    max_participation_rate: float = 0.1
    min_label_coverage: float | None = None
    min_signal_days: int | None = None

    def __post_init__(self) -> None:
        window = self.window
        if type(self.top_n) is not int or self.top_n <= 0:
            raise AlphaFinalHoldoutError("top_n 必须是正整数")
        if (
            isinstance(self.init_cash, bool)
            or not isinstance(self.init_cash, (int, float))
            or float(self.init_cash) <= 0
        ):
            raise AlphaFinalHoldoutError("init_cash 必须是正数")
        if not -0.99 <= float(self.risk_free_rate) <= 1.0:
            raise AlphaFinalHoldoutError("risk_free_rate 必须在 [-0.99, 1]")
        if not 0.0 < float(self.max_participation_rate) <= 1.0:
            raise AlphaFinalHoldoutError("max_participation_rate 必须在 (0, 1]")
        if not 0.0 < float(self.label_coverage) <= 1.0:
            raise AlphaFinalHoldoutError("min_label_coverage 必须在 (0, 1]")
        if type(self.signal_days) is not int or self.signal_days <= 0:
            raise AlphaFinalHoldoutError("min_signal_days 必须是正整数")
        if self.plan_path is not None and not Path(self.plan_path).is_file():
            raise AlphaFinalHoldoutError(f"评估计划不存在：{self.plan_path}")
        if not window.window_id:
            raise AlphaFinalHoldoutError("留出窗口 ID 不能为空")

    def to_dict(self) -> dict:
        """返回稳定字典。"""

        return {
            "window_id": self.window_id,
            "plan_path": self.plan_path,
            "top_n": self.top_n,
            "init_cash": self.init_cash,
            "risk_free_rate": self.risk_free_rate,
            "max_participation_rate": self.max_participation_rate,
            "min_label_coverage": self.label_coverage,
            "min_signal_days": self.signal_days,
        }

    @property
    def window(self) -> HoldoutWindow:
        """返回注册的留出窗口。"""

        return get_holdout_window(
            self.window_id,
            Path(self.window_registry_path),
        )

    @property
    def start(self) -> str:
        """返回窗口起始日。"""

        return self.window.start

    @property
    def end(self) -> str:
        """返回窗口截止日。"""

        return self.window.end

    @property
    def label_coverage(self) -> float:
        """返回标签覆盖门槛。"""

        return (
            self.window.min_label_coverage
            if self.min_label_coverage is None
            else float(self.min_label_coverage)
        )

    @property
    def signal_days(self) -> int:
        """返回最少完整标签信号日。"""

        return (
            self.window.min_signal_days
            if self.min_signal_days is None
            else int(self.min_signal_days)
        )

    @property
    def dataset_name(self) -> str:
        """返回窗口对应数据集名称。"""

        suffix = self.window_id.removeprefix("holdout_")
        return f"weekly_final_holdout_{suffix}_v1"

    @property
    def evaluation_plan(self) -> EvaluationPlan:
        """读取本规格引用的评估计划。"""

        if not self.plan_path:
            raise AlphaFinalHoldoutError("留出验证尚未绑定评估计划")
        return EvaluationPlan.from_file(Path(self.plan_path))

    @property
    def primary_model_id(self) -> str:
        plan = self.evaluation_plan
        name = plan.selected or plan.candidates[0].name
        return plan.by_name(name).source.id

    @property
    def ridge_model_id(self) -> str:
        return self.evaluation_plan.by_name(self.evaluation_plan.reference).source.id

    @property
    def frozen_fold_id(self) -> str:
        fold_id = self.evaluation_plan.frozen_fold_id
        if not fold_id:
            raise AlphaFinalHoldoutError("评估计划缺少 frozen_fold_id")
        return fold_id


def build_final_holdout_dataset(
    spec: FinalHoldoutSpec,
    *,
    force: bool = False,
) -> DatasetArtifact:
    """构建指定留出窗口的周频数据集。"""

    calendar = load_calendar(spec.start, spec.end)
    if calendar.empty:
        raise AlphaFinalHoldoutError(
            f"留出窗口 {spec.window_id} 没有可用行情："
            f"{spec.start}～{spec.end}"
        )
    return build_dataset(
        DatasetSpec(
            name=spec.dataset_name,
            start=spec.start,
            end=spec.end,
            allow_holdout=True,
        ),
        force=force,
    )


def complete_signal_dates(
    data: pd.DataFrame,
    spec: FinalHoldoutSpec,
) -> pd.DatetimeIndex:
    """返回 5 日标签覆盖达到门槛的周频信号日。"""

    required = {"trade_date", "eligible", "label_raw"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise AlphaFinalHoldoutError(f"留出集缺少列：{missing}")
    work = data.loc[:, list(required)].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["eligible"] = work["eligible"].fillna(False).astype(bool)
    eligible_count = work.loc[work["eligible"]].groupby("trade_date").size()
    label_count = (
        work.loc[work["eligible"] & work["label_raw"].notna()]
        .groupby("trade_date")
        .size()
        .reindex(eligible_count.index, fill_value=0)
    )
    coverage = label_count.div(eligible_count.replace(0, np.nan))
    selected = coverage[
        (coverage >= float(spec.label_coverage)) & (eligible_count >= 100)
    ].index
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(selected)))
    if len(dates) < spec.signal_days:
        raise AlphaFinalHoldoutError(
            f"完整标签信号日不足：{len(dates)}；"
            f"窗口 {spec.window_id} 要求至少 {spec.signal_days}"
        )
    return dates


def _preprocess_holdout(
    data: pd.DataFrame,
    manifest: dict,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """按冻结模型的预处理参数处理留出信号日。"""

    work = data.loc[
        pd.to_datetime(data["trade_date"]).isin(dates)
    ].copy()
    settings = manifest.get("preprocess") or {}
    config = PreprocessConfig(
        method=str(settings["method"]),
        missing_indicator=bool(settings["missing_indicator"]),
        winsorize=bool(settings["winsorize"]),
        neutralize=tuple(settings["neutralize"]),
        clip_z=float(settings["clip_z"]),
    )
    preprocessed = transform(
        work,
        config,
        feature_columns=tuple(manifest["raw_features"])
        + tuple(manifest["context_features"]),
    )
    features = tuple(manifest["features"])
    if preprocessed.feature_names != features:
        raise AlphaFinalHoldoutError("留出预处理特征与冻结模型 manifest 不一致")
    return preprocessed.data, features


def _frozen_fold(manifest: dict, fold_id: str) -> dict:
    """取得最后开发和重拟合折。"""

    matches = [
        fold
        for fold in manifest.get("folds") or []
        if fold.get("fold_id") == fold_id
    ]
    if len(matches) != 1:
        raise AlphaFinalHoldoutError(
            f"模型 {manifest.get('model_id')} 缺少唯一的 {fold_id} 折"
        )
    return matches[0]


def _prediction_frame(rows: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """组装标准预测表。"""

    result = rows.loc[:, ["trade_date", "stock_code"]].copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result["stock_code"] = result["stock_code"].astype(str)
    result["score"] = np.asarray(scores, dtype="float64")
    result = result.sort_values(
        ["trade_date", "stock_code"], kind="mergesort"
    ).reset_index(drop=True)
    if result.duplicated(["trade_date", "stock_code"]).any():
        raise AlphaFinalHoldoutError("留出预测存在重复日期和股票")
    if not np.isfinite(result["score"].to_numpy(dtype="float64")).all():
        raise AlphaFinalHoldoutError("留出预测包含非有限分数")
    return result


def _score_model(
    model_id: str,
    rows: pd.DataFrame,
    features: tuple[str, ...],
    root: Path,
    fold_id: str,
) -> pd.DataFrame:
    """通过通用后端为冻结折生成留出分数。"""

    try:
        return predict_model_rows(
            model_id,
            rows,
            features,
            fold_id=fold_id,
            root=root,
        )
    except Exception as error:
        raise AlphaFinalHoldoutError(
            f"模型 {model_id} 冻结折预测失败：{error}"
        ) from error


def _score_b4(
    data: pd.DataFrame,
    dates: pd.DatetimeIndex,
    factor_names: tuple[str, ...],
    factor_dir: Path | str = FACTOR_DIR,
) -> pd.DataFrame:
    """按 P0 口径生成全因子方向调整 rank 基线。"""

    required = {"trade_date", "stock_code", "eligible", *factor_names}
    missing = sorted(required - set(data.columns))
    if missing:
        raise AlphaFinalHoldoutError(f"B4 留出数据缺少列：{missing}")
    work = data.loc[
        pd.to_datetime(data["trade_date"]).isin(dates)
        & data["eligible"].fillna(False).astype(bool)
    ].copy()
    if work.empty:
        raise AlphaFinalHoldoutError("B4 留出数据没有可交易股票")
    total = pd.Series(0.0, index=work.index)
    count = pd.Series(0.0, index=work.index)
    dates_series = pd.to_datetime(work["trade_date"])
    for name in factor_names:
        values = pd.to_numeric(work[name], errors="coerce")
        rank = values.groupby(dates_series).rank(pct=True, method="average")
        direction = int(factors.resolve(name, factor_dir).direction)
        total = total.add(rank * direction, fill_value=0.0)
        count = count.add(rank.notna().astype("float64"), fill_value=0.0)
    frame = work.loc[:, ["trade_date", "stock_code"]].copy()
    frame["score"] = total.div(count.replace(0.0, np.nan))
    frame = frame.dropna(subset=["score"])
    return _prediction_frame(frame, frame["score"].to_numpy())


def score_frozen_holdout(
    data: pd.DataFrame,
    dataset_manifest: dict,
    dates: pd.DatetimeIndex,
    spec: FinalHoldoutSpec,
    *,
    root: Path | None = None,
    factor_dir: Path | str = FACTOR_DIR,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """生成主模型、Ridge 和 B4 的冻结留出预测。"""

    output_root = Path(root or MODEL_DATA_DIR)
    plan = spec.evaluation_plan
    selected_name = plan.selected or plan.candidates[0].name
    primary_model_id = plan.by_name(selected_name).source.id
    reference_model_id = plan.by_name(plan.reference).source.id
    primary_manifest = load_model_manifest(primary_model_id, output_root)
    ridge_manifest = load_model_manifest(reference_model_id, output_root)
    _frozen_fold(primary_manifest, spec.frozen_fold_id)
    _frozen_fold(ridge_manifest, spec.frozen_fold_id)
    if primary_manifest.get("dataset_id") != ridge_manifest.get("dataset_id"):
        raise AlphaFinalHoldoutError("主模型与 Ridge 的训练数据集不一致")
    if list(primary_manifest.get("raw_features") or []) != list(
        ridge_manifest.get("raw_features") or []
    ) or list(primary_manifest.get("context_features") or []) != list(
        ridge_manifest.get("context_features") or []
    ):
        raise AlphaFinalHoldoutError("主模型与 Ridge 的原始特征不一致")

    preprocessed, features = _preprocess_holdout(data, primary_manifest, dates)
    rows = preprocessed.loc[preprocessed["eligible"].astype(bool)].copy()
    if rows.empty:
        raise AlphaFinalHoldoutError("留出预处理后没有可交易股票")
    predictions = {
        "lightgbm": _score_model(
            primary_model_id,
            rows,
            features,
            output_root,
            spec.frozen_fold_id,
        ),
        "ridge": _score_model(
            reference_model_id,
            rows,
            features,
            output_root,
            spec.frozen_fold_id,
        ),
    }
    factor_names = tuple(dataset_manifest.get("features") or [])
    predictions["b4"] = _score_b4(
        data,
        dates,
        factor_names,
        factor_dir,
    )
    metadata = {
        "frozen_fold_id": spec.frozen_fold_id,
        "primary_model_id": primary_model_id,
        "ridge_model_id": reference_model_id,
        "factor_count": len(factor_names),
        "feature_count": len(features),
    }
    return predictions, metadata


def load_holdout_dataset(dataset_id: str) -> tuple[pd.DataFrame, dict]:
    """读取留出数据集并确认允许范围。"""

    data, manifest = load_dataset(dataset_id)
    spec = manifest.get("spec") or {}
    if spec.get("allow_holdout") is not True:
        raise AlphaFinalHoldoutError("数据集不是显式留出集")
    return data, manifest


def prediction_manifest(
    predictions: dict[str, pd.DataFrame],
) -> dict:
    """生成预测文件与内容哈希。"""

    records = {}
    for name in PREDICTION_NAMES:
        frame = predictions[name]
        records[name] = {
            "sha256": prediction_sha256(frame),
            "rows": int(len(frame)),
            "dates": int(frame["trade_date"].nunique()),
            "start": str(frame["trade_date"].min().date()),
            "end": str(frame["trade_date"].max().date()),
        }
    return records


def read_final_holdout_json(path: Path, label: str) -> dict:
    """读取最终留出集 JSON 对象。"""

    if not path.is_file():
        raise AlphaFinalHoldoutError(f"{label}不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AlphaFinalHoldoutError(f"无法读取{label}：{path}") from error
    if not isinstance(payload, dict):
        raise AlphaFinalHoldoutError(f"{label}必须是 JSON 对象：{path}")
    return payload


__all__ = [
    "FINAL_HOLDOUT_SCHEMA_VERSION",
    "FinalHoldoutSpec",
    "PREDICTION_NAMES",
    "build_final_holdout_dataset",
    "complete_signal_dates",
    "load_holdout_dataset",
    "prediction_manifest",
    "read_final_holdout_json",
    "score_frozen_holdout",
]
