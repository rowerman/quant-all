"""LightGBM 训练、RankIC 早停与原生模型读写。"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from alpha.errors import AlphaModelError
from alpha.preprocess import PreprocessConfig


@dataclass(frozen=True)
class GbdtConfig:
    """一次 LightGBM 训练和产物生成的固定配置。"""

    target: str = "label_rank"
    top_n: int = 30
    quantiles: int = 10
    seed: int = 0
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = 6
    min_data_in_leaf: int = 1000
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 20.0
    max_rounds: int = 1000
    early_stopping_rounds: int = 100
    num_threads: int = 4
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

    def __post_init__(self) -> None:
        if self.target != "label_rank":
            raise AlphaModelError("LightGBM 目标固定为 label_rank")
        if type(self.top_n) is not int or self.top_n <= 0:
            raise AlphaModelError("top_n 必须是正整数")
        if type(self.quantiles) is not int or self.quantiles < 2:
            raise AlphaModelError("quantiles 必须是至少为 2 的整数")
        if type(self.seed) is not int or self.seed < 0:
            raise AlphaModelError("seed 必须是非负整数")
        for name in (
            "num_leaves",
            "max_depth",
            "min_data_in_leaf",
            "bagging_freq",
            "max_rounds",
            "early_stopping_rounds",
            "num_threads",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise AlphaModelError(f"{name} 必须是正整数")
        for name in ("learning_rate", "feature_fraction", "bagging_fraction", "lambda_l2"):
            value = getattr(self, name)
            if not np.isfinite(value) or float(value) <= 0:
                raise AlphaModelError(f"{name} 必须是正有限数")
        if float(self.learning_rate) > 1.0:
            raise AlphaModelError("learning_rate 不能大于 1")
        for name in ("feature_fraction", "bagging_fraction"):
            if not 0.0 < float(getattr(self, name)) <= 1.0:
                raise AlphaModelError(f"{name} 必须在 0 到 1 之间")

    def to_dict(self) -> dict:
        """转换为稳定字典。"""

        return asdict(self)


@dataclass(frozen=True)
class FittedGbdt:
    """一折完成早停的 LightGBM 模型。"""

    booster: lgb.Booster
    best_iteration: int
    validation_rank_ic: float


def _validate_arrays(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """校验训练矩阵、标签和样本权重。"""

    matrix = np.asarray(X, dtype=np.float32)
    target = np.asarray(y, dtype=np.float32)
    weights = np.asarray(sample_weight, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise AlphaModelError("训练特征必须是非空二维矩阵")
    if target.shape != (matrix.shape[0],) or weights.shape != (matrix.shape[0],):
        raise AlphaModelError("标签或样本权重长度与特征行数不一致")
    if not np.isfinite(matrix).all():
        raise AlphaModelError("训练特征包含非有限值")
    if not np.isfinite(target).all():
        raise AlphaModelError("训练标签包含非有限值")
    if not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
        raise AlphaModelError("样本权重必须有限、非负且至少一个为正")
    return matrix, target, weights


def _params(config: GbdtConfig) -> dict:
    """转换为 LightGBM 参数，默认 metric 关闭以只按 RankIC 早停。"""

    return {
        "objective": "regression",
        "metric": "None",
        "learning_rate": float(config.learning_rate),
        "num_leaves": int(config.num_leaves),
        "max_depth": int(config.max_depth),
        "min_data_in_leaf": int(config.min_data_in_leaf),
        "feature_fraction": float(config.feature_fraction),
        "bagging_fraction": float(config.bagging_fraction),
        "bagging_freq": int(config.bagging_freq),
        "lambda_l2": float(config.lambda_l2),
        "seed": int(config.seed),
        "num_threads": int(config.num_threads),
        "verbosity": -1,
        "deterministic": True,
        "force_row_wise": True,
    }


def _date_slices(dates: np.ndarray) -> np.ndarray:
    """返回按日期连续分组后的切片边界。"""

    stamps = pd.to_datetime(dates).to_numpy()
    if len(stamps) == 0:
        raise AlphaModelError("验证集日期不能为空")
    if np.any(stamps[1:] < stamps[:-1]):
        raise AlphaModelError("验证集必须按交易日排序")
    return np.r_[0, np.flatnonzero(stamps[1:] != stamps[:-1]) + 1, len(stamps)]


def _rank_ic_metric(
    predictions: np.ndarray,
    labels: np.ndarray,
    boundaries: np.ndarray,
) -> float:
    """计算逐日截面 RankIC 均值。"""

    scores = np.asarray(predictions, dtype="float64")
    if scores.ndim != 1 or scores.shape != labels.shape:
        raise AlphaModelError("验证集预测长度与标签不一致")
    daily: list[float] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        left = rankdata(scores[start:end])
        right = rankdata(labels[start:end])
        left = left - left.mean()
        right = right - right.mean()
        denominator = float(np.sqrt((left @ left) * (right @ right)))
        if denominator > 0:
            daily.append(float(left @ right / denominator))
    if not daily:
        raise AlphaModelError("验证集 RankIC 无定义")
    value = float(np.mean(daily))
    if not np.isfinite(value):
        raise AlphaModelError("验证集 RankIC 不是有限值")
    return value


def fit_gbdt_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    weight_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    weight_valid: np.ndarray,
    valid_dates: np.ndarray,
    features: tuple[str, ...],
    config: GbdtConfig,
) -> FittedGbdt:
    """训练单折模型并按验证集 RankIC 早停。"""

    train_x, train_y, train_weight = _validate_arrays(
        X_train, y_train, weight_train
    )
    valid_x, valid_y, valid_weight = _validate_arrays(
        X_valid, y_valid, weight_valid
    )
    if train_x.shape[1] != valid_x.shape[1] or train_x.shape[1] != len(features):
        raise AlphaModelError("训练、验证特征数与特征清单不一致")
    boundaries = _date_slices(np.asarray(valid_dates))
    labels = np.asarray(valid_y, dtype="float64")

    def metric(predictions: np.ndarray, _dataset: lgb.Dataset) -> tuple[str, float, bool]:
        value = _rank_ic_metric(predictions, labels, boundaries)
        return "valid_rank_ic", value, True

    train_set = lgb.Dataset(
        train_x,
        label=train_y,
        weight=train_weight,
        feature_name=list(features),
    )
    valid_set = lgb.Dataset(
        valid_x,
        label=valid_y,
        weight=valid_weight,
        reference=train_set,
        feature_name=list(features),
    )
    try:
        booster = lgb.train(
            _params(config),
            train_set,
            num_boost_round=int(config.max_rounds),
            valid_sets=[valid_set],
            feval=metric,
            callbacks=[
                lgb.early_stopping(
                    int(config.early_stopping_rounds),
                    first_metric_only=True,
                    verbose=False,
                ),
                lgb.log_evaluation(period=0),
            ],
        )
    except Exception as error:
        raise AlphaModelError(f"LightGBM 早停训练失败：{error}") from error
    best_iteration = int(booster.best_iteration)
    if best_iteration <= 0:
        raise AlphaModelError("LightGBM 没有可用的早停轮次")
    validation = float(booster.best_score["valid_0"]["valid_rank_ic"])
    if not np.isfinite(validation):
        raise AlphaModelError("LightGBM 验证集 RankIC 不是有限值")
    return FittedGbdt(
        booster=booster,
        best_iteration=best_iteration,
        validation_rank_ic=validation,
    )


def refit_gbdt(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    features: tuple[str, ...],
    config: GbdtConfig,
    rounds: int,
) -> lgb.Booster:
    """按早停轮数在 train+valid 上重拟合。"""

    matrix, target, weights = _validate_arrays(X, y, sample_weight)
    if matrix.shape[1] != len(features):
        raise AlphaModelError("重拟合特征数与特征清单不一致")
    if type(rounds) is not int or rounds <= 0:
        raise AlphaModelError("重拟合轮数必须是正整数")
    dataset = lgb.Dataset(
        matrix,
        label=target,
        weight=weights,
        feature_name=list(features),
    )
    try:
        return lgb.train(
            _params(config),
            dataset,
            num_boost_round=rounds,
        )
    except Exception as error:
        raise AlphaModelError(f"LightGBM 重拟合失败：{error}") from error


def predict_gbdt(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """生成 LightGBM 分数并校验输出。"""

    matrix = np.asarray(X, dtype="float32")
    if matrix.ndim != 2 or matrix.shape[1] != booster.num_feature():
        raise AlphaModelError(
            f"预测特征形状 {matrix.shape} 与模型特征数 {booster.num_feature()} 不一致"
        )
    if not np.isfinite(matrix).all():
        raise AlphaModelError("预测特征包含非有限值")
    try:
        scores = np.asarray(booster.predict(matrix), dtype="float64")
    except Exception as error:
        raise AlphaModelError(f"LightGBM 预测失败：{error}") from error
    if scores.shape != (matrix.shape[0],) or not np.isfinite(scores).all():
        raise AlphaModelError("LightGBM 输出形状异常或包含非有限值")
    return scores


def save_gbdt_model(
    booster: lgb.Booster,
    path: Path,
    *,
    num_iteration: int | None = None,
) -> None:
    """原子保存 LightGBM 原生文本模型。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.write_text(
            booster.model_to_string(num_iteration=num_iteration),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_gbdt_model(path: Path) -> lgb.Booster:
    """读取 LightGBM 原生文本模型。"""

    if not path.is_file():
        raise AlphaModelError(f"LightGBM 模型文件不存在：{path}")
    try:
        return lgb.Booster(model_file=str(path))
    except Exception as error:
        raise AlphaModelError(f"LightGBM 模型文件损坏或格式不兼容：{path}") from error
