"""可复现的线性模型拟合、预测与数组化落盘。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

from alpha.errors import AlphaModelError

MODEL_TYPES = ("ridge", "elastic_net")


@dataclass(frozen=True)
class LinearModel:
    """一个已完成标准化的线性模型。"""

    model_type: str
    alpha: float
    l1_ratio: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coef: np.ndarray
    intercept: float

    @property
    def feature_count(self) -> int:
        """返回模型特征数。"""

        return int(self.coef.size)


def _validate_arrays(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """校验训练矩阵形状、有限性和样本权重。"""

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


def fit_linear(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    model_type: str,
    alpha: float,
    l1_ratio: float = 0.1,
    seed: int = 0,
) -> LinearModel:
    """拟合 Ridge 或 ElasticNet，并保存标准化参数。"""

    matrix, target, weights = _validate_arrays(X, y, sample_weight)
    if model_type not in MODEL_TYPES:
        raise AlphaModelError(f"未知线性模型：{model_type}")
    if not np.isfinite(alpha) or alpha < 0:
        raise AlphaModelError("alpha 必须是非负有限数")
    if model_type == "elastic_net" and not (0.0 <= l1_ratio <= 1.0):
        raise AlphaModelError("l1_ratio 必须在 0 到 1 之间")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix, sample_weight=weights)
    if model_type == "ridge":
        estimator = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-5, max_iter=3000)
    else:
        estimator = ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            tol=1e-4,
            max_iter=3000,
            random_state=int(seed),
        )
    try:
        estimator.fit(scaled, target, sample_weight=weights)
    except Exception as error:
        raise AlphaModelError(f"{model_type} 拟合失败：{error}") from error
    coef = np.asarray(estimator.coef_, dtype="float64")
    if coef.shape != (matrix.shape[1],) or not np.isfinite(coef).all():
        raise AlphaModelError("模型系数形状异常或包含非有限值")
    return LinearModel(
        model_type=model_type,
        alpha=float(alpha),
        l1_ratio=float(l1_ratio) if model_type == "elastic_net" else 0.0,
        feature_mean=np.asarray(scaler.mean_, dtype="float64"),
        feature_scale=np.asarray(scaler.scale_, dtype="float64"),
        coef=coef,
        intercept=float(estimator.intercept_),
    )


def predict_linear(model: LinearModel, X: np.ndarray) -> np.ndarray:
    """按保存的标准化参数计算线性分数。"""

    matrix = np.asarray(X, dtype="float64")
    if matrix.ndim != 2 or matrix.shape[1] != model.feature_count:
        raise AlphaModelError(
            f"预测特征形状 {matrix.shape} 与模型特征数 {model.feature_count} 不一致"
        )
    if not np.isfinite(matrix).all():
        raise AlphaModelError("预测特征包含非有限值")
    scale = np.where(model.feature_scale == 0.0, 1.0, model.feature_scale)
    scaled = (matrix - model.feature_mean) / scale
    scores = scaled @ model.coef + model.intercept
    if not np.isfinite(scores).all():
        raise AlphaModelError("模型输出包含非有限值")
    return scores.astype("float64")


def save_model_bundle(models: dict[str, LinearModel], path: Path) -> None:
    """把多折模型原子写入一个 NPZ 文件。"""

    if not models:
        raise AlphaModelError("没有可写入的模型")
    arrays: dict[str, np.ndarray] = {
        "fold_ids": np.asarray(sorted(models), dtype="U32"),
    }
    for fold_id, model in models.items():
        prefix = f"{fold_id}__"
        arrays[f"{prefix}model_type"] = np.asarray(model.model_type)
        arrays[f"{prefix}alpha"] = np.asarray(model.alpha, dtype="float64")
        arrays[f"{prefix}l1_ratio"] = np.asarray(model.l1_ratio, dtype="float64")
        arrays[f"{prefix}feature_mean"] = model.feature_mean
        arrays[f"{prefix}feature_scale"] = model.feature_scale
        arrays[f"{prefix}coef"] = model.coef
        arrays[f"{prefix}intercept"] = np.asarray(model.intercept, dtype="float64")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_model_bundle(path: Path) -> dict[str, LinearModel]:
    """读取 NPZ 模型并恢复每折参数。"""

    if not path.is_file():
        raise AlphaModelError(f"模型文件不存在：{path}")
    try:
        with np.load(path, allow_pickle=False) as bundle:
            fold_ids = [str(item) for item in bundle["fold_ids"]]
            models: dict[str, LinearModel] = {}
            for fold_id in fold_ids:
                prefix = f"{fold_id}__"
                models[fold_id] = LinearModel(
                    model_type=str(bundle[f"{prefix}model_type"].item()),
                    alpha=float(bundle[f"{prefix}alpha"].item()),
                    l1_ratio=float(bundle[f"{prefix}l1_ratio"].item()),
                    feature_mean=np.asarray(
                        bundle[f"{prefix}feature_mean"], dtype="float64"
                    ),
                    feature_scale=np.asarray(
                        bundle[f"{prefix}feature_scale"], dtype="float64"
                    ),
                    coef=np.asarray(bundle[f"{prefix}coef"], dtype="float64"),
                    intercept=float(bundle[f"{prefix}intercept"].item()),
                )
    except Exception as error:
        raise AlphaModelError(f"模型文件损坏或格式不兼容：{path}") from error
    return models
