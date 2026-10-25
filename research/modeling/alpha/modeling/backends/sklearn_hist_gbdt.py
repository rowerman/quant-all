"""Scikit-learn 直方图梯度提升后端。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from alpha.evaluate import rank_ic_series, rank_ic_summary
from alpha.errors import AlphaModelError
from alpha.modeling.contracts import FoldFitResult
from alpha.modeling.registry import register_backend


@dataclass(frozen=True)
class HistGradientBoostingConfig:
    """直方图梯度提升的固定训练配置。"""

    model_type: str = "sklearn_hist_gbdt"
    learning_rate: float = 0.05
    max_iter: int = 300
    max_leaf_nodes: int = 31
    max_depth: int | None = 6
    min_samples_leaf: int = 1000
    l2_regularization: float = 1.0
    max_bins: int = 255
    validation_fraction: float = 0.1
    n_iter_no_change: int = 20
    seed: int = 0


def _positive_int(value: Any, name: str) -> int:
    """校验正整数配置。"""

    if type(value) is not int or value <= 0:
        raise AlphaModelError(f"{name} 必须是正整数")
    return value


class SklearnHistGbdtBackend:
    """使用 Scikit-learn 直方图梯度提升训练截面排序模型。"""

    name = "sklearn_hist_gbdt"
    model_suffix = ".joblib"

    def build_config(
        self,
        payload: Mapping[str, Any],
    ) -> HistGradientBoostingConfig:
        """校验并构造后端配置。"""

        model_type = str(payload.get("model_type", "sklearn_hist_gbdt"))
        if model_type != "sklearn_hist_gbdt":
            raise AlphaModelError(
                "直方图 GBDT 后端 model_type 只支持 sklearn_hist_gbdt"
            )
        max_depth_value = payload.get("max_depth", 6)
        max_depth = (
            None
            if max_depth_value is None
            else _positive_int(max_depth_value, "max_depth")
        )
        learning_rate = float(payload.get("learning_rate", 0.05))
        l2_regularization = float(payload.get("l2_regularization", 1.0))
        validation_fraction = float(payload.get("validation_fraction", 0.1))
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise AlphaModelError("learning_rate 必须是正有限数")
        if not np.isfinite(l2_regularization) or l2_regularization < 0:
            raise AlphaModelError("l2_regularization 必须是非负有限数")
        if not 0.0 < validation_fraction < 1.0:
            raise AlphaModelError("validation_fraction 必须在 (0, 1) 内")
        max_bins = _positive_int(payload.get("max_bins", 255), "max_bins")
        if max_bins < 2:
            raise AlphaModelError("max_bins 必须至少为 2")
        seed = int(payload.get("seed", 0))
        if seed < 0:
            raise AlphaModelError("seed 必须是非负整数")
        return HistGradientBoostingConfig(
            model_type=model_type,
            learning_rate=learning_rate,
            max_iter=_positive_int(payload.get("max_iter", 300), "max_iter"),
            max_leaf_nodes=_positive_int(
                payload.get("max_leaf_nodes", 31),
                "max_leaf_nodes",
            ),
            max_depth=max_depth,
            min_samples_leaf=_positive_int(
                payload.get("min_samples_leaf", 1000),
                "min_samples_leaf",
            ),
            l2_regularization=l2_regularization,
            max_bins=max_bins,
            validation_fraction=validation_fraction,
            n_iter_no_change=_positive_int(
                payload.get("n_iter_no_change", 20),
                "n_iter_no_change",
            ),
            seed=seed,
        )

    def config_dict(self, config: HistGradientBoostingConfig) -> dict:
        """返回可序列化配置。"""

        return {
            "model_type": config.model_type,
            "learning_rate": config.learning_rate,
            "max_iter": config.max_iter,
            "max_leaf_nodes": config.max_leaf_nodes,
            "max_depth": config.max_depth,
            "min_samples_leaf": config.min_samples_leaf,
            "l2_regularization": config.l2_regularization,
            "max_bins": config.max_bins,
            "validation_fraction": config.validation_fraction,
            "n_iter_no_change": config.n_iter_no_change,
            "seed": config.seed,
        }

    def with_seed(
        self,
        config: HistGradientBoostingConfig,
        seed: int,
    ) -> HistGradientBoostingConfig:
        """替换随机种子。"""

        return HistGradientBoostingConfig(
            **{**self.config_dict(config), "seed": int(seed)}
        )

    def _estimator(
        self,
        config: HistGradientBoostingConfig,
        *,
        max_iter: int | None = None,
        early_stopping: bool = True,
    ) -> HistGradientBoostingRegressor:
        """构造固定配置的估计器。"""

        return HistGradientBoostingRegressor(
            learning_rate=config.learning_rate,
            max_iter=max_iter or config.max_iter,
            max_leaf_nodes=config.max_leaf_nodes,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            l2_regularization=config.l2_regularization,
            max_bins=config.max_bins,
            early_stopping=early_stopping,
            validation_fraction=config.validation_fraction,
            n_iter_no_change=config.n_iter_no_change,
            random_state=config.seed,
        )

    def fit_fold(
        self,
        train_rows: pd.DataFrame,
        valid_rows: pd.DataFrame,
        features: tuple[str, ...],
        config: HistGradientBoostingConfig,
    ) -> FoldFitResult:
        """拟合一折，并记录早停轮数和验证集 RankIC。"""

        model = self._estimator(config)
        model.fit(
            train_rows.loc[:, features].to_numpy(dtype="float32"),
            train_rows["label_rank"].to_numpy(dtype="float32"),
            sample_weight=train_rows["sample_weight"].to_numpy(dtype="float32"),
        )
        panel = valid_rows.loc[:, ["trade_date"]].copy()
        panel["score"] = self.predict(
            model,
            valid_rows.loc[:, features].to_numpy(dtype="float32"),
        )
        panel["label"] = valid_rows["label_rank"].to_numpy(dtype="float32")
        validation = rank_ic_summary(rank_ic_series(panel))
        iterations = int(model.n_iter_)
        if iterations <= 0:
            raise AlphaModelError("直方图 GBDT 没有产生有效迭代")
        return FoldFitResult(
            model=model,
            selection={
                "iterations": iterations,
                "validation_rank_ic": validation["rank_ic_mean"],
            },
        )

    def refit(
        self,
        rows: pd.DataFrame,
        features: tuple[str, ...],
        config: HistGradientBoostingConfig,
        fitted: FoldFitResult,
    ) -> HistGradientBoostingRegressor:
        """按验证集早停轮数在 train+valid 上重拟合。"""

        iterations = int(fitted.selection.get("iterations", 0))
        if iterations <= 0:
            raise AlphaModelError("直方图 GBDT 缺少有效早停轮数")
        model = self._estimator(
            config,
            max_iter=iterations,
            early_stopping=False,
        )
        model.fit(
            rows.loc[:, features].to_numpy(dtype="float32"),
            rows["label_rank"].to_numpy(dtype="float32"),
            sample_weight=rows["sample_weight"].to_numpy(dtype="float32"),
        )
        return model

    def predict(
        self,
        model: HistGradientBoostingRegressor,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """生成并校验有限预测分数。"""

        scores = np.asarray(model.predict(matrix), dtype="float64")
        if scores.ndim != 1 or not np.isfinite(scores).all():
            raise AlphaModelError("直方图 GBDT 预测分数必须是有限一维数组")
        return scores

    def save_model(
        self,
        model: HistGradientBoostingRegressor,
        path: Path,
        *,
        fold_id: str,
    ) -> None:
        """保存单折模型。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path, compress=3)

    def load_model(
        self,
        path: Path,
        *,
        fold_id: str,
    ) -> HistGradientBoostingRegressor:
        """读取单折模型并校验类型。"""

        try:
            model = joblib.load(path)
        except (OSError, ValueError, TypeError) as error:
            raise AlphaModelError(f"无法读取直方图 GBDT 模型：{path}") from error
        if not isinstance(model, HistGradientBoostingRegressor):
            raise AlphaModelError(f"模型文件类型错误：{path}")
        return model

    def manifest_metadata(self, config: HistGradientBoostingConfig) -> dict:
        """返回 manifest 的后端专属字段。"""

        return {"config": self.config_dict(config)}


register_backend(SklearnHistGbdtBackend())
