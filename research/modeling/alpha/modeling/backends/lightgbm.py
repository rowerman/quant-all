"""LightGBM 训练后端。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from alpha.errors import AlphaModelError
from alpha.gbdt import (
    FittedGbdt,
    GbdtConfig,
    fit_gbdt_fold,
    load_gbdt_model,
    predict_gbdt,
    refit_gbdt,
    save_gbdt_model,
)
from alpha.modeling.contracts import FoldFitResult
from alpha.modeling.registry import register_backend


class LightGbmBackend:
    """LightGBM 单模型后端。"""

    name = "lightgbm"
    model_suffix = ".txt"

    def build_config(self, payload: Mapping[str, Any]) -> GbdtConfig:
        return GbdtConfig(**{str(key): value for key, value in payload.items()})

    def config_dict(self, config: GbdtConfig) -> dict:
        return config.to_dict()

    def with_seed(self, config: GbdtConfig, seed: int) -> GbdtConfig:
        return GbdtConfig(**{**config.to_dict(), "seed": int(seed)})

    def fit_fold(
        self,
        train_rows: pd.DataFrame,
        valid_rows: pd.DataFrame,
        features: tuple[str, ...],
        config: GbdtConfig,
    ) -> FoldFitResult:
        fitted = fit_gbdt_fold(
            train_rows.loc[:, features].to_numpy(dtype="float32"),
            train_rows["label_rank"].to_numpy(dtype="float32"),
            train_rows["sample_weight"].to_numpy(dtype="float32"),
            valid_rows.loc[:, features].to_numpy(dtype="float32"),
            valid_rows["label_rank"].to_numpy(dtype="float32"),
            valid_rows["sample_weight"].to_numpy(dtype="float32"),
            pd.to_datetime(valid_rows["trade_date"]).to_numpy(),
            features,
            config,
        )
        return FoldFitResult(
            model=fitted,
            selection={
                "best_iteration": int(fitted.best_iteration),
                "validation_rank_ic": float(fitted.validation_rank_ic),
            },
        )

    def refit(
        self,
        rows: pd.DataFrame,
        features: tuple[str, ...],
        config: GbdtConfig,
        fitted: FoldFitResult,
    ) -> object:
        early = fitted.model
        if not isinstance(early, FittedGbdt):
            raise AlphaModelError("LightGBM 单折结果类型错误")
        return refit_gbdt(
            rows.loc[:, features].to_numpy(dtype="float32"),
            rows["label_rank"].to_numpy(dtype="float32"),
            rows["sample_weight"].to_numpy(dtype="float32"),
            features,
            config,
            int(early.best_iteration),
        )

    def predict(self, model: object, matrix: np.ndarray) -> np.ndarray:
        if isinstance(model, FittedGbdt):
            model = model.booster
        return predict_gbdt(model, matrix)

    def save_model(self, model: object, path: Path, *, fold_id: str) -> None:
        save_gbdt_model(model, path)

    def load_model(self, path: Path, *, fold_id: str) -> object:
        return load_gbdt_model(path)

    def manifest_metadata(self, config: GbdtConfig) -> dict:
        return {"config": self.config_dict(config)}


register_backend(LightGbmBackend())
