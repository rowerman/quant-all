"""Ridge 和 ElasticNet 训练后端。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from alpha.evaluate import rank_ic_series, rank_ic_summary
from alpha.errors import AlphaModelError
from alpha.linear import (
    LinearModel,
    fit_linear,
    load_model_bundle,
    predict_linear,
    save_model_bundle,
)
from alpha.modeling.contracts import FoldFitResult
from alpha.modeling.registry import register_backend


@dataclass(frozen=True)
class LinearBackendConfig:
    """线性后端配置。"""

    model_type: str
    alpha_grid: tuple[float, ...]
    l1_ratio: float
    seed: int


class LinearBackend:
    """线性模型后端。"""

    name = "linear"
    model_suffix = ".npz"

    def build_config(self, payload: Mapping[str, Any]) -> LinearBackendConfig:
        model_type = str(payload.get("model_type", "ridge"))
        if model_type not in {"ridge", "elastic_net"}:
            raise AlphaModelError("线性后端 model_type 只支持 ridge/elastic_net")
        alphas = tuple(float(item) for item in payload.get("alpha_grid", (0.1, 1.0, 10.0, 100.0)))
        if not alphas or any(not np.isfinite(item) or item < 0 for item in alphas):
            raise AlphaModelError("alpha_grid 必须是非负有限数")
        return LinearBackendConfig(
            model_type=model_type,
            alpha_grid=alphas,
            l1_ratio=float(payload.get("l1_ratio", 0.1)),
            seed=int(payload.get("seed", 0)),
        )

    def config_dict(self, config: LinearBackendConfig) -> dict:
        return {
            "model_type": config.model_type,
            "alpha_grid": list(config.alpha_grid),
            "l1_ratio": config.l1_ratio,
            "seed": config.seed,
        }

    def with_seed(self, config: LinearBackendConfig, seed: int) -> LinearBackendConfig:
        return LinearBackendConfig(
            config.model_type, config.alpha_grid, config.l1_ratio, seed
        )

    def _fit(
        self,
        rows: pd.DataFrame,
        features: tuple[str, ...],
        config: LinearBackendConfig,
        alpha: float,
    ) -> LinearModel:
        return fit_linear(
            rows.loc[:, features].to_numpy(dtype="float32"),
            rows["label_rank"].to_numpy(dtype="float32"),
            rows["sample_weight"].to_numpy(dtype="float32"),
            model_type=config.model_type,
            alpha=float(alpha),
            l1_ratio=config.l1_ratio,
            seed=config.seed,
        )

    def fit_fold(
        self,
        train_rows: pd.DataFrame,
        valid_rows: pd.DataFrame,
        features: tuple[str, ...],
        config: LinearBackendConfig,
    ) -> FoldFitResult:
        candidates = []
        best_model = None
        for alpha in config.alpha_grid:
            model = self._fit(train_rows, features, config, alpha)
            panel = valid_rows.loc[:, ["trade_date"]].copy()
            panel["score"] = self.predict(
                model, valid_rows.loc[:, features].to_numpy(dtype="float32")
            )
            panel["label"] = valid_rows["label_rank"].to_numpy(dtype="float32")
            stats = rank_ic_summary(rank_ic_series(panel))
            if stats["rank_ic_mean"] is None:
                raise AlphaModelError(f"alpha={alpha} 的验证集 RankIC 无定义")
            candidates.append({"alpha": float(alpha), **stats})
            if max(
                candidates,
                key=lambda item: (float(item["rank_ic_mean"]), float(item["alpha"])),
            ) is candidates[-1]:
                best_model = model
        selected = max(
            candidates,
            key=lambda item: (float(item["rank_ic_mean"]), float(item["alpha"])),
        )
        if best_model is None:
            raise AlphaModelError("线性后端没有可用候选")
        return FoldFitResult(
            model=best_model,
            selection={"selected": selected, "candidates": candidates},
        )

    def refit(
        self,
        rows: pd.DataFrame,
        features: tuple[str, ...],
        config: LinearBackendConfig,
        fitted: FoldFitResult,
    ) -> LinearModel:
        return self._fit(
            rows, features, config, float(fitted.selection["selected"]["alpha"])
        )

    def predict(self, model: LinearModel, matrix: np.ndarray) -> np.ndarray:
        return predict_linear(model, matrix)

    def save_model(self, model: LinearModel, path: Path, *, fold_id: str) -> None:
        save_model_bundle({fold_id: model}, path)

    def load_model(self, path: Path, *, fold_id: str) -> LinearModel:
        bundle = load_model_bundle(path)
        if fold_id not in bundle:
            raise AlphaModelError(f"线性模型缺少 {fold_id}")
        return bundle[fold_id]

    def manifest_metadata(self, config: LinearBackendConfig) -> dict:
        return {"config": self.config_dict(config)}


register_backend(LinearBackend())
