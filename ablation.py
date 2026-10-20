"""分族重训和单因子置换消融。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from alpha.errors import AlphaModelError
from alpha.explain import permute_group_within_dates
from alpha.feature_groups import FeatureGroup
from alpha.gbdt import (
    GbdtConfig,
    fit_gbdt_fold,
    predict_gbdt,
    refit_gbdt,
    save_gbdt_model,
)
from alpha.preprocess import PreprocessConfig, PreprocessedFrame
from alpha.split import Fold, fold_masks


@dataclass(frozen=True)
class FamilyAblationResult:
    """一次分族重训的预测和模型记录。"""

    group: FeatureGroup
    predictions: pd.DataFrame
    model_records: tuple[dict, ...]
    remaining_features: tuple[str, ...]


def folds_from_manifest(manifest: dict) -> tuple[Fold, ...]:
    """从模型 manifest 还原六折切分。"""

    result = []
    for item in manifest.get("folds", ()):
        result.append(
            Fold(
                fold_id=str(item["fold_id"]),
                train_start=str(item["train_start"]),
                train_end=str(item["train_end"]),
                valid_start=str(item["valid_start"]),
                valid_end=str(item["valid_end"]),
                test_start=str(item["test_start"]),
                test_end=str(item["test_end"]),
                purge_days=int(item["purge_days"]),
                embargo_days=int(item["embargo_days"]),
            )
        )
    if not result:
        raise AlphaModelError("模型 manifest 没有可用折")
    return tuple(result)


def gbdt_config_from_manifest(
    manifest: dict,
    *,
    seed: int,
    num_threads: int | None = None,
) -> GbdtConfig:
    """按 manifest 超参数构造单成员配置。"""

    raw = manifest.get("model_params")
    if not isinstance(raw, dict):
        raise AlphaModelError("模型 manifest 缺少 model_params")
    params = dict(raw)
    preprocess = params.get("preprocess")
    if not isinstance(preprocess, dict):
        raise AlphaModelError("模型 manifest 缺少预处理配置")
    params["preprocess"] = PreprocessConfig(**preprocess)
    params["seed"] = int(seed)
    if num_threads is not None:
        params["num_threads"] = int(num_threads)
    return GbdtConfig(**params)


def fit_family_ablation(
    preprocessed: PreprocessedFrame,
    manifest: dict,
    group: FeatureGroup,
    *,
    output_dir: Path,
    seed: int = 0,
    num_threads: int | None = None,
) -> FamilyAblationResult:
    """删除一个特征组后按原六折重训 LightGBM。"""

    all_features = tuple(str(name) for name in manifest["features"])
    removed = set(group.transformed_features)
    remaining = tuple(name for name in all_features if name not in removed)
    if not remaining:
        raise AlphaModelError(f"删除 {group.key} 后没有剩余特征")
    if preprocessed.feature_names != all_features:
        raise AlphaModelError("预处理特征与模型 manifest 不一致")
    config = gbdt_config_from_manifest(
        manifest,
        seed=seed,
        num_threads=num_threads,
    )
    folds = folds_from_manifest(manifest)
    predictions: list[pd.DataFrame] = []
    records: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold in sorted(folds, key=lambda item: item.test_start):
        masks = fold_masks(preprocessed.data, fold)
        usable = (
            preprocessed.data["sample_weight"].gt(0)
            & preprocessed.data[config.target].notna()
        )
        train_rows = preprocessed.data.loc[masks["train"] & usable]
        valid_rows = preprocessed.data.loc[masks["valid"] & usable].sort_values(
            ["trade_date", "stock_code"], kind="mergesort"
        )
        refit_rows = preprocessed.data.loc[
            (masks["train"] | masks["valid"]) & usable
        ]
        test_rows = preprocessed.data.loc[
            masks["test"] & preprocessed.data["eligible"].astype(bool)
        ]
        if train_rows.empty or valid_rows.empty or test_rows.empty:
            raise AlphaModelError(f"{fold.fold_id} 消融训练、验证或测试样本为空")
        fitted = fit_gbdt_fold(
            train_rows.loc[:, remaining].to_numpy(dtype="float32"),
            train_rows[config.target].to_numpy(dtype="float32"),
            train_rows["sample_weight"].to_numpy(dtype="float32"),
            valid_rows.loc[:, remaining].to_numpy(dtype="float32"),
            valid_rows[config.target].to_numpy(dtype="float32"),
            valid_rows["sample_weight"].to_numpy(dtype="float32"),
            pd.to_datetime(valid_rows["trade_date"]).to_numpy(),
            remaining,
            config,
        )
        booster = refit_gbdt(
            refit_rows.loc[:, remaining].to_numpy(dtype="float32"),
            refit_rows[config.target].to_numpy(dtype="float32"),
            refit_rows["sample_weight"].to_numpy(dtype="float32"),
            remaining,
            config,
            fitted.best_iteration,
        )
        scores = predict_gbdt(
            booster,
            test_rows.loc[:, remaining].to_numpy(dtype="float32"),
        )
        part = test_rows.loc[:, ["trade_date", "stock_code"]].copy()
        part["trade_date"] = pd.to_datetime(part["trade_date"])
        part["score"] = scores
        predictions.append(part)
        model_path = output_dir / f"{fold.fold_id}.txt"
        save_gbdt_model(booster, model_path)
        records.append(
            {
                "group": group.key,
                "fold_id": fold.fold_id,
                "seed": int(seed),
                "best_iteration": int(fitted.best_iteration),
                "validation_rank_ic": float(fitted.validation_rank_ic),
                "model_file": str(Path(group.key) / f"{fold.fold_id}.txt"),
            }
        )
    result = pd.concat(predictions, ignore_index=True).sort_values(
        ["trade_date", "stock_code"], kind="mergesort"
    )
    result = result.reset_index(drop=True)
    if result.duplicated(["trade_date", "stock_code"]).any():
        raise AlphaModelError("分族消融预测存在重复键")
    return FamilyAblationResult(
        group=group,
        predictions=result,
        model_records=tuple(records),
        remaining_features=remaining,
    )


def predict_permuted_group(
    preprocessed: PreprocessedFrame,
    manifest: dict,
    group: FeatureGroup,
    models: Mapping[str, object],
    *,
    permutation_seed: int,
) -> pd.DataFrame:
    """固定模型，在测试截面内置换一个特征组并重新预测。"""

    features = tuple(str(name) for name in manifest["features"])
    if preprocessed.feature_names != features:
        raise AlphaModelError("预处理特征与模型 manifest 不一致")
    positions = [features.index(name) for name in group.transformed_features]
    folds = folds_from_manifest(manifest)
    parts: list[pd.DataFrame] = []
    for fold_index, fold in enumerate(sorted(folds, key=lambda item: item.test_start)):
        model = models.get(fold.fold_id)
        if model is None:
            raise AlphaModelError(f"置换消融缺少 {fold.fold_id} 的 seed=0 模型")
        masks = fold_masks(preprocessed.data, fold)
        rows = preprocessed.data.loc[
            masks["test"] & preprocessed.data["eligible"].astype(bool)
        ]
        if rows.empty:
            raise AlphaModelError(f"{fold.fold_id} 置换消融测试样本为空")
        matrix = rows.loc[:, features].to_numpy(dtype="float32")
        permuted = permute_group_within_dates(
            matrix,
            pd.to_datetime(rows["trade_date"]).to_numpy(),
            positions,
            seed=int(permutation_seed) + fold_index * 1009,
        )
        part = rows.loc[:, ["trade_date", "stock_code"]].copy()
        part["trade_date"] = pd.to_datetime(part["trade_date"])
        part["score"] = predict_gbdt(model, permuted)
        parts.append(part)
    result = pd.concat(parts, ignore_index=True).sort_values(
        ["trade_date", "stock_code"], kind="mergesort"
    )
    return result.reset_index(drop=True)


__all__ = [
    "FamilyAblationResult",
    "fit_family_ablation",
    "folds_from_manifest",
    "gbdt_config_from_manifest",
    "predict_permuted_group",
]
