"""线性模型与 LightGBM 的训练编排和预测产物生成。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from alpha.artifacts import (
    MODEL_SCHEMA_VERSION,
    file_sha256,
    model_dir,
    prediction_sha256,
    validate_dev_range,
    write_json,
)
from alpha.config import DEFAULT_CONTRACT, MODEL_DATA_DIR
from alpha.dataset import load_dataset
from alpha.errors import AlphaModelError
from alpha.evaluate import rank_ic_series, rank_ic_summary, yearly_rank_ic
from alpha.gbdt import (
    GbdtConfig,
    fit_gbdt_fold,
    predict_gbdt,
    refit_gbdt,
    save_gbdt_model,
)
from alpha.linear import (
    LinearModel,
    MODEL_TYPES,
    fit_linear,
    predict_linear,
    save_model_bundle,
)
from alpha.logging_conf import get_logger
from alpha.preprocess import PreprocessConfig, transform
from alpha.split import Fold, fold_masks

logger = get_logger("alpha.train")


@dataclass(frozen=True)
class TrainConfig:
    """一次线性模型训练的固定配置。"""

    model_type: str = "ridge"
    target: str = "label_rank"
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    l1_ratio: float = 0.1
    top_n: int = 30
    quantiles: int = 10
    seed: int = 0
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

    def __post_init__(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise AlphaModelError(f"model_type 只支持 {MODEL_TYPES}")
        if self.target != "label_rank":
            raise AlphaModelError("P2 目标固定为 label_rank")
        if not self.alpha_grid or any(
            not np.isfinite(value) or value < 0 for value in self.alpha_grid
        ):
            raise AlphaModelError("alpha_grid 必须是至少一个非负有限数")
        if len(set(float(value) for value in self.alpha_grid)) != len(self.alpha_grid):
            raise AlphaModelError("alpha_grid 不能含重复值")
        if not 0.0 <= self.l1_ratio <= 1.0:
            raise AlphaModelError("l1_ratio 必须在 0 到 1 之间")
        if type(self.top_n) is not int or self.top_n <= 0:
            raise AlphaModelError("top_n 必须是正整数")
        if type(self.quantiles) is not int or self.quantiles < 2:
            raise AlphaModelError("quantiles 必须是至少为 2 的整数")
        if type(self.seed) is not int:
            raise AlphaModelError("seed 必须是整数")

    def to_dict(self) -> dict:
        """转换为稳定字典。"""

        payload = asdict(self)
        payload["alpha_grid"] = [float(value) for value in self.alpha_grid]
        return payload


@dataclass(frozen=True)
class ModelArtifact:
    """训练完成的模型索引。"""

    model_id: str
    path: Path
    manifest: dict
    metrics: dict
    row_count: int
    date_count: int
    prediction_sha256: str


def _code_sha256() -> str:
    """汇总模型链路的代码哈希。"""

    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "artifacts.py",
        "config.py",
        "dataset.py",
        "evaluate.py",
        "gbdt.py",
        "linear.py",
        "preprocess.py",
        "split.py",
        "train.py",
    ):
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _model_id(
    dataset_id: str,
    dataset_manifest: dict,
    folds: Sequence[Fold],
    config: TrainConfig,
    code_sha256: str,
) -> str:
    """由数据、折表和配置生成确定性模型 ID。"""

    payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_versions": dataset_manifest.get("versions"),
        "dataset_code_sha256": dataset_manifest.get("code_sha256"),
        "folds": [fold.to_dict() for fold in folds],
        "config": config.to_dict(),
        "code_sha256": code_sha256,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{config.model_type}_{config.target}_{digest}"


def _gbdt_model_id(
    dataset_id: str,
    dataset_manifest: dict,
    folds: Sequence[Fold],
    config: GbdtConfig,
    code_sha256: str,
) -> str:
    """由数据、折表和 LightGBM 配置生成确定性模型 ID。"""

    payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_versions": dataset_manifest.get("versions"),
        "dataset_code_sha256": dataset_manifest.get("code_sha256"),
        "folds": [fold.to_dict() for fold in folds],
        "config": config.to_dict(),
        "code_sha256": code_sha256,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"lightgbm_{config.target}_{digest}"


def load_folds(path: Path | None = None) -> tuple[Fold, ...]:
    """读取 P1 生成的 walk-forward 切分表。"""

    target = path or MODEL_DATA_DIR / "splits" / "walk_forward_v1.csv"
    if not target.is_file():
        raise AlphaModelError(f"切分表不存在：{target}")
    frame = pd.read_csv(target)
    required = {
        "fold_id",
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
        "test_start",
        "test_end",
        "purge_days",
        "embargo_days",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AlphaModelError(f"切分表缺少列：{missing}")
    folds = tuple(
        Fold(
            fold_id=str(row.fold_id),
            train_start=str(row.train_start),
            train_end=str(row.train_end),
            valid_start=str(row.valid_start),
            valid_end=str(row.valid_end),
            test_start=str(row.test_start),
            test_end=str(row.test_end),
            purge_days=int(row.purge_days),
            embargo_days=int(row.embargo_days),
        )
        for row in frame.itertuples(index=False)
    )
    if not folds:
        raise AlphaModelError("切分表没有可用折")
    return folds


def _fit_candidate(
    train_rows: pd.DataFrame,
    valid_rows: pd.DataFrame,
    features: tuple[str, ...],
    config: TrainConfig,
    alpha: float,
) -> tuple[LinearModel, dict]:
    """在训练集拟合一个候选参数并计算验证集 RankIC。"""

    model = fit_linear(
        train_rows.loc[:, features].to_numpy(dtype="float32"),
        train_rows[config.target].to_numpy(dtype="float32"),
        train_rows["sample_weight"].to_numpy(dtype="float32"),
        model_type=config.model_type,
        alpha=alpha,
        l1_ratio=config.l1_ratio,
        seed=config.seed,
    )
    scores = predict_linear(model, valid_rows.loc[:, features].to_numpy(dtype="float32"))
    panel = valid_rows.loc[:, ["trade_date"]].copy()
    panel["score"] = scores
    panel["label"] = valid_rows[config.target].to_numpy(dtype="float32")
    stats = rank_ic_summary(rank_ic_series(panel))
    if stats["rank_ic_mean"] is None:
        raise AlphaModelError(f"alpha={alpha} 的验证集 RankIC 无定义")
    return model, stats


def _select_alpha(
    train_rows: pd.DataFrame,
    valid_rows: pd.DataFrame,
    features: tuple[str, ...],
    config: TrainConfig,
) -> tuple[float, dict, list[dict]]:
    """按验证集 RankIC 选参，平局选更大的 alpha。"""

    candidates = []
    for alpha in config.alpha_grid:
        _, stats = _fit_candidate(
            train_rows, valid_rows, features, config, float(alpha)
        )
        candidates.append({"alpha": float(alpha), **stats})
    best = max(
        candidates,
        key=lambda item: (float(item["rank_ic_mean"]), float(item["alpha"])),
    )
    return float(best["alpha"]), best, candidates


def _write_prediction_parts(frame: pd.DataFrame, folder: Path) -> dict[str, str]:
    """按年原子写入预测文件并返回相对路径哈希。"""

    hashes: dict[str, str] = {}
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    for year, group in work.groupby(work["trade_date"].dt.year, sort=True):
        target = folder / f"year={int(year)}" / "part.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.tmp")
        group.loc[:, ["trade_date", "stock_code", "score"]].to_parquet(
            temporary, index=False
        )
        os.replace(temporary, target)
        hashes[str(int(year))] = prediction_sha256(group)
    return hashes


def _best_single_factor(
    data: pd.DataFrame,
    fold: Fold,
    factor_names: tuple[str, ...],
) -> dict:
    """只用该折验证集选出下一年测试的单因子基线。"""

    dates = pd.to_datetime(data["trade_date"])
    mask = (
        (dates >= pd.Timestamp(fold.valid_start))
        & (dates <= pd.Timestamp(fold.valid_end))
        & data["eligible"].astype(bool)
        & data["label_raw"].notna()
    )
    valid = data.loc[mask]
    rows: list[dict] = []
    for name in factor_names:
        panel = valid.loc[:, ["trade_date"]].copy()
        panel["score"] = pd.to_numeric(valid[name], errors="coerce")
        panel["label"] = valid["label_raw"]
        panel = panel.replace([np.inf, -np.inf], np.nan).dropna()
        series = rank_ic_series(panel)
        stats = rank_ic_summary(series)
        if stats["rank_ic_mean"] is not None:
            rows.append({"factor": name, **stats})
    if not rows:
        raise AlphaModelError(f"{fold.fold_id} 没有可用的单因子验证结果")
    best = min(
        rows,
        key=lambda item: (-abs(float(item["rank_ic_mean"])), str(item["factor"])),
    )
    return {
        "fold_id": fold.fold_id,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "factor": str(best["factor"]),
        "sign": 1.0 if float(best["rank_ic_mean"]) >= 0 else -1.0,
        "validation_rank_ic": float(best["rank_ic_mean"]),
    }


def _prediction_from_scores(
    rows: pd.DataFrame,
    scores: np.ndarray,
) -> pd.DataFrame:
    """把模型分数整理为标准预测表。"""

    result = rows.loc[:, ["trade_date", "stock_code"]].copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result["score"] = np.asarray(scores, dtype="float64")
    result = result.sort_values(
        ["trade_date", "stock_code"], kind="mergesort"
    ).reset_index(drop=True)
    if result.duplicated(["trade_date", "stock_code"]).any():
        raise AlphaModelError("预测表存在重复日期和股票")
    if not np.isfinite(result["score"].to_numpy(dtype="float64")).all():
        raise AlphaModelError("预测表包含非有限分数")
    return result


def _prediction_frame(
    rows: pd.DataFrame,
    model: LinearModel,
    features: tuple[str, ...],
) -> pd.DataFrame:
    """给一组样本生成标准线性模型预测表。"""

    scores = predict_linear(
        model, rows.loc[:, features].to_numpy(dtype="float32")
    )
    return _prediction_from_scores(rows, scores)


def _fit_loaded(
    data: pd.DataFrame,
    dataset_manifest: dict,
    folds: Sequence[Fold],
    config: TrainConfig,
    *,
    dataset_id: str,
    output_root: Path,
    force: bool = False,
) -> ModelArtifact:
    """在已加载数据上完成六折训练与产物写入。"""

    code_sha256 = _code_sha256()
    model_id = _model_id(
        dataset_id, dataset_manifest, folds, config, code_sha256
    )
    folder = model_dir(model_id, output_root)
    manifest_path = folder / "manifest.json"
    metrics_path = folder / "metrics.json"
    if manifest_path.is_file() and metrics_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return ModelArtifact(
            model_id=model_id,
            path=folder,
            manifest=manifest,
            metrics=metrics,
            row_count=int(manifest["prediction_rows"]),
            date_count=int(manifest["prediction_dates"]),
            prediction_sha256=str(manifest["prediction_sha256"]),
        )
    raw_features = tuple(str(name) for name in dataset_manifest["features"])
    context_features = tuple(
        str(name) for name in dataset_manifest["context_features"]
    )
    preprocessed = transform(
        data,
        config.preprocess,
        feature_columns=raw_features + context_features,
    )
    features = preprocessed.feature_names
    if not features:
        raise AlphaModelError("预处理后没有模型特征")
    predictions: list[pd.DataFrame] = []
    fold_models: dict[str, LinearModel] = {}
    fold_details: list[dict] = []
    validation: dict[str, dict] = {}
    single_factors: list[dict] = []
    for fold in sorted(folds, key=lambda item: item.test_start):
        masks = fold_masks(preprocessed.data, fold)
        train_mask = (
            masks["train"]
            & preprocessed.data["sample_weight"].gt(0)
            & preprocessed.data[config.target].notna()
        )
        valid_mask = (
            masks["valid"]
            & preprocessed.data["sample_weight"].gt(0)
            & preprocessed.data[config.target].notna()
        )
        if not train_mask.any() or not valid_mask.any():
            raise AlphaModelError(f"{fold.fold_id} 训练或验证样本为空")
        train_rows = preprocessed.data.loc[train_mask]
        valid_rows = preprocessed.data.loc[valid_mask]
        alpha, selected, candidates = _select_alpha(
            train_rows, valid_rows, features, config
        )
        refit_mask = (masks["train"] | masks["valid"]) & (
            preprocessed.data["sample_weight"].gt(0)
        ) & preprocessed.data[config.target].notna()
        refit_rows = preprocessed.data.loc[refit_mask]
        model = fit_linear(
            refit_rows.loc[:, features].to_numpy(dtype="float32"),
            refit_rows[config.target].to_numpy(dtype="float32"),
            refit_rows["sample_weight"].to_numpy(dtype="float32"),
            model_type=config.model_type,
            alpha=alpha,
            l1_ratio=config.l1_ratio,
            seed=config.seed,
        )
        test_rows = preprocessed.data.loc[
            masks["test"] & preprocessed.data["eligible"].astype(bool)
        ]
        if test_rows.empty:
            raise AlphaModelError(f"{fold.fold_id} 测试样本为空")
        fold_predictions = _prediction_frame(test_rows, model, features)
        predictions.append(fold_predictions)
        fold_models[fold.fold_id] = model
        best_single = _best_single_factor(data, fold, raw_features)
        single_factors.append(best_single)
        validation[fold.fold_id] = {
            "alpha": alpha,
            "selected": selected,
            "candidates": candidates,
            "train_rows": int(len(train_rows)),
            "valid_rows": int(len(valid_rows)),
            "test_rows": int(len(test_rows)),
        }
        fold_details.append(
            {
                **fold.to_dict(),
                "selected_alpha": alpha,
                "validation_rank_ic": selected["rank_ic_mean"],
                "best_single_factor": best_single,
            }
        )
        logger.info(
            "%s 完成：alpha=%s 验证 RankIC=%.4f 测试行=%d",
            fold.fold_id,
            alpha,
            selected["rank_ic_mean"],
            len(test_rows),
        )
    predictions_frame = pd.concat(predictions, ignore_index=True).sort_values(
        ["trade_date", "stock_code"], kind="mergesort"
    )
    predictions_frame = predictions_frame.reset_index(drop=True)
    prediction_hash = prediction_sha256(predictions_frame)
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / "model.npz"
    save_model_bundle(fold_models, model_path)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    prediction_paths = _write_prediction_parts(
        predictions_frame, folder / "predictions"
    )
    labeled = predictions_frame.merge(
        data.loc[:, ["trade_date", "stock_code", "label_raw"]],
        on=["trade_date", "stock_code"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"label_raw": "label"})
    test_series = rank_ic_series(labeled)
    test_summary = rank_ic_summary(test_series)
    metrics = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "evaluated": False,
        "validation": validation,
        "test_model": {
            **test_summary,
            "yearly": yearly_rank_ic(test_series),
        },
    }
    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_id": model_id,
        "model_type": config.model_type,
        "target": config.target,
        "dataset_id": dataset_id,
        "label": DEFAULT_CONTRACT.label,
        "universe": DEFAULT_CONTRACT.universe,
        "benchmark": DEFAULT_CONTRACT.benchmark,
        "top_n": config.top_n,
        "quantiles": config.quantiles,
        "preprocess": config.to_dict()["preprocess"],
        "raw_features": list(raw_features),
        "context_features": list(context_features),
        "features": list(features),
        "feature_count": len(features),
        "folds": fold_details,
        "best_single_factors": single_factors,
        "data_versions": dataset_manifest.get("versions", {}),
        "dataset_code_sha256": dataset_manifest.get("code_sha256"),
        "code_sha256": code_sha256,
        "model_sha256": model_hash,
        "prediction_sha256": prediction_hash,
        "prediction_files": prediction_paths,
        "prediction_rows": int(len(predictions_frame)),
        "prediction_dates": int(predictions_frame["trade_date"].nunique()),
        "prediction_start": str(predictions_frame["trade_date"].min().date()),
        "prediction_end": str(predictions_frame["trade_date"].max().date()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "trained",
    }
    write_json(folder / "config.json", config.to_dict())
    write_json(
        folder / "features.json",
        {
            "raw": list(raw_features),
            "context": list(context_features),
            "transformed": list(features),
        },
    )
    write_json(metrics_path, metrics)
    write_json(manifest_path, manifest)
    _append_experiment(manifest, metrics)
    logger.info(
        "模型完成：%s 预测行=%d 日期=%d",
        model_id,
        len(predictions_frame),
        predictions_frame["trade_date"].nunique(),
    )
    return ModelArtifact(
        model_id=model_id,
        path=folder,
        manifest=manifest,
        metrics=metrics,
        row_count=len(predictions_frame),
        date_count=int(predictions_frame["trade_date"].nunique()),
        prediction_sha256=prediction_hash,
    )


def _fit_gbdt_loaded(
    data: pd.DataFrame,
    dataset_manifest: dict,
    folds: Sequence[Fold],
    config: GbdtConfig,
    *,
    dataset_id: str,
    output_root: Path,
    force: bool = False,
) -> ModelArtifact:
    """在已加载数据上完成 LightGBM 六折训练与产物写入。"""

    code_sha256 = _code_sha256()
    model_id = _gbdt_model_id(
        dataset_id, dataset_manifest, folds, config, code_sha256
    )
    folder = model_dir(model_id, output_root)
    manifest_path = folder / "manifest.json"
    metrics_path = folder / "metrics.json"
    if manifest_path.is_file() and metrics_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return ModelArtifact(
            model_id=model_id,
            path=folder,
            manifest=manifest,
            metrics=metrics,
            row_count=int(manifest["prediction_rows"]),
            date_count=int(manifest["prediction_dates"]),
            prediction_sha256=str(manifest["prediction_sha256"]),
        )
    raw_features = tuple(str(name) for name in dataset_manifest["features"])
    context_features = tuple(
        str(name) for name in dataset_manifest["context_features"]
    )
    preprocessed = transform(
        data,
        config.preprocess,
        feature_columns=raw_features + context_features,
    )
    features = preprocessed.feature_names
    if not features:
        raise AlphaModelError("预处理后没有模型特征")
    predictions: list[pd.DataFrame] = []
    fold_records: list[dict] = []
    fold_details: list[dict] = []
    validation: dict[str, dict] = {}
    single_factors: list[dict] = []
    folder.mkdir(parents=True, exist_ok=True)
    for fold in sorted(folds, key=lambda item: item.test_start):
        masks = fold_masks(preprocessed.data, fold)
        usable = (
            preprocessed.data["sample_weight"].gt(0)
            & preprocessed.data[config.target].notna()
        )
        train_mask = masks["train"] & usable
        valid_mask = masks["valid"] & usable
        if not train_mask.any() or not valid_mask.any():
            raise AlphaModelError(f"{fold.fold_id} 训练或验证样本为空")
        train_rows = preprocessed.data.loc[train_mask]
        valid_rows = preprocessed.data.loc[valid_mask].sort_values(
            ["trade_date", "stock_code"], kind="mergesort"
        )
        fitted = fit_gbdt_fold(
            train_rows.loc[:, features].to_numpy(dtype="float32"),
            train_rows[config.target].to_numpy(dtype="float32"),
            train_rows["sample_weight"].to_numpy(dtype="float32"),
            valid_rows.loc[:, features].to_numpy(dtype="float32"),
            valid_rows[config.target].to_numpy(dtype="float32"),
            valid_rows["sample_weight"].to_numpy(dtype="float32"),
            pd.to_datetime(valid_rows["trade_date"]).to_numpy(),
            features,
            config,
        )
        refit_rows = preprocessed.data.loc[
            (masks["train"] | masks["valid"]) & usable
        ]
        booster = refit_gbdt(
            refit_rows.loc[:, features].to_numpy(dtype="float32"),
            refit_rows[config.target].to_numpy(dtype="float32"),
            refit_rows["sample_weight"].to_numpy(dtype="float32"),
            features,
            config,
            fitted.best_iteration,
        )
        test_rows = preprocessed.data.loc[
            masks["test"] & preprocessed.data["eligible"].astype(bool)
        ]
        if test_rows.empty:
            raise AlphaModelError(f"{fold.fold_id} 测试样本为空")
        scores = predict_gbdt(
            booster,
            test_rows.loc[:, features].to_numpy(dtype="float32"),
        )
        predictions.append(_prediction_from_scores(test_rows, scores))
        model_relative = f"models/{fold.fold_id}.txt"
        model_path = folder / model_relative
        save_gbdt_model(booster, model_path)
        model_hash = file_sha256(model_path)
        fold_record = {
            "fold_id": fold.fold_id,
            "test_year": int(pd.Timestamp(fold.test_start).year),
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "train_rows": int(len(train_rows)),
            "valid_rows": int(len(valid_rows)),
            "test_rows": int(len(test_rows)),
            "best_iteration": int(fitted.best_iteration),
            "validation_rank_ic": float(fitted.validation_rank_ic),
            "model_file": model_relative,
            "model_sha256": model_hash,
        }
        fold_records.append(fold_record)
        best_single = _best_single_factor(data, fold, raw_features)
        single_factors.append(best_single)
        validation[fold.fold_id] = {
            "best_iteration": int(fitted.best_iteration),
            "validation_rank_ic": float(fitted.validation_rank_ic),
            "train_rows": int(len(train_rows)),
            "valid_rows": int(len(valid_rows)),
            "refit_rows": int(len(refit_rows)),
            "test_rows": int(len(test_rows)),
        }
        fold_details.append(
            {
                **fold.to_dict(),
                **fold_record,
                "best_single_factor": best_single,
            }
        )
        logger.info(
            "%s 完成：最佳轮=%d 验证 RankIC=%.4f 测试行=%d",
            fold.fold_id,
            fitted.best_iteration,
            fitted.validation_rank_ic,
            len(test_rows),
        )
    predictions_frame = pd.concat(predictions, ignore_index=True).sort_values(
        ["trade_date", "stock_code"], kind="mergesort"
    )
    predictions_frame = predictions_frame.reset_index(drop=True)
    prediction_hash = prediction_sha256(predictions_frame)
    prediction_paths = _write_prediction_parts(
        predictions_frame, folder / "predictions"
    )
    labeled = predictions_frame.merge(
        data.loc[:, ["trade_date", "stock_code", "label_raw"]],
        on=["trade_date", "stock_code"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"label_raw": "label"})
    test_series = rank_ic_series(labeled)
    test_summary = rank_ic_summary(test_series)
    aggregate = hashlib.sha256()
    for record in fold_records:
        aggregate.update(str(record["fold_id"]).encode("utf-8"))
        aggregate.update(str(record["model_sha256"]).encode("ascii"))
    metrics = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "evaluated": False,
        "validation": validation,
        "test_model": {
            **test_summary,
            "yearly": yearly_rank_ic(test_series),
        },
    }
    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_id": model_id,
        "model_type": "lightgbm",
        "target": config.target,
        "dataset_id": dataset_id,
        "label": DEFAULT_CONTRACT.label,
        "universe": DEFAULT_CONTRACT.universe,
        "benchmark": DEFAULT_CONTRACT.benchmark,
        "top_n": config.top_n,
        "quantiles": config.quantiles,
        "model_params": config.to_dict(),
        "preprocess": config.to_dict()["preprocess"],
        "raw_features": list(raw_features),
        "context_features": list(context_features),
        "features": list(features),
        "feature_count": len(features),
        "folds": fold_details,
        "fold_models": fold_records,
        "best_single_factors": single_factors,
        "data_versions": dataset_manifest.get("versions", {}),
        "dataset_code_sha256": dataset_manifest.get("code_sha256"),
        "code_sha256": code_sha256,
        "model_sha256": aggregate.hexdigest(),
        "prediction_sha256": prediction_hash,
        "prediction_files": prediction_paths,
        "prediction_rows": int(len(predictions_frame)),
        "prediction_dates": int(predictions_frame["trade_date"].nunique()),
        "prediction_start": str(predictions_frame["trade_date"].min().date()),
        "prediction_end": str(predictions_frame["trade_date"].max().date()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "trained",
    }
    fold_frame = pd.DataFrame(fold_records)
    temporary = folder / "fold_models.csv.tmp"
    fold_frame.to_csv(temporary, index=False)
    os.replace(temporary, folder / "fold_models.csv")
    write_json(folder / "config.json", config.to_dict())
    write_json(
        folder / "features.json",
        {
            "raw": list(raw_features),
            "context": list(context_features),
            "transformed": list(features),
        },
    )
    write_json(metrics_path, metrics)
    write_json(manifest_path, manifest)
    _append_experiment(
        manifest,
        metrics,
        batch="P3",
        notes="P3 LightGBM; development years only",
    )
    logger.info(
        "LightGBM 模型完成：%s 预测行=%d 日期=%d",
        model_id,
        len(predictions_frame),
        predictions_frame["trade_date"].nunique(),
    )
    return ModelArtifact(
        model_id=model_id,
        path=folder,
        manifest=manifest,
        metrics=metrics,
        row_count=len(predictions_frame),
        date_count=int(predictions_frame["trade_date"].nunique()),
        prediction_sha256=prediction_hash,
    )


def _append_experiment(
    manifest: dict,
    metrics: dict,
    *,
    batch: str = "P2",
    notes: str = "P2 linear model; development years only",
) -> None:
    """幂等追加模型实验记录。"""

    path = MODEL_DATA_DIR / "experiments.csv"
    model_id = manifest["model_id"]
    existing = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    if not existing.empty and model_id in set(existing["experiment_id"].astype(str)):
        return
    model_metrics = metrics.get("test_model") or {}
    row = {
        "experiment_id": model_id,
        "created_at": manifest["created_at"],
        "batch": batch,
        "name": model_id,
        "kind": "model",
        "stage": "development",
        "start": manifest["prediction_start"],
        "end": manifest["prediction_end"],
        "config_hash": manifest["model_id"],
        "feature_set": f"all_implemented_{manifest['feature_count']}",
        "data_version": manifest["data_versions"].get("data_version"),
        "factor_version": manifest["data_versions"].get("factor_version"),
        "snapshot_id": None,
        "metrics_json": json.dumps(
            {
                "rank_ic_mean": model_metrics.get("rank_ic_mean"),
                "icir": model_metrics.get("icir"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "notes": notes,
    }
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.is_file(), index=False)


def fit(
    dataset_id: str,
    config: TrainConfig | None = None,
    *,
    folds: Sequence[Fold] | None = None,
    output_root: Path | None = None,
    force: bool = False,
) -> ModelArtifact:
    """训练线性模型并返回可复现产物。"""

    settings = config or TrainConfig()
    data, manifest = load_dataset(dataset_id)
    fold_list = tuple(folds or load_folds())
    if not fold_list:
        raise AlphaModelError("没有可训练的折")
    for fold in fold_list:
        validate_dev_range(fold.test_start, fold.test_end)
    return _fit_loaded(
        data,
        manifest,
        fold_list,
        settings,
        dataset_id=dataset_id,
        output_root=output_root or MODEL_DATA_DIR,
        force=force,
    )


def fit_gbdt(
    dataset_id: str,
    config: GbdtConfig | None = None,
    *,
    folds: Sequence[Fold] | None = None,
    output_root: Path | None = None,
    force: bool = False,
) -> ModelArtifact:
    """训练 LightGBM 六折模型并返回可复现产物。"""

    settings = config or GbdtConfig()
    data, manifest = load_dataset(dataset_id)
    fold_list = tuple(folds or load_folds())
    if not fold_list:
        raise AlphaModelError("没有可训练的折")
    for fold in fold_list:
        validate_dev_range(fold.test_start, fold.test_end)
    return _fit_gbdt_loaded(
        data,
        manifest,
        fold_list,
        settings,
        dataset_id=dataset_id,
        output_root=output_root or MODEL_DATA_DIR,
        force=force,
    )
