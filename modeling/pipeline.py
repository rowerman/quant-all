"""模型无关的 walk-forward 训练和产物写入流水线。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from alpha.artifacts import (
    MODEL_SCHEMA_VERSION_V2,
    file_sha256,
    model_dir,
    prediction_sha256,
    write_json,
)
from alpha.config import MODEL_DATA_DIR
from alpha.dataset import load_dataset
from alpha.ensemble import combine_rank_predictions
from alpha.errors import AlphaModelError
from alpha.evaluate import rank_ic_series, rank_ic_summary, yearly_rank_ic
from alpha.modeling.artifacts import (
    prediction_from_scores,
    prediction_metadata,
    write_prediction_parts,
)
from alpha.modeling.contracts import (
    ModelArtifact,
    TrainingSpec,
    preprocess_from_payload,
)
from alpha.modeling.folds import load_folds, validate_folds
from alpha.modeling.provenance import (
    append_experiment,
    model_id as build_model_id,
    source_sha256,
)
from alpha.modeling.registry import get_backend
from alpha.preprocess import transform
from alpha.split import Fold, fold_masks


def _resume(folder: Path, model_id: str) -> ModelArtifact | None:
    manifest_path = folder / "manifest.json"
    metrics_path = folder / "metrics.json"
    if not manifest_path.is_file() or not metrics_path.is_file():
        return None
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


def _combine(predictions: list[pd.DataFrame], names: tuple[str, ...]) -> pd.DataFrame:
    if len(predictions) == 1:
        return predictions[0]
    return combine_rank_predictions(predictions, names=names)


def fit_model_loaded(
    data: pd.DataFrame,
    dataset_manifest: dict,
    dataset_id: str,
    spec: TrainingSpec,
    *,
    folds: Sequence[Fold] | None = None,
    output_root: Path | None = None,
    force: bool = False,
) -> ModelArtifact:
    """在已加载数据上使用指定后端完成训练。"""

    backend = get_backend(spec.backend)
    fold_list = tuple(folds or load_folds())
    validate_folds(fold_list)
    config = backend.build_config(spec.config)
    code_hash = source_sha256(spec.backend)
    resolved_model_type = spec.resolved_model_type(config)
    model_id = build_model_id(
        dataset_id,
        dataset_manifest,
        fold_list,
        spec,
        code_hash,
        resolved_model_type=resolved_model_type,
    )
    root = Path(output_root or MODEL_DATA_DIR)
    folder = model_dir(model_id, root, include_archived=False)
    if not force:
        resumed = _resume(folder, model_id)
        if resumed is not None:
            return resumed

    raw_features = tuple(str(name) for name in dataset_manifest["features"])
    context_features = tuple(
        str(name) for name in dataset_manifest.get("context_features") or ()
    )
    preprocess_config = preprocess_from_payload(spec.preprocess)
    preprocessed = transform(
        data,
        preprocess_config,
        feature_columns=raw_features + context_features,
    )
    features = preprocessed.feature_names
    if not features:
        raise AlphaModelError("预处理后没有模型特征")
    folder.mkdir(parents=True, exist_ok=True)

    predictions: list[pd.DataFrame] = []
    member_predictions: dict[int, list[pd.DataFrame]] = {
        seed: [] for seed in spec.seeds
    }
    fold_details: list[dict] = []
    fold_models: list[dict] = []
    validation: dict[str, dict] = {}
    seed_values: tuple[int | None, ...] = spec.seeds or (None,)
    seed_names = tuple(
        f"seed={seed}" if seed is not None else "model" for seed in seed_values
    )

    for fold in sorted(fold_list, key=lambda item: item.test_start):
        masks = fold_masks(preprocessed.data, fold)
        usable = (
            preprocessed.data["sample_weight"].gt(0)
            & preprocessed.data[spec.target].notna()
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
        if train_rows.empty or valid_rows.empty or refit_rows.empty or test_rows.empty:
            raise AlphaModelError(f"{fold.fold_id} 训练、验证或测试样本为空")

        valid_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []
        member_records: list[dict] = []
        member_validation: dict[str, dict] = {}
        valid_matrix = valid_rows.loc[:, features].to_numpy(dtype="float32")
        test_matrix = test_rows.loc[:, features].to_numpy(dtype="float32")
        for seed in seed_values:
            member_config = (
                config if seed is None else backend.with_seed(config, int(seed))
            )
            fitted = backend.fit_fold(
                train_rows, valid_rows, features, member_config
            )
            valid_part = prediction_from_scores(
                valid_rows, backend.predict(fitted.model, valid_matrix)
            )
            valid_parts.append(valid_part)
            model = backend.refit(
                refit_rows, features, member_config, fitted
            )
            test_part = prediction_from_scores(
                test_rows, backend.predict(model, test_matrix)
            )
            test_parts.append(test_part)
            if seed is not None:
                member_predictions[seed].append(test_part)
            seed_label = "model" if seed is None else f"seed={seed}"
            model_relative = (
                f"models/{seed_label}/{fold.fold_id}{backend.model_suffix}"
                if seed is not None
                else f"models/{fold.fold_id}{backend.model_suffix}"
            )
            model_path = folder / model_relative
            backend.save_model(model, model_path, fold_id=fold.fold_id)
            record = {
                "fold_id": fold.fold_id,
                "test_year": int(pd.Timestamp(fold.test_start).year),
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "model_file": model_relative,
                "model_sha256": file_sha256(model_path),
                "selection": fitted.selection,
            }
            if seed is not None:
                record["seed"] = int(seed)
            member_records.append(record)
            fold_models.append(record)
            member_validation[seed_label] = fitted.selection

        fold_prediction = _combine(test_parts, seed_names)
        predictions.append(fold_prediction)
        fold_details.append(
            {
                **fold.to_dict(),
                "train_rows": int(len(train_rows)),
                "valid_rows": int(len(valid_rows)),
                "refit_rows": int(len(refit_rows)),
                "test_rows": int(len(test_rows)),
                "members": member_records,
            }
        )
        validation[fold.fold_id] = {
            "train_rows": int(len(train_rows)),
            "valid_rows": int(len(valid_rows)),
            "refit_rows": int(len(refit_rows)),
            "test_rows": int(len(test_rows)),
            "members": member_validation,
        }

    predictions_frame = pd.concat(predictions, ignore_index=True).sort_values(
        ["trade_date", "stock_code"], kind="mergesort"
    ).reset_index(drop=True)
    prediction_hash = prediction_sha256(predictions_frame)
    prediction_paths = write_prediction_parts(
        predictions_frame, folder / "predictions"
    )
    member_metadata: dict[str, dict] = {}
    for seed in spec.seeds:
        member_frame = pd.concat(
            member_predictions[seed], ignore_index=True
        ).sort_values(["trade_date", "stock_code"], kind="mergesort")
        member_frame = member_frame.reset_index(drop=True)
        files = write_prediction_parts(
            member_frame, folder / "member_predictions" / f"seed={seed}"
        )
        member_metadata[str(seed)] = prediction_metadata(member_frame, files)

    labeled = predictions_frame.merge(
        data.loc[:, ["trade_date", "stock_code", "label_raw"]],
        on=["trade_date", "stock_code"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"label_raw": "label"})
    test_series = rank_ic_series(labeled)
    test_summary = rank_ic_summary(test_series)
    aggregate = hashlib.sha256()
    for record in fold_models:
        aggregate.update(str(record["fold_id"]).encode("utf-8"))
        aggregate.update(str(record.get("seed", "")).encode("ascii"))
        aggregate.update(str(record["model_sha256"]).encode("ascii"))

    metrics = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "evaluated": False,
        "validation": validation,
        "test_model": {**test_summary, "yearly": yearly_rank_ic(test_series)},
    }
    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION_V2,
        "model_id": model_id,
        "model_type": resolved_model_type,
        "backend": {
            "name": spec.backend,
            "source_sha256": code_hash,
        },
        "target": spec.target,
        "dataset_id": dataset_id,
        "label": str(data.attrs.get("label") or dataset_manifest.get("spec", {}).get("label", "fwd_ret_open_5")),
        "universe": "all_a",
        "benchmark": "H00300.CSI",
        "top_n": spec.top_n,
        "quantiles": spec.quantiles,
        "seeds": list(spec.seeds),
        "member_type": spec.backend if spec.seeds else None,
        "ensemble": (
            {
                "seeds": list(spec.seeds),
                "aggregation": {
                    "method": "datewise_rank_mean",
                    "weights": "equal",
                },
            }
            if spec.seeds
            else None
        ),
        "training_spec": spec.to_dict(),
        "backend_metadata": backend.manifest_metadata(config),
        "preprocess": spec.preprocess or preprocess_config.__dict__,
        "raw_features": list(raw_features),
        "context_features": list(context_features),
        "features": list(features),
        "feature_count": len(features),
        "folds": fold_details,
        "fold_models": fold_models,
        "member_predictions": member_metadata,
        "data_versions": dataset_manifest.get("versions", {}),
        "dataset_code_sha256": dataset_manifest.get("code_sha256"),
        "code_sha256": code_hash,
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
    write_json(folder / "config.json", spec.to_dict())
    pd.DataFrame(fold_models).to_csv(folder / "fold_models.csv", index=False)
    write_json(
        folder / "features.json",
        {
            "raw": list(raw_features),
            "context": list(context_features),
            "transformed": list(features),
        },
    )
    write_json(folder / "metrics.json", metrics)
    write_json(folder / "manifest.json", manifest)
    append_experiment(manifest, metrics, root=root)
    return ModelArtifact(
        model_id=model_id,
        path=folder,
        manifest=manifest,
        metrics=metrics,
        row_count=len(predictions_frame),
        date_count=int(predictions_frame["trade_date"].nunique()),
        prediction_sha256=prediction_hash,
    )


def fit_model(
    dataset_id: str,
    spec: TrainingSpec,
    *,
    folds: Sequence[Fold] | None = None,
    output_root: Path | None = None,
    force: bool = False,
) -> ModelArtifact:
    """读取数据集并训练任一已注册模型后端。"""

    data, manifest = load_dataset(dataset_id)
    return fit_model_loaded(
        data,
        manifest,
        dataset_id,
        spec,
        folds=folds,
        output_root=output_root,
        force=force,
    )
