"""按已登记模型 ID 重放样本外预测。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.artifacts import model_dir, prediction_sha256, validate_dev_range
from alpha.config import MODEL_DATA_DIR
from alpha.dataset import load_dataset
from alpha.errors import AlphaArtifactError
from alpha.linear import load_model_bundle, predict_linear
from alpha.preprocess import PreprocessConfig, transform


def _manifest(model_id: str, root: Path) -> dict:
    """读取并校验模型 manifest。"""

    path = model_dir(model_id, root) / "manifest.json"
    if not path.is_file():
        raise AlphaArtifactError(f"模型不存在：{model_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model_id") != model_id:
        raise AlphaArtifactError(
            f"模型 manifest ID 不匹配：目录 {model_id}，manifest {payload.get('model_id')}"
        )
    return payload


def _predict_loaded(
    data: pd.DataFrame,
    manifest: dict,
    *,
    root: Path,
    start: str,
    end: str,
) -> pd.DataFrame:
    """在已加载数据集上重放指定测试区间。"""

    settings = manifest["preprocess"]
    config = PreprocessConfig(
        method=settings["method"],
        missing_indicator=bool(settings["missing_indicator"]),
        winsorize=bool(settings["winsorize"]),
        neutralize=tuple(settings["neutralize"]),
        clip_z=float(settings["clip_z"]),
    )
    preprocessed = transform(
        data,
        config,
        feature_columns=tuple(manifest["raw_features"])
        + tuple(manifest["context_features"]),
    )
    features = tuple(manifest["features"])
    if preprocessed.feature_names != features:
        raise AlphaArtifactError("预处理特征与模型 manifest 不一致")
    bundle = load_model_bundle(model_dir(manifest["model_id"], root) / "model.npz")
    start_stamp = pd.Timestamp(start)
    end_stamp = pd.Timestamp(end)
    parts: list[pd.DataFrame] = []
    dates = pd.to_datetime(preprocessed.data["trade_date"])
    for fold in manifest["folds"]:
        fold_id = str(fold["fold_id"])
        if fold_id not in bundle:
            raise AlphaArtifactError(f"模型文件缺少 {fold_id}")
        lo = max(start_stamp, pd.Timestamp(fold["test_start"]))
        hi = min(end_stamp, pd.Timestamp(fold["test_end"]))
        if lo > hi:
            continue
        mask = (
            (dates >= lo)
            & (dates <= hi)
            & preprocessed.data["eligible"].astype(bool)
        )
        rows = preprocessed.data.loc[mask]
        if rows.empty:
            continue
        scores = predict_linear(
            bundle[fold_id],
            rows.loc[:, features].to_numpy(dtype="float32"),
        )
        part = rows.loc[:, ["trade_date", "stock_code"]].copy()
        part["trade_date"] = pd.to_datetime(part["trade_date"])
        part["score"] = scores
        parts.append(part)
    if not parts:
        raise AlphaArtifactError(f"区间 {start} ~ {end} 没有可重放的样本外测试日")
    result = pd.concat(parts, ignore_index=True).sort_values(
        ["trade_date", "stock_code"], kind="mergesort"
    )
    result = result.reset_index(drop=True)
    if not np.isfinite(result["score"].to_numpy(dtype="float64")).all():
        raise AlphaArtifactError("重放预测包含非有限分数")
    return result


def predict(
    model_id: str,
    start: str,
    end: str,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    """重放模型在指定测试区间的预测并校验 SHA256。"""

    validate_dev_range(start, end)
    output_root = root or MODEL_DATA_DIR
    manifest = _manifest(model_id, output_root)
    data, _ = load_dataset(manifest["dataset_id"])
    result = _predict_loaded(
        data,
        manifest,
        root=output_root,
        start=start,
        end=end,
    )
    full_range = (
        pd.Timestamp(start) <= pd.Timestamp(manifest["prediction_start"])
        and pd.Timestamp(end) >= pd.Timestamp(manifest["prediction_end"])
    )
    if full_range:
        digest = prediction_sha256(result)
        if digest != manifest["prediction_sha256"]:
            raise AlphaArtifactError(
                f"预测重放哈希不一致：{digest} != {manifest['prediction_sha256']}"
            )
    return result
