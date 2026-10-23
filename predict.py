"""按已登记模型 ID 重放样本外预测。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from alpha.artifacts import (
    PREDICTION_COLUMNS,
    file_sha256,
    model_dir,
    prediction_sha256,
    validate_dev_range,
)
from alpha.config import MODEL_DATA_DIR
from alpha.dataset import load_dataset
from alpha.ensemble import combine_rank_predictions
from alpha.errors import AlphaArtifactError
from alpha.modeling.manifests import backend_from_manifest, records_by_fold
from alpha.modeling.registry import get_backend
from alpha.preprocess import PreprocessConfig, transform

ARTIFACT_SNAPSHOT_SCHEMA_VERSION = "alpha_model_artifact_snapshot_v1"


def load_model_manifest(model_id: str, root: Path | None = None) -> dict:
    """读取并校验模型 manifest。"""

    path = model_dir(model_id, root or MODEL_DATA_DIR) / "manifest.json"
    if not path.is_file():
        raise AlphaArtifactError(f"模型不存在：{model_id}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlphaArtifactError(f"无法读取模型 manifest：{path}") from error
    if not isinstance(payload, dict):
        raise AlphaArtifactError(f"模型 manifest 必须是对象：{path}")
    if payload.get("model_id") != model_id:
        raise AlphaArtifactError(
            f"模型 manifest ID 不匹配：目录 {model_id}，manifest {payload.get('model_id')}"
        )
    return payload


def _normalize_prediction(
    frame: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """校验并规范化一张预测分片。"""

    if not isinstance(frame, pd.DataFrame):
        raise AlphaArtifactError(f"{label}必须是 DataFrame")
    missing = sorted(set(PREDICTION_COLUMNS) - set(frame.columns))
    if missing:
        raise AlphaArtifactError(f"{label}缺少列：{missing}")
    work = frame.loc[:, list(PREDICTION_COLUMNS)].copy()
    try:
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="raise")
    except (TypeError, ValueError) as error:
        raise AlphaArtifactError(f"{label}包含无效交易日期") from error
    if work["trade_date"].isna().any():
        raise AlphaArtifactError(f"{label}交易日期不能为空")
    codes = work["stock_code"]
    if codes.isna().any() or codes.astype(str).str.strip().eq("").any():
        raise AlphaArtifactError(f"{label}股票代码不能为空")
    work["stock_code"] = codes.astype(str)
    scores = pd.to_numeric(work["score"], errors="coerce")
    if not np.isfinite(scores.to_numpy(dtype="float64")).all():
        raise AlphaArtifactError(f"{label}分数必须全部为有限值")
    work["score"] = scores.astype("float64")
    work = work.sort_values(
        ["trade_date", "stock_code"], kind="mergesort"
    ).reset_index(drop=True)
    if work.duplicated(["trade_date", "stock_code"]).any():
        raise AlphaArtifactError(f"{label}存在重复日期和股票")
    return work


def load_prediction_parts(
    root: Path,
    files: Mapping[str, str],
    expected_sha256: str,
    *,
    label: str = "预测",
) -> pd.DataFrame:
    """读取一组年度预测分片并校验分片及合并哈希。"""

    if not isinstance(files, Mapping) or not files:
        raise AlphaArtifactError(f"{label}没有预测分片")
    expected = str(expected_sha256 or "")
    if not expected:
        raise AlphaArtifactError(f"{label}缺少合并哈希")
    parts: list[pd.DataFrame] = []
    for year, part_hash in sorted(files.items(), key=lambda item: str(item[0])):
        key = str(year)
        if len(key) != 4 or not key.isdigit():
            raise AlphaArtifactError(f"{label}年度分片键非法：{key!r}")
        path = root / f"year={key}" / "part.parquet"
        if not path.is_file():
            raise AlphaArtifactError(f"{label}分片缺失：{path}")
        try:
            part = pd.read_parquet(path)
        except Exception as error:
            raise AlphaArtifactError(f"无法读取{label}分片：{path}") from error
        normalized = _normalize_prediction(part, label=f"{label} {key}")
        years = normalized["trade_date"].dt.year
        if not years.eq(int(key)).all():
            raise AlphaArtifactError(f"{label}分片 {key} 包含其他年度日期")
        digest = prediction_sha256(normalized)
        if digest != str(part_hash or ""):
            raise AlphaArtifactError(f"{label}分片哈希不一致：{path}")
        parts.append(normalized)
    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(
        ["trade_date", "stock_code"], kind="mergesort"
    ).reset_index(drop=True)
    if result.duplicated(["trade_date", "stock_code"]).any():
        raise AlphaArtifactError(f"合并{label}存在重复日期和股票")
    digest = prediction_sha256(result)
    if digest != expected:
        raise AlphaArtifactError(f"合并{label}哈希与 manifest 不一致")
    return result


def load_predictions(
    model_id: str,
    start: str | None = None,
    end: str | None = None,
    *,
    root: Path | None = None,
) -> tuple[Path, dict, pd.DataFrame]:
    """读取并完整校验模型已落盘预测，再按闭区间截取。"""

    output_root = root or MODEL_DATA_DIR
    folder = model_dir(model_id, output_root)
    manifest = load_model_manifest(model_id, output_root)
    predictions = load_prediction_parts(
        folder / "predictions",
        manifest.get("prediction_files") or {},
        str(manifest.get("prediction_sha256") or ""),
        label=f"模型 {model_id} 预测",
    )
    expected_rows = manifest.get("prediction_rows")
    expected_dates = manifest.get("prediction_dates")
    if type(expected_rows) is not int or expected_rows != len(predictions):
        raise AlphaArtifactError(
            f"预测行数与 manifest 不一致：{len(predictions)} != {expected_rows}"
        )
    unique_dates = predictions["trade_date"].drop_duplicates()
    if type(expected_dates) is not int or expected_dates != len(unique_dates):
        raise AlphaArtifactError(
            f"预测日期数与 manifest 不一致：{len(unique_dates)} != {expected_dates}"
        )
    actual_start = str(predictions["trade_date"].min().date())
    actual_end = str(predictions["trade_date"].max().date())
    if actual_start != manifest.get("prediction_start"):
        raise AlphaArtifactError(
            f"预测起始日与 manifest 不一致：{actual_start} != "
            f"{manifest.get('prediction_start')}"
        )
    if actual_end != manifest.get("prediction_end"):
        raise AlphaArtifactError(
            f"预测截止日与 manifest 不一致：{actual_end} != "
            f"{manifest.get('prediction_end')}"
        )
    if start is None and end is None:
        return folder, manifest, predictions
    try:
        start_stamp = pd.Timestamp(
            manifest["prediction_start"] if start is None else start
        )
        end_stamp = pd.Timestamp(manifest["prediction_end"] if end is None else end)
    except (TypeError, ValueError) as error:
        raise AlphaArtifactError(f"非法预测区间：{start!r} ~ {end!r}") from error
    if start_stamp > end_stamp:
        raise AlphaArtifactError(
            f"预测起始日晚于截止日：{start_stamp.date()} ~ {end_stamp.date()}"
        )
    selected = predictions.loc[
        predictions["trade_date"].between(start_stamp, end_stamp, inclusive="both")
    ].reset_index(drop=True)
    if selected.empty:
        raise AlphaArtifactError(
            f"模型 {model_id} 在区间 {start_stamp.date()} ~ {end_stamp.date()} "
            "没有已落盘预测"
        )
    return folder, manifest, selected


def _canonical_sha256(payload: object) -> str:
    """计算稳定 JSON 内容的 SHA256。"""

    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _artifact_sha256(path: Path, label: str) -> str:
    """计算产物文件哈希并把读取失败转换为模型产物错误。"""

    try:
        return file_sha256(path)
    except OSError as error:
        raise AlphaArtifactError(f"无法读取{label}：{path}") from error


def _artifact_file(
    folder: Path,
    relative: str,
    *,
    declared_sha256: str | None = None,
) -> dict:
    """读取模型目录内文件并记录实际哈希。"""

    if not relative or Path(relative).is_absolute():
        raise AlphaArtifactError(f"模型产物路径非法：{relative!r}")
    base = folder.resolve()
    path = (folder / relative).resolve()
    if base not in path.parents:
        raise AlphaArtifactError(f"模型产物路径越界：{relative!r}")
    if not path.is_file():
        raise AlphaArtifactError(f"模型产物文件缺失：{path}")
    actual = _artifact_sha256(path, "模型产物文件")
    declared = None if declared_sha256 is None else str(declared_sha256)
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": actual,
        "declared_sha256": declared,
        "matches_manifest": declared is None or actual == declared,
    }


def _model_file_snapshots(folder: Path, manifest: dict) -> list[dict]:
    """按 manifest 定位并校验模型主文件的快照字段。"""

    records = manifest.get("fold_models")
    if records:
        files = []
        for record in records:
            relative = str(record.get("model_file") or "")
            files.append(
                _artifact_file(
                    folder,
                    relative,
                    declared_sha256=str(record.get("model_sha256") or ""),
                )
            )
        return sorted(files, key=lambda item: item["path"])
    if (folder / "model.npz").is_file():
        return [
            _artifact_file(
                folder,
                "model.npz",
                declared_sha256=str(manifest.get("model_sha256") or ""),
            )
        ]
    raise AlphaArtifactError(
        f"模型 {manifest.get('model_id')} 没有可定位的模型文件"
    )


def _dataset_snapshot(output_root: Path, dataset_id: object) -> dict:
    """读取训练数据集及其 manifest，并记录实际内容哈希。"""

    key = str(dataset_id or "")
    if not key or Path(key).name != key:
        raise AlphaArtifactError(f"非法数据集 ID：{dataset_id!r}")
    folder = output_root / "datasets" / key
    manifest_path = folder / "manifest.json"
    data_path = folder / "data.parquet"
    if not manifest_path.is_file():
        raise AlphaArtifactError(f"数据集 manifest 缺失：{manifest_path}")
    if not data_path.is_file():
        raise AlphaArtifactError(f"数据集文件缺失：{data_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlphaArtifactError(
            f"无法读取数据集 manifest：{manifest_path}"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("dataset_id") != key:
        raise AlphaArtifactError(f"数据集 manifest ID 不匹配：{key}")
    return {
        "dataset_id": key,
        "manifest_sha256": _artifact_sha256(manifest_path, "数据集 manifest"),
        "data_sha256": _artifact_sha256(data_path, "数据集文件"),
        "dataset_code_sha256": manifest.get("code_sha256"),
        "spec": manifest.get("spec"),
        "versions": manifest.get("versions") or {},
        "features": list(manifest.get("features") or []),
        "context_features": list(manifest.get("context_features") or []),
        "row_count": manifest.get("row_count"),
        "date_count": manifest.get("date_count"),
        "stock_count": manifest.get("stock_count"),
        "date_min": manifest.get("date_min"),
        "date_max": manifest.get("date_max"),
        "eligible_rows": manifest.get("eligible_rows"),
        "label_coverage": manifest.get("label_coverage"),
    }


def model_artifact_snapshot(model_id: str, root: Path | None = None) -> dict:
    """返回模型、预测和训练数据的可复现摘要。

    参数:
        model_id: 已落盘模型 ID。
        root: 可选的模型产物根目录，默认是 ``model_data/``。
    """

    output_root = root or MODEL_DATA_DIR
    folder, manifest, predictions = load_predictions(model_id, root=output_root)
    dataset = _dataset_snapshot(output_root, manifest.get("dataset_id"))
    folds = [
        {
            "fold_id": fold.get("fold_id"),
            "train_start": fold.get("train_start"),
            "train_end": fold.get("train_end"),
            "valid_start": fold.get("valid_start"),
            "valid_end": fold.get("valid_end"),
            "test_start": fold.get("test_start"),
            "test_end": fold.get("test_end"),
            "purge_days": fold.get("purge_days"),
            "embargo_days": fold.get("embargo_days"),
        }
        for fold in manifest.get("folds") or []
    ]
    model = {
        "manifest_sha256": _artifact_sha256(
            folder / "manifest.json",
            "模型 manifest",
        ),
        "model_id": manifest.get("model_id"),
        "model_type": manifest.get("model_type"),
        "member_type": manifest.get("member_type"),
        "label": manifest.get("label"),
        "target": manifest.get("target"),
        "universe": manifest.get("universe"),
        "benchmark": manifest.get("benchmark"),
        "top_n": manifest.get("top_n"),
        "quantiles": manifest.get("quantiles"),
        "model_params": manifest.get("model_params") or {},
        "ensemble": manifest.get("ensemble"),
        "preprocess": manifest.get("preprocess") or {},
        "seeds": list(manifest.get("seeds") or []),
        "raw_features": list(manifest.get("raw_features") or []),
        "context_features": list(manifest.get("context_features") or []),
        "features": list(manifest.get("features") or []),
        "feature_count": manifest.get("feature_count"),
        "folds": folds,
        "files": _model_file_snapshots(folder, manifest),
        "code_sha256": manifest.get("code_sha256"),
        "declared_model_sha256": manifest.get("model_sha256"),
    }
    predictions_payload = {
        "sha256": manifest.get("prediction_sha256"),
        "files": manifest.get("prediction_files") or {},
        "rows": int(len(predictions)),
        "dates": int(predictions["trade_date"].nunique()),
        "start": str(predictions["trade_date"].min().date()),
        "end": str(predictions["trade_date"].max().date()),
        "member_predictions": manifest.get("member_predictions") or {},
    }
    content = {
        "schema_version": ARTIFACT_SNAPSHOT_SCHEMA_VERSION,
        "model": model,
        "dataset": dataset,
        "predictions": predictions_payload,
    }
    return {**content, "artifact_sha256": _canonical_sha256(content)}


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
    folder = model_dir(manifest["model_id"], root)
    backend = get_backend(backend_from_manifest(manifest))
    grouped = records_by_fold(manifest)
    start_stamp = pd.Timestamp(start)
    end_stamp = pd.Timestamp(end)
    parts: list[pd.DataFrame] = []
    dates = pd.to_datetime(preprocessed.data["trade_date"])
    for fold in manifest["folds"]:
        fold_id = str(fold["fold_id"])
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
        matrix = rows.loc[:, features].to_numpy(dtype="float32")
        records = grouped.get(fold_id) or []
        if not records:
            raise AlphaArtifactError(f"模型文件缺少 {fold_id}")
        member_parts: list[pd.DataFrame] = []
        for record in sorted(records, key=lambda item: int(item.get("seed", 0))):
            relative = str(record.get("model_file") or "")
            path = folder / relative
            expected = str(record.get("model_sha256") or "")
            if not relative or not path.is_file():
                raise AlphaArtifactError(f"模型文件缺失：{path}")
            if not expected or file_sha256(path) != expected:
                raise AlphaArtifactError(f"模型文件哈希不一致：{path}")
            try:
                model = backend.load_model(path, fold_id=fold_id)
                scores = backend.predict(model, matrix)
            except Exception as error:
                raise AlphaArtifactError(f"无法重放模型 {path}：{error}") from error
            member_part = rows.loc[:, ["trade_date", "stock_code"]].copy()
            member_part["trade_date"] = pd.to_datetime(
                member_part["trade_date"]
            )
            member_part["score"] = scores
            member_parts.append(member_part)
        part = (
            member_parts[0]
            if len(member_parts) == 1
            else combine_rank_predictions(
                member_parts,
                names=tuple(
                    f"seed={record.get('seed', index)}"
                    for index, record in enumerate(records)
                ),
            )
        )
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
    manifest = load_model_manifest(model_id, output_root)
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
