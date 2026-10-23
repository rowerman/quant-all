"""最终留出集的冻结来源、身份、预测写入和完成校验。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from alpha.artifacts import file_sha256, model_dir
from alpha.config import FINAL_HOLDOUT_RUNS_DIR
from alpha.errors import AlphaFinalHoldoutError
from alpha.final_holdout_data import (
    FINAL_HOLDOUT_SCHEMA_VERSION,
    FinalHoldoutSpec,
    prediction_manifest,
    read_final_holdout_json,
)
from alpha.predict import load_model_manifest
from scripts.build_alpha.backtest_accept_artifacts import (
    atomic_write_frame,
    atomic_write_json,
)

FINAL_HOLDOUT_ROOT = FINAL_HOLDOUT_RUNS_DIR


def _canonical(payload: object) -> str:
    """生成稳定 JSON 文本。"""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def code_sha256() -> str:
    """汇总最终验证链路的源码哈希。"""

    project = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in (
        "alpha/final_holdout.py",
        "alpha/final_holdout_artifacts.py",
        "alpha/final_holdout_data.py",
        "alpha/final_holdout_eval.py",
        "alpha/final_holdout_ledger.py",
        "alpha/final_holdout_report.py",
        "alpha/final_holdout_runner.py",
        "alpha/exposure.py",
        "alpha/monitor_inputs.py",
        "alpha/monitor_metrics.py",
        "alpha/lifecycle_engine.py",
        "backtest/strategies/_final_holdout.py",
        "scripts/build_alpha/backtest_accept.py",
        "scripts/build_alpha/backtest_accept_artifacts.py",
        "scripts/build_alpha/backtest_accept_report.py",
        "backtest/chunked.py",
        "backtest/engine.py",
        "backtest/ledger.py",
        "backtest/allocation.py",
        "backtest/signals.py",
    ):
        path = project / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def model_sources(model_id: str, *, fold_id: str) -> dict:
    """读取冻结模型和预测来源摘要。"""

    folder = model_dir(model_id)
    manifest_path = folder / "manifest.json"
    manifest = load_model_manifest(model_id)
    fold = next(
        (
            item
            for item in manifest.get("folds") or []
            if item.get("fold_id") == fold_id
        ),
        None,
    )
    if fold is None:
        raise AlphaFinalHoldoutError(f"{model_id} 缺少 {fold_id} 折")
    return {
        "model_id": model_id,
        "manifest_sha256": file_sha256(manifest_path),
        "model_sha256": manifest.get("model_sha256"),
        "prediction_sha256": manifest.get("prediction_sha256"),
        "dataset_id": manifest.get("dataset_id"),
        "frozen_fold_id": fold_id,
        "frozen_fold": fold,
    }


def holdout_id(
    spec: FinalHoldoutSpec,
    dataset_manifest: dict,
    dataset_path: Path,
    sources: dict,
    predictions: dict,
    code_hash: str,
) -> str:
    """由数据、模型、预测、门槛和代码生成验证 ID。"""

    payload = {
        "schema_version": FINAL_HOLDOUT_SCHEMA_VERSION,
        "spec": spec.to_dict(),
        "gate": {
            "required_improvements": 2,
            "turnover_multiple_limit": 2.0,
            "expected_years": list(
                range(
                    pd.Timestamp(spec.start).year,
                    pd.Timestamp(spec.end).year + 1,
                )
            ),
        },
        "evaluation_plan": spec.evaluation_plan.to_dict(),
        "dataset": {
            "dataset_id": dataset_manifest.get("dataset_id"),
            "manifest_sha256": file_sha256(dataset_path.parent / "manifest.json"),
            "data_sha256": file_sha256(dataset_path),
            "versions": dataset_manifest.get("versions"),
        },
        "sources": sources,
        "predictions": prediction_manifest(predictions),
        "code_sha256": code_hash,
    }
    digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return f"final_holdout_{digest[:12]}"


def write_predictions(
    root: Path,
    predictions: dict[str, pd.DataFrame],
) -> dict:
    """写入留出预测和哈希 manifest。"""

    folder = root / "predictions"
    if folder.exists():
        raise AlphaFinalHoldoutError(f"留出预测目录已存在：{folder}")
    folder.mkdir(parents=True)
    for name, frame in predictions.items():
        atomic_write_frame(folder / f"{name}.parquet", frame.reset_index(drop=True))
    payload = {
        "schema_version": "alpha_final_holdout_predictions_v1",
        "predictions": prediction_manifest(predictions),
    }
    atomic_write_json(folder / "manifest.json", payload)
    return payload


def load_final_holdout(
    holdout_id: str,
    *,
    root: Path | None = None,
) -> dict:
    """读取并校验已完成的最终留出集结果。"""

    if not holdout_id.startswith("final_holdout_") or Path(holdout_id).name != holdout_id:
        raise AlphaFinalHoldoutError(f"非法留出集 ID：{holdout_id!r}")
    output_root = Path(root or FINAL_HOLDOUT_ROOT)
    folder = output_root / holdout_id
    manifest = read_final_holdout_json(folder / "manifest.json", "留出集 manifest")
    complete = read_final_holdout_json(folder / "complete.json", "留出集完成标记")
    if (
        manifest.get("holdout_id") != holdout_id
        or complete.get("holdout_id") != holdout_id
        or complete.get("complete") is not True
    ):
        raise AlphaFinalHoldoutError("留出集身份或完成状态不一致")
    files = complete.get("files") or {}
    if not files:
        raise AlphaFinalHoldoutError("留出集完成标记没有文件哈希")
    for relative, expected in files.items():
        path = folder / str(relative)
        if not path.is_file() or file_sha256(path) != str(expected):
            raise AlphaFinalHoldoutError(f"留出集文件缺失或哈希不一致：{path}")
    return {
        "folder": folder,
        "manifest": manifest,
        "complete": complete,
    }


__all__ = [
    "FINAL_HOLDOUT_ROOT",
    "code_sha256",
    "holdout_id",
    "load_final_holdout",
    "model_sources",
    "write_predictions",
]
