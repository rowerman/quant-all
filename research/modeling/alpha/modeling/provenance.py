"""模型代码摘要、确定性 ID 和实验索引。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from alpha.artifacts import MODEL_SCHEMA_VERSION_V2, write_json
from alpha.config import MODEL_DATA_DIR
from alpha.modeling.contracts import TrainingSpec
from alpha.split import Fold
from common.paths import PROJECT_ROOT


def source_sha256(backend_name: str) -> str:
    """汇总通用流水线和指定后端源码。"""

    from alpha.modeling import registry

    backend = registry.get_backend(backend_name)
    module = __import__(backend.__class__.__module__, fromlist=["__file__"])
    files = [Path(module.__file__)]
    modeling_root = Path(__file__).resolve().parent
    files.extend(
        path
        for path in sorted(modeling_root.rglob("*.py"))
        if path.name != "provenance.py"
    )
    root = modeling_root.parent
    files.extend(
        root / name
        for name in (
            "artifacts.py",
            "config.py",
            "dataset.py",
            "ensemble.py",
            "evaluate.py",
            "linear.py",
            "gbdt.py",
            "preprocess.py",
            "split.py",
            "kernel_ops.py",
        )
    )
    files.extend(
        PROJECT_ROOT / name
        for name in (
            "core/kernels/backend.py",
            "core/kernels/fallback.py",
            "core/kernels/native.py",
            "core/kernels/preprocess.py",
            "native/include/quant_native/daily.hpp",
            "native/src/bindings.cpp",
            "native/src/rank.cpp",
            "native/src/standardize.cpp",
        )
    )
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = str(path)
        digest.update(label.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def model_id(
    dataset_id: str,
    dataset_manifest: dict,
    folds: Sequence[Fold],
    spec: TrainingSpec,
    code_sha256: str,
    *,
    resolved_model_type: str | None = None,
) -> str:
    """生成与后端、数据和配置相关的确定性模型 ID。"""

    payload = {
        "schema_version": MODEL_SCHEMA_VERSION_V2,
        "dataset_id": dataset_id,
        "dataset_versions": dataset_manifest.get("versions"),
        "dataset_code_sha256": dataset_manifest.get("code_sha256"),
        "folds": [fold.to_dict() for fold in folds],
        "spec": spec.to_dict(),
        "resolved_model_type": resolved_model_type or spec.resolved_model_type(),
        "code_sha256": code_sha256,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    model_type = resolved_model_type or spec.resolved_model_type()
    return f"{model_type}_{spec.target}_{digest}"


def append_experiment(
    manifest: Mapping,
    metrics: Mapping,
    *,
    root: Path | None = None,
) -> None:
    """幂等追加模型实验记录。"""

    path = Path(root or MODEL_DATA_DIR) / "experiments.csv"
    model_id_value = str(manifest["model_id"])
    existing = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    if not existing.empty and model_id_value in set(
        existing["experiment_id"].astype(str)
    ):
        return
    model_metrics = metrics.get("test_model") or {}
    row = {
        "experiment_id": model_id_value,
        "created_at": manifest["created_at"],
        "batch": "modeling",
        "name": model_id_value,
        "kind": "model",
        "stage": "development",
        "start": manifest["prediction_start"],
        "end": manifest["prediction_end"],
        "config_hash": model_id_value,
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
        "notes": "modeling backend; development years only",
    }
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not path.is_file(), index=False
    )
