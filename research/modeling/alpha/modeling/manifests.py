"""历史与当前模型 manifest 的统一读取。"""

from __future__ import annotations

from collections import defaultdict

from alpha.errors import AlphaModelError
from alpha.modeling.registry import backend_name_for_manifest


def backend_from_manifest(manifest: dict) -> str:
    """返回 manifest 对应后端名称。"""

    return backend_name_for_manifest(manifest)


def model_records(manifest: dict) -> list[dict]:
    """把 v1/v2 manifest 归一化为折内模型记录。"""

    model_type = str(manifest.get("model_type") or "")
    if isinstance(manifest.get("backend"), dict):
        records = manifest.get("fold_models") or []
        if not records:
            raise AlphaModelError("v2 manifest 缺少 fold_models")
        return [dict(item) for item in records]
    if model_type in {"ridge", "elastic_net"}:
        path = "model.npz"
        digest = str(manifest.get("model_sha256") or "")
        return [
            {
                "fold_id": str(fold["fold_id"]),
                "model_file": path,
                "model_sha256": digest,
            }
            for fold in manifest.get("folds") or []
        ]
    if model_type == "lightgbm":
        records = []
        for fold in manifest.get("folds") or []:
            records.append(
                {
                    "fold_id": str(fold["fold_id"]),
                    "model_file": str(fold["model_file"]),
                    "model_sha256": str(fold["model_sha256"]),
                }
            )
        return records
    if model_type == "lightgbm_ensemble":
        records = []
        for fold in manifest.get("folds") or []:
            for member in fold.get("members") or []:
                records.append(
                    {
                        "fold_id": str(fold["fold_id"]),
                        "seed": int(member["seed"]),
                        "model_file": str(member["model_file"]),
                        "model_sha256": str(member["model_sha256"]),
                    }
                )
        return records
    raise AlphaModelError(f"无法归一化历史模型 manifest：{model_type}")


def records_by_fold(manifest: dict) -> dict[str, list[dict]]:
    """按 fold_id 分组模型记录。"""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in model_records(manifest):
        grouped[str(record["fold_id"])].append(record)
    return dict(grouped)
