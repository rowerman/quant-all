"""Alpha 模型登记与历史模型索引。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from alpha.artifacts import file_sha256, write_json
from alpha.config import MODEL_DATA_DIR
from alpha.errors import AlphaArtifactError, AlphaRegistryError
from alpha.logging_conf import get_logger
from alpha.predict import model_artifact_snapshot
from alpha.registry_runs import DOWNSTREAM_KINDS, scan_downstream_runs

REGISTRY_SCHEMA_VERSION = "alpha_model_registry_v3"
REGISTRY_FILENAME = "index.json"

logger = get_logger("alpha.registry")


def _read_json_object(
    path: Path,
    label: str,
    error_type: type[AlphaArtifactError] | type[AlphaRegistryError],
) -> dict:
    """读取 JSON 对象并把文件或格式错误转换为指定业务错误。"""

    if not path.is_file():
        raise error_type(f"{label}不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise error_type(f"无法读取{label}：{path}") from error
    if not isinstance(payload, dict):
        raise error_type(f"{label}必须是 JSON 对象：{path}")
    return payload


def _read_artifact_json(path: Path, label: str) -> dict:
    """读取模型产物 JSON。"""

    return _read_json_object(path, label, AlphaArtifactError)


def _feature_payload(manifest: dict, features: dict) -> dict:
    """校验 features.json 与 manifest 的特征清单一致。"""

    expected = {
        "raw": manifest.get("raw_features") or [],
        "context": manifest.get("context_features") or [],
        "transformed": manifest.get("features") or [],
    }
    for name, values in expected.items():
        actual = features.get(name)
        if not isinstance(actual, list) or actual != values:
            raise AlphaArtifactError(f"features.json 的 {name} 与模型 manifest 不一致")
    return {
        **expected,
        "count": len(expected["transformed"]),
    }


def _fold_window(folds: list[dict]) -> dict:
    """汇总多折训练和测试窗口。"""

    if not folds:
        raise AlphaArtifactError("模型 manifest 没有可用折")
    try:
        return {
            "train_start": min(str(fold["train_start"]) for fold in folds),
            "train_end": max(str(fold["train_end"]) for fold in folds),
            "test_start": min(str(fold["test_start"]) for fold in folds),
            "test_end": max(str(fold["test_end"]) for fold in folds),
            "fold_count": len(folds),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise AlphaArtifactError("模型折窗口字段缺失或非法") from error


def _metric_summary(metrics: dict) -> dict:
    """提取索引需要的核心指标，完整指标仍保存在 metrics.json。"""

    model_level = metrics.get("test_model") or {}
    portfolio = metrics.get("portfolio") or {}
    return {
        "evaluated": bool(metrics.get("evaluated", False)),
        "rank_ic_mean": model_level.get("rank_ic_mean"),
        "icir": model_level.get("icir"),
        "rank_ic_win_rate": model_level.get("rank_ic_win_rate"),
        "days": model_level.get("days"),
        "annual_return": portfolio.get("annual_return"),
        "excess_annual_return": portfolio.get("excess_annual_return"),
        "max_drawdown": portfolio.get("max_drawdown"),
        "sharpe": portfolio.get("sharpe"),
        "turnover_two_way_annual": portfolio.get("turnover_two_way_annual"),
        "periods": portfolio.get("periods"),
    }


def _validate_config(manifest: dict, config: dict) -> None:
    """校验 config 中存在的公共字段与 manifest 一致。"""

    comparisons = {
        "model_type": manifest.get("model_type"),
        "target": manifest.get("target"),
        "top_n": manifest.get("top_n"),
        "quantiles": manifest.get("quantiles"),
    }
    for name, expected in comparisons.items():
        if name in config and config[name] != expected:
            raise AlphaArtifactError(f"config.json 的 {name} 与模型 manifest 不一致")


def _validate_metrics(manifest: dict, metrics: dict) -> None:
    """校验 metrics 的模型和数据标识与 manifest 一致。"""

    for name in ("model_id", "dataset_id"):
        expected = manifest.get(name)
        if name in metrics and metrics[name] != expected:
            raise AlphaArtifactError(f"metrics.json 的 {name} 与模型 manifest 不一致")


def _model_record(
    folder: Path,
    manifest: dict,
    root: Path,
    *,
    storage: str,
) -> dict:
    """把单个模型产物归一化为稳定的登记记录。"""

    model_id = str(manifest.get("model_id") or "")
    model_type = str(manifest.get("model_type") or "")
    status = manifest.get("status")
    if not model_id or not model_type:
        raise AlphaArtifactError(f"模型 manifest 缺少 model_id 或 model_type：{folder}")
    if not isinstance(status, str) or not status:
        raise AlphaArtifactError(f"模型 {model_id} 缺少 status")
    decision = manifest.get("decision")
    if decision is not None and not isinstance(decision, str):
        raise AlphaArtifactError(f"模型 {model_id} 的 decision 必须是字符串或 null")

    config_path = folder / "config.json"
    features_path = folder / "features.json"
    metrics_path = folder / "metrics.json"
    config = _read_artifact_json(config_path, f"模型 {model_id} config")
    features_file = _read_artifact_json(features_path, f"模型 {model_id} features")
    metrics = _read_artifact_json(metrics_path, f"模型 {model_id} metrics")
    _validate_config(manifest, config)
    _validate_metrics(manifest, metrics)
    features = _feature_payload(manifest, features_file)

    declared_evaluation = manifest.get("evaluation_sha256")
    if declared_evaluation:
        actual_evaluation = file_sha256(metrics_path)
        if actual_evaluation != str(declared_evaluation):
            raise AlphaArtifactError(f"模型 {model_id} metrics 哈希与 manifest 不一致")

    snapshot = model_artifact_snapshot(model_id, root=root)
    model_snapshot = snapshot["model"]
    for item in model_snapshot["files"]:
        if item.get("matches_manifest") is not True:
            raise AlphaArtifactError(
                f"模型 {model_id} 文件哈希与 manifest 不一致：{item.get('path')}"
            )
    folds = list(model_snapshot.get("folds") or [])

    relative_folder = folder.relative_to(root).as_posix()
    return {
        "model_id": model_id,
        "model_dir": relative_folder,
        "storage": storage,
        "active": storage == "active",
        "status_scope": "training",
        "model_type": model_type,
        "member_type": manifest.get("member_type"),
        "status": status,
        "decision": decision,
        "decision_record": metrics.get("decision"),
        "created_at": manifest.get("created_at"),
        "evaluated_at": manifest.get("evaluated_at"),
        "label": manifest.get("label"),
        "target": manifest.get("target"),
        "universe": manifest.get("universe"),
        "benchmark": manifest.get("benchmark"),
        "dataset_id": manifest.get("dataset_id"),
        "features": features,
        "window": _fold_window(folds),
        "folds": folds,
        "seeds": list(manifest.get("seeds") or []),
        "parameters": config,
        "metrics": _metric_summary(metrics),
        "artifacts": {
            "analysis_dir": (
                f"{relative_folder}/analysis"
                if (folder / "analysis").is_dir()
                else None
            ),
            "manifest_sha256": model_snapshot["manifest_sha256"],
            "config_sha256": file_sha256(config_path),
            "features_sha256": file_sha256(features_path),
            "metrics_sha256": file_sha256(metrics_path),
            "model_files": model_snapshot["files"],
            "predictions": snapshot["predictions"],
            "dataset": snapshot["dataset"],
            "artifact_sha256": snapshot["artifact_sha256"],
        },
    }


def _registry_root(root: Path | None) -> Path:
    """返回并校验模型根目录。"""

    target = Path(root or MODEL_DATA_DIR)
    if not target.is_dir():
        raise AlphaRegistryError(f"模型根目录不存在：{target}")
    return target


def _discover_models(root: Path) -> list[tuple[Path, dict]]:
    """发现模型目录并校验目录名与 manifest ID 一致。"""

    discovered: list[tuple[Path, dict]] = []
    directories = []
    for base, storage in (
        (root / "models", "active"),
        (root / "archive" / "models", "archived"),
    ):
        if not base.is_dir():
            continue
        try:
            entries = sorted(base.iterdir(), key=lambda path: path.name)
        except OSError as error:
            raise AlphaRegistryError(f"无法扫描模型根目录：{base}") from error
        directories.extend((folder, storage) for folder in entries if folder.is_dir())
    if not directories:
        reserved = {
            "archive",
            "audit",
            "baselines",
            "datasets",
            "models",
            "runs",
            "splits",
        }
        try:
            entries = sorted(root.iterdir(), key=lambda path: path.name)
        except OSError as error:
            raise AlphaRegistryError(f"无法扫描模型根目录：{root}") from error
        directories.extend(
            (folder, "legacy")
            for folder in entries
            if folder.is_dir() and folder.name not in reserved
        )
    for folder, storage in directories:
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_artifact_json(manifest_path, f"模型 {folder.name} manifest")
        if "model_id" not in manifest and "model_type" not in manifest:
            continue
        model_id = manifest.get("model_id")
        if not model_id or not manifest.get("model_type"):
            raise AlphaArtifactError(f"模型 manifest 缺少 model_id 或 model_type：{folder}")
        if str(model_id) != folder.name:
            raise AlphaRegistryError(
                f"模型目录与 manifest ID 不一致：{folder.name} != {model_id}"
            )
        discovered.append((folder, manifest))
    if not discovered:
        raise AlphaRegistryError(f"模型根目录下没有可登记模型：{root}")
    identifiers = [str(manifest["model_id"]) for _, manifest in discovered]
    if len(set(identifiers)) != len(identifiers):
        raise AlphaRegistryError("模型根目录存在重复 model_id")
    return discovered


def rebuild_registry(root: Path | None = None) -> dict:
    """扫描全部模型，完整校验后原子重建模型索引。"""

    output_root = _registry_root(root)
    discovered = _discover_models(output_root)
    downstream, run_link_count = scan_downstream_runs(
        output_root,
        {
            str(manifest["model_id"]): folder
            for folder, manifest in discovered
        },
    )
    records = []
    for folder, manifest in discovered:
        if (output_root / "archive" / "models") in folder.parents:
            storage = "archived"
        elif (output_root / "models") in folder.parents:
            storage = "active"
        else:
            storage = "legacy"
        record = _model_record(
            folder,
            manifest,
            output_root,
            storage=storage,
        )
        record["downstream"] = downstream[str(record["model_id"])]
        records.append(record)
    records.sort(
        key=lambda item: (str(item.get("created_at") or ""), item["model_id"])
    )
    payload = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_count": len(records),
        "active_model_count": sum(1 for item in records if item["active"]),
        "run_link_count": int(run_link_count),
        "models": records,
    }
    try:
        write_json(output_root / REGISTRY_FILENAME, payload)
    except OSError as error:
        raise AlphaRegistryError(f"无法写入模型索引：{output_root}") from error
    logger.info("模型索引重建完成：%d 个模型", len(records))
    return payload


def load_registry(root: Path | None = None) -> dict:
    """读取并校验模型索引结构。"""

    output_root = _registry_root(root)
    path = output_root / REGISTRY_FILENAME
    payload = _read_json_object(path, "模型索引", AlphaRegistryError)
    if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise AlphaRegistryError(
            f"模型索引 schema 不支持：{payload.get('schema_version')!r}"
        )
    models = payload.get("models")
    if not isinstance(models, list):
        raise AlphaRegistryError("模型索引 models 必须是数组")
    if type(payload.get("model_count")) is not int or payload["model_count"] != len(
        models
    ):
        raise AlphaRegistryError("模型索引 model_count 与记录数不一致")
    identifiers: set[str] = set()
    for item in models:
        if not isinstance(item, dict):
            raise AlphaRegistryError("模型索引记录必须是对象")
        required = {
            "model_id",
            "model_dir",
            "model_type",
            "status",
            "status_scope",
            "features",
            "window",
            "parameters",
            "metrics",
            "artifacts",
            "downstream",
        }
        missing = sorted(required - set(item))
        if missing:
            raise AlphaRegistryError(f"模型索引记录缺少字段：{missing}")
        model_id = item.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise AlphaRegistryError("模型索引记录缺少 model_id")
        if item.get("storage") == "active":
            expected_dir = f"models/{model_id}"
        elif item.get("storage") == "archived":
            expected_dir = f"archive/models/{model_id}"
        else:
            expected_dir = model_id
        if item.get("model_dir") != expected_dir:
            raise AlphaRegistryError(f"模型索引目录与 ID 不一致：{model_id}")
        if model_id in identifiers:
            raise AlphaRegistryError(f"模型索引存在重复 ID：{model_id}")
        if item.get("status_scope") != "training":
            raise AlphaRegistryError(
                f"模型 {model_id} 的 status_scope 非法"
            )
        downstream = item.get("downstream")
        if not isinstance(downstream, dict) or set(downstream) != set(
            DOWNSTREAM_KINDS
        ):
            raise AlphaRegistryError(
                f"模型 {model_id} 的 downstream 结构非法"
            )
        for name in DOWNSTREAM_KINDS:
            links = downstream.get(name)
            if not isinstance(links, list):
                raise AlphaRegistryError(
                    f"模型 {model_id} 的 downstream.{name} 必须是数组"
                )
            for link in links:
                if not isinstance(link, dict):
                    raise AlphaRegistryError(
                        f"模型 {model_id} 的 downstream.{name} 记录非法"
                    )
                digest = link.get("manifest_sha256")
                if not isinstance(digest, str) or len(digest) != 64:
                    raise AlphaRegistryError(
                        f"模型 {model_id} 的运行 manifest_sha256 非法"
                    )
        identifiers.add(model_id)
        artifact_sha256 = (item.get("artifacts") or {}).get("artifact_sha256")
        if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
            raise AlphaRegistryError(f"模型 {model_id} 的 artifact_sha256 非法")
    expected_links = sum(
        len(links)
        for item in models
        for links in item["downstream"].values()
    )
    if (
        type(payload.get("run_link_count")) is not int
        or payload["run_link_count"] != expected_links
    ):
        raise AlphaRegistryError("模型索引 run_link_count 与运行链接数不一致")
    return payload


def get_registered_model(model_id: str, root: Path | None = None) -> dict:
    """按模型 ID 返回登记记录。"""

    if not isinstance(model_id, str) or not model_id:
        raise AlphaRegistryError(f"非法模型 ID：{model_id!r}")
    for item in load_registry(root)["models"]:
        if item["model_id"] == model_id:
            return item
    raise AlphaRegistryError(f"模型未登记：{model_id}")
