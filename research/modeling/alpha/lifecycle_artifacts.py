"""模型生命周期来源校验、产物写入和不可变读取。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from alpha.artifacts import file_sha256, model_dir, prediction_sha256, write_json
from alpha.config import MODEL_DATA_DIR
from alpha.errors import AlphaLifecycleError
from alpha.lifecycle_engine import evaluate_lifecycle
from alpha.lifecycle_policy import (
    DEFAULT_FALLBACK_MODEL_ID,
    DEFAULT_MONITOR_ID,
    DEFAULT_PRIMARY_MODEL_ID,
    LIFECYCLE_COMPLETE_SCHEMA_VERSION,
    LIFECYCLE_RESULT_FILES,
    LIFECYCLE_SCHEMA_VERSION,
    LifecyclePolicy,
)
from alpha.lifecycle_report import write_lifecycle_package
from alpha.logging_conf import get_logger
from alpha.monitor_policy import MONITOR_SCHEMA_VERSION, number
from alpha.predict import load_model_manifest, load_predictions
from alpha.registry import get_registered_model
from common.paths import PROJECT_ROOT

logger = get_logger("alpha.lifecycle")


def _read_json(path: Path, label: str) -> dict:
    """读取 JSON 对象。"""

    if not path.is_file():
        raise AlphaLifecycleError(f"{label}不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlphaLifecycleError(f"无法读取{label}：{path}") from error
    if not isinstance(payload, dict):
        raise AlphaLifecycleError(f"{label}必须是 JSON 对象：{path}")
    return payload


def _verify_files(folder: Path, declared: dict, label: str) -> None:
    """校验一组声明文件。"""

    if not isinstance(declared, dict) or not declared:
        raise AlphaLifecycleError(f"{label}没有文件哈希")
    for relative, expected in declared.items():
        path = folder / str(relative)
        if not path.is_file():
            raise AlphaLifecycleError(f"{label}文件缺失：{path}")
        if file_sha256(path) != str(expected):
            raise AlphaLifecycleError(f"{label}文件哈希不一致：{path}")


def _code_sha256() -> str:
    """汇总生命周期计算和入口代码哈希。"""

    root = PROJECT_ROOT
    digest = hashlib.sha256()
    for relative in (
        "research/modeling/alpha/lifecycle.py",
        "research/modeling/alpha/lifecycle_policy.py",
        "research/modeling/alpha/lifecycle_engine.py",
        "research/modeling/alpha/lifecycle_report.py",
        "research/modeling/alpha/lifecycle_artifacts.py",
        "research/modeling/alpha/monitor_policy.py",
        "core/data/scripts/build_alpha/lifecycle.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _lifecycle_id(payload: dict) -> str:
    """由不可变来源和策略生成生命周期 ID。"""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "p5_3_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """原子写入 CSV。"""

    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _complete_payload(lifecycle_id: str, folder: Path) -> dict:
    """构造完成标记。"""

    return {
        "schema_version": LIFECYCLE_COMPLETE_SCHEMA_VERSION,
        "lifecycle_id": lifecycle_id,
        "files": {
            name: file_sha256(folder / name)
            for name in LIFECYCLE_RESULT_FILES
            if (folder / name).is_file()
        },
        "complete": True,
    }


def _install_directory(temporary: Path, target: Path) -> None:
    """用完整临时目录替换目标；失败时保留旧目录。"""

    backup = target.with_name(f"{target.name}.old")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        target.rename(backup)
    try:
        temporary.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _load_sources(
    model_id: str,
    fallback_model_id: str,
    monitor_id: str,
    root: Path,
) -> dict:
    """读取并校验主模型、回滚模型、监控和账本来源。"""

    primary_record = get_registered_model(model_id, root)
    fallback_record = get_registered_model(fallback_model_id, root)
    primary_manifest = load_model_manifest(model_id, root)
    fallback_manifest = load_model_manifest(fallback_model_id, root)
    fallback_decision = str(fallback_manifest.get("decision") or "")
    if not fallback_decision.startswith("pass"):
        raise AlphaLifecycleError("回滚模型没有通过既有决策门")

    model_folder = model_dir(model_id, root)
    monitor_folder = model_folder / "monitoring" / monitor_id
    monitor_manifest = _read_json(
        monitor_folder / "manifest.json", "P5-2 监控 manifest"
    )
    monitor_complete = _read_json(
        monitor_folder / "complete.json", "P5-2 监控完成标记"
    )
    if (
        monitor_manifest.get("schema_version") != MONITOR_SCHEMA_VERSION
        or monitor_manifest.get("monitor_id") != monitor_id
        or monitor_complete.get("complete") is not True
    ):
        raise AlphaLifecycleError("P5-2 监控包状态或 ID 不一致")
    _verify_files(
        monitor_folder,
        monitor_complete.get("files") or {},
        "P5-2 监控",
    )
    monitor_sources = monitor_manifest.get("sources") or {}
    if monitor_sources.get("model_id") != model_id:
        raise AlphaLifecycleError("P5-2 监控对象与生命周期主模型不一致")
    if monitor_sources.get("model_manifest_sha256") != (
        (primary_record.get("artifacts") or {}).get("manifest_sha256")
    ):
        raise AlphaLifecycleError("主模型 manifest 与登记索引不一致")
    if monitor_sources.get("model_artifact_sha256") != (
        primary_record.get("artifacts") or {}
    ).get("artifact_sha256"):
        raise AlphaLifecycleError("主模型产物与监控来源不一致")

    _, _, primary_predictions = load_predictions(model_id, root=root)
    _, _, fallback_predictions = load_predictions(fallback_model_id, root=root)
    primary_hash = prediction_sha256(primary_predictions)
    fallback_hash = prediction_sha256(fallback_predictions)
    if primary_hash != monitor_sources.get("prediction_sha256"):
        raise AlphaLifecycleError("主模型预测与监控来源不一致")
    primary_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(primary_predictions["trade_date"]).unique())
    )
    fallback_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(fallback_predictions["trade_date"]).unique())
    )
    if not primary_dates.equals(fallback_dates):
        raise AlphaLifecycleError("主模型与回滚模型预测日期不一致")

    acceptance_id = str(monitor_sources.get("acceptance_id") or "")
    candidate = str(monitor_sources.get("acceptance_candidate") or "")
    metrics_path = (
        root / "runs" / "acceptance" / acceptance_id / candidate / "metrics.json"
    )
    strategy_path = (
        root / "runs" / "acceptance" / acceptance_id / candidate / "strategy.json"
    )
    acceptance_metrics = _read_json(metrics_path, "P4 候选指标")
    if file_sha256(metrics_path) != monitor_sources.get("metrics_sha256"):
        raise AlphaLifecycleError("P4 指标与监控来源不一致")
    if file_sha256(strategy_path) != monitor_sources.get("strategy_sha256"):
        raise AlphaLifecycleError("P4 策略与监控来源不一致")
    reference_turnover = number(
        (acceptance_metrics.get("metrics") or {}).get("turnover_two_way_annual")
    )
    if reference_turnover is None or reference_turnover <= 0:
        raise AlphaLifecycleError("P4 候选缺少基准换手")

    timeseries_path = monitor_folder / "timeseries.csv"
    timeseries = pd.read_csv(timeseries_path)
    if "trade_date" not in timeseries:
        raise AlphaLifecycleError("P5-2 监控时序缺少 trade_date")
    timeseries["trade_date"] = pd.to_datetime(timeseries["trade_date"])
    monitor_dates = pd.DatetimeIndex(timeseries["trade_date"])
    if not monitor_dates.equals(primary_dates):
        raise AlphaLifecycleError("P5-2 监控日期与模型预测日期不一致")

    sources = {
        "model_id": model_id,
        "fallback_model_id": fallback_model_id,
        "monitor_id": monitor_id,
        "monitor_manifest_sha256": file_sha256(monitor_folder / "manifest.json"),
        "monitor_timeseries_sha256": file_sha256(timeseries_path),
        "primary_prediction_sha256": primary_hash,
        "fallback_prediction_sha256": fallback_hash,
        "primary_artifact_sha256": primary_record["artifacts"]["artifact_sha256"],
        "fallback_artifact_sha256": fallback_record["artifacts"]["artifact_sha256"],
        "primary_manifest_sha256": file_sha256(
            model_folder / "manifest.json"
        ),
        "fallback_manifest_sha256": file_sha256(
            model_dir(fallback_model_id, root) / "manifest.json"
        ),
        "acceptance_id": acceptance_id,
        "acceptance_candidate": candidate,
        "acceptance_metrics_sha256": file_sha256(metrics_path),
        "acceptance_strategy_sha256": file_sha256(strategy_path),
    }
    return {
        "primary_manifest": primary_manifest,
        "fallback_manifest": fallback_manifest,
        "timeseries": timeseries,
        "reference_turnover": reference_turnover,
        "sources": sources,
    }


def load_model_lifecycle(
    lifecycle_id: str,
    *,
    model_id: str = DEFAULT_PRIMARY_MODEL_ID,
    root: Path | None = None,
) -> dict:
    """读取并校验完整生命周期包。"""

    output_root = root or MODEL_DATA_DIR
    folder = model_dir(model_id, output_root) / "lifecycle" / lifecycle_id
    manifest = _read_json(folder / "manifest.json", "生命周期 manifest")
    complete = _read_json(folder / "complete.json", "生命周期完成标记")
    if (
        manifest.get("schema_version") != LIFECYCLE_SCHEMA_VERSION
        or manifest.get("lifecycle_id") != lifecycle_id
        or complete.get("schema_version") != LIFECYCLE_COMPLETE_SCHEMA_VERSION
        or complete.get("lifecycle_id") != lifecycle_id
        or complete.get("complete") is not True
    ):
        raise AlphaLifecycleError("生命周期包 schema、ID 或完成状态不一致")
    if set(complete.get("files") or {}) != set(LIFECYCLE_RESULT_FILES):
        raise AlphaLifecycleError("生命周期完成标记文件清单不完整")
    _verify_files(folder, complete.get("files") or {}, "生命周期")
    sources = manifest.get("sources") or {}
    if (
        sources.get("model_id") != model_id
        or sources.get("model_id") != manifest.get("policy", {}).get(
            "primary_model_id"
        )
    ):
        raise AlphaLifecycleError("生命周期 manifest 的主模型不一致")
    timeseries = pd.read_csv(folder / "timeseries.csv")
    timeseries["trade_date"] = pd.to_datetime(timeseries["trade_date"])
    return {
        "folder": folder,
        "manifest": manifest,
        "timeseries": timeseries,
        "events": pd.read_csv(folder / "events.csv"),
    }


def run_model_lifecycle(
    model_id: str = DEFAULT_PRIMARY_MODEL_ID,
    *,
    fallback_model_id: str = DEFAULT_FALLBACK_MODEL_ID,
    monitor_id: str = DEFAULT_MONITOR_ID,
    policy: LifecyclePolicy | None = None,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
    root: Path | None = None,
) -> dict:
    """校验来源，计算生命周期并原子写入不可变产物。"""

    output_root = Path(root or MODEL_DATA_DIR)
    settings = policy or LifecyclePolicy(
        primary_model_id=model_id,
        fallback_model_id=fallback_model_id,
    )
    if (
        settings.primary_model_id != model_id
        or settings.fallback_model_id != fallback_model_id
    ):
        raise AlphaLifecycleError("生命周期策略中的模型与函数参数不一致")
    source = _load_sources(model_id, fallback_model_id, monitor_id, output_root)
    result = evaluate_lifecycle(
        source["timeseries"],
        reference_turnover=source["reference_turnover"],
        folds=source["primary_manifest"].get("folds") or [],
        policy=settings,
        start=start,
        end=end,
    )
    code_hash = _code_sha256()
    identity = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "model_id": model_id,
        "start": result.summary["start"],
        "end": result.summary["end"],
        "sources": source["sources"],
        "policy": settings.to_dict(),
        "code_sha256": code_hash,
    }
    lifecycle_id = _lifecycle_id(identity)
    target = model_dir(model_id, output_root) / "lifecycle" / lifecycle_id
    if target.is_dir() and not force:
        return load_model_lifecycle(lifecycle_id, model_id=model_id, root=output_root)[
            "manifest"
        ]
    if target.exists() and not force:
        raise AlphaLifecycleError(f"生命周期目录已存在但不完整：{target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    _write_csv(result.timeseries, temporary / "timeseries.csv")
    _write_csv(result.events, temporary / "events.csv")
    payload = write_lifecycle_package(
        temporary,
        lifecycle_id=lifecycle_id,
        identity=identity,
        result=result,
        sources=source["sources"],
    )
    write_json(temporary / "manifest.json", payload)
    write_json(
        temporary / "complete.json",
        _complete_payload(lifecycle_id, temporary),
    )
    _install_directory(temporary, target)
    logger.info("模型生命周期已写入：%s", target)
    return payload


__all__ = ["load_model_lifecycle", "run_model_lifecycle"]
