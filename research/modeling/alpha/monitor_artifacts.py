"""模型监控来源校验、产物编排和命令行共用入口。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha.artifacts import (
    file_sha256,
    model_dir,
    prediction_sha256,
    validate_dev_range,
    write_json,
)
from alpha.config import MODEL_DATA_DIR
from alpha.errors import AlphaMonitorError
from alpha.logging_conf import get_logger
from alpha.monitor_inputs import DEPLOYED_SMOOTHING_WEEKS
from alpha.monitor_metrics import evaluate_monitor_frames, simulate_score_decay
from alpha.monitor_plan_sources import load_plan_monitor_source
from alpha.monitor_policy import (
    DEFAULT_ACCEPTANCE_ID,
    DEFAULT_EXPOSURE_ANALYSIS_ID,
    DEFAULT_MODEL_ID,
    MONITOR_SCHEMA_VERSION,
    MonitorPolicy,
    number,
)
from alpha.predict import load_predictions
from alpha.registry import get_registered_model
from common.paths import PROJECT_ROOT

MONITOR_COMPLETE_SCHEMA_VERSION = "alpha_model_monitor_complete_v1"
MONITOR_RESULT_FILES = (
    "alerts.csv",
    "decay_simulation.csv",
    "manifest.json",
    "report.json",
    "report.md",
    "timeseries.csv",
)

logger = get_logger("alpha.monitor")


def _read_json(path: Path, label: str) -> dict:
    """读取监控输入 JSON 对象。"""

    if not path.is_file():
        raise AlphaMonitorError(f"{label}不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlphaMonitorError(f"无法读取{label}：{path}") from error
    if not isinstance(payload, dict):
        raise AlphaMonitorError(f"{label}必须是 JSON 对象：{path}")
    return payload


def _load_dataset(root: Path, dataset_id: str) -> tuple[pd.DataFrame, dict]:
    """从指定模型根目录读取数据集并校验 manifest ID。"""

    folder = root / "datasets" / dataset_id
    manifest = _read_json(folder / "manifest.json", "数据集 manifest")
    path = folder / "data.parquet"
    if manifest.get("dataset_id") != dataset_id or not path.is_file():
        raise AlphaMonitorError(f"数据集产物不完整：{dataset_id}")
    return pd.read_parquet(path), manifest


def _verify_declared_files(folder: Path, files: dict, label: str) -> None:
    """校验 manifest 声明的文件哈希。"""

    if not isinstance(files, dict) or not files:
        raise AlphaMonitorError(f"{label}没有文件哈希")
    for relative, expected in files.items():
        path = folder / str(relative)
        if not path.is_file():
            raise AlphaMonitorError(f"{label}文件缺失：{path}")
        if file_sha256(path) != str(expected):
            raise AlphaMonitorError(f"{label}文件哈希不一致：{path}")


def _monitor_code_sha256() -> str:
    """汇总监控计算与报告代码哈希。"""

    root = PROJECT_ROOT
    digest = hashlib.sha256()
    for relative in (
        "research/modeling/alpha/monitor.py",
        "research/modeling/alpha/monitor_policy.py",
        "research/modeling/alpha/monitor_inputs.py",
        "research/modeling/alpha/monitor_metrics.py",
        "research/modeling/alpha/monitor_plan_sources.py",
        "research/modeling/alpha/monitor_artifacts.py",
        "research/modeling/alpha/monitor_report.py",
        "research/modeling/alpha/portfolio.py",
        "research/modeling/alpha/evaluate.py",
        "research/modeling/alpha/evaluation_plan.py",
        "research/modeling/alpha/ledger_acceptance.py",
        "core/data/scripts/build_alpha/monitor.py",
    ):
        path = root / relative
        if path.is_file():
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _monitor_id(payload: dict) -> str:
    """由不可变输入生成监控 ID。"""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "p5_2_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _result_files(folder: Path) -> dict:
    """返回监控结果文件哈希。"""

    return {
        name: file_sha256(folder / name)
        for name in MONITOR_RESULT_FILES
        if (folder / name).is_file()
    }


def _complete_payload(monitor_id: str, folder: Path) -> dict:
    """构造完成标记。"""

    return {
        "schema_version": MONITOR_COMPLETE_SCHEMA_VERSION,
        "monitor_id": monitor_id,
        "files": _result_files(folder),
        "complete": True,
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """原子写入 CSV。"""

    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _install_directory(temporary: Path, target: Path) -> None:
    """用完整临时目录替换目标，失败时保留旧目录。"""

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


def _source_and_frames(
    model_id: str,
    acceptance_id: str,
    exposure_analysis_id: str,
    *,
    root: Path,
    start: str | None,
    end: str | None,
):
    """读取并校验正式模型、账本和暴露来源。"""

    record = get_registered_model(model_id, root)
    model_folder = model_dir(model_id, root)
    manifest_path = model_folder / "manifest.json"
    manifest = _read_json(manifest_path, "模型 manifest")
    model_manifest_sha256 = file_sha256(manifest_path)
    registered_manifest = (record.get("artifacts") or {}).get("manifest_sha256")
    registered_prediction = (
        (record.get("artifacts") or {}).get("predictions") or {}
    ).get("sha256")
    if registered_manifest != model_manifest_sha256:
        raise AlphaMonitorError("模型 manifest 与登记索引不一致，请先重建索引")
    if registered_prediction != manifest.get("prediction_sha256"):
        raise AlphaMonitorError("模型预测哈希与登记索引不一致，请先重建索引")
    _, _, predictions = load_predictions(
        model_id,
        start=start,
        end=end,
        root=root,
    )
    dataset_id = str(manifest.get("dataset_id") or "")
    data, dataset_manifest = _load_dataset(root, dataset_id)
    selected_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(predictions["trade_date"]).unique())
    )

    acceptance_folder = root / "runs" / "acceptance" / acceptance_id
    acceptance_manifest = _read_json(
        acceptance_folder / "manifest.json",
        "P4 验收 manifest",
    )
    if (
        acceptance_manifest.get("analysis_id") != acceptance_id
        or acceptance_manifest.get("status") != "pass_p4_4a"
    ):
        raise AlphaMonitorError(f"P4 验收不是正式通过状态：{acceptance_id}")
    candidates = acceptance_manifest.get("candidates") or {}
    candidate_names = [
        name
        for name, item in candidates.items()
        if (item.get("spec") or {}).get("source") == model_id
        and item.get("complete")
        and item.get("passed")
    ]
    if len(candidate_names) != 1:
        raise AlphaMonitorError("P4 验收中无法唯一确定模型对应的通过候选")
    candidate = candidate_names[0]
    candidate_folder = acceptance_folder / candidate
    complete = _read_json(candidate_folder / "complete.json", "P4 候选完成标记")
    if complete.get("analysis_id") != acceptance_id or complete.get("candidate") != candidate:
        raise AlphaMonitorError("P4 候选完成标记与验收 ID 不一致")
    _verify_declared_files(candidate_folder, complete.get("files") or {}, "P4 候选")
    strategy = _read_json(candidate_folder / "strategy.json", "P4 策略")
    provider_params = (strategy.get("signal") or {}).get("provider", {}).get("params", {})
    if provider_params.get("model_id") != model_id:
        raise AlphaMonitorError("P4 策略 provider 模型与监控模型不一致")
    rebalance_dates = pd.DatetimeIndex(
        pd.to_datetime(provider_params.get("rebalance_dates") or [])
    )
    rebalance_dates = rebalance_dates[
        (rebalance_dates >= selected_dates.min())
        & (rebalance_dates <= selected_dates.max())
    ]
    if len(rebalance_dates) != len(selected_dates) or not rebalance_dates.difference(
        selected_dates
    ).empty:
        raise AlphaMonitorError("P4 策略信号日与模型预测日期不一致")
    candidate_metrics = _read_json(candidate_folder / "metrics.json", "P4 指标")
    checks = candidate_metrics.get("checks") or {}
    if not checks.get("passed"):
        raise AlphaMonitorError("P4 候选验收检查未通过")
    reference_turnover = number(
        (candidate_metrics.get("metrics") or {}).get("turnover_two_way_annual")
    )
    if reference_turnover is None or reference_turnover <= 0:
        raise AlphaMonitorError("P4 候选缺少基准换手")

    exposure_folder = model_folder / "analysis" / exposure_analysis_id
    exposure_manifest = _read_json(exposure_folder / "manifest.json", "P3-4 manifest")
    if (
        exposure_manifest.get("analysis_id") != exposure_analysis_id
        or exposure_manifest.get("status") != "complete"
        or exposure_manifest.get("model_id") != model_id
        or exposure_manifest.get("prediction_sha256")
        != manifest.get("prediction_sha256")
    ):
        raise AlphaMonitorError("P3-4 暴露来源与模型预测不一致")
    style_path = exposure_folder / "portfolio_style_series.csv"
    declared_style_hash = (exposure_manifest.get("files") or {}).get(
        "portfolio_style_series.csv"
    )
    if not declared_style_hash or file_sha256(style_path) != str(declared_style_hash):
        raise AlphaMonitorError("P3-4 风格暴露文件哈希不一致")
    style_series = pd.read_csv(style_path)
    ledger = pd.read_parquet(candidate_folder / "daily.parquet")
    holdings = pd.read_parquet(candidate_folder / "holdings.parquet")
    sources = {
        "model_id": model_id,
        "model_manifest_sha256": model_manifest_sha256,
        "model_artifact_sha256": record["artifacts"]["artifact_sha256"],
        "prediction_sha256": prediction_sha256(predictions),
        "dataset_id": dataset_id,
        "dataset_manifest_sha256": file_sha256(
            root / "datasets" / dataset_id / "manifest.json"
        ),
        "dataset_data_sha256": file_sha256(
            root / "datasets" / dataset_id / "data.parquet"
        ),
        "acceptance_id": acceptance_id,
        "acceptance_candidate": candidate,
        "acceptance_complete_sha256": file_sha256(candidate_folder / "complete.json"),
        "strategy_sha256": file_sha256(candidate_folder / "strategy.json"),
        "ledger_sha256": file_sha256(candidate_folder / "daily.parquet"),
        "holdings_sha256": file_sha256(candidate_folder / "holdings.parquet"),
        "metrics_sha256": file_sha256(candidate_folder / "metrics.json"),
        "exposure_analysis_id": exposure_analysis_id,
        "exposure_manifest_sha256": file_sha256(exposure_folder / "manifest.json"),
        "style_series_sha256": file_sha256(style_path),
    }
    return {
        "record": record,
        "manifest": manifest,
        "dataset_manifest": dataset_manifest,
        "predictions": predictions,
        "data": data,
        "rebalance_dates": rebalance_dates,
        "style_series": style_series,
        "ledger": ledger,
        "holdings": holdings,
        "reference_turnover": reference_turnover,
        "sources": sources,
        "evaluation_id": None,
        "candidate": candidate,
        "smoothing_weeks": DEPLOYED_SMOOTHING_WEEKS,
        "style_kind": "p3_4_daily",
    }


def run_model_monitor(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    acceptance_id: str = DEFAULT_ACCEPTANCE_ID,
    exposure_analysis_id: str = DEFAULT_EXPOSURE_ANALYSIS_ID,
    evaluation_id: str | None = None,
    candidate: str | None = None,
    start: str | None = None,
    end: str | None = None,
    policy: MonitorPolicy | None = None,
    force: bool = False,
    root: Path | None = None,
) -> dict:
    """校验来源、计算监控并原子写入不可变监控包。"""

    from alpha.monitor_report import write_monitor_package

    output_root = root or MODEL_DATA_DIR
    settings = policy or MonitorPolicy()
    source = (
        load_plan_monitor_source(
            model_id,
            acceptance_id,
            evaluation_id,
            candidate,
            root=output_root,
            start=start,
            end=end,
        )
        if evaluation_id
        else _source_and_frames(
            model_id,
            acceptance_id,
            exposure_analysis_id,
            root=output_root,
            start=start,
            end=end,
        )
    )
    dates = source["rebalance_dates"]
    monitor_start = start or str(dates.min().date())
    monitor_end = end or str(dates.max().date())
    validate_dev_range(monitor_start, monitor_end)
    code_hash = _monitor_code_sha256()
    identity = {
        "schema_version": MONITOR_SCHEMA_VERSION,
        "model_id": model_id,
        "start": monitor_start,
        "end": monitor_end,
        "sources": source["sources"],
        "policy": settings.to_dict(),
        "smoothing_weeks": source["smoothing_weeks"],
        "evaluation_id": source["evaluation_id"],
        "candidate": source["candidate"],
        "style_kind": source["style_kind"],
        "code_sha256": code_hash,
    }
    monitor_id = _monitor_id(identity)
    target = model_dir(model_id, output_root) / "monitoring" / monitor_id
    complete_path = target / "complete.json"
    if complete_path.is_file() and not force:
        payload = _read_json(complete_path, "监控完成标记")
        if payload.get("schema_version") == MONITOR_COMPLETE_SCHEMA_VERSION:
            if payload.get("files") != _result_files(target):
                raise AlphaMonitorError("已有监控包文件哈希不一致")
            logger.info("复用已有模型监控：%s", monitor_id)
            return _read_json(target / "manifest.json", "监控 manifest")
    if target.exists() and not force:
        raise AlphaMonitorError(f"监控目录已存在但未完整生成：{target}")

    result = evaluate_monitor_frames(
        source["predictions"],
        source["data"],
        source["ledger"],
        source["rebalance_dates"],
        source["style_series"],
        source["holdings"],
        reference_turnover=source["reference_turnover"],
        policy=settings,
        start=monitor_start,
        end=monitor_end,
        smoothing_weeks=source["smoothing_weeks"],
        style_already_rolling=source["style_kind"] == "evaluation_rolling",
    )
    decay = simulate_score_decay(
        source["predictions"],
        source["data"],
        source["rebalance_dates"],
        policy=settings,
        start=monitor_start,
        end=monitor_end,
        smoothing_weeks=source["smoothing_weeks"],
    )
    if not decay["alert_triggered"]:
        raise AlphaMonitorError("衰减演练未触发 RankIC alert")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    _write_csv(result.timeseries.reset_index(), temporary / "timeseries.csv")
    _write_csv(result.alerts, temporary / "alerts.csv")
    _write_csv(decay["timeseries"], temporary / "decay_simulation.csv")
    payload = write_monitor_package(
        temporary,
        monitor_id=monitor_id,
        identity=identity,
        result=result,
        decay=decay,
        sources=source["sources"],
    )
    write_json(temporary / "manifest.json", payload)
    write_json(
        temporary / "complete.json",
        _complete_payload(monitor_id, temporary),
    )
    _install_directory(temporary, target)
    logger.info("模型监控已写入：%s", target)
    return payload


__all__ = ["MONITOR_COMPLETE_SCHEMA_VERSION", "run_model_monitor"]
