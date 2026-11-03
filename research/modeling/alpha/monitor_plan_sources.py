"""通用评估与完整账本的监控来源读取。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha.artifacts import file_sha256, model_dir, prediction_sha256
from alpha.errors import AlphaMonitorError
from alpha.evaluation_plan import EvaluationPlan
from alpha.ledger_acceptance import LEDGER_ACCEPTANCE_SCHEMA_VERSION
from alpha.monitor_policy import number
from alpha.predict import load_predictions
from alpha.registry import get_registered_model


def _read_json(path: Path, label: str) -> dict:
    """读取监控来源 JSON。"""

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
    """读取模型训练数据集。"""

    folder = root / "datasets" / dataset_id
    manifest = _read_json(folder / "manifest.json", "数据集 manifest")
    path = folder / "data.parquet"
    if manifest.get("dataset_id") != dataset_id or not path.is_file():
        raise AlphaMonitorError(f"数据集产物不完整：{dataset_id}")
    return pd.read_parquet(path), manifest


def _verify_files(folder: Path, files: dict, label: str) -> None:
    """校验产物完成标记中的文件哈希。"""

    if not isinstance(files, dict) or not files:
        raise AlphaMonitorError(f"{label}没有文件哈希")
    for relative, expected in files.items():
        path = folder / str(relative)
        if not path.is_file() or file_sha256(path) != str(expected):
            raise AlphaMonitorError(f"{label}文件缺失或哈希不一致：{path}")


def load_plan_monitor_source(
    model_id: str,
    acceptance_id: str,
    evaluation_id: str,
    candidate_name: str | None,
    *,
    root: Path,
    start: str | None,
    end: str | None,
) -> dict:
    """读取计划评估、通用账本和候选风险时序。"""

    record = get_registered_model(model_id, root)
    model_folder = model_dir(model_id, root)
    manifest_path = model_folder / "manifest.json"
    manifest = _read_json(manifest_path, "模型 manifest")
    model_manifest_sha256 = file_sha256(manifest_path)
    registered_manifest = (record.get("artifacts") or {}).get(
        "manifest_sha256"
    )
    registered_prediction = (
        (record.get("artifacts") or {}).get("predictions") or {}
    ).get("sha256")
    if registered_manifest != model_manifest_sha256:
        raise AlphaMonitorError("模型 manifest 与登记索引不一致，请先重建索引")
    if registered_prediction != manifest.get("prediction_sha256"):
        raise AlphaMonitorError("模型预测哈希与登记索引不一致，请先重建索引")

    acceptance_folder = root / "runs" / "acceptance" / acceptance_id
    acceptance_manifest = _read_json(
        acceptance_folder / "manifest.json",
        "通用账本 manifest",
    )
    if (
        acceptance_manifest.get("schema_version")
        != LEDGER_ACCEPTANCE_SCHEMA_VERSION
        or acceptance_manifest.get("analysis_id") != acceptance_id
        or acceptance_manifest.get("status") != "pass_acceptance"
    ):
        raise AlphaMonitorError("通用账本不是正式通过状态")
    if acceptance_manifest.get("evaluation_id") != evaluation_id:
        raise AlphaMonitorError("通用账本与评估 ID 不一致")
    candidates = acceptance_manifest.get("candidates") or {}
    matching = [
        name
        for name, item in candidates.items()
        if (item.get("source") or {}).get("kind") == "model"
        and (item.get("source") or {}).get("id") == model_id
        and item.get("complete")
        and item.get("passed")
    ]
    candidate = candidate_name or (
        matching[0] if len(matching) == 1 else None
    )
    if not candidate or candidate not in candidates:
        raise AlphaMonitorError("通用账本中无法唯一确定模型候选")
    candidate_folder = acceptance_folder / candidate
    complete = _read_json(
        candidate_folder / "complete.json",
        "通用账本候选完成标记",
    )
    if (
        complete.get("analysis_id") != acceptance_id
        or complete.get("candidate") != candidate
    ):
        raise AlphaMonitorError("通用账本候选完成标记不一致")
    _verify_files(
        candidate_folder,
        complete.get("files") or {},
        "通用账本候选",
    )
    candidate_metrics = _read_json(
        candidate_folder / "metrics.json",
        "通用账本候选指标",
    )
    if not (candidate_metrics.get("checks") or {}).get("passed"):
        raise AlphaMonitorError("通用账本候选检查未通过")
    reference_turnover = number(
        (candidate_metrics.get("metrics") or {}).get(
            "turnover_two_way_annual"
        )
    )
    if reference_turnover is None or reference_turnover <= 0:
        raise AlphaMonitorError("通用账本候选缺少基准换手")

    evaluation_folder = root / "runs" / "evaluation" / evaluation_id
    evaluation_manifest = _read_json(
        evaluation_folder / "manifest.json",
        "评估 manifest",
    )
    if (
        evaluation_manifest.get("analysis_id") != evaluation_id
        or evaluation_manifest.get("status") != "pass_evaluation"
    ):
        raise AlphaMonitorError("评估状态或 ID 不一致")
    plan = EvaluationPlan.from_dict(evaluation_manifest.get("plan") or {})
    candidate_spec = plan.by_name(candidate)
    if (
        candidate_spec.source.kind != "model"
        or candidate_spec.source.id != model_id
    ):
        raise AlphaMonitorError("评估候选来源与监控模型不一致")

    style_relative = f"risk/{candidate}/style_series.csv"
    style_path = evaluation_folder / style_relative
    declared_style_hash = (evaluation_manifest.get("files") or {}).get(
        style_relative
    )
    if (
        not style_path.is_file()
        or not declared_style_hash
        or file_sha256(style_path) != str(declared_style_hash)
    ):
        raise AlphaMonitorError("评估风险风格文件缺失或哈希不一致")
    style_wide = pd.read_csv(style_path)
    style_series = style_wide.melt(
        id_vars="trade_date",
        var_name="style",
        value_name="exposure",
    ).dropna(subset=["exposure"])
    style_series["coverage"] = 1.0
    style_series["holdings"] = int(candidate_spec.holding_policy["top_n"])

    rebalance_dates = pd.DatetimeIndex(
        pd.to_datetime(
            (acceptance_manifest.get("scope") or {}).get(
                "rebalance_dates"
            )
            or []
        )
    )
    if rebalance_dates.empty:
        raise AlphaMonitorError("通用账本没有调仓日")
    if start is not None:
        rebalance_dates = rebalance_dates[
            rebalance_dates >= pd.Timestamp(start)
        ]
    if end is not None:
        rebalance_dates = rebalance_dates[
            rebalance_dates <= pd.Timestamp(end)
        ]
    _, _, predictions = load_predictions(
        model_id,
        start=start,
        end=end,
        root=root,
    )
    dataset_id = str(manifest.get("dataset_id") or "")
    data, dataset_manifest = _load_dataset(root, dataset_id)
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
        "acceptance_complete_sha256": file_sha256(
            candidate_folder / "complete.json"
        ),
        "strategy_sha256": file_sha256(
            candidate_folder / "strategy.json"
        ),
        "ledger_sha256": file_sha256(candidate_folder / "daily.parquet"),
        "holdings_sha256": file_sha256(
            candidate_folder / "holdings.parquet"
        ),
        "metrics_sha256": file_sha256(
            candidate_folder / "metrics.json"
        ),
        "evaluation_id": evaluation_id,
        "evaluation_manifest_sha256": file_sha256(
            evaluation_folder / "manifest.json"
        ),
        "risk_style_series_sha256": file_sha256(style_path),
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
        "evaluation_id": evaluation_id,
        "candidate": candidate,
        "smoothing_weeks": candidate_spec.score_policy.smoothing_weeks,
        "style_kind": "evaluation_rolling",
    }


__all__ = ["load_plan_monitor_source"]
