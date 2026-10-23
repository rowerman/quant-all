"""最终留出集验证的总编排。"""

from __future__ import annotations

import shutil
from pathlib import Path

from alpha.artifacts import file_sha256
from alpha.config import MODEL_DATA_DIR
from alpha.errors import AlphaFinalHoldoutError
from alpha.final_holdout_artifacts import (
    FINAL_HOLDOUT_ROOT,
    code_sha256,
    holdout_id,
    load_final_holdout,
    model_sources,
    write_predictions,
)
from alpha.final_holdout_data import (
    FINAL_HOLDOUT_SCHEMA_VERSION,
    FinalHoldoutSpec,
    build_final_holdout_dataset,
    complete_signal_dates,
    load_holdout_dataset,
    prediction_manifest,
    read_final_holdout_json,
    score_frozen_holdout,
)
from alpha.final_holdout_eval import (
    compute_holdout_exposure,
    evaluate_final_holdout_gate,
    evaluate_holdout_models,
    model_metrics_for_registry,
)
from alpha.final_holdout_ledger import load_candidate_payload, run_candidate
from alpha.final_holdout_report import (
    build_final_holdout_report,
    combined_yearly_frame,
    comparison_frame,
    incomplete_report,
    result_files,
)
from alpha.lifecycle import evaluate_lifecycle
from alpha.lifecycle_policy import LifecyclePolicy
from alpha.monitor import MonitorPolicy
from alpha.monitor_metrics import evaluate_monitor_frames as evaluate_monitor
from scripts.build_alpha.backtest_accept_artifacts import (
    atomic_write_frame,
    atomic_write_json,
    atomic_write_text,
)

def _candidate_map(spec: FinalHoldoutSpec) -> dict[str, tuple[str, int]]:
    """从评估计划生成内部预测映射。"""

    plan = spec.evaluation_plan
    selected_name = plan.selected or plan.candidates[0].name
    selected = plan.by_name(selected_name)
    reference = plan.by_name(plan.reference)
    control = plan.controls[0]
    return {
        selected.name: ("lightgbm", selected.score_policy.smoothing_weeks),
        reference.name: ("ridge", reference.score_policy.smoothing_weeks),
        control.name: ("b4", control.score_policy.smoothing_weeks),
        "ML_LIFECYCLE": ("lifecycle", selected.score_policy.smoothing_weeks),
    }


def run_final_holdout(
    spec: FinalHoldoutSpec | None = None,
    *,
    force: bool = False,
    resume_from: str | None = None,
    root: Path | None = None,
) -> dict:
    """执行一次不可调参的最终留出集验证。"""

    settings = spec or FinalHoldoutSpec()
    candidates = _candidate_map(settings)
    plan = settings.evaluation_plan
    selected_name = plan.selected or plan.candidates[0].name
    selected_weeks = plan.by_name(selected_name).score_policy.smoothing_weeks
    reference_name = plan.reference
    control_name = plan.controls[0].name
    output_root = Path(root or FINAL_HOLDOUT_ROOT)
    dataset = build_final_holdout_dataset(settings)
    data, dataset_manifest = load_holdout_dataset(dataset.dataset_id)
    dates = complete_signal_dates(data, settings)
    predictions, prediction_metadata = score_frozen_holdout(
        data,
        dataset_manifest,
        dates,
        settings,
    )
    source_payload = {
        "primary": model_sources(
            settings.primary_model_id, fold_id=settings.frozen_fold_id
        ),
        "ridge": model_sources(
            settings.ridge_model_id, fold_id=settings.frozen_fold_id
        ),
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "manifest_sha256": file_sha256(dataset.path.parent / "manifest.json"),
            "data_sha256": file_sha256(dataset.path),
        },
        "prediction_metadata": prediction_metadata,
    }
    current_code_hash = code_sha256()
    initial_code_hash = current_code_hash
    computed_id = holdout_id(
        settings,
        dataset_manifest,
        dataset.path,
        source_payload,
        predictions,
        current_code_hash,
    )
    verification_id = resume_from or computed_id
    if (
        not verification_id.startswith("final_holdout_")
        or Path(verification_id).name != verification_id
    ):
        raise AlphaFinalHoldoutError(f"非法续跑 ID：{verification_id!r}")
    target = output_root / verification_id
    if target.exists():
        complete_path = target / "complete.json"
        if complete_path.is_file() and not force and not resume_from:
            return load_final_holdout(verification_id, root=output_root)
        if not resume_from and not force:
            raise AlphaFinalHoldoutError(f"留出集目录已存在但未完成：{target}")
        if force:
            if target.parent.resolve() != output_root.resolve():
                raise AlphaFinalHoldoutError("拒绝删除输出目录之外的路径")
            shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=resume_from is not None)

    try:
        if resume_from:
            previous_sources = read_final_holdout_json(
                target / "sources.json",
                "续跑来源",
            )
            initial_code_hash = str(
                previous_sources.get("initial_code_sha256")
                or previous_sources.get("code_sha256")
                or current_code_hash
            )
            for name in ("primary", "ridge", "dataset", "prediction_metadata"):
                if previous_sources.get(name) != source_payload.get(name):
                    raise AlphaFinalHoldoutError(
                        f"续跑来源与当前留出输入不一致：{name}"
                    )
            previous_predictions = read_final_holdout_json(
                target / "predictions" / "manifest.json",
                "续跑预测 manifest",
            )
            if previous_predictions.get("predictions") != prediction_manifest(
                predictions
            ):
                raise AlphaFinalHoldoutError("续跑预测哈希与当前冻结预测不一致")
        else:
            write_predictions(target, predictions)
        atomic_write_json(
            target / "sources.json",
            {
                **source_payload,
                "initial_code_sha256": initial_code_hash,
                "code_sha256": current_code_hash,
                "holdout_id": verification_id,
                "resume_from": resume_from,
            },
        )
        model_metrics = evaluate_holdout_models(
            predictions,
            data,
            dates,
            smoothing_weeks=selected_weeks,
            selected_name=selected_name,
            reference_name=reference_name,
            control_name=control_name,
        )
        atomic_write_json(
            target / "model_metrics.json",
            model_metrics_for_registry(model_metrics),
        )

        candidate_results: dict[str, dict] = {}
        for candidate_name in (selected_name, reference_name, control_name):
            if resume_from:
                candidate_results[candidate_name] = load_candidate_payload(
                    target,
                    candidate_name,
                )
            else:
                prediction_name, weeks = candidates[candidate_name]
                candidate_results[candidate_name] = run_candidate(
                    root=target,
                    holdout_id=verification_id,
                    candidate_name=candidate_name,
                    prediction_name=prediction_name,
                    smoothing_weeks=weeks,
                    spec=settings,
                    dates=dates,
                )

        exposure = compute_holdout_exposure(
            predictions["lightgbm"],
            data,
            dates,
            top_n=settings.top_n,
            smoothing_weeks=selected_weeks,
        )
        exposure_folder = target / "exposure"
        atomic_write_frame(
            exposure_folder / "style_summary.csv",
            exposure.style_summary,
        )
        atomic_write_frame(
            exposure_folder / "style_series.csv",
            exposure.style_series,
        )
        atomic_write_frame(
            exposure_folder / "industry_summary.csv",
            exposure.industry_summary,
        )
        atomic_write_json(exposure_folder / "summary.json", exposure.summary)

        selected_result = candidate_results[selected_name]["result"]
        reference_turnover = float(
            candidate_results[selected_name]["metrics"][
                "turnover_two_way_annual"
            ]
        )
        monitor = evaluate_monitor(
            predictions["lightgbm"],
            data,
            selected_result.daily.reset_index(),
            dates,
            exposure.style_series,
            selected_result.holdings,
            reference_turnover=reference_turnover,
            policy=MonitorPolicy(),
            start=str(dates.min().date()),
            end=str(dates.max().date()),
            allow_holdout=True,
        )
        monitor_folder = target / "monitor"
        atomic_write_frame(
            monitor_folder / "timeseries.csv",
            monitor.timeseries.reset_index(),
        )
        atomic_write_frame(monitor_folder / "alerts.csv", monitor.alerts)
        atomic_write_json(monitor_folder / "summary.json", monitor.summary)

        lifecycle = evaluate_lifecycle(
            monitor.timeseries.reset_index(),
            reference_turnover=reference_turnover,
            folds=[],
            policy=LifecyclePolicy(),
            start=str(dates.min().date()),
            end=str(dates.max().date()),
            allow_holdout=True,
            include_retrain_schedule=False,
        )
        lifecycle_folder = target / "lifecycle"
        atomic_write_frame(
            lifecycle_folder / "timeseries.csv",
            lifecycle.timeseries,
        )
        atomic_write_frame(
            lifecycle_folder / "events.csv",
            lifecycle.events,
        )
        atomic_write_json(lifecycle_folder / "summary.json", lifecycle.summary)

        lifecycle_folder_path = target / "acceptance" / "ML_LIFECYCLE"
        if resume_from and lifecycle_folder_path.is_dir():
            candidate_results["ML_LIFECYCLE"] = load_candidate_payload(
                target,
                "ML_LIFECYCLE",
            )
        else:
            prediction_name, weeks = candidates["ML_LIFECYCLE"]
            candidate_results["ML_LIFECYCLE"] = run_candidate(
                root=target,
                holdout_id=verification_id,
                candidate_name="ML_LIFECYCLE",
                prediction_name=prediction_name,
                smoothing_weeks=weeks,
                spec=settings,
                dates=dates,
            )

        ledger_metrics = {
            name: payload["metrics"]
            for name, payload in candidate_results.items()
        }
        yearly = combined_yearly_frame(candidate_results)
        atomic_write_frame(target / "yearly.csv", yearly)
        gate = evaluate_final_holdout_gate(
            model_metrics,
            ledger_metrics,
            yearly,
            selected_name=selected_name,
            reference_name=reference_name,
        )
        atomic_write_json(target / "gate.json", gate)
        comparison = comparison_frame(candidate_results)
        atomic_write_frame(target / "comparison.csv", comparison)
        all_structure_passed = all(
            payload["checks"]["passed"]
            for payload in candidate_results.values()
        )
        status = (
            "pass_final_holdout"
            if gate["passed"] and all_structure_passed
            else "fail_final_holdout"
        )
        scope = {
            "data_start": settings.start,
            "data_end": settings.end,
            "signal_start": str(dates.min().date()),
            "signal_end": str(dates.max().date()),
            "signal_days": len(dates),
            "ledger_start": str(dates.min().date()),
            "ledger_end": settings.end,
            "init_cash": float(settings.init_cash),
            "top_n": settings.top_n,
            "smoothing_weeks": selected_weeks,
            "benchmark": "H00300.CSI",
        }
        report = build_final_holdout_report(
            holdout_id=verification_id,
            status=status,
            scope=scope,
            sources=source_payload,
            model_metrics=model_metrics,
            gate=gate,
            candidates=candidate_results,
            monitor_summary=monitor.summary,
            lifecycle_summary=lifecycle.summary,
            exposure_summary=exposure.summary,
        )
        atomic_write_text(target / "report.md", report)
        output_names = [
            str(Path(path).relative_to(target))
            for path in result_files(target).values()
        ]
        outputs = {
            name: file_sha256(target / name)
            for name in output_names
        }
        manifest = {
            "schema_version": FINAL_HOLDOUT_SCHEMA_VERSION,
            "holdout_id": verification_id,
            "status": status,
            "scope": scope,
            "gate": gate,
            "candidates": {
                name: {
                    "checks": payload["checks"],
                    "snapshot_id": payload["run_manifest"].get("snapshot_id"),
                }
                for name, payload in candidate_results.items()
            },
            "sources": source_payload,
            "outputs": outputs,
        }
        atomic_write_json(target / "manifest.json", manifest)
        complete = {
            "schema_version": FINAL_HOLDOUT_SCHEMA_VERSION,
            "holdout_id": verification_id,
            "status": status,
            "files": {
                **outputs,
                "manifest.json": file_sha256(target / "manifest.json"),
            },
            "complete": True,
        }
        atomic_write_json(target / "complete.json", complete)
        return {
            "holdout_id": verification_id,
            "status": status,
            "path": str(target),
            "report": str(target / "report.md"),
            "gate": gate,
        }
    except Exception as error:
        atomic_write_text(
            target / "report.md",
            incomplete_report(verification_id, str(error)),
        )
        atomic_write_json(
            target / "manifest.json",
            {
                "schema_version": FINAL_HOLDOUT_SCHEMA_VERSION,
                "holdout_id": verification_id,
                "status": "incomplete_final_holdout",
                "error": f"{type(error).__name__}: {error}",
                "sources": source_payload,
            },
        )
        if isinstance(error, AlphaFinalHoldoutError):
            raise
        raise AlphaFinalHoldoutError(
            f"最终留出集验证失败：{type(error).__name__}: {error}"
        ) from error


__all__ = ["run_final_holdout"]
