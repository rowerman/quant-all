"""P4-4B1：固定 100 万资金的八候选完整账本对照。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from alpha.artifacts import file_sha256
from alpha.config import ACCEPTANCE_RUNS_DIR, DEFAULT_CONTRACT
from alpha.errors import AlphaConfigError
from alpha.evaluation_plan import current_model_ids
from alpha.logging_conf import get_logger
from alpha.predict import load_predictions
from backtest.chunked import run_chunked
from backtest.providers import validate_strategy_provider
from backtest.specs import (
    BacktestParams,
    BenchmarkSpec,
    PortfolioSpec,
    ProviderSpec,
    RebalanceSpec,
    SignalSpec,
    StrategySpec,
)
from backtest.strategies._factor_linear import (
    ensure_registered as ensure_linear_registered,
)
from backtest.strategies._factor_rank_v2 import (
    ensure_registered as ensure_factor_rank_registered,
)
from scripts.build_alpha.backtest_accept import (
    DEFAULT_END,
    DEFAULT_START,
    acceptance_scope,
    attach_rebalance_dates,
)
from scripts.build_alpha.backtest_accept_artifacts import (
    atomic_write_frame,
    atomic_write_json,
    atomic_write_text,
    read_candidate_completion as read_p4_4a_candidate_completion,
    read_json,
)
from scripts.build_alpha.backtest_accept_b1_artifacts import (
    candidate_is_complete,
    read_candidate_completion,
    write_candidate_artifacts,
)
from scripts.build_alpha.backtest_accept_b1_report import (
    build_report,
    capacity_frame,
    combined_yearly_frame,
    comparison_frame,
    output_hashes,
)
from scripts.build_alpha.backtest_accept_report import (
    acceptance_checks,
    capacity_summary,
    yearly_summary,
)
from scripts.build_alpha.baselines import baseline_strategy
from scripts.build_alpha.panel import implemented_factor_names
from common.paths import PROJECT_ROOT

logger = get_logger("scripts.build_alpha.backtest_accept_b1")

SCHEMA_VERSION = "p4_4b1_acceptance_v1"
ROUND = "p4-4b1"
ROUND_LABEL = "P4-4B1"
P4_4A_ANALYSIS_ID = "p4_4a_23c7e101aee9"
TOP_N = DEFAULT_CONTRACT.top_n
INIT_CASH = 1_000_000.0
RISK_FREE_RATE = 0.02
ENSEMBLE_MODEL_ID, RIDGE_MODEL_ID = current_model_ids()
SINGLE_LIGHTGBM_MODEL_ID = "lightgbm_label_rank_aaae94773921"
ML_MODEL_IDS = (
    RIDGE_MODEL_ID,
    SINGLE_LIGHTGBM_MODEL_ID,
    ENSEMBLE_MODEL_ID,
)


@dataclass(frozen=True)
class CandidateSpec:
    """一个 P4-4B1 固定候选。"""

    name: str
    kind: str
    source: str
    smoothing_weeks: int | None = None
    reuse: bool = False

    def to_dict(self) -> dict:
        """转换为稳定字典。"""

        return asdict(self)


CANDIDATES = (
    CandidateSpec(
        name="RIDGE_RAW",
        kind="model",
        source=RIDGE_MODEL_ID,
        smoothing_weeks=1,
    ),
    CandidateSpec(
        name="LIGHTGBM_SINGLE_RAW",
        kind="model",
        source=SINGLE_LIGHTGBM_MODEL_ID,
        smoothing_weeks=1,
    ),
    CandidateSpec(
        name="LIGHTGBM_ENSEMBLE_RAW",
        kind="model",
        source=ENSEMBLE_MODEL_ID,
        smoothing_weeks=1,
    ),
    CandidateSpec(
        name="LIGHTGBM_SMOOTH_4W",
        kind="model",
        source=ENSEMBLE_MODEL_ID,
        smoothing_weeks=4,
        reuse=True,
    ),
    CandidateSpec(name="B1_MOM20", kind="baseline", source="B1_MOM20"),
    CandidateSpec(name="B2_MOM_EP_VOL", kind="baseline", source="B2_MOM_EP_VOL"),
    CandidateSpec(name="B3_ALL_ZSCORE", kind="baseline", source="B3_ALL_ZSCORE"),
    CandidateSpec(
        name="B4_ALL_RANK",
        kind="baseline",
        source="B4_ALL_RANK",
        reuse=True,
    ),
)

CANDIDATE_GROUPS = {
    "raw-models": ("RIDGE_RAW", "LIGHTGBM_SINGLE_RAW", "LIGHTGBM_ENSEMBLE_RAW"),
    "baselines": ("B1_MOM20", "B2_MOM_EP_VOL", "B3_ALL_ZSCORE", "B4_ALL_RANK"),
}


def candidate_specs(selection: str = "all") -> tuple[CandidateSpec, ...]:
    """按名称或分组返回固定候选。"""

    mapping = {spec.name: spec for spec in CANDIDATES}
    if selection == "all":
        return CANDIDATES
    if selection in CANDIDATE_GROUPS:
        return tuple(mapping[name] for name in CANDIDATE_GROUPS[selection])
    if selection not in mapping:
        choices = sorted(
            {"all", *CANDIDATE_GROUPS, *mapping}
        )
        raise AlphaConfigError(f"未知候选 {selection!r}；可选 {choices}")
    return (mapping[selection],)


def _linear_provider_params(
    strategy: StrategySpec,
    top_n: int,
    rebalance_dates: list[str] | tuple[str, ...],
) -> dict:
    """从现有基线策略提取线性 provider 参数。"""

    return {
        "top_n": top_n,
        "factor_names": list(strategy.signal.factors),
        "factor_weights": list(strategy.signal.weights),
        "standardize": strategy.signal.standardize,
        "winsorize": strategy.signal.winsorize,
        "rebalance_dates": list(rebalance_dates),
    }


def build_candidate_strategy(
    spec: CandidateSpec,
    top_n: int,
    rebalance_dates: list[str] | tuple[str, ...],
    *,
    round_label: str = ROUND_LABEL,
) -> StrategySpec:
    """按指定轮次和固定口径构造候选策略。"""

    if spec.kind == "model":
        if spec.smoothing_weeks is None:
            raise AlphaConfigError(f"{spec.name} 缺少平滑周数")
        strategy = StrategySpec(
            name=f"{round_label} {spec.name} Top30",
            universe=DEFAULT_CONTRACT.universe,
            signal=SignalSpec(
                factors=(),
                weights=(),
                provider=ProviderSpec(
                    name="ml_rank",
                    version="2",
                    params={
                        "model_id": spec.source,
                        "top_n": top_n,
                        "smoothing_weeks": spec.smoothing_weeks,
                        "rebalance_dates": list(rebalance_dates),
                    },
                ),
            ),
            portfolio=PortfolioSpec(selection="provider", top_n=top_n),
            rebalance=RebalanceSpec(frequency="custom"),
            benchmark=BenchmarkSpec(DEFAULT_CONTRACT.benchmark),
        )
    elif spec.kind == "baseline":
        base = baseline_strategy(spec.source, implemented_factor_names())
        factor_names = tuple(base.signal.factors)
        if spec.source == "B4_ALL_RANK":
            ensure_factor_rank_registered()
            provider = ProviderSpec(
                name="factor_rank",
                version="2",
                params={
                    **base.signal.provider.params,
                    "rebalance_dates": list(rebalance_dates),
                },
            )
        else:
            ensure_linear_registered()
            provider = ProviderSpec(
                name="factor_linear",
                version="1",
                params=_linear_provider_params(
                    base,
                    top_n,
                    rebalance_dates,
                ),
            )
        strategy = replace(
            base,
            name=f"{round_label} {spec.name} Top30",
            signal=SignalSpec(
                factors=factor_names,
                weights=(),
                provider=provider,
            ),
            portfolio=PortfolioSpec(selection="provider", top_n=top_n),
            rebalance=RebalanceSpec(frequency="custom"),
        )
    else:
        raise AlphaConfigError(f"未知候选类型：{spec.kind}")
    validate_strategy_provider(strategy)
    return strategy


def _code_sha256() -> str:
    """汇总 P4-4B1 验收链路源码哈希。"""

    root = PROJECT_ROOT
    digest = hashlib.sha256()
    for relative in (
        "core/data/scripts/build_alpha/backtest_accept.py",
        "core/data/scripts/build_alpha/backtest_accept_artifacts.py",
        "core/data/scripts/build_alpha/backtest_accept_report.py",
        "core/data/scripts/build_alpha/backtest_accept_b1.py",
        "core/data/scripts/build_alpha/backtest_accept_b1_artifacts.py",
        "core/data/scripts/build_alpha/backtest_accept_b1_report.py",
        "trading/backtest/chunked.py",
        "trading/backtest/engine.py",
        "trading/backtest/ledger.py",
        "trading/backtest/allocation.py",
        "trading/backtest/signals.py",
        "trading/backtest/strategies/ml_rank_v2.py",
        "trading/backtest/strategies/_factor_rank_v2.py",
        "trading/backtest/strategies/_factor_linear.py",
        "research/modeling/alpha/portfolio.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def acceptance_analysis_id(
    scope: dict,
    specs: Iterable[CandidateSpec],
    model_manifests: dict[str, dict],
    source_manifest_sha256: str,
    resume_source_id: str,
    resume_source_sha256: str,
    code_sha256: str,
) -> str:
    """生成同时绑定候选、模型、复用来源和代码的分析 ID。"""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "round": ROUND,
        "scope": scope,
        "candidates": [spec.to_dict() for spec in specs],
        "models": {
            model_id: {
                "model_sha256": manifest.get("model_sha256"),
                "prediction_sha256": manifest.get("prediction_sha256"),
                "dataset_id": manifest.get("dataset_id"),
                "prediction_start": manifest.get("prediction_start"),
                "prediction_end": manifest.get("prediction_end"),
                "prediction_rows": manifest.get("prediction_rows"),
                "seeds": manifest.get("seeds"),
            }
            for model_id, manifest in sorted(model_manifests.items())
        },
        "source_analysis_id": P4_4A_ANALYSIS_ID,
        "source_manifest_sha256": source_manifest_sha256,
        "resume_source_id": resume_source_id,
        "resume_source_sha256": resume_source_sha256,
        "code_sha256": code_sha256,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "p4_4b1_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _load_model_inputs() -> tuple[dict[str, dict], pd.DatetimeIndex]:
    """读取三个模型 manifest，并校验预测日期完全一致。"""

    manifests: dict[str, dict] = {}
    reference: pd.DatetimeIndex | None = None
    for model_id in ML_MODEL_IDS:
        _, manifest, predictions = load_predictions(model_id)
        dates = pd.DatetimeIndex(
            sorted(pd.to_datetime(predictions["trade_date"]).unique())
        )
        if reference is None:
            reference = dates
        elif not reference.equals(dates):
            raise AlphaConfigError(
                f"{model_id} 的预测日期与其他模型不一致"
            )
        manifests[model_id] = manifest
        del predictions
    if reference is None or reference.empty:
        raise AlphaConfigError("三个模型都没有可用预测日期")
    return manifests, reference


def _source_reference() -> tuple[dict, str]:
    """读取并校验 P4-4A 最终分析。"""

    folder = ACCEPTANCE_RUNS_DIR / P4_4A_ANALYSIS_ID
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise AlphaConfigError(f"缺少 P4-4A 来源 manifest：{manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "pass_p4_4a":
        raise AlphaConfigError(
            f"P4-4A 来源状态不是 pass_p4_4a：{manifest.get('status')!r}"
        )
    return manifest, file_sha256(manifest_path)


def _validate_source_scope(source_manifest: dict, scope: dict) -> None:
    """要求复用来源与 P4-4B1 使用完全相同的账本口径。"""

    source_scope = source_manifest.get("scope") or {}
    _validate_scope_payload(source_scope, scope, "P4-4A")


def _validate_scope_payload(source_scope: dict, scope: dict, label: str) -> None:
    """校验一个候选或分析范围与当前运行一致。"""

    keys = (
        "start",
        "end",
        "init_cash",
        "risk_free_rate",
        "top_n",
        "benchmark",
        "rebalance_dates_sha256",
    )
    changed = [
        key
        for key in keys
        if source_scope.get(key) != scope.get(key)
    ]
    if changed:
        raise AlphaConfigError(
            f"{label} 来源口径与 P4-4B1 不一致：" + ", ".join(changed)
        )


def load_reused_candidate(
    spec: CandidateSpec,
    source_manifest: dict,
    scope: dict,
) -> dict:
    """读取 P4-4A 已完成候选，并验证完成标记与验收检查。"""

    if not spec.reuse:
        raise AlphaConfigError(f"{spec.name} 不是复用候选")
    _validate_source_scope(source_manifest, scope)
    folder = ACCEPTANCE_RUNS_DIR / P4_4A_ANALYSIS_ID / spec.name
    completion = read_p4_4a_candidate_completion(folder)
    if not completion:
        raise AlphaConfigError(f"P4-4A 来源候选完成标记无效：{folder}")
    if (
        completion.get("analysis_id") != P4_4A_ANALYSIS_ID
        or completion.get("candidate") != spec.name
    ):
        raise AlphaConfigError(f"P4-4A 来源候选身份不一致：{folder}")
    metrics = read_json(folder / "metrics.json")
    run_manifest = read_json(folder / "run_manifest.json")
    checks = metrics.get("checks") or {}
    if checks.get("passed") is not True:
        raise AlphaConfigError(f"P4-4A 来源候选未通过检查：{spec.name}")
    if not (run_manifest.get("acceptance_checks") or {}).get("passed"):
        raise AlphaConfigError(f"P4-4A 来源运行 manifest 未通过：{spec.name}")
    return {
        "metrics": metrics["metrics"],
        "yearly": pd.read_csv(folder / "yearly.csv"),
        "capacity": metrics["capacity"],
        "checks": checks,
        "run_manifest": run_manifest,
        "snapshot_id": metrics["snapshot_id"],
        "reused": True,
        "origin": "reused_p4_4a",
        "source": {
            "analysis_id": P4_4A_ANALYSIS_ID,
            "candidate": spec.name,
            "path": str(folder),
            "files": completion["files"],
        },
    }


def _resume_source_fingerprint(
    source_id: str,
    specs: tuple[CandidateSpec, ...],
) -> tuple[dict[str, dict], str]:
    """读取上一分析中已完整落盘的候选标记并生成确定性指纹。"""

    markers: dict[str, dict] = {}
    root = ACCEPTANCE_RUNS_DIR / source_id
    for spec in specs:
        if spec.reuse:
            continue
        completion = read_candidate_completion(root / spec.name)
        if not completion:
            continue
        if (
            completion.get("analysis_id") != source_id
            or completion.get("candidate") != spec.name
        ):
            raise AlphaConfigError(
                f"续跑来源候选身份不一致：{spec.name}"
            )
        markers[spec.name] = completion
    if not markers:
        raise AlphaConfigError(f"续跑来源没有完整候选：{source_id}")
    encoded = json.dumps(
        markers,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return markers, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_resumed_candidate(
    spec: CandidateSpec,
    source_id: str,
    scope: dict,
) -> dict:
    """从同一轮上一分析复用已通过检查的候选。"""

    folder = ACCEPTANCE_RUNS_DIR / source_id / spec.name
    completion = read_candidate_completion(folder)
    if not completion or completion.get("analysis_id") != source_id:
        raise AlphaConfigError(f"续跑来源候选完成标记无效：{folder}")
    metrics = read_json(folder / "metrics.json")
    run_manifest = read_json(folder / "run_manifest.json")
    _validate_scope_payload(metrics.get("scope") or {}, scope, "P4-4B1")
    checks = metrics.get("checks") or {}
    if checks.get("passed") is not True:
        raise AlphaConfigError(f"续跑来源候选未通过检查：{spec.name}")
    if not (run_manifest.get("acceptance_checks") or {}).get("passed"):
        raise AlphaConfigError(f"续跑来源运行 manifest 未通过：{spec.name}")
    return {
        "metrics": metrics["metrics"],
        "yearly": pd.read_csv(folder / "yearly.csv"),
        "capacity": metrics["capacity"],
        "checks": checks,
        "run_manifest": run_manifest,
        "snapshot_id": metrics["snapshot_id"],
        "reused": True,
        "origin": "resumed_p4_4b1",
        "source": {
            "analysis_id": source_id,
            "candidate": spec.name,
            "path": str(folder),
            "files": completion["files"],
        },
    }


def _progress(stage: str, payload: dict) -> None:
    """记录长回测阶段。"""

    if stage in {"meta", "load", "signals", "done"}:
        logger.info(
            "P4-4B1 %s：%s",
            stage,
            json.dumps(payload, ensure_ascii=False, default=str),
        )
    else:
        logger.debug(
            "P4-4B1 %s：%s",
            stage,
            json.dumps(payload, ensure_ascii=False, default=str),
        )


def candidate_payload(
    spec: CandidateSpec,
    strategy: StrategySpec,
    scope: dict,
    result,
    *,
    schema_version: str = SCHEMA_VERSION,
    round_name: str = ROUND,
) -> tuple[dict, pd.DataFrame, dict]:
    """把完整账本结果整理成指定轮次的落盘载荷。"""

    yearly = yearly_summary(result.daily)
    capacity = capacity_summary(result.metadata)
    signal_dates = pd.DatetimeIndex(scope["rebalance_dates"])
    daily_dates = pd.DatetimeIndex(result.daily.index).sort_values()
    positions = daily_dates.get_indexer(signal_dates)
    execution_dates = pd.DatetimeIndex(
        [
            daily_dates[position + 1]
            for position in positions
            if position >= 0 and position + 1 < len(daily_dates)
        ]
    )
    checks = acceptance_checks(
        result.metrics,
        result.metadata,
        result.daily,
        dev_start=DEFAULT_CONTRACT.dev_start,
        dev_end=DEFAULT_CONTRACT.dev_end,
        expected_signal_days=len(signal_dates),
        expected_execution_days=len(execution_dates),
        actual_signal_days=int(
            (result.metadata.get("signal") or {}).get("rebalance_days", -1)
        ),
    )
    metrics = {
        "schema_version": schema_version,
        "round": round_name,
        "candidate": spec.name,
        "kind": spec.kind,
        "source": spec.source,
        "engine": "ledger",
        "scope": scope,
        "strategy_hash": strategy.config_hash,
        "snapshot_id": result.metadata.get("snapshot_id"),
        "metrics": result.metrics,
        "yearly": yearly.to_dict(orient="records"),
        "capacity": capacity,
        "checks": checks,
        "elapsed_seconds": result.metadata.get("elapsed_seconds"),
    }
    run_manifest = {
        **result.metadata,
        "candidate": spec.name,
        "kind": spec.kind,
        "source": spec.source,
        "acceptance_checks": checks,
    }
    return metrics, yearly, run_manifest


def _run_candidate(
    spec: CandidateSpec,
    scope: dict,
    folder: Path,
    analysis_id: str,
) -> dict:
    """串行运行并落盘一个候选。"""

    strategy = build_candidate_strategy(
        spec,
        scope["top_n"],
        scope["rebalance_dates"],
    )
    params = BacktestParams(
        start=scope["start"],
        end=scope["end"],
        init_cash=scope["init_cash"],
        risk_free_rate=scope["risk_free_rate"],
    )
    logger.info("开始 %s：%s ~ %s", spec.name, params.start, params.end)
    result = run_chunked(strategy, params, progress=_progress)
    metrics, yearly, run_manifest = candidate_payload(
        spec,
        strategy,
        scope,
        result,
    )
    write_candidate_artifacts(
        folder,
        analysis_id=analysis_id,
        candidate=spec.name,
        metrics=metrics,
        strategy=strategy.to_dict(),
        yearly=yearly,
        daily=result.daily,
        orders=result.orders,
        holdings=result.holdings,
        run_manifest=run_manifest,
    )
    logger.info(
        "完成 %s：订单=%d 耗时=%ss 检查=%s",
        spec.name,
        len(result.orders),
        result.metadata.get("elapsed_seconds"),
        "通过" if metrics["checks"]["passed"] else "失败",
    )
    return metrics


def _write_round_outputs(
    root: Path,
    analysis_id: str,
    scope: dict,
    specs: tuple[CandidateSpec, ...],
    loaded: dict[str, dict],
    model_manifests: dict[str, dict],
    source_manifest_sha256: str,
    resume_source_id: str,
    resume_source_sha256: str,
) -> dict:
    """写出汇总表、报告、来源和总 manifest。"""

    comparison = comparison_frame(loaded)
    yearly = combined_yearly_frame(loaded)
    capacity = capacity_frame(loaded)
    report = build_report(
        analysis_id=analysis_id,
        scope=scope,
        candidates=loaded,
        expected_candidates=len(specs),
        source_analysis_id=P4_4A_ANALYSIS_ID,
    )
    atomic_write_frame(root / "comparison.csv", comparison)
    atomic_write_frame(root / "yearly.csv", yearly)
    atomic_write_frame(root / "capacity.csv", capacity)
    atomic_write_text(root / "report.md", report)
    sources = {
        "p4_4a": {
            "analysis_id": P4_4A_ANALYSIS_ID,
            "manifest_sha256": source_manifest_sha256,
            "candidates": {
                name: payload["source"]
                for name, payload in loaded.items()
                if payload.get("reused")
                and payload.get("origin") == "reused_p4_4a"
            },
        },
        "resumed_p4_4b1": {
            "analysis_id": resume_source_id,
            "fingerprint_sha256": resume_source_sha256,
            "candidates": {
                name: payload["source"]
                for name, payload in loaded.items()
                if payload.get("origin") == "resumed_p4_4b1"
            },
        },
        "models": {
            model_id: {
                "manifest": manifest,
            }
            for model_id, manifest in model_manifests.items()
        },
    }
    atomic_write_json(root / "sources.json", sources)
    output_names = (
        "comparison.csv",
        "yearly.csv",
        "capacity.csv",
        "report.md",
        "sources.json",
    )
    all_complete = len(loaded) == len(specs)
    all_passed = bool(loaded) and all(
        payload["checks"]["passed"] for payload in loaded.values()
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "round": ROUND,
        "status": (
            "pass_p4_4b1"
            if all_complete and all_passed
            else "failed_p4_4b1"
            if all_complete
            else "partial_p4_4b1"
        ),
        "scope": scope,
        "candidates": {
            spec.name: {
                "spec": spec.to_dict(),
                "complete": spec.name in loaded,
                "passed": loaded.get(spec.name, {})
                .get("checks", {})
                .get("passed", False),
                "reused": spec.reuse,
                "snapshot_id": loaded.get(spec.name, {}).get("snapshot_id"),
            }
            for spec in specs
        },
        "source_analysis_id": P4_4A_ANALYSIS_ID,
        "source_manifest_sha256": source_manifest_sha256,
        "resume_source_analysis_id": resume_source_id,
        "resume_source_sha256": resume_source_sha256,
        "model_prediction_sha256": {
            model_id: manifest.get("prediction_sha256")
            for model_id, manifest in model_manifests.items()
        },
        "outputs": output_hashes(root, output_names),
    }
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


def execute_acceptance(
    *,
    candidate: str = "all",
    resume: bool = False,
    resume_source_analysis: str | None = None,
    output_root: Path | None = None,
) -> dict:
    """执行 P4-4B1，返回分析目录和验收状态。"""

    scope = acceptance_scope(
        start=DEFAULT_START,
        end=DEFAULT_END,
        init_cash=INIT_CASH,
        risk_free_rate=RISK_FREE_RATE,
    )
    model_manifests, prediction_dates = _load_model_inputs()
    prediction_dates = prediction_dates[
        (prediction_dates >= pd.Timestamp(scope["start"]))
        & (prediction_dates <= pd.Timestamp(scope["end"]))
    ]
    scope = attach_rebalance_dates(scope, prediction_dates)
    source_manifest, source_manifest_sha256 = _source_reference()
    specs = candidate_specs(candidate)
    resume_source_id = resume_source_analysis or ""
    resume_markers: dict[str, dict] = {}
    resume_source_sha256 = ""
    if resume_source_id:
        resume_markers, resume_source_sha256 = _resume_source_fingerprint(
            resume_source_id,
            specs,
        )
    analysis_id = acceptance_analysis_id(
        scope,
        specs,
        model_manifests,
        source_manifest_sha256,
        resume_source_id,
        resume_source_sha256,
        _code_sha256(),
    )
    root = (output_root or ACCEPTANCE_RUNS_DIR) / analysis_id
    if root.exists() and not resume:
        raise AlphaConfigError(
            f"验收目录已存在：{root}；确认要续跑时使用 --resume"
        )
    root.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, dict] = {}
    statuses: dict[str, str] = {}
    for spec in specs:
        folder = root / spec.name
        if spec.reuse:
            logger.info("复用 P4-4A 候选 %s", spec.name)
            loaded[spec.name] = load_reused_candidate(
                spec,
                source_manifest,
                scope,
            )
            statuses[spec.name] = "reused_p4_4a"
            continue
        if spec.name in resume_markers:
            logger.info(
                "从 %s 续跑候选 %s",
                resume_source_id,
                spec.name,
            )
            loaded[spec.name] = load_resumed_candidate(
                spec,
                resume_source_id,
                scope,
            )
            statuses[spec.name] = "resumed_from_p4_4b1"
            continue
        if resume and candidate_is_complete(folder, analysis_id, spec.name):
            logger.info("跳过已完成候选 %s", spec.name)
            metrics = read_json(folder / "metrics.json")
            loaded[spec.name] = {
                "metrics": metrics["metrics"],
                "yearly": pd.read_csv(folder / "yearly.csv"),
                "capacity": metrics["capacity"],
                "checks": metrics["checks"],
                "run_manifest": read_json(folder / "run_manifest.json"),
                "snapshot_id": metrics["snapshot_id"],
                "reused": False,
                "source": {
                    "analysis_id": analysis_id,
                    "candidate": spec.name,
                    "path": str(folder),
                },
            }
            statuses[spec.name] = "skipped_complete"
            continue
        payload = _run_candidate(spec, scope, folder, analysis_id)
        loaded[spec.name] = {
            "metrics": payload["metrics"],
            "yearly": pd.read_csv(folder / "yearly.csv"),
            "capacity": payload["capacity"],
            "checks": payload["checks"],
            "run_manifest": read_json(folder / "run_manifest.json"),
            "snapshot_id": payload["snapshot_id"],
            "reused": False,
            "source": {
                "analysis_id": analysis_id,
                "candidate": spec.name,
                "path": str(folder),
            },
        }
        statuses[spec.name] = "completed"
    manifest = _write_round_outputs(
        root,
        analysis_id,
        scope,
        specs,
        loaded,
        model_manifests,
        source_manifest_sha256,
        resume_source_id,
        resume_source_sha256,
    )
    return {
        "analysis_id": analysis_id,
        "output_directory": str(root),
        "status": manifest["status"],
        "candidates": statuses,
        "report": str(root / "report.md"),
    }


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="P4-4B1 完整账本对照")
    parser.add_argument("--candidate", default="all")
    parser.add_argument("--resume", action="store_true", help="跳过已完整落盘的候选")
    parser.add_argument(
        "--resume-from-analysis",
        help="复用同轮上一分析中已通过检查的候选",
    )
    return parser


def main() -> None:
    """命令行入口。"""

    args = _parser().parse_args()
    result = execute_acceptance(
        candidate=args.candidate,
        resume=args.resume,
        resume_source_analysis=args.resume_from_analysis,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass_p4_4b1":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
