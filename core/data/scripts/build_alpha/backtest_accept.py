"""P4-4A：用完整账本验收选中 ML 模型与 B4 基线。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from alpha.config import ACCEPTANCE_RUNS_DIR, DEFAULT_CONTRACT
from alpha.errors import AlphaConfigError
from alpha.evaluation_plan import load_current_plan
from alpha.logging_conf import get_logger
from alpha.predict import load_model_manifest, load_predictions
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
from backtest.strategies._factor_rank_v2 import ensure_registered
from common.paths import PROJECT_ROOT
from scripts.build_alpha.backtest_accept_artifacts import (
    atomic_write_frame,
    atomic_write_json,
    atomic_write_text,
    candidate_is_complete,
    read_json,
    write_candidate_artifacts,
)
from scripts.build_alpha.backtest_accept_report import (
    acceptance_checks,
    build_markdown_report,
    capacity_summary,
    combined_yearly_frame,
    comparison_frame,
    load_p3_weekly_reference,
    output_hashes,
    yearly_summary,
)
from scripts.build_alpha.baselines import baseline_strategy
from scripts.build_alpha.panel import implemented_factor_names

logger = get_logger("scripts.build_alpha.backtest_accept")

SCHEMA_VERSION = "p4_4a_acceptance_v1"
ROUND = "p4-4a"
DEFAULT_START = "2018-01-12"
DEFAULT_END = "2023-12-29"
TOP_N = DEFAULT_CONTRACT.top_n
INIT_CASH = 1_000_000.0
RISK_FREE_RATE = 0.02


@dataclass(frozen=True)
class CandidateSpec:
    """一个 P4-4A 固定候选。"""

    name: str
    kind: str
    source: str
    smoothing_weeks: int = 1

    def to_dict(self) -> dict:
        """转换为稳定字典。"""

        return asdict(self)


_PLAN = load_current_plan()
_SELECTED = _PLAN.by_name(_PLAN.selected or _PLAN.candidates[0].name)
_CONTROL = _PLAN.controls[0]
CANDIDATES = (
    CandidateSpec(
        name=_SELECTED.name,
        kind="selected",
        source=_SELECTED.source.id,
        smoothing_weeks=_SELECTED.score_policy.smoothing_weeks,
    ),
    CandidateSpec(
        name=_CONTROL.name,
        kind="baseline",
        source=_CONTROL.source.id,
    ),
)


def _iso_date(value: str, name: str) -> str:
    """校验 ISO 日期。"""

    try:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise AlphaConfigError(f"{name} 必须是 YYYY-MM-DD：{value!r}") from error
    return value


def acceptance_scope(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    *,
    init_cash: float = INIT_CASH,
    risk_free_rate: float = RISK_FREE_RATE,
    top_n: int = TOP_N,
) -> dict:
    """校验并返回 P4-4A 固定范围。"""

    start = _iso_date(start, "start")
    end = _iso_date(end, "end")
    if start > end:
        raise AlphaConfigError("start 不能晚于 end")
    if start < DEFAULT_CONTRACT.dev_start:
        raise AlphaConfigError("P4-4A 起始日不能早于开发期")
    if end > DEFAULT_CONTRACT.dev_end:
        raise AlphaConfigError("P4-4A 不得读取 2024-2026 最终测试集")
    if isinstance(init_cash, bool) or not isinstance(init_cash, (int, float)):
        raise AlphaConfigError("init_cash 必须是正数")
    if not 0 < float(init_cash):
        raise AlphaConfigError("init_cash 必须大于 0")
    if isinstance(risk_free_rate, bool) or not isinstance(risk_free_rate, (int, float)):
        raise AlphaConfigError("risk_free_rate 必须是数字")
    if not -0.99 <= float(risk_free_rate) <= 1.0:
        raise AlphaConfigError("risk_free_rate 必须在 [-0.99, 1] 之间")
    if type(top_n) is not int or top_n <= 0:
        raise AlphaConfigError("top_n 必须是正整数")
    return {
        "start": start,
        "end": end,
        "init_cash": float(init_cash),
        "risk_free_rate": float(risk_free_rate),
        "top_n": top_n,
        "universe": DEFAULT_CONTRACT.universe,
        "weighting": DEFAULT_CONTRACT.weighting,
        "rebalance": DEFAULT_CONTRACT.rebalance,
        "benchmark": DEFAULT_CONTRACT.benchmark,
        "max_participation_rate": 0.1,
    }


def attach_rebalance_dates(scope: dict, dates) -> dict:
    """把显式调仓日加入验收范围并校验覆盖。"""

    normalized = sorted(
        dict.fromkeys(
            str(pd.Timestamp(item).date())
            for item in pd.DatetimeIndex(dates)
        )
    )
    if not normalized:
        raise AlphaConfigError("显式调仓日不能为空")
    outside = [
        item
        for item in normalized
        if item < scope["start"] or item > scope["end"]
    ]
    if outside:
        raise AlphaConfigError(
            "调仓日超出验收区间："
            + ", ".join(outside[:5])
        )
    encoded = json.dumps(
        normalized,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **scope,
        "rebalance_dates": normalized,
        "rebalance_dates_count": len(normalized),
        "rebalance_dates_start": normalized[0],
        "rebalance_dates_end": normalized[-1],
        "rebalance_dates_sha256": hashlib.sha256(encoded).hexdigest(),
        "rebalance_dates_source": "model_prediction_dates",
    }


def candidate_specs(selection: str = "all") -> tuple[CandidateSpec, ...]:
    """返回本次请求的固定候选。"""

    mapping = {
        "selected": CANDIDATES[0],
        "b4": CANDIDATES[1],
    }
    if selection == "all":
        return CANDIDATES
    if selection not in mapping:
        raise AlphaConfigError(f"未知候选 {selection!r}")
    return (mapping[selection],)


def build_candidate_strategy(
    spec: CandidateSpec,
    top_n: int,
    rebalance_dates: list[str] | tuple[str, ...],
) -> StrategySpec:
    """按固定口径构造候选策略。"""

    if spec.kind == "baseline":
        ensure_registered()
        strategy = baseline_strategy(spec.source, implemented_factor_names())
        strategy = replace(
            strategy,
            signal=replace(
                strategy.signal,
                provider=ProviderSpec(
                    name="factor_rank",
                    version="2",
                    params={
                        **strategy.signal.provider.params,
                        "rebalance_dates": list(rebalance_dates),
                    },
                ),
            ),
        )
    elif spec.kind == "selected":
        strategy = StrategySpec(
            name="P4-4A LightGBM 五种子 4 周平滑 Top30",
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
    else:
        raise AlphaConfigError(f"未知候选类型：{spec.kind}")
    validate_strategy_provider(strategy)
    return strategy


def _code_sha256() -> str:
    """汇总验收链路源码哈希。"""

    root = PROJECT_ROOT
    digest = hashlib.sha256()
    for relative in (
        "core/data/scripts/build_alpha/backtest_accept.py",
        "core/data/scripts/build_alpha/backtest_accept_artifacts.py",
        "core/data/scripts/build_alpha/backtest_accept_report.py",
        "trading/backtest/chunked.py",
        "trading/backtest/engine.py",
        "trading/backtest/ledger.py",
        "trading/backtest/attribution.py",
        "trading/backtest/signals.py",
        "trading/backtest/trading_rules.py",
        "trading/backtest/data.py",
        "trading/backtest/costs.py",
        "trading/backtest/strategies/ml_rank.py",
        "trading/backtest/strategies/_factor_rank.py",
        "research/modeling/alpha/portfolio.py",
        "research/modeling/alpha/predict.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def acceptance_analysis_id(
    scope: dict,
    specs: Iterable[CandidateSpec],
    model_manifest: dict,
    code_sha256: str,
) -> str:
    """由固定范围、候选、模型和代码生成确定性分析 ID。"""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "round": ROUND,
        "scope": scope,
        "candidates": [spec.to_dict() for spec in specs],
        "model": {
            "model_id": model_manifest.get("model_id"),
            "model_sha256": model_manifest.get("model_sha256"),
            "prediction_sha256": model_manifest.get("prediction_sha256"),
            "dataset_id": model_manifest.get("dataset_id"),
            "data_versions": model_manifest.get("data_versions"),
            "feature_count": model_manifest.get("feature_count"),
            "seeds": model_manifest.get("seeds"),
        },
        "code_sha256": code_sha256,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "p4_4a_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _progress(stage: str, payload: dict) -> None:
    """记录长回测阶段；高频明细只写 debug。"""

    if stage in {"meta", "load", "signals", "done"}:
        logger.info("P4-4A %s：%s", stage, json.dumps(payload, ensure_ascii=False, default=str))
    else:
        logger.debug("P4-4A %s：%s", stage, json.dumps(payload, ensure_ascii=False, default=str))


def _candidate_payload(
    spec: CandidateSpec,
    strategy: StrategySpec,
    scope: dict,
    result,
) -> tuple[dict, pd.DataFrame, dict]:
    """把完整账本结果整理成可落盘载荷。"""

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
    expected_signals = len(signal_dates)
    actual_signals = int(
        (result.metadata.get("signal") or {}).get("rebalance_days", -1)
    )
    checks = acceptance_checks(
        result.metrics,
        result.metadata,
        result.daily,
        dev_start=DEFAULT_CONTRACT.dev_start,
        dev_end=DEFAULT_CONTRACT.dev_end,
        expected_signal_days=expected_signals,
        expected_execution_days=len(execution_dates),
        actual_signal_days=actual_signals,
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
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
    """执行并落盘一个候选。"""

    strategy = build_candidate_strategy(
        spec,
        scope["top_n"],
        scope["rebalance_dates"],
    )
    if spec.kind == "baseline":
        ensure_registered()
    params = BacktestParams(
        start=scope["start"],
        end=scope["end"],
        init_cash=scope["init_cash"],
        risk_free_rate=scope["risk_free_rate"],
    )
    logger.info("开始 %s：%s ~ %s", spec.name, params.start, params.end)
    result = run_chunked(strategy, params, progress=_progress)
    metrics, yearly, run_manifest = _candidate_payload(spec, strategy, scope, result)
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
) -> dict:
    """写出汇总对照、年度表、报告和 manifest。"""

    comparison = comparison_frame(loaded)
    yearly = combined_yearly_frame(loaded)
    names = {spec.name for spec in specs}
    weekly_candidates, weekly_yearly = load_p3_weekly_reference(names)
    complete = len(loaded) == len(specs) and set(spec.name for spec in specs) == {
        spec.name for spec in CANDIDATES
    }
    report = build_markdown_report(
        analysis_id=analysis_id,
        scope=scope,
        candidates=loaded,
        comparison=comparison,
        yearly=yearly,
        weekly_candidates=weekly_candidates,
        weekly_yearly=weekly_yearly,
        complete=complete,
    )
    atomic_write_frame(root / "comparison.csv", comparison)
    atomic_write_frame(root / "yearly.csv", yearly)
    atomic_write_text(root / "report.md", report)
    output_names = ("comparison.csv", "yearly.csv", "report.md")
    passed = bool(loaded) and all(
        payload["checks"]["passed"] for payload in loaded.values()
    )
    all_candidates_complete = len(loaded) == len(specs)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "round": ROUND,
        "status": (
            "pass_p4_4a" if complete and passed
            else "failed_p4_4a" if all_candidates_complete
            else "partial_p4_4a"
        ),
        "scope": scope,
        "candidates": {
            spec.name: {
                "spec": spec.to_dict(),
                "complete": spec.name in loaded,
                "passed": loaded.get(spec.name, {}).get("checks", {}).get("passed", False),
                "snapshot_id": loaded.get(spec.name, {}).get("snapshot_id"),
            }
            for spec in specs
        },
        "p3_weekly_reference": {
            "candidate_rows": int(len(weekly_candidates)),
            "yearly_rows": int(len(weekly_yearly)),
        },
        "outputs": output_hashes(root, output_names),
    }
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


def execute_acceptance(
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    candidate: str = "all",
    resume: bool = False,
    output_root: Path | None = None,
) -> dict:
    """执行 P4-4A，返回分析目录和验收状态。"""

    scope = acceptance_scope(start=start, end=end)
    _, _, predictions = load_predictions(CANDIDATES[0].source)
    prediction_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(predictions["trade_date"]).unique())
    )
    prediction_dates = prediction_dates[
        (prediction_dates >= pd.Timestamp(scope["start"]))
        & (prediction_dates <= pd.Timestamp(scope["end"]))
    ]
    scope = attach_rebalance_dates(scope, prediction_dates)
    specs = candidate_specs(candidate)
    model_manifest = load_model_manifest(CANDIDATES[0].source)
    analysis_id = acceptance_analysis_id(
        scope,
        specs,
        model_manifest,
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
        if resume and candidate_is_complete(folder, analysis_id, spec.name):
            logger.info("跳过已完成候选 %s", spec.name)
            loaded[spec.name] = {
                "metrics": read_json(folder / "metrics.json")["metrics"],
                "yearly": pd.read_csv(folder / "yearly.csv"),
                "capacity": read_json(folder / "metrics.json")["capacity"],
                "checks": read_json(folder / "metrics.json")["checks"],
                "run_manifest": read_json(folder / "run_manifest.json"),
                "snapshot_id": read_json(folder / "metrics.json")["snapshot_id"],
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
        }
        statuses[spec.name] = "completed"
    manifest = _write_round_outputs(root, analysis_id, scope, specs, loaded)
    return {
        "analysis_id": analysis_id,
        "output_directory": str(root),
        "status": manifest["status"],
        "candidates": statuses,
        "report": str(root / "report.md"),
    }


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="P4-4A 完整账本验收")
    parser.add_argument("--round", choices=(ROUND,), default=ROUND)
    parser.add_argument("--candidate", choices=("all", "selected", "b4"), default="all")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--resume", action="store_true", help="跳过已完整落盘的候选")
    return parser


def main() -> None:
    """命令行入口。"""

    args = _parser().parse_args()
    result = execute_acceptance(
        start=args.start,
        end=args.end,
        candidate=args.candidate,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass_p4_4a":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
