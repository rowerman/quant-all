"""P5-3 生命周期规则完整账本验收。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from alpha.artifacts import file_sha256
from alpha.config import MODEL_DATA_DIR
from alpha.errors import AlphaConfigError
from alpha.lifecycle import (
    DEFAULT_FALLBACK_MODEL_ID,
    DEFAULT_PRIMARY_MODEL_ID,
    load_model_lifecycle,
)
from alpha.logging_conf import get_logger
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
from scripts.build_alpha.backtest_accept import (
    DEFAULT_END,
    DEFAULT_START,
    acceptance_scope,
    attach_rebalance_dates,
)
from scripts.build_alpha.backtest_accept_report import (
    acceptance_checks,
    capacity_summary,
    yearly_summary,
)
from scripts.build_alpha.lifecycle_accept_artifacts import (
    CANDIDATE,
    atomic_frame,
    atomic_json,
    atomic_text,
    candidate_is_complete,
    comparison_frame,
    json_default,
    read_json,
    write_candidate,
)
from scripts.build_alpha.lifecycle_accept_report import build_report
from common.paths import PROJECT_ROOT

logger = get_logger("scripts.build_alpha.lifecycle_accept")

SCHEMA_VERSION = "p5_3_acceptance_v1"
TOP_N = 30
INIT_CASH = 1_000_000.0
RISK_FREE_RATE = 0.02


def _code_sha256() -> str:
    """汇总生命周期验收代码哈希。"""

    root = PROJECT_ROOT
    digest = hashlib.sha256()
    for relative in (
        "core/data/scripts/build_alpha/lifecycle_accept.py",
        "core/data/scripts/build_alpha/lifecycle_accept_artifacts.py",
        "core/data/scripts/build_alpha/lifecycle_accept_report.py",
        "trading/backtest/strategies/ml_rank_v3.py",
        "trading/backtest/strategies/ml_rank_v2.py",
        "trading/backtest/chunked.py",
        "trading/backtest/engine.py",
        "trading/backtest/ledger.py",
        "research/modeling/alpha/lifecycle.py",
        "research/modeling/alpha/lifecycle_policy.py",
        "research/modeling/alpha/lifecycle_engine.py",
        "research/modeling/alpha/lifecycle_artifacts.py",
        "research/modeling/alpha/portfolio.py",
        "research/modeling/alpha/predict.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _analysis_id(
    lifecycle_manifest: dict,
    scope: dict,
    code_sha256: str,
    *,
    model_id: str = DEFAULT_PRIMARY_MODEL_ID,
    fallback_model_id: str = DEFAULT_FALLBACK_MODEL_ID,
    smoothing_weeks: int = 4,
    top_n: int = TOP_N,
) -> str:
    """由生命周期、范围和代码生成确定性验收 ID。"""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "lifecycle_id": lifecycle_manifest["lifecycle_id"],
        "lifecycle_files": lifecycle_manifest.get("files"),
        "lifecycle_sources": lifecycle_manifest.get("sources"),
        "scope": scope,
        "model_id": model_id,
        "fallback_model_id": fallback_model_id,
        "smoothing_weeks": smoothing_weeks,
        "top_n": top_n,
        "code_sha256": code_sha256,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    )
    return "p5_3_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _strategy(
    lifecycle_id: str,
    rebalance_dates: list[str],
    *,
    model_id: str = DEFAULT_PRIMARY_MODEL_ID,
    fallback_model_id: str = DEFAULT_FALLBACK_MODEL_ID,
    smoothing_weeks: int = 4,
    top_n: int = TOP_N,
) -> StrategySpec:
    """构造生命周期 provider 策略。"""

    strategy = StrategySpec(
        name=f"P5-3 生命周期 {model_id} Top{top_n}",
        signal=SignalSpec(
            factors=(),
            weights=(),
            provider=ProviderSpec(
                name="ml_rank",
                version="3",
                params={
                    "lifecycle_id": lifecycle_id,
                    "model_id": model_id,
                    "fallback_model_id": fallback_model_id,
                    "top_n": top_n,
                    "smoothing_weeks": smoothing_weeks,
                    "rebalance_dates": rebalance_dates,
                },
            ),
        ),
        portfolio=PortfolioSpec(selection="provider", top_n=top_n),
        rebalance=RebalanceSpec(frequency="custom"),
        benchmark=BenchmarkSpec("H00300.CSI"),
    )
    validate_strategy_provider(strategy)
    return strategy


def execute_acceptance(
    lifecycle_id: str,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    model_id: str = DEFAULT_PRIMARY_MODEL_ID,
    fallback_model_id: str = DEFAULT_FALLBACK_MODEL_ID,
    smoothing_weeks: int = 4,
    top_n: int = TOP_N,
    resume: bool = False,
    output_root: Path | None = None,
) -> dict:
    """执行 P5-3 完整账本并原子写入验收产物。"""

    root = Path(output_root or MODEL_DATA_DIR)
    lifecycle = load_model_lifecycle(lifecycle_id, model_id=model_id)
    scope = acceptance_scope(
        start=start,
        end=end,
        init_cash=INIT_CASH,
        risk_free_rate=RISK_FREE_RATE,
        top_n=top_n,
    )
    lifecycle_dates = pd.DatetimeIndex(lifecycle["timeseries"]["trade_date"])
    lifecycle_dates = lifecycle_dates[
        (lifecycle_dates >= pd.Timestamp(scope["start"]))
        & (lifecycle_dates <= pd.Timestamp(scope["end"]))
    ]
    scope = attach_rebalance_dates(scope, lifecycle_dates)
    analysis_id = _analysis_id(
        lifecycle["manifest"],
        scope,
        _code_sha256(),
        model_id=model_id,
        fallback_model_id=fallback_model_id,
        smoothing_weeks=smoothing_weeks,
        top_n=top_n,
    )
    acceptance_folder = root / "runs" / "acceptance" / analysis_id
    candidate_folder = acceptance_folder / CANDIDATE
    if acceptance_folder.exists() and not resume:
        raise AlphaConfigError(
            f"P5-3 验收目录已存在：{acceptance_folder}；续跑请使用 --resume"
        )
    acceptance_folder.mkdir(parents=True, exist_ok=True)
    if resume and candidate_is_complete(candidate_folder, analysis_id):
        metrics = read_json(candidate_folder / "metrics.json")
        yearly = pd.read_csv(candidate_folder / "yearly.csv")
        logger.info("复用已完成 P5-3 候选")
    else:
        strategy = _strategy(
            lifecycle_id,
            scope["rebalance_dates"],
            model_id=model_id,
            fallback_model_id=fallback_model_id,
            smoothing_weeks=smoothing_weeks,
            top_n=top_n,
        )
        params = BacktestParams(
            start=scope["start"],
            end=scope["end"],
            init_cash=scope["init_cash"],
            risk_free_rate=scope["risk_free_rate"],
        )
        logger.info("开始 P5-3 完整账本：%s ~ %s", params.start, params.end)
        result = run_chunked(
            strategy,
            params,
            progress=lambda stage, payload: logger.info(
                "P5-3 %s：%s",
                stage,
                json.dumps(payload, ensure_ascii=False, default=json_default),
            ),
        )
        yearly = yearly_summary(result.daily)
        capacity = capacity_summary(result.metadata)
        daily_dates = pd.DatetimeIndex(result.daily.index).sort_values()
        signal_dates = pd.DatetimeIndex(scope["rebalance_dates"])
        positions = daily_dates.get_indexer(signal_dates)
        execution_days = sum(
            1 <= position < len(daily_dates) for position in positions
        )
        checks = acceptance_checks(
            result.metrics,
            result.metadata,
            result.daily,
            dev_start="2012-01-01",
            dev_end="2023-12-31",
            expected_signal_days=len(signal_dates),
            expected_execution_days=int(execution_days),
            actual_signal_days=int(
                (result.metadata.get("signal") or {}).get("rebalance_days", -1)
            ),
        )
        event_counts = lifecycle["manifest"]["summary"]["event_counts"]
        checks["checks"]["lifecycle_actions_present"] = all(
            event_counts.get(name, 0) > 0
            for name in ("reduce", "rollback", "pause", "recover")
        )
        provider_artifacts = (
            (
                (result.metadata.get("snapshot") or {})
                .get("components", {})
                .get("rules", {})
                .get("provider", {})
                .get("artifacts", {})
            )
            if isinstance(result.metadata, dict)
            else {}
        )
        checks["checks"]["lifecycle_snapshot_present"] = bool(
            provider_artifacts.get("lifecycle_id") == lifecycle_id
        )
        checks["passed"] = bool(checks["passed"] and all(checks["checks"].values()))
        metrics = {
            "schema_version": SCHEMA_VERSION,
            "candidate": CANDIDATE,
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
        write_candidate(
            candidate_folder,
            analysis_id=analysis_id,
            strategy=strategy,
            result=result,
            metrics=metrics,
            yearly=yearly,
        )

    comparison = comparison_frame(metrics, lifecycle["manifest"], root)
    atomic_frame(acceptance_folder / "comparison.csv", comparison)
    status = "pass_p5_3" if metrics["checks"]["passed"] else "failed_p5_3"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "status": status,
        "scope": scope,
        "lifecycle": {
            "lifecycle_id": lifecycle_id,
            "summary": lifecycle["manifest"]["summary"],
            "manifest_sha256": file_sha256(
                lifecycle["folder"] / "manifest.json"
            ),
            "complete_sha256": file_sha256(
                lifecycle["folder"] / "complete.json"
            ),
        },
        "candidate": {
            "name": CANDIDATE,
            "complete": candidate_is_complete(candidate_folder, analysis_id),
            "passed": metrics["checks"]["passed"],
            "snapshot_id": metrics["snapshot_id"],
        },
        "outputs": {
            "comparison.csv": file_sha256(acceptance_folder / "comparison.csv"),
        },
    }
    atomic_text(
        acceptance_folder / "report.md",
        build_report(manifest, metrics, comparison),
    )
    manifest["outputs"]["report.md"] = file_sha256(
        acceptance_folder / "report.md"
    )
    atomic_json(acceptance_folder / "manifest.json", manifest)
    return {
        "analysis_id": analysis_id,
        "lifecycle_id": lifecycle_id,
        "status": status,
        "output_directory": str(acceptance_folder),
        "report": str(acceptance_folder / "report.md"),
    }


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="P5-3 生命周期完整账本验收")
    parser.add_argument("--lifecycle-id", required=True)
    parser.add_argument("--model-id", default=DEFAULT_PRIMARY_MODEL_ID)
    parser.add_argument(
        "--fallback-model-id",
        default=DEFAULT_FALLBACK_MODEL_ID,
    )
    parser.add_argument("--smoothing-weeks", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = _parser().parse_args(argv)
    result = execute_acceptance(
        args.lifecycle_id,
        start=args.start,
        end=args.end,
        model_id=args.model_id,
        fallback_model_id=args.fallback_model_id,
        smoothing_weeks=args.smoothing_weeks,
        top_n=args.top_n,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass_p5_3":
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
