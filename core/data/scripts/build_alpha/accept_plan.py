"""按通用评估计划执行完整账本验收。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from alpha.artifacts import file_sha256
from alpha.config import DEFAULT_CONTRACT, MODEL_DATA_DIR
from alpha.errors import AlphaConfigError
from alpha.ledger_acceptance import (
    LEDGER_ACCEPTANCE_SCHEMA_VERSION,
    LEDGER_CANDIDATE_SCHEMA_VERSION,
    acceptance_scope,
    candidate_names,
    candidate_strategy,
    ledger_analysis_id,
    load_evaluation,
)
from alpha.logging_conf import get_logger
from backtest.chunked import run_chunked
from backtest.providers import validate_strategy_provider
from backtest.specs import BacktestParams
from backtest.strategies._plan_rank import ensure_registered
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
    capacity_summary,
    yearly_summary,
)
from scripts.build_alpha.ledger_accept_report import (
    build_report,
    comparison_frame,
    yearly_frame,
)

logger = get_logger("scripts.build_alpha.accept_plan")


def _code_sha256() -> str:
    """汇总计划、provider、执行器和产物代码哈希。"""

    root = PROJECT_ROOT
    digest = hashlib.sha256()
    for relative in (
        "research/modeling/alpha/ledger_acceptance.py",
        "trading/backtest/strategies/_plan_rank.py",
        "trading/backtest/chunked.py",
        "trading/backtest/engine.py",
        "trading/backtest/ledger.py",
        "trading/backtest/allocation.py",
        "trading/backtest/signals.py",
        "core/data/scripts/build_alpha/accept_plan.py",
        "core/data/scripts/build_alpha/backtest_accept_artifacts.py",
        "core/data/scripts/build_alpha/backtest_accept_report.py",
        "core/data/scripts/build_alpha/ledger_accept_report.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _candidate_dates(
    evaluation_folder: Path,
    evaluation_manifest: dict,
    name: str,
) -> pd.DatetimeIndex:
    """读取并校验一个候选的不可变信号日。"""

    relative = f"scores/{name}.parquet"
    path = evaluation_folder / relative
    expected = (evaluation_manifest.get("files") or {}).get(relative)
    if not path.is_file() or not expected:
        raise AlphaConfigError(f"候选分数不存在：{path}")
    if file_sha256(path) != str(expected):
        raise AlphaConfigError(f"候选分数哈希不一致：{path}")
    frame = pd.read_parquet(path, columns=["trade_date"])
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame["trade_date"]).unique()))
    if dates.empty:
        raise AlphaConfigError(f"候选分数没有信号日：{name}")
    return dates


def _scope_with_dates(
    evaluation_manifest: dict,
    dates: pd.DatetimeIndex,
    *,
    init_cash: float,
    risk_free_rate: float,
) -> dict:
    """把共同候选信号日写入账本范围。"""

    scope = acceptance_scope(
        evaluation_manifest,
        init_cash=init_cash,
        risk_free_rate=risk_free_rate,
    )
    in_range = dates[
        (dates >= pd.Timestamp(scope["start"]))
        & (dates <= pd.Timestamp(scope["end"]))
    ]
    if in_range.empty:
        raise AlphaConfigError("候选没有账本范围内的信号日")
    normalized = [str(day.date()) for day in in_range]
    encoded = json.dumps(
        normalized,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **scope,
        "rebalance_dates": normalized,
        "rebalance_dates_count": len(normalized),
        "rebalance_dates_sha256": hashlib.sha256(encoded).hexdigest(),
        "rebalance_dates_source": "evaluation_candidate_scores",
    }


def _candidate_payload(
    *,
    candidate,
    strategy,
    scope: dict,
    result,
) -> tuple[dict, pd.DataFrame, dict]:
    """整理单个候选的标准账本产物。"""

    yearly = yearly_summary(result.daily)
    capacity = capacity_summary(result.metadata)
    daily_dates = pd.DatetimeIndex(result.daily.index).sort_values()
    signal_dates = pd.DatetimeIndex(scope["rebalance_dates"])
    positions = daily_dates.get_indexer(signal_dates)
    execution_dates = [
        daily_dates[position + 1]
        for position in positions
        if position >= 0 and position + 1 < len(daily_dates)
    ]
    checks = acceptance_checks(
        result.metrics,
        result.metadata,
        result.daily,
        dev_start=DEFAULT_CONTRACT.dev_start,
        dev_end=DEFAULT_CONTRACT.dev_end,
        expected_signal_days=len(signal_dates),
        expected_execution_days=len(execution_dates),
        actual_signal_days=int(
            (result.metadata.get("signal") or {}).get(
                "rebalance_days", -1
            )
        ),
    )
    metrics = {
        "schema_version": LEDGER_CANDIDATE_SCHEMA_VERSION,
        "candidate": candidate.name,
        "source": candidate.source.to_dict(),
        "score_policy": {
            "kind": candidate.score_policy.kind,
            "smoothing_weeks": candidate.score_policy.smoothing_weeks,
        },
        "holding_policy": dict(candidate.holding_policy),
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
        "candidate": candidate.name,
        "source": candidate.source.to_dict(),
        "acceptance_checks": checks,
    }
    return metrics, yearly, run_manifest


def _load_candidate(
    folder: Path,
    analysis_id: str,
    name: str,
) -> dict:
    """读取已完成的通用账本候选。"""

    if not candidate_is_complete(
        folder,
        analysis_id,
        name,
        schema_version=LEDGER_CANDIDATE_SCHEMA_VERSION,
    ):
        raise AlphaConfigError(f"候选完成标记无效：{folder}")
    metrics = read_json(folder / "metrics.json")
    return {
        "metrics": metrics["metrics"],
        "source": metrics["source"],
        "score_policy": metrics["score_policy"],
        "holding_policy": metrics["holding_policy"],
        "yearly": pd.read_csv(folder / "yearly.csv"),
        "checks": metrics["checks"],
        "run_manifest": read_json(folder / "run_manifest.json"),
        "snapshot_id": metrics["snapshot_id"],
    }


def _write_outputs(
    folder: Path,
    *,
    analysis_id: str,
    evaluation_id: str,
    evaluation_manifest: dict,
    evaluation_folder: Path,
    names: tuple[str, ...],
    scope: dict,
    candidates: dict[str, dict],
    code_hash: str,
) -> dict:
    """写出通用账本汇总产物和总 manifest。"""

    comparison = comparison_frame(candidates)
    yearly = yearly_frame(candidates)
    all_complete = len(candidates) == len(names)
    all_passed = all_complete and all(
        payload["checks"].get("passed") for payload in candidates.values()
    )
    status = (
        "pass_acceptance"
        if all_passed
        else "failed_acceptance"
        if all_complete
        else "partial_acceptance"
    )
    report = build_report(
        analysis_id=analysis_id,
        evaluation_id=evaluation_id,
        scope=scope,
        candidates=candidates,
        status=status,
    )
    atomic_write_frame(folder / "comparison.csv", comparison)
    atomic_write_frame(folder / "yearly.csv", yearly)
    atomic_write_text(folder / "report.md", report)
    output_names = ("comparison.csv", "yearly.csv", "report.md")
    sources = {
        "evaluation_id": evaluation_id,
        "evaluation_manifest_sha256": file_sha256(
            evaluation_folder / "manifest.json"
        ),
        "evaluation_manifest": evaluation_manifest,
        "candidate_scores": {
            name: (evaluation_manifest.get("files") or {}).get(
                f"scores/{name}.parquet"
            )
            for name in names
        },
    }
    atomic_write_json(folder / "sources.json", sources)
    output_names = (*output_names, "sources.json")
    manifest = {
        "schema_version": LEDGER_ACCEPTANCE_SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "status": status,
        "evaluation_id": evaluation_id,
        "selected_candidate": evaluation_manifest.get(
            "selected_candidate"
        ),
        "scope": scope,
        "candidates": {
            name: {
                "complete": name in candidates,
                "passed": candidates[name]["checks"].get("passed")
                if name in candidates
                else False,
                "snapshot_id": candidates[name]["snapshot_id"]
                if name in candidates
                else None,
                "source": candidates[name]["source"]
                if name in candidates
                else None,
                "score_policy": candidates[name]["score_policy"]
                if name in candidates
                else None,
                "holding_policy": candidates[name]["holding_policy"]
                if name in candidates
                else None,
            }
            for name in names
        },
        "sources": sources,
        "code_sha256": code_hash,
        "outputs": {
            name: file_sha256(folder / name) for name in output_names
        },
    }
    atomic_write_json(folder / "manifest.json", manifest)
    return manifest


def run_acceptance(
    evaluation_id: str,
    *,
    candidate: str = "selected",
    init_cash: float = 1_000_000.0,
    risk_free_rate: float = 0.02,
    resume: bool = False,
    root: Path | None = None,
) -> dict:
    """执行评估计划候选的完整账本验收。"""

    ensure_registered()
    evaluation_folder, evaluation_manifest, plan = load_evaluation(
        evaluation_id,
        root=(Path(root) / "runs" / "evaluation") if root else None,
    )
    names = candidate_names(plan, evaluation_manifest, candidate)
    date_sets = [
        _candidate_dates(evaluation_folder, evaluation_manifest, name)
        for name in names
    ]
    for name, dates in zip(names[1:], date_sets[1:], strict=True):
        if not date_sets[0].equals(dates):
            raise AlphaConfigError(f"候选 {name} 的信号日与其他候选不一致")
    scope = _scope_with_dates(
        evaluation_manifest,
        date_sets[0],
        init_cash=init_cash,
        risk_free_rate=risk_free_rate,
    )
    code_hash = _code_sha256()
    analysis_id = ledger_analysis_id(
        evaluation_id,
        file_sha256(evaluation_folder / "manifest.json"),
        names,
        scope,
        code_hash,
    )
    output_root = Path(root or MODEL_DATA_DIR) / "runs" / "acceptance"
    folder = output_root / analysis_id
    if folder.exists() and not resume:
        raise AlphaConfigError(
            f"账本目录已存在：{folder}；续跑请使用 --resume"
        )
    folder.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, dict] = {}
    statuses: dict[str, str] = {}
    for name in names:
        candidate_folder = folder / name
        if resume and candidate_is_complete(
            candidate_folder,
            analysis_id,
            name,
            schema_version=LEDGER_CANDIDATE_SCHEMA_VERSION,
        ):
            loaded[name] = _load_candidate(
                candidate_folder,
                analysis_id,
                name,
            )
            statuses[name] = "skipped_complete"
            continue
        candidate = plan.by_name(name)
        strategy = candidate_strategy(
            evaluation_id,
            candidate,
            scope["rebalance_dates"],
        )
        validate_strategy_provider(strategy)
        params = BacktestParams(
            start=scope["start"],
            end=scope["end"],
            init_cash=scope["init_cash"],
            risk_free_rate=scope["risk_free_rate"],
        )
        logger.info(
            "开始通用账本候选 %s：%s ~ %s",
            name,
            params.start,
            params.end,
        )
        result = run_chunked(
            strategy,
            params,
            progress=lambda stage, payload: logger.debug(
                "通用账本 %s %s：%s",
                name,
                stage,
                json.dumps(payload, ensure_ascii=False, default=str),
            ),
        )
        metrics, yearly, run_manifest = _candidate_payload(
            candidate=candidate,
            strategy=strategy,
            scope=scope,
            result=result,
        )
        write_candidate_artifacts(
            candidate_folder,
            analysis_id=analysis_id,
            candidate=name,
            metrics=metrics,
            strategy=strategy.to_dict(),
            yearly=yearly,
            daily=result.daily,
            orders=result.orders,
            holdings=result.holdings,
            run_manifest=run_manifest,
            schema_version=LEDGER_CANDIDATE_SCHEMA_VERSION,
        )
        loaded[name] = _load_candidate(candidate_folder, analysis_id, name)
        statuses[name] = "completed"
        logger.info(
            "完成通用账本候选 %s：检查=%s",
            name,
            "通过" if loaded[name]["checks"].get("passed") else "失败",
        )

    manifest = _write_outputs(
        folder,
        analysis_id=analysis_id,
        evaluation_id=evaluation_id,
        evaluation_manifest=evaluation_manifest,
        evaluation_folder=evaluation_folder,
        names=names,
        scope=scope,
        candidates=loaded,
        code_hash=code_hash,
    )
    return {
        "analysis_id": analysis_id,
        "evaluation_id": evaluation_id,
        "status": manifest["status"],
        "output_directory": str(folder),
        "candidates": statuses,
        "report": str(folder / "report.md"),
    }


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="通用模型完整账本验收")
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument(
        "--candidate",
        default="selected",
        help="selected/reference/control/all 或评估计划中的候选名",
    )
    parser.add_argument("--init-cash", type=float, default=1_000_000.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.02)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = _parser().parse_args(argv)
    payload = run_acceptance(
        args.evaluation_id,
        candidate=args.candidate,
        init_cash=args.init_cash,
        risk_free_rate=args.risk_free_rate,
        resume=args.resume,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "pass_acceptance":
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
