"""P4-4B2：选中策略在 100 万、1000 万和 1 亿资金下的完整账本容量曲线。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from alpha.artifacts import file_sha256
from alpha.config import ACCEPTANCE_RUNS_DIR, DEFAULT_CONTRACT, MODEL_DATA_DIR
from alpha.errors import AlphaConfigError
from alpha.logging_conf import get_logger
from alpha.predict import load_predictions
from backtest.chunked import run_chunked
from backtest.specs import BacktestParams
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
    read_json,
)
from scripts.build_alpha.backtest_accept_b1 import (
    CANDIDATES,
    P4_4A_ANALYSIS_ID,
    RISK_FREE_RATE,
    TOP_N,
    build_candidate_strategy,
    candidate_payload,
    load_reused_candidate,
)
from scripts.build_alpha.backtest_accept_b2_artifacts import (
    SCHEMA_VERSION,
    read_tier_completion,
    tier_is_complete,
    write_tier_artifacts,
)
from scripts.build_alpha.backtest_accept_b2_report import (
    BASE_CASH,
    CAPITAL_TIERS,
    TIER_CANDIDATE,
    build_report,
    capacity_curve,
    economic_capacity,
    output_hashes,
    yearly_curve,
)
from common.paths import PROJECT_ROOT

logger = get_logger("scripts.build_alpha.backtest_accept_b2")

ROUND = "p4-4b2"
ROUND_LABEL = "P4-4B2"
B1_ANALYSIS_ID = "p4_4b1_e9c029453c2d"
SELECTED_SPEC = next(
    spec for spec in CANDIDATES if spec.name == TIER_CANDIDATE
)
MODEL_ID = SELECTED_SPEC.source


def _source_reference() -> tuple[dict, str]:
    """读取并校验 P4-4B1 正式来源。"""

    folder = ACCEPTANCE_RUNS_DIR / B1_ANALYSIS_ID
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise AlphaConfigError(f"缺少 P4-4B1 来源 manifest：{manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "pass_p4_4b1":
        raise AlphaConfigError(
            f"P4-4B1 来源状态不是 pass_p4_4b1：{manifest.get('status')!r}"
        )
    candidate = (manifest.get("candidates") or {}).get(TIER_CANDIDATE) or {}
    if candidate.get("complete") is not True or candidate.get("passed") is not True:
        raise AlphaConfigError(
            f"P4-4B1 选中候选未通过：{TIER_CANDIDATE}"
        )
    return manifest, file_sha256(manifest_path)


def _load_model_input() -> tuple[dict, pd.DatetimeIndex]:
    """读取选中模型的 manifest 和预测日期。"""

    _, manifest, predictions = load_predictions(MODEL_ID)
    dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(predictions["trade_date"]).unique())
    )
    if dates.empty:
        raise AlphaConfigError(f"{MODEL_ID} 没有可用预测日期")
    return manifest, dates


def _build_scopes(prediction_dates: pd.DatetimeIndex) -> dict[float, dict]:
    """为三个资金档位构造同一日期口径的验收范围。"""

    scopes = {}
    for cash in CAPITAL_TIERS:
        scope = acceptance_scope(
            start=DEFAULT_START,
            end=DEFAULT_END,
            init_cash=cash,
            risk_free_rate=RISK_FREE_RATE,
            top_n=TOP_N,
        )
        selected_dates = prediction_dates[
            (prediction_dates >= pd.Timestamp(scope["start"]))
            & (prediction_dates <= pd.Timestamp(scope["end"]))
        ]
        scopes[float(cash)] = attach_rebalance_dates(scope, selected_dates)
    references = {
        scope["rebalance_dates_sha256"] for scope in scopes.values()
    }
    if len(references) != 1:
        raise AlphaConfigError("P4-4B2 三个资金档位的调仓日期不一致")
    return scopes


def _load_source_tier(
    scope: dict,
    b1_manifest: dict,
    source_manifest_sha256: str,
) -> dict:
    """从 P4-4B1/P4-4A 读取 100 万档并核对快照。"""

    p4a_manifest_path = (
        ACCEPTANCE_RUNS_DIR / P4_4A_ANALYSIS_ID / "manifest.json"
    )
    if not p4a_manifest_path.is_file():
        raise AlphaConfigError(f"缺少 P4-4A 来源 manifest：{p4a_manifest_path}")
    p4a_manifest = read_json(p4a_manifest_path)
    payload = load_reused_candidate(SELECTED_SPEC, p4a_manifest, scope)
    b1_candidate = (b1_manifest.get("candidates") or {}).get(TIER_CANDIDATE) or {}
    if b1_candidate.get("snapshot_id") != payload["snapshot_id"]:
        raise AlphaConfigError("P4-4B1 与 P4-4A 的选中候选快照不一致")
    source_scope = b1_manifest.get("scope") or {}
    if source_scope.get("init_cash") != BASE_CASH:
        raise AlphaConfigError("P4-4B1 来源不是固定 100 万资金口径")
    if (
        source_scope.get("rebalance_dates_sha256")
        != scope["rebalance_dates_sha256"]
    ):
        raise AlphaConfigError("P4-4B1 来源调仓日期与 P4-4B2 不一致")
    payload["source"]["p4_4b1_analysis_id"] = B1_ANALYSIS_ID
    payload["source"]["p4_4b1_manifest_sha256"] = source_manifest_sha256
    return payload


def _code_sha256() -> str:
    """汇总 P4-4B2 验收链路源码哈希。"""

    root = PROJECT_ROOT
    digest = hashlib.sha256()
    for relative in (
        "core/data/scripts/build_alpha/backtest_accept.py",
        "core/data/scripts/build_alpha/backtest_accept_artifacts.py",
        "core/data/scripts/build_alpha/backtest_accept_report.py",
        "core/data/scripts/build_alpha/backtest_accept_b1.py",
        "core/data/scripts/build_alpha/backtest_accept_b1_artifacts.py",
        "core/data/scripts/build_alpha/backtest_accept_b2.py",
        "core/data/scripts/build_alpha/backtest_accept_b2_artifacts.py",
        "core/data/scripts/build_alpha/backtest_accept_b2_report.py",
        "trading/backtest/chunked.py",
        "trading/backtest/engine.py",
        "trading/backtest/ledger.py",
        "trading/backtest/allocation.py",
        "trading/backtest/signals.py",
        "trading/backtest/strategies/ml_rank_v2.py",
        "research/modeling/alpha/portfolio.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def acceptance_analysis_id(
    scopes: dict[float, dict],
    model_manifest: dict,
    b1_manifest_sha256: str,
    source_snapshot_id: str,
    code_sha256: str,
) -> str:
    """生成绑定来源、模型、资金档位和代码的分析 ID。"""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "round": ROUND,
        "candidate": SELECTED_SPEC.to_dict(),
        "capital_tiers": list(CAPITAL_TIERS),
        "scopes": {
            str(int(cash)): scope for cash, scope in scopes.items()
        },
        "model": {
            "model_id": MODEL_ID,
            "model_sha256": model_manifest.get("model_sha256"),
            "prediction_sha256": model_manifest.get("prediction_sha256"),
            "dataset_id": model_manifest.get("dataset_id"),
            "prediction_rows": model_manifest.get("prediction_rows"),
            "prediction_start": model_manifest.get("prediction_start"),
            "prediction_end": model_manifest.get("prediction_end"),
            "seeds": model_manifest.get("seeds"),
        },
        "p4_4b1_analysis_id": B1_ANALYSIS_ID,
        "p4_4b1_manifest_sha256": b1_manifest_sha256,
        "source_snapshot_id": source_snapshot_id,
        "code_sha256": code_sha256,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "p4_4b2_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _progress(stage: str, payload: dict) -> None:
    """记录 P4-4B2 长回测阶段。"""

    level = "info" if stage in {"meta", "load", "signals", "done"} else "debug"
    getattr(logger, level)(
        "P4-4B2 %s：%s",
        stage,
        json.dumps(payload, ensure_ascii=False, default=str),
    )


def _load_complete_tier(
    folder: Path,
    analysis_id: str,
    scope: dict,
) -> dict:
    """读取一个已完成的资金档位。"""

    completion = read_tier_completion(folder)
    if (
        not completion
        or completion.get("analysis_id") != analysis_id
        or completion.get("candidate") != TIER_CANDIDATE
    ):
        raise AlphaConfigError(f"P4-4B2 档位完成标记无效：{folder}")
    metrics = read_json(folder / "metrics.json")
    run_manifest = read_json(folder / "run_manifest.json")
    if metrics.get("scope", {}).get("init_cash") != scope["init_cash"]:
        raise AlphaConfigError(f"P4-4B2 档位资金口径不一致：{folder}")
    return {
        "metrics": metrics["metrics"],
        "yearly": pd.read_csv(folder / "yearly.csv"),
        "capacity": metrics["capacity"],
        "checks": metrics["checks"],
        "run_manifest": run_manifest,
        "snapshot_id": metrics["snapshot_id"],
        "reused": False,
        "source": {
            "analysis_id": analysis_id,
            "candidate": TIER_CANDIDATE,
            "path": str(folder),
            "files": completion["files"],
        },
    }


def _run_tier(
    cash: float,
    scope: dict,
    folder: Path,
    analysis_id: str,
) -> dict:
    """运行并落盘一个新增资金档位。"""

    strategy = build_candidate_strategy(
        SELECTED_SPEC,
        scope["top_n"],
        scope["rebalance_dates"],
        round_label=ROUND_LABEL,
    )
    params = BacktestParams(
        start=scope["start"],
        end=scope["end"],
        init_cash=scope["init_cash"],
        risk_free_rate=scope["risk_free_rate"],
    )
    logger.info(
        "开始 %s：资金 %.0f，%s ~ %s",
        TIER_CANDIDATE,
        cash,
        params.start,
        params.end,
    )
    result = run_chunked(strategy, params, progress=_progress)
    metrics, yearly, run_manifest = candidate_payload(
        SELECTED_SPEC,
        strategy,
        scope,
        result,
        schema_version=SCHEMA_VERSION,
        round_name=ROUND,
    )
    metrics["init_cash"] = cash
    run_manifest["round"] = ROUND
    run_manifest["init_cash"] = cash
    write_tier_artifacts(
        folder,
        analysis_id=analysis_id,
        candidate=TIER_CANDIDATE,
        metrics=metrics,
        strategy=strategy.to_dict(),
        yearly=yearly,
        daily=result.daily,
        orders=result.orders,
        holdings=result.holdings,
        run_manifest=run_manifest,
    )
    logger.info(
        "完成资金档位 %.0f：订单=%d 耗时=%ss 检查=%s",
        cash,
        len(result.orders),
        result.metadata.get("elapsed_seconds"),
        "通过" if metrics["checks"]["passed"] else "失败",
    )
    return _load_complete_tier(folder, analysis_id, scope)


def _write_outputs(
    *,
    root: Path,
    analysis_id: str,
    candidates: dict[float, dict],
    statuses: dict[float, str],
    model_manifest: dict,
    b1_manifest_sha256: str,
) -> dict:
    """写出容量曲线、分年结果、报告、来源和总 manifest。"""

    curve = capacity_curve(candidates)
    yearly = yearly_curve(candidates)
    report = build_report(
        analysis_id=analysis_id,
        candidates=candidates,
        expected_tiers=len(CAPITAL_TIERS),
        source_analysis_id=B1_ANALYSIS_ID,
        source_manifest_sha256=b1_manifest_sha256,
    )
    atomic_write_frame(root / "capacity_curve.csv", curve)
    atomic_write_frame(root / "yearly.csv", yearly)
    atomic_write_text(root / "report.md", report)
    sources = {
        "p4_4b1": {
            "analysis_id": B1_ANALYSIS_ID,
            "manifest_sha256": b1_manifest_sha256,
            "candidate": TIER_CANDIDATE,
        },
        "model": {
            "model_id": MODEL_ID,
            "manifest": model_manifest,
        },
        "tiers": {
            str(int(cash)): {
                "status": statuses[cash],
                "source": candidates[cash]["source"],
            }
            for cash in sorted(candidates)
        },
    }
    atomic_write_json(root / "sources.json", sources)
    output_names = (
        "capacity_curve.csv",
        "yearly.csv",
        "report.md",
        "sources.json",
    )
    all_complete = len(candidates) == len(CAPITAL_TIERS)
    all_passed = bool(candidates) and all(
        payload["checks"].get("passed") is True
        for payload in candidates.values()
    )
    status = (
        "pass_p4_4b2"
        if all_complete and all_passed
        else "failed_p4_4b2"
        if all_complete
        else "partial_p4_4b2"
    )
    capacity_cash = economic_capacity(curve)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "round": ROUND,
        "status": status,
        "candidate": SELECTED_SPEC.to_dict(),
        "capital_tiers": list(CAPITAL_TIERS),
        "economic_capacity_cash": capacity_cash,
        "tiers": {
            str(int(cash)): {
                "complete": cash in candidates,
                "passed": candidates.get(cash, {})
                .get("checks", {})
                .get("passed", False),
                "status": statuses.get(cash, "missing"),
                "snapshot_id": candidates.get(cash, {}).get("snapshot_id"),
                "elapsed_seconds": candidates.get(cash, {})
                .get("run_manifest", {})
                .get("elapsed_seconds"),
            }
            for cash in CAPITAL_TIERS
        },
        "p4_4b1_analysis_id": B1_ANALYSIS_ID,
        "p4_4b1_manifest_sha256": b1_manifest_sha256,
        "model_prediction_sha256": model_manifest.get("prediction_sha256"),
        "outputs": output_hashes(root, output_names),
    }
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


def execute_acceptance(
    *,
    resume: bool = False,
    output_root: Path | None = None,
) -> dict:
    """执行 P4-4B2，返回分析目录和验收状态。"""

    b1_manifest, b1_manifest_sha256 = _source_reference()
    model_manifest, prediction_dates = _load_model_input()
    scopes = _build_scopes(prediction_dates)
    source = _load_source_tier(
        scopes[BASE_CASH],
        b1_manifest,
        b1_manifest_sha256,
    )
    analysis_id = acceptance_analysis_id(
        scopes,
        model_manifest,
        b1_manifest_sha256,
        source["snapshot_id"],
        _code_sha256(),
    )
    root = (output_root or ACCEPTANCE_RUNS_DIR) / analysis_id
    if root.exists() and not resume:
        raise AlphaConfigError(
            f"验收目录已存在：{root}；确认要续跑时使用 --resume"
        )
    root.mkdir(parents=True, exist_ok=True)
    candidates: dict[float, dict] = {BASE_CASH: source}
    statuses: dict[float, str] = {BASE_CASH: "reused_p4_4b1"}
    for cash in CAPITAL_TIERS:
        if cash == BASE_CASH:
            logger.info("复用 P4-4B1 的 100 万档")
            continue
        folder = root / f"cash_{int(cash)}"
        if resume and tier_is_complete(
            folder,
            analysis_id,
            TIER_CANDIDATE,
        ):
            logger.info("跳过已完成资金档位 %.0f", cash)
            candidates[cash] = _load_complete_tier(
                folder,
                analysis_id,
                scopes[cash],
            )
            statuses[cash] = "skipped_complete"
            continue
        candidates[cash] = _run_tier(
            cash,
            scopes[cash],
            folder,
            analysis_id,
        )
        statuses[cash] = "completed"
    manifest = _write_outputs(
        root=root,
        analysis_id=analysis_id,
        candidates=candidates,
        statuses=statuses,
        model_manifest=model_manifest,
        b1_manifest_sha256=b1_manifest_sha256,
    )
    return {
        "analysis_id": analysis_id,
        "output_directory": str(root),
        "status": manifest["status"],
        "tiers": statuses,
        "economic_capacity_cash": manifest["economic_capacity_cash"],
        "report": str(root / "report.md"),
    }


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="P4-4B2 三档容量曲线")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过当前分析目录中已完整落盘的 1000 万和 1 亿档位",
    )
    return parser


def main() -> None:
    """命令行入口。"""

    args = _parser().parse_args()
    result = execute_acceptance(resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass_p4_4b2":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
