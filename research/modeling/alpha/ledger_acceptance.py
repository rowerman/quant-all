"""计划驱动的完整账本候选、范围和策略定义。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from alpha.config import EVALUATION_RUNS_DIR
from alpha.errors import AlphaArtifactError, AlphaConfigError
from alpha.evaluation_plan import EvaluationCandidate, EvaluationPlan
from backtest.specs import (
    BenchmarkSpec,
    PortfolioSpec,
    ProviderSpec,
    RebalanceSpec,
    SignalSpec,
    StrategySpec,
)

LEDGER_ACCEPTANCE_SCHEMA_VERSION = "alpha_ledger_acceptance_v1"
LEDGER_CANDIDATE_SCHEMA_VERSION = "alpha_ledger_candidate_v1"
EVALUATION_SCHEMA_VERSION = "alpha_model_evaluation_v1"


def evaluation_folder(
    evaluation_id: str,
    *,
    root: Path | None = None,
) -> Path:
    """返回并校验评估目录。"""

    if not evaluation_id or Path(evaluation_id).name != evaluation_id:
        raise AlphaArtifactError(f"非法评估 ID：{evaluation_id!r}")
    base = Path(root or EVALUATION_RUNS_DIR)
    folder = base / evaluation_id
    if not (folder / "manifest.json").is_file():
        raise AlphaArtifactError(f"评估产物不存在：{folder}")
    return folder


def load_evaluation(
    evaluation_id: str,
    *,
    root: Path | None = None,
) -> tuple[Path, dict, EvaluationPlan]:
    """读取评估 manifest 和计划，并校验状态与身份。"""

    folder = evaluation_folder(evaluation_id, root=root)
    try:
        manifest = json.loads(
            (folder / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise AlphaArtifactError(f"无法读取评估 manifest：{folder}") from error
    if not isinstance(manifest, dict):
        raise AlphaArtifactError("评估 manifest 必须是 JSON 对象")
    if (
        manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION
        or manifest.get("analysis_id") != evaluation_id
    ):
        raise AlphaArtifactError("评估 manifest 的 schema 或 ID 不一致")
    if manifest.get("status") != "pass_evaluation":
        raise AlphaConfigError(
            f"评估没有通过，不能进入完整账本：{manifest.get('status')}"
        )
    plan = EvaluationPlan.from_dict(manifest.get("plan") or {})
    return folder, manifest, plan


def candidate_names(
    plan: EvaluationPlan,
    evaluation_manifest: dict,
    selection: str,
) -> tuple[str, ...]:
    """按评估计划展开本次账本候选。"""

    all_names = tuple(
        item.name for item in (*plan.candidates, *plan.controls)
    )
    if selection == "all":
        return all_names
    if selection == "selected":
        selected = str(
            evaluation_manifest.get("selected_candidate")
            or plan.selected
            or plan.reference
        )
    elif selection == "reference":
        selected = plan.reference
    elif selection == "control":
        if not plan.controls:
            raise AlphaConfigError("评估计划没有控制候选")
        selected = plan.controls[0].name
    elif selection in all_names:
        selected = selection
    else:
        raise AlphaConfigError(
            f"未知账本候选 {selection!r}；可选 selected/reference/control/all "
            f"或 {list(all_names)}"
        )
    return (selected,)


def acceptance_scope(
    evaluation_manifest: dict,
    *,
    init_cash: float,
    risk_free_rate: float,
) -> dict:
    """构造通用账本的共同范围和资金参数。"""

    if init_cash <= 0:
        raise AlphaConfigError("init_cash 必须大于 0")
    if not -0.99 <= risk_free_rate <= 1.0:
        raise AlphaConfigError("risk_free_rate 必须在 [-0.99, 1] 之间")
    start = str(evaluation_manifest.get("start") or "")
    end = str(evaluation_manifest.get("end") or "")
    if not start or not end or start > end:
        raise AlphaConfigError("评估 manifest 缺少有效区间")
    return {
        "start": start,
        "end": end,
        "init_cash": float(init_cash),
        "risk_free_rate": float(risk_free_rate),
        "benchmark": "H00300.CSI",
        "max_participation_rate": 0.1,
    }


def candidate_strategy(
    evaluation_id: str,
    candidate: EvaluationCandidate,
    rebalance_dates: list[str] | tuple[str, ...],
) -> StrategySpec:
    """按评估候选构造统一的计划回测策略。"""

    top_n = int(candidate.holding_policy["top_n"])
    return StrategySpec(
        name=f"{evaluation_id} {candidate.name} Top{top_n}",
        signal=SignalSpec(
            factors=(),
            weights=(),
            provider=ProviderSpec(
                name="plan_rank",
                version="1",
                params={
                    "evaluation_id": evaluation_id,
                    "candidate": candidate.name,
                    "rebalance_dates": list(rebalance_dates),
                },
            ),
        ),
        portfolio=PortfolioSpec(selection="provider", top_n=top_n),
        rebalance=RebalanceSpec(frequency="custom"),
        benchmark=BenchmarkSpec("H00300.CSI"),
    )


def ledger_analysis_id(
    evaluation_id: str,
    evaluation_manifest_sha256: str,
    candidates: tuple[str, ...],
    scope: dict,
    code_sha256: str,
) -> str:
    """生成绑定评估来源、候选、范围和代码的账本 ID。"""

    payload = {
        "schema_version": LEDGER_ACCEPTANCE_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "candidates": list(candidates),
        "scope": scope,
        "code_sha256": code_sha256,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "acceptance_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "LEDGER_ACCEPTANCE_SCHEMA_VERSION",
    "LEDGER_CANDIDATE_SCHEMA_VERSION",
    "acceptance_scope",
    "candidate_names",
    "candidate_strategy",
    "evaluation_folder",
    "ledger_analysis_id",
    "load_evaluation",
]
