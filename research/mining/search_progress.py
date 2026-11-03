"""V2 搜索运行的可恢复进度和审核摘要。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mining.search_budget import SearchBudget, TrialCounts
from mining.search_control import SearchControlPolicy
from mining.storage_v2 import MiningV2Store


@dataclass(frozen=True)
class SearchProgress:
    """从 V2 数据库和轨迹恢复的当前进度。"""

    run_id: str
    status: str
    current_stage: str
    current_turn: int
    current_branch_id: str | None
    current_action: str | None
    stop_reason: str | None
    interrupted_from: str | None
    turns_requested: int
    turns_completed: int
    hypotheses: int
    candidates: int
    failed_candidates: int
    trial_counts: TrialCounts
    budget: SearchBudget
    control_policy: SearchControlPolicy

    def to_dict(self) -> dict[str, Any]:
        """返回页面和 API 使用的稳定结构。"""

        return {
            "run_id": self.run_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "current_turn": self.current_turn,
            "current_branch_id": self.current_branch_id,
            "current_action": self.current_action,
            "stop_reason": self.stop_reason,
            "interrupted_from": self.interrupted_from,
            "turns_requested": self.turns_requested,
            "turns_completed": self.turns_completed,
            "hypotheses": self.hypotheses,
            "candidates": self.candidates,
            "failed_candidates": self.failed_candidates,
            "attempts": self.trial_counts.to_dict(),
            "budget": self.budget.to_dict(),
            "control_policy": self.control_policy.to_dict(),
        }


def progress_for_search(
    store: MiningV2Store,
    run_id: str,
) -> SearchProgress:
    """读取当前运行的阶段、树、尝试和预算状态。"""

    run = store.get_run(run_id)
    control = store.search_control_state(run_id)
    trace = store.get_search_trace(run_id)
    turns = trace["turns"]
    hypotheses = trace["hypotheses"]
    candidates = trace["candidates"]
    current_branch = (
        hypotheses[-1]["branch_id"] if hypotheses else None
    )
    current_turn = (
        int(turns[-1]["turn_index"]) if turns else 0
    )
    return SearchProgress(
        run_id=run_id,
        status=control["status"],
        current_stage=(
            control["interrupted_from"] or control["status"]
        ),
        current_turn=current_turn,
        current_branch_id=current_branch,
        current_action=control["current_action"],
        stop_reason=control["stop_reason"],
        interrupted_from=control["interrupted_from"],
        turns_requested=len(turns),
        turns_completed=sum(item["status"] == "completed" for item in turns),
        hypotheses=len(hypotheses),
        candidates=len(candidates),
        failed_candidates=sum(
            item["status"]
            in {"technical_failed", "screened_out", "holdout_failed", "failed"}
            for item in candidates
        ),
        trial_counts=store.trial_counts(run_id),
        budget=run["context"]["budget"]
        if isinstance(run["context"]["budget"], SearchBudget)
        else SearchBudget.from_dict(run["context"]["budget"]),
        control_policy=run["context"]["control_policy"]
        if isinstance(run["context"]["control_policy"], SearchControlPolicy)
        else SearchControlPolicy.from_dict(run["context"]["control_policy"]),
    )
