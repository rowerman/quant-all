"""V2/V3 搜索运行详情的共享投影。"""

from __future__ import annotations

from typing import Any, Mapping

from mining.free_exploration import direction_from_hypothesis
from mining.search_cards import HypothesisCard
from mining.search_contract_types import FREE_EXPLORATION_INPUT_MODE
from mining.search_evidence import EvidencePacket
from mining.search_progress import progress_for_search
from mining.search_research_quality import evaluate_research_quality
from mining.search_scorecard import build_run_scorecards
from mining.search_statistical_diagnostics import MULTIPLE_TESTING_STAGE


def build_search_run_detail(
    store: Any,
    run_id: str,
    run_view: Mapping[str, Any],
) -> dict[str, Any]:
    """从运行轨迹组装 V2/V3 共用的页面详情。"""

    trace = store.get_search_trace(run_id)
    progress = progress_for_search(store, run_id)
    scorecard_objects = {
        item.candidate_id: item
        for item in build_run_scorecards(store, run_id)
    }
    scorecards = {
        candidate_id: card.to_dict()
        for candidate_id, card in scorecard_objects.items()
    }
    decisions = {
        int(item["turn_index"]): item for item in trace["decisions"]
    }
    evidence = {
        int(item["turn_index"]): item for item in trace["evidence"]
    }
    tool_calls = store.list_analysis_tool_calls(run_id)
    tool_calls_by_turn: dict[int, list[dict[str, Any]]] = {}
    analysis_count_by_candidate: dict[str, int] = {}
    for call in tool_calls:
        turn_index = int(call["turn_index"])
        tool_calls_by_turn.setdefault(turn_index, []).append(
            {
                "call_index": call["call_index"],
                "tool_name": call["tool_name"],
                "tool_version": call["tool_version"],
                "candidate_id": call.get("candidate_id"),
                "status": call["status"],
                "cache_hit": bool(call["cache_hit"]),
                "duration_ms": call["duration_ms"],
                "error_code": call.get("error_code"),
                "error_message": call.get("error_message"),
                "input_hash": call.get("input_hash"),
                "result_hash": call.get("result_hash"),
                "arguments_summary": call.get("arguments_summary", {}),
                "evidence_class": (
                    "supplemental"
                    if call["status"] in {"succeeded", "cache_hit"}
                    else "failure"
                ),
            }
        )
        candidate_id = call.get("candidate_id")
        if candidate_id is not None:
            analysis_count_by_candidate[str(candidate_id)] = (
                analysis_count_by_candidate.get(str(candidate_id), 0) + 1
            )
    turns = []
    for item in trace["turns"]:
        turn_index = int(item["turn_index"])
        decision = decisions.get(turn_index)
        packet = evidence.get(turn_index)
        turns.append(
            {
                "turn_index": turn_index,
                "status": item["status"],
                "hypothesis_id": item.get("hypothesis_id"),
                "action": decision["action"] if decision else None,
                "reason": decision["reason"][:500] if decision else None,
                "evidence_hash": (
                    EvidencePacket.from_dict(packet).evidence_hash
                    if packet
                    else None
                ),
                "feedback": (
                    _feedback_summary(packet)
                    if packet is not None
                    else None
                ),
                "tool_calls": tool_calls_by_turn.get(turn_index, []),
            }
        )
    candidates = []
    for item in trace["candidates"]:
        card = scorecards.get(item["candidate_id"], {})
        card_object = scorecard_objects.get(item["candidate_id"])
        selection_quality = (
            evaluate_research_quality(
                card_object,
                progress.control_policy,
                stage="selection",
            ).to_dict()
            if card_object is not None
            else None
        )
        research_stage = (
            "full"
            if card_object is not None
            and "full" in card_object.evidence_status_by_stage
            else "selection"
        )
        research_quality = (
            evaluate_research_quality(
                card_object,
                progress.control_policy,
                stage=research_stage,
            ).to_dict()
            if card_object is not None
            else None
        )
        multiple_testing = card.get("multiple_testing") or {}
        statistical_result = next(
            (
                candidate
                for candidate in multiple_testing.get("candidates", [])
                if candidate.get("candidate_id") == item["candidate_id"]
            ),
            None,
        )
        candidates.append(
            {
                "candidate_id": item["candidate_id"],
                "name": item["name"],
                "status": item["status"],
                "parent_candidate_id": item["parent_candidate_id"],
                "hypothesis_id": item["hypothesis_id"],
                "batch_id": item["batch_id"],
                "missing_fields": card.get("missing_fields", []),
                "scorecard_hash": card.get("scorecard_hash"),
                "decision_stage": card.get("decision_stage"),
                "metrics": card.get("metrics", {}),
                "selection_diagnostics": card.get(
                    "selection_diagnostics"
                ),
                "stability": card.get("stability"),
                "selection_quality": selection_quality,
                "research_quality": research_quality,
                "multiple_testing": statistical_result,
                "attempts": card.get("trial_counts", {}),
                "analysis_count": analysis_count_by_candidate.get(
                    item["candidate_id"],
                    0,
                ),
            }
        )
    finalist_event = next(
        (
            event
            for event in reversed(store.list_events(run_id))
            if event["event_type"] == "finalists_selected"
        ),
        None,
    )
    branch_count = len(
        {str(item["branch_id"]) for item in trace["hypotheses"]}
    )
    statistical_rows = store.list_evaluation_evidence(
        run_id,
        stage=MULTIPLE_TESTING_STAGE,
    )
    statistical_summary = (
        statistical_rows[-1]["evidence"]["metrics"]
        if statistical_rows
        else None
    )
    research_summary = store.load_research_summary(run_id)
    research_state = store.research_run_state(run_id)
    return {
        "run": dict(run_view),
        "exploration": (
            exploration_detail(trace)
            if run_view["input_mode"] == FREE_EXPLORATION_INPUT_MODE
            else None
        ),
        "progress": progress.to_dict(),
        "hypotheses": [
            {
                "hypothesis_id": HypothesisCard.from_dict(
                    item
                ).hypothesis_id,
                "parent_hypothesis_id": item["parent_hypothesis_id"],
                "turn_index": item["turn_index"],
                "branch_id": item["branch_id"],
                "statement": item["statement"],
                "mechanism": item["mechanism"],
                "expected_direction": item["expected_direction"],
            }
            for item in trace["hypotheses"]
        ],
        "turns": turns,
        "candidates": candidates,
        "attempts": progress.trial_counts.to_dict(),
        "branch_count": branch_count,
        "portfolio": (
            finalist_event["payload"].get("portfolio")
            if finalist_event is not None
            else None
        ),
        "multiple_testing": statistical_summary,
        "analysis_tools": {
            "catalog_version": "mining_analysis_catalog_v1",
            "calls": tool_calls_by_turn,
            "total_calls": len(tool_calls),
            "failed_calls": sum(
                item["status"] == "failed" for item in tool_calls
            ),
        },
        "research": (
            {
                **research_state,
                "summary": (
                    None
                    if research_summary is None
                    else research_summary.to_dict()
                ),
            }
            if research_state is not None
            else None
        ),
    }


def exploration_detail(trace: Mapping[str, Any]) -> dict[str, Any]:
    """返回已冻结的自由探索首轮方向安全摘要。"""

    for row in trace.get("hypotheses", ()):
        card = HypothesisCard.from_dict(row)
        if card.metadata.get("source") != "free_exploration":
            continue
        direction = direction_from_hypothesis(card)
        if direction is None:
            continue
        return {
            "initial_direction": {
                "direction_id": direction.direction_id,
                "direction_hash": direction.direction_hash,
                "semantic_hash": direction.semantic_hash,
                "mechanism_category": direction.mechanism_category,
                "statement": direction.statement,
                "mechanism": direction.mechanism,
                "expected_direction": direction.expected_direction,
                "falsification_criteria": list(
                    direction.falsification_criteria
                ),
                "data_dependencies": list(
                    direction.data_dependencies
                ),
                "expression_complexity": direction.expression_complexity,
                "max_expression_depth": direction.max_expression_depth,
                "seed": direction.seed,
                "seed_source": direction.seed_source,
            }
        }
    return {"initial_direction": None}


def _feedback_summary(packet: Mapping[str, Any]) -> dict[str, Any]:
    metrics = packet.get("metrics") or {}
    return {
        "candidate_count": len(metrics),
        "failure_count": len(packet.get("failures") or []),
        "quality_candidates": [
            candidate_id
            for candidate_id, values in metrics.items()
            if values.get("gate_passed") is True
            and values.get("rank_ic") is not None
            and values.get("rank_ic_ir") is not None
            and abs(float(values["rank_ic"])) >= 0.02
            and abs(float(values["rank_ic_ir"])) >= 0.20
            and float(values.get("year_sign_consistency") or 0.0) >= 0.60
            and float(values.get("coverage") or 0.0) >= 0.30
        ],
        "trial_counts": packet.get("trial_counts"),
    }


__all__ = ["build_search_run_detail", "exploration_detail"]
