"""资料驱动 V2 运行的种子候选登记和首轮评估。"""

from __future__ import annotations

from typing import Any, Mapping

from mining.errors import ResearchRunFailure
from mining.research_run_contracts import ResearchRunSummary
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_evidence import EvidencePacket
from mining.search_evidence_builder import build_development_evidence
from mining.search_control import SearchControlState, decide_search_action
from mining.search_orchestrator_support import (
    batch_input_hash,
    candidate_payload,
    merge_evidence,
    trial_record,
)


class SearchResearchMixin:
    """把资料转换结果写成 V2 首个种子批次。"""

    def _research_seed_if_needed(
        self,
        run: Mapping[str, Any],
        context: Any,
    ) -> bool:
        """完成资料种子；返回 True 表示运行已迁移到后续阶段。"""

        research_input = self.store.load_research_input(run["run_id"])
        if research_input is None:
            return False
        trace = self.store.get_search_trace(run["run_id"])
        completed = any(
            item["event_key"] == "research_seed_completed"
            for item in self.store.list_events(run["run_id"])
        )
        if completed:
            return False

        summary = self.store.load_research_summary(run["run_id"])
        if summary is None:
            preparation = self._prepare_research_run(
                research_input,
                context,
            )
            self.store.save_research_summary(
                preparation.summary,
                preparation.provenances,
            )
            summary = preparation.summary
            if summary.status != "ready":
                raise ResearchRunFailure(
                    "资料批次没有产生可进入 V2 验证的候选"
                )
        provenances = self.store.list_research_candidates(run["run_id"])
        if not provenances:
            raise ResearchRunFailure(
                "资料批次没有产生可进入 V2 验证的候选"
            )
        if not isinstance(summary, ResearchRunSummary):
            raise ResearchRunFailure("资料运行摘要缺失")

        current = self._research_hypothesis(run, summary)
        candidates = tuple(item.candidate for item in provenances)
        batch = ExperimentBatch(
            run_id=run["run_id"],
            hypothesis_id=current.hypothesis_id,
            turn_index=1,
            branch_id=current.branch_id,
            candidate_ids=tuple(item.candidate_id for item in candidates),
            parent_candidate_ids=(),
            development_start=context.windows.development_start,
            development_end=context.windows.development_end,
            input_hash=batch_input_hash(
                request_hash=(
                    summary.conversion_hash
                    or summary.input.manifest.manifest_hash
                ),
                candidates=candidates,
            ),
            parameters={
                "source_kind": "research_report",
                "conversion_id": summary.conversion_id,
                "conversion_hash": summary.conversion_hash,
                "manifest_hash": summary.input.manifest.manifest_hash,
            },
        )
        self.store.begin_turn(run["run_id"], current, batch)
        for candidate in candidates:
            self.store.record_candidate(
                run["run_id"],
                candidate_id=candidate.candidate_id,
                candidate_hash=candidate.content_hash,
                semantic_hash=candidate.semantic_hash,
                name=candidate.name,
                status="proposed",
                hypothesis_id=current.hypothesis_id,
                batch_id=batch.batch_id,
                payload=candidate_payload(candidate),
            )
        self._record_research_trials(
            run=run,
            hypothesis=current,
            batch=batch,
            summary=summary,
        )

        candidate_evidence = self._evaluate_turn_candidates(
            run=run,
            context=context,
            current=current,
            batch=batch,
            candidates=candidates,
        )
        existing_evidence = [
            item
            for item in trace["evidence"]
            if int(item["turn_index"]) == 1
        ]
        if existing_evidence:
            evidence = EvidencePacket.from_dict(existing_evidence[-1])
        else:
            control_state = self.store.search_control_state(run["run_id"])
            evidence = build_development_evidence(
                hypothesis=current,
                batch=batch,
                candidates=tuple(candidate_evidence),
                trial_counts=self.store.trial_counts(run["run_id"]),
                budget_state={
                    "turns_used": 1,
                    "branches_used": 1,
                    "refines_in_branch": 0,
                    "candidates_proposed": self.store.trial_counts(
                        run["run_id"]
                    ).proposed,
                    "candidates_evaluated": self.store.trial_counts(
                        run["run_id"]
                    ).evaluated,
                    "full_evaluated": self.store.trial_counts(
                        run["run_id"]
                    ).full_evaluated,
                    "no_improvement_turns": control_state[
                        "no_improvement_turns"
                    ],
                    "remaining_candidates": max(
                        0,
                        context.budget.max_candidates_per_run
                        - self.store.trial_counts(run["run_id"]).proposed,
                    ),
                    "cancelled": bool(control_state["cancel_requested"]),
                    "snapshot_stale": False,
                },
            )
            self.store.record_evidence(evidence)
        control = self.store.search_control_state(run["run_id"])
        result = decide_search_action(
            evidence,
            context.budget,
            context.control_policy,
            SearchControlState(
                branches_used=int(evidence.budget_state["branches_used"]),
                refines_in_branch=int(
                    evidence.budget_state["refines_in_branch"]
                ),
                no_improvement_turns=int(control["no_improvement_turns"]),
                previous_best=control["previous_best"],
                proposed_action=(
                    "promote"
                    if int(evidence.budget_state["remaining_candidates"]) == 0
                    else "refine"
                ),
                cancelled=bool(control["cancel_requested"]),
                snapshot_stale=False,
            ),
        )
        self.store.record_decision(result.decision)
        self.store.transition_turn(run["run_id"], 1, "completed")
        self.store.update_search_control_progress(
            run["run_id"],
            no_improvement_turns=result.no_improvement_turns,
            previous_best=(
                result.best_candidate.ordering[:4]
                if result.best_candidate is not None
                else None
            ),
        )
        self.store.append_event(
            run["run_id"],
            event_key="research_seed_control_decided",
            stage="searching",
            event_type="research_seed_decided",
            payload={
                "candidate_count": len(candidates),
                "action": result.decision.action,
                "reason": result.policy_reason,
            },
        )
        self.store.append_event(
            run["run_id"],
            event_key="research_seed_completed",
            stage="searching",
            event_type="research_seed_completed",
            payload={
                "candidate_count": len(candidates),
                "conversion_id": summary.conversion_id,
            },
        )
        if result.decision.action in {"stop", "promote"}:
            if control["cancel_requested"]:
                self.store.transition_search_run(
                    run["run_id"],
                    "cancelled",
                    event_key="run_cancelled",
                    stage="searching",
                    stop_reason="用户取消",
                )
            else:
                self.store.transition_search_run(
                    run["run_id"],
                    "selecting",
                    event_key="run_selecting",
                    stage="selecting",
                    payload={
                        "candidate_ids": list(result.quality_candidates),
                    },
                    stop_reason=result.policy_reason,
                    current_action=result.decision.action,
                )
            return True
        return False

    def _record_research_trials(
        self,
        *,
        run: Mapping[str, Any],
        hypothesis: HypothesisCard,
        batch: ExperimentBatch,
        summary: ResearchRunSummary,
    ) -> None:
        """为每个研报草稿幂等写入一次种子尝试。"""

        for decision in summary.decisions:
            if decision.admitted:
                kind = "proposed"
                reason = None
            elif decision.status == "accepted":
                kind = "rejected"
                reason = "admission_budget_exhausted"
            else:
                kind = {
                    "rejected": "rejected",
                    "invalid": "invalid",
                    "duplicate": "duplicate",
                }[decision.status]
                reason = decision.reason_code
            trial = trial_record(
                run_id=run["run_id"],
                trial_key=f"research-{decision.draft_id}",
                kind=kind,
                turn_index=1,
                hypothesis_id=hypothesis.hypothesis_id,
                batch_id=batch.batch_id,
                candidate_id=decision.candidate_id,
                reason_code=reason,
                payload={
                    "source_kind": "research_report",
                    "document_id": decision.document_id,
                    "draft_id": decision.draft_id,
                    "factor_hash": decision.factor_hash,
                    "screening_status": decision.status,
                    "admitted": decision.admitted,
                },
            )
            self.store.record_research_seed_trial(trial)

    @staticmethod
    def _research_hypothesis(
        run: Mapping[str, Any],
        summary: ResearchRunSummary,
    ) -> HypothesisCard:
        """构造资料候选批次的聚合假设，单候选假设保存在来源记录中。"""

        return HypothesisCard(
            run_id=run["run_id"],
            turn_index=1,
            branch_id="branch-0001",
            statement=(
                f"资料驱动研究：从 {len(summary.input.manifest.materials)} 份资料"
                f"准入 {summary.admitted_count} 个受控候选"
            ),
            mechanism="研报字段、公式和来源页码经过确定性筛查与 DSL 转换",
            expected_direction="non_monotonic",
            falsification_criteria=(
                "开发期质量未达到冻结门槛",
                "选择期或完整评估未能复核研究结论",
                "来源证据与候选定义不再一致",
            ),
            metadata={
                "source_kind": "research_report",
                "manifest_hash": summary.input.manifest.manifest_hash,
                "conversion_id": summary.conversion_id,
                "conversion_hash": summary.conversion_hash,
            },
        )
