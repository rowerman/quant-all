"""V2 逐轮搜索执行。"""

from __future__ import annotations

from typing import Any, Mapping

from mining.contract_candidates import CandidateDefinition
from mining.errors import MiningEvaluationError
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_control import SearchControlState, decide_search_action
from mining.search_evidence import EvidencePacket
from mining.search_evidence_builder import build_development_evidence
from mining.search_orchestrator_support import (
    batch_input_hash,
    initial_hypothesis,
    merge_evidence,
    next_hypothesis,
    trial_record,
)
from mining.search_worker_client import (
    generate_search_candidates,
    normalize_search_response,
    wait_for_search_response,
)
from mining_worker.protocol_v2 import (
    WorkerSearchCandidate,
    WorkerSearchHypothesis,
    WorkerSearchRequest,
)


class SearchTurnMixin:
    """执行证据驱动的多轮搜索。"""

    def _search(self, run: Mapping[str, Any], context: Any) -> None:
        trace = self.store.get_search_trace(run["run_id"])
        current = self._current_hypothesis(run, trace)
        while True:
            trace = self.store.get_search_trace(run["run_id"])
            turns = list(trace["turns"])
            active_turn = (
                turns[-1]
                if turns and turns[-1]["status"] != "completed"
                else None
            )
            turn_index = (
                int(active_turn["turn_index"])
                if active_turn is not None
                else len(turns) + 1
            )
            if turn_index > context.budget.max_search_turns:
                raise MiningEvaluationError("搜索 turn 预算已耗尽但尚未收口")
            previous_evidence_rows = [
                item
                for item in trace["evidence"]
                if int(item["turn_index"]) < turn_index
            ]
            previous_evidence = (
                EvidencePacket.from_dict(previous_evidence_rows[-1])
                if previous_evidence_rows
                else None
            )
            if active_turn is not None:
                current = HypothesisCard.from_dict(active_turn["hypothesis"])
                batch = ExperimentBatch.from_dict(
                    active_turn["experiment_batch"]
                )
            worker_request = self._worker_request(
                run=run,
                context=context,
                current=current,
                turn_index=turn_index,
                previous_evidence=previous_evidence,
                trace=trace,
                active_batch_id=(
                    batch.batch_id if active_turn is not None else None
                ),
            )
            events = {
                item["event_key"]: item
                for item in self.store.list_events(run["run_id"])
            }
            if active_turn is None:
                self.store.append_event(
                    run["run_id"],
                    event_key=f"turn_{turn_index}_worker_requested",
                    stage="searching",
                    event_type="search_turn_requested",
                    payload={
                        "turn_index": turn_index,
                        "branch_id": current.branch_id,
                        "request_hash": worker_request.payload_hash,
                        "max_candidates": worker_request.max_candidates,
                    },
                )
                normalized = generate_search_candidates(
                    self._worker(),
                    worker_request,
                    should_cancel=lambda: self.store.cancel_requested(
                        run["run_id"]
                    ),
                    on_submitted=lambda job: self.store.append_event(
                        run["run_id"],
                        event_key=f"turn_{turn_index}_job_submitted",
                        stage="searching",
                        event_type="search_worker_submitted",
                        payload={"turn_index": turn_index, "job_id": job.job_id},
                    ),
                )
                parent_ids = self._parent_candidate_ids(
                    trace=trace,
                    current_action=(
                        trace["decisions"][-1]["action"]
                        if trace["decisions"]
                        else None
                    ),
                )
                batch = ExperimentBatch(
                    run_id=run["run_id"],
                    hypothesis_id=current.hypothesis_id,
                    turn_index=turn_index,
                    branch_id=current.branch_id,
                    candidate_ids=tuple(
                        item.candidate_id for item in normalized.accepted
                    ),
                    parent_candidate_ids=parent_ids,
                    development_start=context.windows.development_start,
                    development_end=context.windows.development_end,
                    input_hash=batch_input_hash(
                        request_hash=worker_request.payload_hash,
                        candidates=normalized.accepted,
                    ),
                    parameters={
                        "worker_job_id": normalized.job_id,
                        "rejected_count": len(normalized.rejected),
                    },
                )
                self.store.begin_turn(run["run_id"], current, batch)
            else:
                job_event = events.get(f"turn_{turn_index}_job_submitted")
                if job_event is not None:
                    response = wait_for_search_response(
                        self._worker(),
                        str(job_event["payload"]["job_id"]),
                        should_cancel=lambda: self.store.cancel_requested(
                            run["run_id"]
                        ),
                    )
                    normalized = normalize_search_response(
                        worker_request,
                        response,
                        str(job_event["payload"]["job_id"]),
                    )
                else:
                    normalized = generate_search_candidates(
                        self._worker(),
                        worker_request,
                        should_cancel=lambda: self.store.cancel_requested(
                            run["run_id"]
                        ),
                        on_submitted=lambda job: self.store.append_event(
                            run["run_id"],
                            event_key=f"turn_{turn_index}_job_submitted",
                            stage="searching",
                            event_type="search_worker_submitted",
                            payload={
                                "turn_index": turn_index,
                                "job_id": job.job_id,
                            },
                        ),
                    )
            for candidate in normalized.accepted:
                self.store.record_candidate(
                    run["run_id"],
                    candidate_id=candidate.candidate_id,
                    candidate_hash=candidate.content_hash,
                    semantic_hash=candidate.semantic_hash,
                    name=candidate.name,
                    status="proposed",
                    hypothesis_id=current.hypothesis_id,
                    batch_id=batch.batch_id,
                    payload=self._candidate_payload(candidate),
                )
                self.store.record_trial(
                    trial_record(
                        run_id=run["run_id"],
                        trial_key=f"turn-{turn_index}-proposed-{candidate.candidate_id}",
                        kind="proposed",
                        turn_index=turn_index,
                        hypothesis_id=current.hypothesis_id,
                        batch_id=batch.batch_id,
                        candidate_id=candidate.candidate_id,
                    )
                )
            for rejection in normalized.rejected:
                self.store.record_trial(
                    trial_record(
                        run_id=run["run_id"],
                        trial_key=f"turn-{turn_index}-invalid-{rejection.index}",
                        kind="invalid",
                        turn_index=turn_index,
                        hypothesis_id=current.hypothesis_id,
                        batch_id=batch.batch_id,
                        candidate_id=None,
                        reason_code=rejection.error_code,
                        payload={"index": rejection.index},
                    )
                )

            existing_evidence_rows = [
                item
                for item in trace["evidence"]
                if int(item["turn_index"]) == turn_index
            ]
            if active_turn is not None and active_turn["status"] in {
                "evidence_ready",
                "decided",
            }:
                evidence = EvidencePacket.from_dict(
                    existing_evidence_rows[-1]
                )
            else:
                candidate_evidence = self._evaluate_turn_candidates(
                    run=run,
                    context=context,
                    current=current,
                    batch=batch,
                    candidates=normalized.accepted,
                )
            control = self.store.search_control_state(run["run_id"])
            if not (
                active_turn is not None
                and active_turn["status"] in {"evidence_ready", "decided"}
            ):
                prior_hypotheses = [
                    item
                    for item in trace["hypotheses"]
                    if int(item["turn_index"]) < turn_index
                ]
                evidence = build_development_evidence(
                    hypothesis=current,
                    batch=batch,
                    candidates=tuple(candidate_evidence),
                    trial_counts=self.store.trial_counts(run["run_id"]),
                    budget_state={
                        "turns_used": turn_index,
                        "branches_used": len(
                            {
                                item["branch_id"]
                                for item in prior_hypotheses
                            }
                            | {current.branch_id}
                        ),
                        "refines_in_branch": max(
                            0,
                            sum(
                                item["branch_id"] == current.branch_id
                                for item in prior_hypotheses
                            ),
                        ),
                        "candidates_proposed": self.store.trial_counts(
                            run["run_id"]
                        ).proposed,
                        "candidates_evaluated": self.store.trial_counts(
                            run["run_id"]
                        ).evaluated,
                        "full_evaluated": self.store.trial_counts(
                            run["run_id"]
                        ).full_evaluated,
                        "no_improvement_turns": control["no_improvement_turns"],
                        "remaining_candidates": max(
                            0,
                            context.budget.max_candidates_per_run
                            - self.store.trial_counts(run["run_id"]).proposed,
                        ),
                        "cancelled": bool(control["cancel_requested"]),
                        "snapshot_stale": False,
                    },
                )
                evidence = merge_evidence(previous_evidence, evidence)
                self.store.record_evidence(evidence)
            branch_count = int(evidence.budget_state["branches_used"])
            refine_count = int(evidence.budget_state["refines_in_branch"])
            result = decide_search_action(
                evidence,
                context.budget,
                context.control_policy,
                SearchControlState(
                    branches_used=branch_count,
                    refines_in_branch=refine_count,
                    no_improvement_turns=int(control["no_improvement_turns"]),
                    previous_best=control["previous_best"],
                    proposed_action=normalized.decision.action,
                    cancelled=bool(control["cancel_requested"]),
                    snapshot_stale=False,
                ),
            )
            self.store.record_decision(result.decision)
            if active_turn is None or active_turn["status"] != "completed":
                self.store.transition_turn(
                    run["run_id"],
                    turn_index,
                    "completed",
                )
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
                event_key=f"turn_{turn_index}_control_decided",
                stage="searching",
                event_type="search_control_decided",
                payload=result.to_dict(),
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
                    return
                self.store.transition_search_run(
                    run["run_id"],
                    "selecting",
                    event_key="run_selecting",
                    stage="selecting",
                    payload={"candidate_ids": list(result.quality_candidates)},
                    stop_reason=result.policy_reason,
                    current_action=result.decision.action,
                )
                return

            if normalized.next_hypothesis is None:
                raise MiningEvaluationError("控制器要求继续搜索但缺少下一假设")
            next_branch = current.branch_id
            if result.decision.action == "branch":
                next_branch = f"branch-{branch_count + 1:04d}"
            turn_number = turn_index + 1
            current = next_hypothesis(
                run_id=run["run_id"],
                turn_index=turn_number,
                branch_id=next_branch,
                statement=normalized.next_hypothesis.statement,
                mechanism=normalized.next_hypothesis.mechanism,
                expected_direction=normalized.next_hypothesis.expected_direction,
                falsification_criteria=(
                    normalized.next_hypothesis.falsification_criteria
                ),
                parent_hypothesis_id=current.hypothesis_id,
            )
            self.store.prepare_hypothesis(current)
            self.store.append_event(
                run["run_id"],
                event_key=f"turn_{turn_index}_next_hypothesis",
                stage="searching",
                event_type="next_hypothesis_prepared",
                payload={
                    "next_turn": turn_number,
                    "branch_id": next_branch,
                    "hypothesis_hash": current.card_hash,
                },
            )
            trace = self.store.get_search_trace(run["run_id"])

    def _current_hypothesis(
        self,
        run: Mapping[str, Any],
        trace: Mapping[str, Any],
    ) -> HypothesisCard:
        turns = trace["turns"]
        if not turns:
            return initial_hypothesis(run)
        if trace["hypotheses"]:
            return HypothesisCard.from_dict(trace["hypotheses"][-1])
        return initial_hypothesis(run)

    def _worker_request(
        self,
        *,
        run: Mapping[str, Any],
        context: Any,
        current: HypothesisCard,
        turn_index: int,
        previous_evidence: EvidencePacket | None,
        trace: Mapping[str, Any],
        active_batch_id: str | None = None,
    ) -> WorkerSearchRequest:
        lineage = tuple(
            WorkerSearchCandidate(
                candidate_id=item["candidate_id"],
                name=item["name"],
                expression=item["payload"]["canonical_expression"],
                status=item["status"],
                parent_candidate_id=item["parent_candidate_id"],
            )
            for item in trace["candidates"][-24:]
            if item["batch_id"] != active_batch_id
        )
        remaining = max(
            1,
            context.budget.max_candidates_per_run
            - self.store.trial_counts(run["run_id"]).proposed,
        )
        return WorkerSearchRequest(
            request_id=f"{run['run_id']}:search:{turn_index}",
            run_id=run["run_id"],
            turn_index=turn_index,
            branch_id=current.branch_id,
            context_hash=context.context_hash,
            hypothesis=WorkerSearchHypothesis(
                statement=current.statement,
                mechanism=current.mechanism,
                expected_direction=current.expected_direction,
                falsification_criteria=current.falsification_criteria,
                parent_hypothesis_id=current.parent_hypothesis_id,
            ),
            lineage=lineage,
            evidence=(
                previous_evidence.to_dict()
                if previous_evidence is not None
                else None
            ),
            budget=context.budget.to_dict(),
            allowed_actions=("refine", "branch", "stop", "promote"),
            llm_model=context.llm_model,
            max_candidates=min(
                context.budget.max_candidates_per_turn,
                remaining,
            ),
            max_expression_depth=6,
            excluded_fields=(),
            fields=context.field_catalog,
            operators=context.operator_catalog,
        )

    @staticmethod
    def _parent_candidate_ids(
        *,
        trace: Mapping[str, Any],
        current_action: str | None,
    ) -> tuple[str, ...]:
        if current_action == "branch" or not trace["experiment_batches"]:
            return ()
        batch = trace["experiment_batches"][-1]
        return tuple(batch["candidate_ids"][:4])

    @staticmethod
    def _candidate_payload(candidate: CandidateDefinition) -> dict[str, Any]:
        from mining.search_orchestrator_support import candidate_payload

        return candidate_payload(candidate)
