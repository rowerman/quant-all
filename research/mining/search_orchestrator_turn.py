"""V2 逐轮搜索执行。"""

from __future__ import annotations

from typing import Any, Mapping

from mining.contract_candidates import CandidateDefinition
from mining.errors import MiningContractError, MiningEvaluationError
from mining.free_exploration import (
    direction_from_hypothesis,
    direction_to_hypothesis,
    generate_initial_direction,
)
from mining.free_exploration_contracts import ResearchDirection
from mining.free_exploration_pool import plan_next_branch_direction
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
from mining.search_worker_client_v3 import (
    generate_search_candidates_with_tools,
    resume_search_candidates_with_tools,
    wait_search_turn_response,
)
from mining.errors import WorkerResponseError
from mining.search_contract_types import FREE_EXPLORATION_INPUT_MODE
from mining.search_run import SearchRunRequest
from mining_worker.protocol_v2 import (
    WorkerSearchCandidate,
    WorkerSearchHypothesis,
    WorkerSearchRequest,
)


class SearchTurnMixin:
    """执行证据驱动的多轮搜索。"""

    def _search(self, run: Mapping[str, Any], context: Any) -> None:
        trace = self.store.get_search_trace(run["run_id"])
        current = self._current_hypothesis(run, trace, context=context)
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
                normalized = self._generate_search_candidates(
                    run=run,
                    context=context,
                    turn_index=turn_index,
                    worker_request=worker_request,
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
                    gateway = getattr(self, "analysis_tool_gateway", None)
                    if gateway is None:
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
                        tool_job = events.get(
                            f"turn_{turn_index}_tool_job_submitted"
                        )
                        if tool_job is not None:
                            response = wait_search_turn_response(
                                self._worker(),
                                str(tool_job["payload"]["job_id"]),
                                should_cancel=lambda: self.store.cancel_requested(
                                    run["run_id"]
                                ),
                            )
                            if response.kind != "decision" or response.response is None:
                                raise WorkerResponseError(
                                    "恢复的 Worker v3 finalize 作业未返回最终决策"
                                )
                            normalized = normalize_search_response(
                                worker_request,
                                response.response,
                                str(tool_job["payload"]["job_id"]),
                            )
                        else:
                            normalized = resume_search_candidates_with_tools(
                                self._worker(),
                                worker_request,
                                gateway,
                                context=context,
                                discover_job_id=str(
                                    job_event["payload"]["job_id"]
                                ),
                                should_cancel=lambda: self.store.cancel_requested(
                                    run["run_id"]
                                ),
                                on_submitted=lambda job, phase: (
                                    self.store.append_event(
                                        run["run_id"],
                                        event_key=(
                                            f"turn_{turn_index}_tool_job_submitted"
                                        ),
                                        stage="searching",
                                        event_type="search_worker_submitted",
                                        payload={
                                            "turn_index": turn_index,
                                            "job_id": job.job_id,
                                            "phase": phase,
                                        },
                                    )
                                ),
                            )
                else:
                    normalized = self._generate_search_candidates(
                        run=run,
                        context=context,
                        turn_index=turn_index,
                        worker_request=worker_request,
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

            next_branch = current.branch_id
            if result.decision.action == "branch":
                next_branch = f"branch-{branch_count + 1:04d}"
            turn_number = turn_index + 1
            request = self._run_request(run)
            if (
                result.decision.action == "branch"
                and request.input_mode == FREE_EXPLORATION_INPUT_MODE
            ):
                current = self._prepare_free_branch_hypothesis(
                    run=run,
                    context=context,
                    current=current,
                    decision_turn=turn_index,
                    next_turn=turn_number,
                    next_branch=next_branch,
                )
            else:
                if normalized.next_hypothesis is None:
                    raise MiningEvaluationError("控制器要求继续搜索但缺少下一假设")
                metadata = {"source": "worker"}
                if request.input_mode == FREE_EXPLORATION_INPUT_MODE:
                    direction = direction_from_hypothesis(current)
                    if direction is not None:
                        metadata = {
                            "source": "free_exploration_refine",
                            "direction_role": "refine",
                            "research_direction": direction.to_dict(),
                        }
                current = next_hypothesis(
                    run_id=run["run_id"],
                    turn_index=turn_number,
                    branch_id=next_branch,
                    statement=normalized.next_hypothesis.statement,
                    mechanism=normalized.next_hypothesis.mechanism,
                    expected_direction=(
                        normalized.next_hypothesis.expected_direction
                    ),
                    falsification_criteria=(
                        normalized.next_hypothesis.falsification_criteria
                    ),
                    parent_hypothesis_id=current.hypothesis_id,
                    metadata=metadata,
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

    def _generate_search_candidates(
        self,
        *,
        run: Mapping[str, Any],
        context: Any,
        turn_index: int,
        worker_request: WorkerSearchRequest,
    ):
        """按运行配置选择旧 Worker 或工具补充 Worker。"""

        gateway = getattr(self, "analysis_tool_gateway", None)
        if gateway is None:
            return generate_search_candidates(
                self._worker(),
                worker_request,
                should_cancel=lambda: self.store.cancel_requested(run["run_id"]),
                on_submitted=lambda job: self.store.append_event(
                    run["run_id"],
                    event_key=f"turn_{turn_index}_job_submitted",
                    stage="searching",
                    event_type="search_worker_submitted",
                    payload={"turn_index": turn_index, "job_id": job.job_id},
                ),
            )

        def on_submitted(job, phase: str) -> None:
            suffix = "job_submitted" if phase == "discover" else "tool_job_submitted"
            self.store.append_event(
                run["run_id"],
                event_key=f"turn_{turn_index}_{suffix}",
                stage="searching",
                event_type="search_worker_submitted",
                payload={
                    "turn_index": turn_index,
                    "job_id": job.job_id,
                    "phase": phase,
                },
            )

        return generate_search_candidates_with_tools(
            self._worker(),
            worker_request,
            gateway,
            context=context,
            should_cancel=lambda: self.store.cancel_requested(run["run_id"]),
            on_submitted=on_submitted,
        )

    def _current_hypothesis(
        self,
        run: Mapping[str, Any],
        trace: Mapping[str, Any],
        *,
        context: Any,
    ) -> HypothesisCard:
        turns = trace["turns"]
        if trace["hypotheses"]:
            current = HypothesisCard.from_dict(trace["hypotheses"][-1])
            decisions = trace["decisions"]
            if decisions and decisions[-1]["action"] == "branch":
                decision_turn = int(decisions[-1]["turn_index"])
                next_turn = decision_turn + 1
                request = self._run_request(run)
                if request.input_mode == FREE_EXPLORATION_INPUT_MODE:
                    next_hypothesis = next(
                        (
                            HypothesisCard.from_dict(item)
                            for item in trace["hypotheses"]
                            if int(item["turn_index"]) == next_turn
                        ),
                        None,
                    )
                    if next_hypothesis is not None:
                        if direction_from_hypothesis(next_hypothesis) is None:
                            return next_hypothesis
                        parent = next(
                            (
                                HypothesisCard.from_dict(item)
                                for item in trace["hypotheses"]
                                if item["hypothesis_id"]
                                == next_hypothesis.parent_hypothesis_id
                            ),
                            current,
                        )
                        return self._prepare_free_branch_hypothesis(
                            run=run,
                            context=context,
                            current=parent,
                            decision_turn=decision_turn,
                            next_turn=next_turn,
                            next_branch=next_hypothesis.branch_id,
                        )
                    branch_ids = {
                        str(item["branch_id"])
                        for item in trace["hypotheses"]
                    }
                    return self._prepare_free_branch_hypothesis(
                        run=run,
                        context=context,
                        current=current,
                        decision_turn=decision_turn,
                        next_turn=next_turn,
                        next_branch=f"branch-{len(branch_ids) + 1:04d}",
                    )
            return current
        if turns:
            raise MiningContractError("V2 turn 存在但缺少对应假设")
        return self._initial_search_hypothesis(run, context)

    @staticmethod
    def _run_request(run: Mapping[str, Any]) -> SearchRunRequest:
        """从持久化运行恢复搜索请求。"""

        return SearchRunRequest.from_dict(
            {
                "idempotency_key": run["idempotency_key"],
                **run["request"],
            }
        )

    def _prepare_free_branch_hypothesis(
        self,
        *,
        run: Mapping[str, Any],
        context: Any,
        current: HypothesisCard,
        decision_turn: int,
        next_turn: int,
        next_branch: str,
    ) -> HypothesisCard:
        """规划、记录并冻结一个自由探索分支方向。"""

        request = self._run_request(run)
        trace = self.store.get_search_trace(run["run_id"])
        prior_hypotheses = [
            item
            for item in trace["hypotheses"]
            if int(item["turn_index"]) < next_turn
        ]
        plan = plan_next_branch_direction(
            request,
            context,
            prior_hypotheses,
        )
        existing = next(
            (
                HypothesisCard.from_dict(item)
                for item in trace["hypotheses"]
                if int(item["turn_index"]) == next_turn
            ),
            None,
        )
        if existing is not None:
            existing_direction = direction_from_hypothesis(existing)
            if (
                existing.branch_id != next_branch
                or existing_direction is None
                or existing_direction.direction_hash
                != plan.direction.direction_hash
            ):
                raise MiningContractError(
                    "已保存的自由探索分支与重建计划不一致"
                )
        for duplicate in plan.duplicate_attempts:
            semantic_hash = duplicate.direction.semantic_hash
            self.store.record_trial(
                trial_record(
                    run_id=run["run_id"],
                    trial_key=(
                        f"direction-duplicate-{next_turn}-"
                        f"{semantic_hash[:24]}"
                    ),
                    kind="duplicate",
                    turn_index=decision_turn,
                    hypothesis_id=current.hypothesis_id,
                    batch_id=None,
                    candidate_id=None,
                    reason_code="semantic_duplicate",
                    payload={
                        "attempt_kind": "research_direction",
                        "target_turn": next_turn,
                        "semantic_hash": semantic_hash,
                        "duplicate_of_direction_id": (
                            duplicate.duplicate_of_direction_id
                        ),
                        "mechanism_category": (
                            duplicate.direction.mechanism_category
                        ),
                    },
                )
            )
        hypothesis = (
            existing
            if existing is not None
            else direction_to_hypothesis(
                plan.direction,
                turn_index=next_turn,
                branch_id=next_branch,
                parent_hypothesis_id=current.hypothesis_id,
                direction_role="branch",
            )
        )
        self.store.prepare_hypothesis(hypothesis)
        self.store.append_event(
            run["run_id"],
            event_key=f"turn_{decision_turn}_free_branch_direction_planned",
            stage="searching",
            event_type="free_exploration_direction_planned",
            payload={
                "direction": plan.direction.to_dict(),
                "direction_hash": plan.direction.direction_hash,
                "direction_id": plan.direction.direction_id,
                "semantic_hash": plan.direction.semantic_hash,
                "pool_hash": plan.pool_hash,
                "plan_hash": plan.plan_hash,
                "duplicate_attempts": [
                    {
                        "semantic_hash": item.direction.semantic_hash,
                        "duplicate_of_direction_id": (
                            item.duplicate_of_direction_id
                        ),
                    }
                    for item in plan.duplicate_attempts
                ],
                "target_turn": next_turn,
                "branch_id": next_branch,
            },
        )
        return hypothesis

    def _initial_search_hypothesis(
        self,
        run: Mapping[str, Any],
        context: Any,
    ) -> HypothesisCard:
        """返回手工假设或已冻结的自由探索首轮假设。"""

        request = self._run_request(run)
        if request.input_mode != FREE_EXPLORATION_INPUT_MODE:
            return initial_hypothesis(run)
        event = next(
            (
                item
                for item in reversed(self.store.list_events(run["run_id"]))
                if item["event_type"] == "free_exploration_direction_planned"
            ),
            None,
        )
        if event is not None:
            raw = event["payload"].get("direction")
            direction = ResearchDirection.from_dict(raw)
            if (
                direction.run_id != run["run_id"]
                or direction.request_hash != request.request_hash
                or direction.context_hash != context.context_hash
            ):
                raise MiningContractError("已保存的自由探索方向与当前运行不一致")
            return direction_to_hypothesis(direction)

        direction = generate_initial_direction(request, context)
        self.store.append_event(
            run["run_id"],
            event_key="free_exploration_direction_planned",
            stage="searching",
            event_type="free_exploration_direction_planned",
            payload={
                "direction": direction.to_dict(),
                "direction_hash": direction.direction_hash,
                "direction_id": direction.direction_id,
            },
        )
        return direction_to_hypothesis(direction)

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
        research_direction = direction_from_hypothesis(current)
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
            max_expression_depth=(
                research_direction.max_expression_depth
                if research_direction is not None
                else 6
            ),
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
