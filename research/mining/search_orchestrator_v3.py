"""V3 四类研究来源的统一运行编排。"""

from __future__ import annotations

from typing import Any, Mapping

from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_control import SearchControlState, decide_search_action
from mining.search_evidence_builder import build_development_evidence
from mining.search_orchestrator import MiningSearchOrchestrator
from mining.search_orchestrator_support import (
    batch_input_hash,
    trial_record,
)
from mining.search_run import SearchRunRequest
from mining.storage_v3_runtime import MiningV3RuntimeStore


class MiningV3SearchOrchestrator(MiningSearchOrchestrator):
    """在 V3 runtime 存储上执行统一搜索和阶段评估。"""

    def __init__(self, *, store: MiningV3RuntimeStore, **kwargs: Any) -> None:
        super().__init__(store=store, **kwargs)
        self.store = store

    def create_run(
        self,
        request: SearchRunRequest,
        context: Any,
        *,
        background: bool = True,
        research_input: Any | None = None,
    ) -> dict[str, Any]:
        """CoSTEER 导入不依赖 Worker，其他来源仍检查 Worker。"""

        if self.store.seed_candidate(request.run_id) is None:
            health = self._worker().health()
            if health.get("status") != "ready":
                from mining.errors import MiningStateError

                raise MiningStateError("Worker v2 尚未就绪")
        run = self.store.create_run(
            request,
            context,
            research_input=research_input,
        )
        if background and run["status"] == "queued":
            self._schedule(run["run_id"])
        return run

    def _prepare_research_run(self, research_input: Any, context: Any) -> Any:
        """资料候选保持来源无关身份，来源由 V3 包络表达。"""

        if self._research_preparer is not None:
            return self._research_preparer(
                research_input,
                material_root=self.research_material_root,
                context=context,
                admission_limit=context.budget.max_candidates_per_run,
            )
        from mining.research_factor_conversion_client import (
            HttpResearchFactorConverter,
        )
        from mining.research_material_extractor import HttpResearchFactorExtractor
        from mining.research_run_pipeline import prepare_research_run

        if self._research_extractor_value is None:
            self._research_extractor_value = (
                HttpResearchFactorExtractor.from_environment()
            )
        if self._research_converter_value is None:
            self._research_converter_value = (
                HttpResearchFactorConverter.from_environment()
            )
        return prepare_research_run(
            research_input,
            material_root=self.research_material_root,
            context=context,
            extractor=self._research_extractor_value,
            converter=self._research_converter_value,
            admission_limit=context.budget.max_candidates_per_run,
            enrich_candidates=False,
        )

    def _search(self, run: Mapping[str, Any], context: Any) -> None:
        """CoSTEER 直接导入候选，其余来源沿用 Worker 搜索。"""

        request = SearchRunRequest.from_dict(
            {
                "idempotency_key": run["idempotency_key"],
                **run["request"],
            }
        )
        input_envelope = self.store.load_input_envelope(run["run_id"])
        if input_envelope.input_mode != "costeer":
            return super()._search(run, context)
        self._seed_costeer_candidate(run, context, request)

    def resume(self, run_id: str, *, background: bool = True) -> dict[str, Any]:
        """恢复前重验全部持久化事实，再回到原中断阶段。"""

        self.store.recover_v3_run(run_id)
        return super().resume(run_id, background=background)

    def _seed_costeer_candidate(
        self,
        run: Mapping[str, Any],
        context: Any,
        request: SearchRunRequest,
    ) -> None:
        candidate = self.store.seed_candidate(run["run_id"])
        if candidate is None:
            raise RuntimeError("CoSTEER 运行缺少人工转写候选")
        if any(
            event["event_key"] == "costeer_seed_completed"
            for event in self.store.list_events(run["run_id"])
        ):
            return
        current = HypothesisCard(
            run_id=run["run_id"],
            turn_index=1,
            branch_id="branch-costeer-0001",
            statement=candidate.description,
            mechanism=candidate.mechanism,
            expected_direction=(
                "positive" if candidate.direction > 0 else "negative"
            ),
            falsification_criteria=(
                "任一冻结阶段评估未通过",
                "人工审批证据与候选包络不一致",
            ),
            metadata={
                "source": "costeer",
                "promotion_id": request.idempotency_key,
            },
        )
        batch = ExperimentBatch(
            run_id=run["run_id"],
            hypothesis_id=current.hypothesis_id,
            turn_index=1,
            branch_id=current.branch_id,
            candidate_ids=(candidate.candidate_id,),
            parent_candidate_ids=(),
            development_start=context.windows.development_start,
            development_end=context.windows.development_end,
            input_hash=batch_input_hash(
                request_hash=request.request_hash,
                candidates=(candidate,),
            ),
            parameters={"source_kind": "costeer"},
        )
        self.store.begin_turn(run["run_id"], current, batch)
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
                trial_key=f"costeer-proposed-{candidate.candidate_id}",
                kind="proposed",
                turn_index=1,
                hypothesis_id=current.hypothesis_id,
                batch_id=batch.batch_id,
                candidate_id=candidate.candidate_id,
                payload={"source_kind": "costeer"},
            )
        )
        summaries = self._evaluate_turn_candidates(
            run=run,
            context=context,
            current=current,
            batch=batch,
            candidates=(candidate,),
        )
        control = self.store.search_control_state(run["run_id"])
        counts = self.store.trial_counts(run["run_id"])
        evidence = build_development_evidence(
            hypothesis=current,
            batch=batch,
            candidates=tuple(summaries),
            trial_counts=counts,
            budget_state={
                "turns_used": 1,
                "branches_used": 1,
                "refines_in_branch": 0,
                "candidates_proposed": counts.proposed,
                "candidates_evaluated": counts.evaluated,
                "full_evaluated": counts.full_evaluated,
                "no_improvement_turns": control["no_improvement_turns"],
                "remaining_candidates": 0,
                "cancelled": bool(control["cancel_requested"]),
                "snapshot_stale": False,
            },
        )
        self.store.record_evidence(evidence)
        result = decide_search_action(
            evidence,
            context.budget,
            context.control_policy,
            SearchControlState(
                branches_used=1,
                refines_in_branch=0,
                no_improvement_turns=int(control["no_improvement_turns"]),
                previous_best=control["previous_best"],
                proposed_action="promote",
                cancelled=bool(control["cancel_requested"]),
                snapshot_stale=False,
            ),
        )
        self.store.record_decision(result.decision)
        self.store.transition_turn(run["run_id"], 1, "completed")
        self.store.append_event(
            run["run_id"],
            event_key="costeer_seed_completed",
            stage="searching",
            event_type="costeer_seed_completed",
            payload={
                "candidate_id": candidate.candidate_id,
                "promotion_id": request.idempotency_key,
                "action": result.decision.action,
            },
        )
        if result.quality_candidates:
            self.store.transition_search_run(
                run["run_id"],
                "selecting",
                event_key="run_selecting",
                stage="selecting",
                payload={"candidate_ids": list(result.quality_candidates)},
                current_action="promote",
            )
        else:
            self.store.transition_search_run(
                run["run_id"],
                "cancelled" if control["cancel_requested"] else "failed",
                event_key="run_costeer_seed_failed",
                stage="searching",
                stop_reason="CoSTEER 导入候选未通过开发期评估",
            )


__all__ = ["MiningV3SearchOrchestrator"]
