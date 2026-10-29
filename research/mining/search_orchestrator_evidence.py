"""V2 turn 的开发期评估和恢复摘要。"""

from __future__ import annotations

from typing import Any, Mapping

from mining.candidate_batch import CandidateBatchItem
from mining.contract_candidates import CandidateDefinition
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_evaluation_adapter import (
    development_context,
)
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_evidence_builder import DevelopmentCandidateEvidence
from mining.search_orchestrator_support import (
    evaluation_evidence,
    evaluation_report_mapping,
    trial_record,
)


class SearchEvidenceMixin:
    """执行开发期评估，并从已登记证据恢复未完成 turn。"""

    def _evaluate_turn_candidates(
        self,
        *,
        run: Mapping[str, Any],
        context: Any,
        current: HypothesisCard,
        batch: ExperimentBatch,
        candidates: tuple[CandidateDefinition, ...],
    ) -> list[DevelopmentCandidateEvidence]:
        if not candidates:
            return []
        registered = self.store.list_evaluation_evidence(run["run_id"])
        by_candidate: dict[str, dict[str, SearchEvaluationEvidence]] = {}
        for row in registered:
            payload = row["evidence"]
            if payload["stage"] not in {"technical", "development"}:
                continue
            by_candidate.setdefault(payload["candidate_id"], {})[
                payload["stage"]
            ] = SearchEvaluationEvidence.from_dict(payload)
        result_items: list[DevelopmentCandidateEvidence] = []
        pending: list[CandidateDefinition] = []
        for candidate in candidates:
            status = self.store.get_candidate(
                run["run_id"],
                candidate.candidate_id,
            )["status"]
            if status in {"screening", "technical_failed", "screened_out"}:
                result_items.append(
                    self._registered_candidate_summary(
                        candidate.candidate_id,
                        by_candidate.get(candidate.candidate_id, {}),
                    )
                )
                continue
            existing = by_candidate.get(candidate.candidate_id, {})
            technical = existing.get("technical")
            development = existing.get("development")
            if technical is not None and (
                technical.status != "passed" or development is not None
            ):
                summary = self._registered_candidate_summary(
                    candidate.candidate_id,
                    existing,
                )
                result_items.append(summary)
                self._complete_candidate_from_summary(
                    run=run,
                    current=current,
                    batch=batch,
                    candidate_id=candidate.candidate_id,
                    summary=summary,
                )
                continue
            pending.append(candidate)
        if not pending:
            return result_items
        materialization = self._materializer(
            tuple(
                CandidateBatchItem(
                    candidate=candidate,
                    candidate_root=self.store.candidate_dir(
                        run["run_id"],
                        candidate.candidate_id,
                    )
                    / "development",
                )
                for candidate in pending
            ),
            development_context(context),
            start=context.windows.development_start,
            end=context.windows.development_end,
        )
        failures = {
            item.candidate_id: item for item in materialization.failures
        }
        for candidate in pending:
            self.store.transition_candidate(
                run["run_id"],
                candidate.candidate_id,
                "validating",
            )
            failure = failures.get(candidate.candidate_id)
            if failure is not None:
                result_items.append(
                    DevelopmentCandidateEvidence(
                        candidate_id=candidate.candidate_id,
                        technical_status="failed",
                        quick_status="not_run",
                        gate_passed=False,
                        failure_stage="technical",
                        error_code=failure.error_code,
                        error_message=failure.message,
                    )
                )
                self.store.transition_candidate(
                    run["run_id"],
                    candidate.candidate_id,
                    "technical_failed",
                    trial=trial_record(
                        run_id=run["run_id"],
                        trial_key=(
                            f"turn-{current.turn_index}-gate-failed-"
                            f"{candidate.candidate_id}"
                        ),
                        kind="gate_failed",
                        turn_index=current.turn_index,
                        hypothesis_id=current.hypothesis_id,
                        batch_id=batch.batch_id,
                        candidate_id=candidate.candidate_id,
                        reason_code=failure.error_code,
                    ),
                )
                continue
            evaluation = self._development_evaluator(
                run_id=run["run_id"],
                candidate=candidate,
                context=context,
                stage_root=self.store.candidate_dir(
                    run["run_id"],
                    candidate.candidate_id,
                )
                / "development",
            )
            self.store.record_evaluation_evidence(
                evaluation_evidence(evaluation.technical)
            )
            if evaluation.quick is not None:
                self.store.record_evaluation_evidence(
                    evaluation_evidence(evaluation.quick)
                )
            summary = DevelopmentCandidateEvidence.from_reports(
                candidate.candidate_id,
                gate={"passed": evaluation.gate.passed},
                technical=evaluation_report_mapping(evaluation.technical),
                quick=(
                    evaluation_report_mapping(evaluation.quick)
                    if evaluation.quick is not None
                    else None
                ),
            )
            result_items.append(summary)
            if not evaluation.gate.passed or evaluation.technical.status != "passed":
                self.store.transition_candidate(
                    run["run_id"],
                    candidate.candidate_id,
                    "technical_failed",
                    trial=trial_record(
                        run_id=run["run_id"],
                        trial_key=(
                            f"turn-{current.turn_index}-gate-failed-"
                            f"{candidate.candidate_id}"
                        ),
                        kind="gate_failed",
                        turn_index=current.turn_index,
                        hypothesis_id=current.hypothesis_id,
                        batch_id=batch.batch_id,
                        candidate_id=candidate.candidate_id,
                        reason_code=evaluation.technical.error_code or "MINING-106",
                    ),
                )
            elif evaluation.quick is None or evaluation.quick.status != "passed":
                self.store.transition_candidate(
                    run["run_id"],
                    candidate.candidate_id,
                    "screened_out",
                    trial=trial_record(
                        run_id=run["run_id"],
                        trial_key=(
                            f"turn-{current.turn_index}-rejected-"
                            f"{candidate.candidate_id}"
                        ),
                        kind="rejected",
                        turn_index=current.turn_index,
                        hypothesis_id=current.hypothesis_id,
                        batch_id=batch.batch_id,
                        candidate_id=candidate.candidate_id,
                        reason_code="MINING-106",
                    ),
                )
            else:
                self.store.transition_candidate(
                    run["run_id"],
                    candidate.candidate_id,
                    "screening",
                    trial=trial_record(
                        run_id=run["run_id"],
                        trial_key=(
                            f"turn-{current.turn_index}-evaluated-"
                            f"{candidate.candidate_id}"
                        ),
                        kind="evaluated",
                        turn_index=current.turn_index,
                        hypothesis_id=current.hypothesis_id,
                        batch_id=batch.batch_id,
                        candidate_id=candidate.candidate_id,
                    ),
                )
        return result_items

    @staticmethod
    def _registered_candidate_summary(
        candidate_id: str,
        rows: Mapping[str, SearchEvaluationEvidence],
    ) -> DevelopmentCandidateEvidence:
        technical = rows.get("technical")
        development = rows.get("development")
        values = (
            (technical.metrics.get("values") or {})
            if technical is not None
            else {}
        )
        metrics = development.metrics if development is not None else {}
        if technical is None:
            return DevelopmentCandidateEvidence(
                candidate_id=candidate_id,
                technical_status="failed",
                quick_status="not_run",
                gate_passed=False,
                failure_stage="technical",
                error_code="MINING-106",
                error_message="恢复时缺少技术门禁证据",
            )
        failure_stage = None
        if technical.status != "passed":
            failure_stage = "technical"
        elif development is None or development.status != "passed":
            failure_stage = "quick"
        return DevelopmentCandidateEvidence(
            candidate_id=candidate_id,
            technical_status=technical.status,
            quick_status=(
                development.status if development is not None else "not_run"
            ),
            gate_passed=technical.status == "passed",
            rank_ic=metrics.get("rank_ic_mean"),
            rank_ic_ir=metrics.get("rank_ic_ir"),
            year_sign_consistency=metrics.get("year_sign_consistency"),
            coverage=values.get("coverage"),
            valid_days=metrics.get("valid_days"),
            sample_count=metrics.get("sample_count"),
            failure_stage=failure_stage,
            error_code=(
                (development or technical).error_code
                if failure_stage
                else None
            ),
            error_message=(
                (development or technical).error_message
                if failure_stage
                else None
            ),
        )

    def _complete_candidate_from_summary(
        self,
        *,
        run: Mapping[str, Any],
        current: HypothesisCard,
        batch: ExperimentBatch,
        candidate_id: str,
        summary: DevelopmentCandidateEvidence,
    ) -> None:
        status = self.store.get_candidate(
            run["run_id"],
            candidate_id,
        )["status"]
        if status == "proposed":
            self.store.transition_candidate(
                run["run_id"],
                candidate_id,
                "validating",
            )
        if summary.failure_stage == "technical":
            target = "technical_failed"
            kind = "gate_failed"
            suffix = "gate-failed"
        elif summary.failure_stage == "quick":
            target = "screened_out"
            kind = "rejected"
            suffix = "rejected"
        else:
            target = "screening"
            kind = "evaluated"
            suffix = "evaluated"
        self.store.transition_candidate(
            run["run_id"],
            candidate_id,
            target,
            trial=trial_record(
                run_id=run["run_id"],
                trial_key=(
                    f"turn-{current.turn_index}-{suffix}-{candidate_id}"
                ),
                kind=kind,
                turn_index=current.turn_index,
                hypothesis_id=current.hypothesis_id,
                batch_id=batch.batch_id,
                candidate_id=candidate_id,
                reason_code=(
                    summary.error_code
                    if target != "screening"
                    else None
                ),
            ),
        )
