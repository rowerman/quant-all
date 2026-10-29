"""V2 选择期、最终研究和模块测试段执行。"""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Any, Mapping

from calculate import open_source
from mining.contract_candidates import CandidateDefinition
from mining.evaluation_contract import load_quality_policy
from mining.search_evaluation_adapter import (
    evaluate_final,
    evaluate_holdout,
    evaluate_selection,
)
from mining.search_portfolio import select_portfolio_finalists
from mining.search_selection_diagnostics import (
    SELECTION_DIAGNOSTICS_STAGE,
    CandidateCorrelationReport,
    IncrementalValueReport,
    build_selection_diagnostics,
    compute_candidate_correlations,
    compute_incremental_value,
    load_candidate_values,
    sample_trade_dates,
)
from mining.search_orchestrator_stability import SearchStabilityMixin
from mining.search_orchestrator_support import (
    candidate_from_v2_record,
    evaluation_evidence,
    trial_record,
)
from mining.search_scorecard import build_run_scorecards


class SearchPipelineMixin(SearchStabilityMixin):
    """把开发期入围候选沿隔离窗口推进到审核就绪。"""

    def _select(self, run: Mapping[str, Any], context: Any) -> None:
        trace = self.store.get_search_trace(run["run_id"])
        candidates = [
            item for item in trace["candidates"] if item["status"] == "screening"
        ]
        rejected: set[str] = set()
        selected_results: dict[
            str, tuple[Mapping[str, Any], CandidateDefinition, Any]
        ] = {}
        for item in candidates:
            candidate = candidate_from_v2_record(item)
            result = self._selection_evaluator(
                run_id=run["run_id"],
                candidate=candidate,
                context=context,
                stage_root=self.store.candidate_dir(
                    run["run_id"],
                    candidate.candidate_id,
                )
                / "selection",
            )
            technical = evaluation_evidence(
                result.technical,
                stage="selection",
            )
            self.store.record_evaluation_evidence(technical)
            if result.quick is None or result.quick.status != "passed":
                self._reject_candidate(
                    run=run,
                    trace=trace,
                    candidate_id=candidate.candidate_id,
                    stage="selection",
                    trial_key=f"selection-rejected-{candidate.candidate_id}",
                )
                rejected.add(candidate.candidate_id)
                continue
            quick = evaluation_evidence(result.quick, stage="selection")
            self.store.record_evaluation_evidence(quick)
            selected_results[candidate.candidate_id] = (
                item,
                candidate,
                result,
            )
        self._record_stability_reports(
            run=run,
            context=context,
            selected_results=selected_results,
        )
        self._record_selection_diagnostics(
            run=run,
            context=context,
            selected_results=selected_results,
        )
        scorecards = tuple(
            card
            for card in build_run_scorecards(self.store, run["run_id"])
            if card.candidate_id in selected_results
        )
        portfolio = self._portfolio_selector(
            scorecards,
            context.control_policy,
            max_finalists=context.budget.max_full_evaluations,
        )
        finalists = list(portfolio.finalists)
        finalist_set = set(finalists)
        for item in candidates:
            if item["candidate_id"] in finalist_set:
                self.store.transition_candidate(
                    run["run_id"],
                    item["candidate_id"],
                    "evaluating",
                )
            elif (
                item["status"] == "screening"
                and item["candidate_id"] not in rejected
            ):
                self._reject_candidate(
                    run=run,
                    trace=trace,
                    candidate_id=item["candidate_id"],
                    stage="selection",
                    trial_key=f"selection-not-selected-{item['candidate_id']}",
                )
        self.store.append_event(
            run["run_id"],
            event_key="finalists_selected",
            stage="selecting",
            event_type="finalists_selected",
            payload={
                "candidate_ids": finalists,
                "portfolio": portfolio.to_dict(),
            },
        )
        self.store.transition_search_run(
            run["run_id"],
            "evaluating",
            event_key="run_evaluating",
            stage="evaluating",
            payload={"candidate_ids": finalists},
            current_action="promote",
        )

    def _record_selection_diagnostics(
        self,
        *,
        run: Mapping[str, Any],
        context: Any,
        selected_results: Mapping[
            str, tuple[Mapping[str, Any], CandidateDefinition, Any]
        ],
    ) -> None:
        """为选择期候选登记相关性、正式因子和增量价值证据。"""

        if not selected_results:
            return
        existing = {
            str(row["evidence"]["candidate_id"])
            for row in self.store.list_evaluation_evidence(
                run["run_id"],
                stage=SELECTION_DIAGNOSTICS_STAGE,
            )
        }
        pending = {
            candidate_id: values
            for candidate_id, values in selected_results.items()
            if candidate_id not in existing
        }
        if not pending:
            return
        policy = load_quality_policy()
        try:
            conn = open_source(self.db_file)
            try:
                dates = sample_trade_dates(
                    conn,
                    context.windows.selection_start,
                    context.windows.selection_end,
                    policy.correlation_sample_days,
                )
            finally:
                conn.close()
        except Exception:
            dates = ()
        values: dict[str, Any] = {}
        for candidate_id, (_, candidate, result) in pending.items():
            artifact_glob = str(getattr(result, "factor_glob", ""))
            if artifact_glob and glob(artifact_glob):
                try:
                    values[candidate_id] = load_candidate_values(
                        candidate.name,
                        artifact_glob,
                        dates,
                    )
                except Exception:
                    continue
        correlations = compute_candidate_correlations(
            run_id=run["run_id"],
            window_start=context.windows.selection_start,
            window_end=context.windows.selection_end,
            values=values,
            min_stocks=policy.correlation_min_stocks,
            min_days=policy.correlation_min_days,
            threshold=context.control_policy.correlation_cluster_threshold,
        )
        for candidate_id, (_, candidate, result) in pending.items():
            correlation = correlations.get(candidate_id)
            if correlation is None:
                correlation = CandidateCorrelationReport(
                    run_id=run["run_id"],
                    candidate_id=candidate_id,
                    window_start=context.windows.selection_start,
                    window_end=context.windows.selection_end,
                    sample_dates=tuple(dates),
                    peers=(),
                    max_abs_correlation=None,
                    cluster_id="cluster-0001",
                    cluster_candidate_ids=(candidate_id,),
                    status="not_computable",
                    reason="选择期候选产物不可用",
                )
            technical_metrics = getattr(result.technical, "metrics", {})
            artifact_glob = str(getattr(result, "factor_glob", ""))
            if artifact_glob and glob(artifact_glob):
                incremental = compute_incremental_value(
                    candidate,
                    context,
                    Path(artifact_glob).parents[2],
                    technical_metrics,
                    db_file=self.db_file,
                    factor_dir=self.factor_dir,
                )
            else:
                from mining.search_selection_diagnostics import (
                    formal_factor_incremental_value,
                )

                formal_status, peer, value, maximum = (
                    formal_factor_incremental_value(technical_metrics)
                )
                incremental = IncrementalValueReport(
                    formal_factor_status=formal_status,
                    formal_factor_peer=peer,
                    formal_factor_correlation=value,
                    formal_factor_max_absolute=maximum,
                    raw_rank_ic=None,
                    incremental_rank_ic=None,
                    survivor_share=None,
                    valid_days=None,
                    status="not_computable",
                    reason="选择期候选产物不可用",
                )
            diagnostics = build_selection_diagnostics(
                run_id=run["run_id"],
                candidate_id=candidate_id,
                input_material={
                    "context_hash": run["context_hash"],
                    "candidate_id": candidate_id,
                    "factor_sha256": getattr(result, "factor_sha256", None),
                    "sample_dates": list(dates),
                },
                correlation=correlation,
                incremental=incremental,
            )
            self.store.record_evaluation_evidence(
                diagnostics.to_evaluation_evidence()
            )

    def _evaluate(self, run: Mapping[str, Any], context: Any) -> None:
        trace = self.store.get_search_trace(run["run_id"])
        events = {
            item["event_key"]: item
            for item in self.store.list_events(run["run_id"])
        }
        selected = events.get("finalists_selected")
        if selected is None:
            raise RuntimeError("V2 评估阶段缺少冻结入围批次")
        finalists = list(selected["payload"]["candidate_ids"])
        for candidate_id in finalists:
            item = self.store.get_candidate(run["run_id"], candidate_id)
            if item["status"] != "evaluating":
                continue
            candidate = candidate_from_v2_record(item)
            result = self._final_evaluator(
                run_id=run["run_id"],
                candidate=candidate,
                context=context,
                stage_root=self.store.candidate_dir(
                    run["run_id"],
                    candidate_id,
                )
                / "final",
            )
            self.store.record_evaluation_evidence(
                evaluation_evidence(result.report, stage="full")
            )
            source = self._candidate_source(trace, candidate_id)
            if result.report.status == "passed":
                self.store.transition_candidate(
                    run["run_id"],
                    candidate_id,
                    "evaluated",
                    trial=trial_record(
                        run_id=run["run_id"],
                        trial_key=f"full-evaluated-{candidate_id}",
                        kind="full_evaluated",
                        turn_index=source["turn_index"],
                        hypothesis_id=source["hypothesis_id"],
                        batch_id=source["batch_id"],
                        candidate_id=candidate_id,
                    ),
                )
            else:
                self.store.transition_candidate(
                    run["run_id"],
                    candidate_id,
                    "failed",
                    trial=trial_record(
                        run_id=run["run_id"],
                        trial_key=f"full-failed-{candidate_id}",
                        kind="rejected",
                        turn_index=source["turn_index"],
                        hypothesis_id=source["hypothesis_id"],
                        batch_id=source["batch_id"],
                        candidate_id=candidate_id,
                        reason_code=result.report.error_code or "MINING-106",
                    ),
                )
        self.store.transition_search_run(
            run["run_id"],
            "holding_out",
            event_key="run_holding_out",
            stage="holding_out",
            payload={"candidate_ids": finalists},
        )

    def _holdout(self, run: Mapping[str, Any], context: Any) -> None:
        trace = self.store.get_search_trace(run["run_id"])
        finalists = [
            item["candidate_id"]
            for item in trace["candidates"]
            if item["status"] == "evaluated"
        ]
        events = {
            item["event_key"]: item
            for item in self.store.list_events(run["run_id"])
        }
        claim = events.get("holdout_claimed")
        last_turn = trace["turns"][-1]
        if claim is None:
            self.store.record_trial(
                trial_record(
                    run_id=run["run_id"],
                    trial_key="holdout-claimed-once",
                    kind="holdout_claimed",
                    turn_index=int(last_turn["turn_index"]),
                    hypothesis_id=str(last_turn["hypothesis_id"]),
                    batch_id=str(last_turn["experiment_batch_id"]),
                    candidate_id=None,
                    payload={"candidate_ids": finalists},
                )
            )
            self.store.append_event(
                run["run_id"],
                event_key="holdout_claimed",
                stage="holding_out",
                event_type="holdout_claimed",
                payload={"candidate_ids": finalists},
            )
        else:
            claimed = list(claim["payload"].get("candidate_ids", []))
            if claimed != finalists:
                raise RuntimeError("模块测试段已由不同候选批次领取")

        reviewed: list[str] = []
        for candidate_id in finalists:
            item = self.store.get_candidate(run["run_id"], candidate_id)
            if item["status"] != "evaluated":
                continue
            self.store.transition_candidate(
                run["run_id"],
                candidate_id,
                "holdout_running",
            )
            candidate = candidate_from_v2_record(item)
            result = self._holdout_evaluator(
                run_id=run["run_id"],
                candidate=candidate,
                context=context,
                stage_root=self.store.candidate_dir(
                    run["run_id"],
                    candidate_id,
                )
                / "holdout",
            )
            self.store.record_evaluation_evidence(
                evaluation_evidence(result.report, stage="holdout")
            )
            source = self._candidate_source(trace, candidate_id)
            target = "review_ready" if result.passed else "holdout_failed"
            self.store.transition_candidate(
                run["run_id"],
                candidate_id,
                target,
                trial=trial_record(
                    run_id=run["run_id"],
                    trial_key=f"holdout-{candidate_id}",
                    kind="evaluated" if result.passed else "rejected",
                    turn_index=source["turn_index"],
                    hypothesis_id=source["hypothesis_id"],
                    batch_id=source["batch_id"],
                    candidate_id=candidate_id,
                    reason_code=None if result.passed else "MINING-106",
                ),
            )
            if result.passed:
                reviewed.append(candidate_id)
        self.store.transition_search_run(
            run["run_id"],
            "review_ready",
            event_key="run_review_ready",
            stage="review",
            payload={"candidate_ids": reviewed},
            stop_reason="搜索、最终评估和一次性测试段已完成",
        )

    def _reject_candidate(
        self,
        *,
        run: Mapping[str, Any],
        trace: Mapping[str, Any],
        candidate_id: str,
        stage: str,
        trial_key: str,
    ) -> None:
        source = self._candidate_source(trace, candidate_id)
        self.store.transition_candidate(
            run["run_id"],
            candidate_id,
            "screened_out",
            trial=trial_record(
                run_id=run["run_id"],
                trial_key=trial_key,
                kind="rejected",
                turn_index=int(source["turn_index"]),
                hypothesis_id=str(source["hypothesis_id"]),
                batch_id=str(source["batch_id"]),
                candidate_id=candidate_id,
                reason_code=(
                    "MINING-106"
                    if stage == "selection"
                    else "MINING-105"
                ),
            ),
        )

    def _candidate_source(
        self,
        trace: Mapping[str, Any],
        candidate_id: str,
    ) -> dict[str, Any]:
        item = next(
            candidate
            for candidate in trace["candidates"]
            if candidate["candidate_id"] == candidate_id
        )
        batch = self.store.get_batch(str(item["run_id"]), str(item["batch_id"]))
        return {
            "turn_index": int(batch["turn_index"]),
            "hypothesis_id": str(item["hypothesis_id"]),
            "batch_id": str(item["batch_id"]),
        }
