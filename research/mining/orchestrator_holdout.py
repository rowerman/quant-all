"""挖掘编排的候选冻结与模块测试段阶段。"""

from __future__ import annotations

from threading import Event
from typing import Any, Mapping

from mining.contract_types import HOLDOUT_SEGMENT_ID, sha256_json
from mining.errors import (
    HoldoutError,
    MiningEvaluationError,
    MiningNotFoundError,
)
from mining.orchestration_support import context_from_run, events_by_key


class HoldoutPhaseMixin:
    """冻结候选批次并执行一次性模块测试段。"""

    def _holdout(
        self,
        run: Mapping[str, Any],
        cancellation: Event,
    ) -> None:
        """冻结候选批次并执行一次性模块测试段。"""

        run_id = run["run_id"]
        context = context_from_run(run)
        self._assert_snapshot(context)
        events = events_by_key(self.store, run_id)
        selected = events.get("finalists_selected")
        if selected is None:
            raise HoldoutError("缺少冻结候选批次")
        finalists = list(selected["payload"]["candidate_ids"])
        evidence_hash = self._holdout_evidence_hash(run_id, finalists)
        for candidate_id in finalists:
            self._transition_if_needed(
                run_id,
                candidate_id,
                "holdout_running",
                stage="holdout",
            )
        try:
            usage = self.store.get_holdout_usage(HOLDOUT_SEGMENT_ID)
        except MiningNotFoundError:
            self._check_cancelled(run_id, cancellation)
            usage = self.store.claim_holdout(
                run_id=run_id,
                candidate_ids=finalists,
                evidence_hash=evidence_hash,
            )
        if (
            sorted(usage["candidate_ids"]) != sorted(finalists)
            or usage["evidence_hash"] != evidence_hash
        ):
            raise HoldoutError("模块测试段已被不同候选批次领取")
        batch_candidates = []
        for candidate_id in usage["candidate_ids"]:
            item = self.store.get_candidate(run_id, candidate_id)
            if item["status"] not in {"review_ready", "holdout_failed"}:
                batch_candidates.append(item)
        batch_failures = self._materialize_batches(
            run,
            batch_candidates,
            context,
            start=context.holdout_start,
            end=context.holdout_end,
            holdout=True,
        )
        for candidate_id in usage["candidate_ids"]:
            self._check_cancelled(run_id, cancellation)
            item = self.store.get_candidate(run_id, candidate_id)
            if item["status"] in {"review_ready", "holdout_failed"}:
                continue
            failure = batch_failures.get(candidate_id)
            if failure is not None:
                self._transition_if_needed(
                    run_id,
                    candidate_id,
                    "holdout_failed",
                    stage="holdout",
                    failure_stage="holdout",
                    error_code=failure.error_code,
                    error_message=failure.message,
                )
                continue
            report = self.store.find_evaluation(run_id, candidate_id, "holdout")
            if report is None:
                self._append_stage_started(
                    run_id,
                    candidate_id,
                    "holdout",
                    "holdout_evaluation_started",
                )
                result = self.holdout_evaluator(
                    self._evaluation_input(run, item, context),
                    policy=None,
                )
                self._check_cancelled(run_id, cancellation)
                report = self.store.record_evaluation(result.report)
            if report["status"] == "error":
                self._transition_if_needed(
                    run_id,
                    candidate_id,
                    "holdout_failed",
                    stage="holdout",
                    failure_stage="holdout",
                    error_code="MINING-106",
                    error_message="模块测试段执行失败",
                )
                raise MiningEvaluationError("模块测试段执行失败")
            target = (
                "review_ready"
                if report["status"] == "passed"
                else "holdout_failed"
            )
            self._transition_if_needed(
                run_id,
                candidate_id,
                target,
                stage="holdout",
                failure_stage=None if target == "review_ready" else "holdout",
            )
        self.store.transition_run(
            run_id,
            "review_ready",
            event_key="run_review_ready",
            stage="review",
            payload={"candidates": finalists},
        )

    def _holdout_evidence_hash(
        self,
        run_id: str,
        finalists: list[str],
    ) -> str:
        """生成冻结候选批次进入测试段前的证据哈希。"""

        material = []
        for candidate_id in sorted(finalists):
            gate = self.store.find_technical_gate(run_id, candidate_id)
            reports = {
                stage: self.store.find_evaluation(run_id, candidate_id, stage)
                for stage in ("technical", "quick", "full")
            }
            if gate is None or any(report is None for report in reports.values()):
                raise HoldoutError("候选冻结前证据不完整")
            material.append(
                {
                    "candidate_id": candidate_id,
                    "gate_hash": gate["check_hash"],
                    "report_ids": {
                        stage: report["report_id"]
                        for stage, report in reports.items()
                        if report is not None
                    },
                }
            )
        return sha256_json({"run_id": run_id, "candidates": material})
