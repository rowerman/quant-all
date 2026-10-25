"""挖掘编排的技术门禁与快速筛选阶段。"""

from __future__ import annotations

from threading import Event
from typing import Any, Mapping

from mining.errors import MiningEvaluationError
from mining.orchestration_support import context_from_run

_FAILED_CANDIDATE_STATUSES = {
    "technical_failed",
    "screened_out",
    "holdout_failed",
    "failed",
}


class ScreeningPhaseMixin:
    """执行技术门禁和研究期快速筛选。"""

    def _screen(
        self,
        run: Mapping[str, Any],
        cancellation: Event,
    ) -> None:
        """执行技术门禁和研究期快速筛选。"""

        run_id = run["run_id"]
        context = context_from_run(run)
        self._assert_snapshot(context)
        for item in self.store.list_candidates(run_id):
            self._check_cancelled(run_id, cancellation)
            status = item["status"]
            if status in _FAILED_CANDIDATE_STATUSES or status in {
                "review_ready",
                "cancelled",
                "stale",
            }:
                continue
            candidate_id = item["candidate_id"]
            if status == "proposed":
                item = self.store.transition_candidate(
                    run_id,
                    candidate_id,
                    "validating",
                    event_key=f"candidate_{candidate_id}_validating",
                    stage="technical",
                )
            technical = self.store.find_evaluation(
                run_id, candidate_id, "technical"
            )
            gate = self.store.find_technical_gate(run_id, candidate_id)
            result = None
            if technical is None or gate is None or (
                status == "screening"
                and self.store.find_evaluation(run_id, candidate_id, "quick")
                is None
            ):
                self._append_stage_started(
                    run_id,
                    candidate_id,
                    "technical",
                    "quality_check_started",
                )
                result = self.evaluator(
                    self._evaluation_input(run, item, context),
                    policy=None,
                )
                self._check_cancelled(run_id, cancellation)
                technical = technical or self.store.record_evaluation(
                    result.technical
                )
                gate = gate or self.store.record_technical_gate(result.gate)
            if not gate["passed"] or technical["status"] != "passed":
                self._transition_if_needed(
                    run_id,
                    candidate_id,
                    "technical_failed",
                    stage="technical",
                    failure_stage="technical",
                    error_code="MINING-106",
                    error_message="候选未通过技术门禁",
                )
            else:
                self._transition_if_needed(
                    run_id,
                    candidate_id,
                    "screening",
                    stage="screening",
                )
                quick = self.store.find_evaluation(run_id, candidate_id, "quick")
                if quick is None:
                    if result is None:
                        result = self.evaluator(
                            self._evaluation_input(run, item, context),
                            policy=None,
                        )
                        self._check_cancelled(run_id, cancellation)
                    if result.quick is None:
                        raise MiningEvaluationError("技术门禁通过但缺少快速筛选结果")
                    quick = self.store.record_evaluation(result.quick)
                if quick["status"] != "passed":
                    self._transition_if_needed(
                        run_id,
                        candidate_id,
                        "screened_out",
                        stage="screening",
                        failure_stage="quick",
                        error_code="MINING-106",
                        error_message="候选未通过研究期快速筛选",
                    )
            if self._failure_rate_reached(run_id, context):
                raise MiningEvaluationError(
                    "候选失败率达到冻结阈值，停止后续质量检查"
                )
        passing = [
            item
            for item in self.store.list_candidates(run_id)
            if item["status"] == "screening"
        ]
        if not passing:
            raise MiningEvaluationError("没有候选通过研究期快速筛选")
        self.store.transition_run(
            run_id,
            "evaluating",
            event_key="run_evaluating",
            stage="evaluating",
        )

    def _failure_rate_reached(self, run_id: str, context: Any) -> bool:
        """判断候选失败率是否达到冻结阈值。"""

        checked = [
            item
            for item in self.store.list_candidates(run_id)
            if item["status"]
            in {
                "technical_failed",
                "screened_out",
                "screening",
                "evaluating",
                "evaluated",
                "holdout_running",
                "holdout_failed",
                "review_ready",
            }
        ]
        if len(checked) < context.budget.min_quality_checks_for_failure_stop:
            return False
        failed = sum(
            item["status"] in {"technical_failed", "screened_out"}
            for item in checked
        )
        return failed / len(checked) >= context.budget.max_failure_rate
