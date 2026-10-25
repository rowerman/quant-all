"""挖掘编排的候选冻结与完整评估阶段。"""

from __future__ import annotations

from threading import Event
from typing import Any, Mapping

from mining.errors import MiningEvaluationError
from mining.orchestration_support import context_from_run, events_by_key


class EvaluationPhaseMixin:
    """选择入围候选并执行研究期完整评估。"""

    def _full_evaluate(
        self,
        run: Mapping[str, Any],
        cancellation: Event,
    ) -> None:
        """选择入围候选并执行完整评估。"""

        run_id = run["run_id"]
        context = context_from_run(run)
        self._assert_snapshot(context)
        selected = events_by_key(self.store, run_id).get("finalists_selected")
        if selected is None:
            finalists = self._rank_finalists(run_id, context)
            if not finalists:
                raise MiningEvaluationError("没有可进入完整评估的候选")
            self.store.append_event(
                run_id,
                event_key="finalists_selected",
                stage="evaluating",
                event_type="finalists_selected",
                payload={"candidate_ids": finalists},
            )
            finalist_set = set(finalists)
            for item in self.store.list_candidates(run_id):
                if (
                    item["status"] == "screening"
                    and item["candidate_id"] not in finalist_set
                ):
                    self._transition_if_needed(
                        run_id,
                        item["candidate_id"],
                        "screened_out",
                        stage="screening",
                    )
        else:
            finalists = list(selected["payload"]["candidate_ids"])
        for candidate_id in finalists:
            self._check_cancelled(run_id, cancellation)
            item = self.store.get_candidate(run_id, candidate_id)
            self._transition_if_needed(
                run_id,
                candidate_id,
                "evaluating",
                stage="evaluation",
            )
            report = self.store.find_evaluation(run_id, candidate_id, "full")
            if report is None:
                self._append_stage_started(
                    run_id,
                    candidate_id,
                    "full",
                    "full_evaluation_started",
                )
                result = self.full_evaluator(
                    self._evaluation_input(run, item, context),
                    policy=None,
                )
                self._check_cancelled(run_id, cancellation)
                report = self.store.record_evaluation(result.report)
            if report["status"] != "passed":
                raise MiningEvaluationError(
                    f"候选 {candidate_id} 完整评估失败"
                )
            self._transition_if_needed(
                run_id,
                candidate_id,
                "evaluated",
                stage="evaluation",
            )
        self.store.transition_run(
            run_id,
            "holding_out",
            event_key="run_holding_out",
            stage="holding_out",
        )

    def _rank_finalists(self, run_id: str, context: Any) -> list[str]:
        """按研究期指标确定性选择完整评估候选。"""

        ranked: list[tuple[float, float, float, str, str]] = []
        for item in self.store.list_candidates(run_id):
            if item["status"] != "screening":
                continue
            quick = self.store.find_evaluation(
                run_id, item["candidate_id"], "quick"
            )
            technical = self.store.find_evaluation(
                run_id, item["candidate_id"], "technical"
            )
            if quick is None or technical is None:
                continue
            metrics = quick["metrics"]
            rank_ic = abs(float(metrics.get("rank_ic_mean") or 0.0))
            icir = abs(float(metrics.get("rank_ic_ir") or 0.0))
            coverage = float(
                (technical["metrics"].get("values") or {}).get("coverage", 0.0)
            )
            ranked.append(
                (
                    -rank_ic,
                    -icir,
                    -coverage,
                    item["content_hash"],
                    item["candidate_id"],
                )
            )
        ranked.sort()
        return [item[4] for item in ranked[: context.budget.full_eval_top_k]]
