"""挖掘编排的 Worker 生成阶段。"""

from __future__ import annotations

from threading import Event
from typing import Any, Mapping

from mining.errors import MiningDuplicateError
from mining.orchestration_support import (
    build_worker_request,
    context_from_run,
    events_by_key,
    request_from_run,
)
from mining.worker_client import generate_worker_candidates
from mining_worker.worker_config import load_worker_config


class GenerationPhaseMixin:
    """执行受预算和停止条件约束的多轮生成。"""

    def _generate(
        self,
        run: Mapping[str, Any],
        cancellation: Event,
    ) -> None:
        """执行多轮 Worker 生成，直到预算或停止条件。"""

        run_id = run["run_id"]
        request = request_from_run(run)
        context = context_from_run(run)
        self._assert_snapshot(context)
        max_rounds = min(
            context.budget.max_rounds,
            context.budget.max_llm_calls,
        )
        worker_config = load_worker_config()
        for round_index in range(1, max_rounds + 1):
            self._check_cancelled(run_id, cancellation)
            events = events_by_key(self.store, run_id)
            completed = events.get(f"round_{round_index}_completed")
            if completed is not None:
                if int(completed["payload"].get("new_candidates", 0)) == 0:
                    break
                continue
            remaining = (
                context.budget.max_candidates
                - len(self.store.list_candidates(run_id))
            )
            if remaining <= 0:
                break
            requested_key = f"round_{round_index}_requested"
            requested = events.get(requested_key)
            if requested is None:
                feedback = self._feedback_history(events)
                max_candidates = min(
                    remaining,
                    worker_config.max_candidates_per_round,
                )
                worker_request = build_worker_request(
                    request=request,
                    context=context,
                    round_index=round_index,
                    feedback=feedback,
                    max_candidates=max_candidates,
                )
                self.store.append_event(
                    run_id,
                    event_key=requested_key,
                    stage="generating",
                    event_type="round_requested",
                    payload={
                        "round": round_index,
                        "max_candidates": max_candidates,
                        "feedback": list(feedback),
                        "payload_hash": worker_request.payload_hash,
                    },
                )
                requested = events_by_key(self.store, run_id)[requested_key]
            payload = requested["payload"]
            max_candidates = int(payload["max_candidates"])
            feedback = tuple(payload["feedback"])
            worker_request = build_worker_request(
                request=request,
                context=context,
                round_index=round_index,
                feedback=feedback,
                max_candidates=max_candidates,
            )
            normalized = generate_worker_candidates(
                self._worker_client(),
                worker_request,
                should_cancel=lambda: self._is_cancelled(run_id, cancellation),
                on_submitted=lambda job: self._record_worker_job(
                    run_id,
                    round_index,
                    job.job_id,
                ),
            )
            self._check_cancelled(run_id, cancellation)
            duplicates = 0
            for definition in normalized.accepted:
                try:
                    self.store.add_candidate(run_id, definition)
                except MiningDuplicateError:
                    duplicates += 1
            stored_round = [
                item
                for item in self.store.list_candidates(run_id)
                if int(item["discovery_round"]) == round_index
            ]
            feedback_entry = {
                "round": round_index,
                "stored": len(stored_round),
                "rejected": len(normalized.rejected),
                "duplicates": duplicates,
                "error_codes": sorted(
                    {item.error_code for item in normalized.rejected}
                ),
            }
            self.store.append_event(
                run_id,
                event_key=f"round_{round_index}_completed",
                stage="generating",
                event_type="round_completed",
                payload={
                    "round": round_index,
                    "new_candidates": len(stored_round),
                    "feedback_entry": feedback_entry,
                    "usage": dict(normalized.usage),
                },
            )
            if not stored_round:
                break
        self.store.transition_run(
            run_id,
            "screening",
            event_key="run_screening",
            stage="screening",
        )

    def _feedback_history(
        self,
        events: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        """读取最近完成轮次的安全反馈摘要。"""

        completed = sorted(
            (
                int(event["payload"]["round"]),
                event["payload"]["feedback_entry"],
            )
            for event in events.values()
            if event["event_type"] == "round_completed"
        )
        return tuple(item[1] for item in completed[-19:])

    def _record_worker_job(
        self,
        run_id: str,
        round_index: int,
        job_id: str,
    ) -> None:
        """记录 Worker 作业 ID 供取消使用。"""

        self._worker_jobs[run_id] = job_id
        self.store.append_event(
            run_id,
            event_key=f"round_{round_index}_job_submitted",
            stage="generating",
            event_type="worker_job_submitted",
            payload={"round": round_index, "job_id": job_id},
        )
