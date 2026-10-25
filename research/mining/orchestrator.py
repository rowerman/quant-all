"""自动因子挖掘运行控制与阶段协调。"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event, RLock
from typing import Any, Callable, Mapping

from common.paths import DB_FILE, FACTOR_DIR
from mining.candidate_evaluator import evaluate_candidate
from mining.candidate_stage_evaluator import (
    evaluate_full_candidate,
    evaluate_holdout_candidate,
)
from mining.contract_requests import MiningRunRequest
from mining.errors import (
    MiningBusyError,
    MiningError,
    MiningRecoveryError,
    MiningStateError,
    SnapshotStaleError,
    WorkerCancelledError,
    WorkerConfigurationError,
)
from mining.evaluation_contract import CandidateEvaluationInput
from mining.logging_conf import format_event, get_mining_logger
from mining.orchestration_contracts import MiningProgress
from mining.orchestration_support import (
    build_frozen_context,
    candidate_from_record,
    context_from_run,
    progress_for_run,
)
from mining.orchestrator_evaluation import EvaluationPhaseMixin
from mining.orchestrator_generation import GenerationPhaseMixin
from mining.orchestrator_holdout import HoldoutPhaseMixin
from mining.orchestrator_screening import ScreeningPhaseMixin
from mining.snapshot import DEFAULT_DATA_MANIFEST, assert_context_snapshot
from mining.storage import MiningStore
from mining.worker_client import HttpWorkerClient, WorkerClient

_LOGGER = get_mining_logger("mining.orchestrator")
_TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled", "stale"}


class _RunCancelled(RuntimeError):
    """内部取消信号。"""


class MiningOrchestrator(
    GenerationPhaseMixin,
    ScreeningPhaseMixin,
    EvaluationPhaseMixin,
    HoldoutPhaseMixin,
):
    """单槽位、可恢复的挖掘编排器。"""

    def __init__(
        self,
        *,
        store: MiningStore | None = None,
        worker: WorkerClient | None = None,
        evaluator: Callable[..., Any] = evaluate_candidate,
        full_evaluator: Callable[..., Any] = evaluate_full_candidate,
        holdout_evaluator: Callable[..., Any] = evaluate_holdout_candidate,
        db_file: Path | str = DB_FILE,
        factor_dir: Path | str = FACTOR_DIR,
        manifest_file: Path | str = DEFAULT_DATA_MANIFEST,
    ) -> None:
        self.store = store or MiningStore()
        self.worker = worker
        self.evaluator = evaluator
        self.full_evaluator = full_evaluator
        self.holdout_evaluator = holdout_evaluator
        self.db_file = Path(db_file)
        self.factor_dir = Path(factor_dir)
        self.manifest_file = Path(manifest_file)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mining-orchestrator",
        )
        self._lock = RLock()
        self._active: set[str] = set()
        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._cancel_events: dict[str, Event] = {}
        self._worker_jobs: dict[str, str] = {}

    def create_run(
        self,
        request: MiningRunRequest,
        *,
        background: bool = True,
    ) -> dict[str, Any]:
        """冻结上下文、创建运行并按需提交后台执行。"""

        worker = self._worker_client()
        health = worker.health()
        if health.get("status") != "ready":
            raise WorkerConfigurationError("Worker 尚未就绪")
        context = build_frozen_context(
            request,
            db_file=self.db_file,
            factor_dir=self.factor_dir,
            manifest_file=self.manifest_file,
        )
        run = self.store.create_run(request, context, worker_mode="rdagent")
        if background and run["status"] == "queued":
            self._schedule(run["run_id"])
        return run

    def execute(self, run_id: str) -> dict[str, Any]:
        """同步执行或恢复运行，直到阶段结束或出现终止条件。"""

        with self._lock:
            if run_id in self._active:
                raise MiningStateError(f"运行 {run_id} 已在执行")
            self._active.add(run_id)
            cancellation = self._cancel_events.setdefault(run_id, Event())
        try:
            return self._execute(run_id, cancellation)
        finally:
            with self._lock:
                self._active.discard(run_id)
                self._cancel_events.pop(run_id, None)
                self._worker_jobs.pop(run_id, None)

    def resume(
        self,
        run_id: str,
        *,
        background: bool = True,
    ) -> dict[str, Any]:
        """显式恢复中断运行。"""

        resumed = self.store.resume_run(run_id)
        if background:
            self._schedule(run_id)
        return resumed

    def cancel(self, run_id: str) -> dict[str, Any]:
        """请求取消运行并终止尚未完成的候选。"""

        run = self.store.get_run(run_id)
        if run["status"] in _TERMINAL_RUN_STATUSES or run["status"] == "review_ready":
            return run
        cancellation = self._cancel_events.get(run_id)
        if cancellation is not None:
            cancellation.set()
        job_id = self._worker_jobs.get(run_id) or self._latest_worker_job(run_id)
        if job_id:
            try:
                self._worker_client().cancel(job_id)
            except MiningError as exc:
                _LOGGER.warning(
                    format_event(
                        {
                            "operation": "cancel_worker_failed",
                            "run_id": run_id,
                            "error_code": exc.code,
                        }
                    )
                )
        return self.store.terminate_run(
            run_id,
            "cancelled",
            event_key="run_cancelled",
        )

    def progress(self, run_id: str) -> MiningProgress:
        """读取可恢复的当前阶段和累计预算。"""

        return progress_for_run(self.store, run_id)

    def close(self) -> None:
        """取消当前任务并等待编排线程退出。"""

        with self._lock:
            run_ids = tuple(self._active)
        for run_id in run_ids:
            try:
                self.cancel(run_id)
            except MiningError as exc:
                _LOGGER.error(
                    format_event(
                        {
                            "operation": "close_cancel_failed",
                            "run_id": run_id,
                            "error_code": exc.code,
                        }
                    )
                )
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _schedule(self, run_id: str) -> None:
        """在单线程执行器中提交运行。"""

        with self._lock:
            future = self._futures.get(run_id)
            if future is not None and not future.done():
                return
            self._cancel_events.setdefault(run_id, Event())
            self._futures[run_id] = self._executor.submit(self.execute, run_id)

    def _execute(
        self,
        run_id: str,
        cancellation: Event,
    ) -> dict[str, Any]:
        """执行阶段循环并统一落盘失败、取消和过期状态。"""

        try:
            while True:
                self._check_cancelled(run_id, cancellation)
                run = self.store.get_run(run_id)
                status = run["status"]
                if status in _TERMINAL_RUN_STATUSES or status == "review_ready":
                    return run
                if status == "interrupted":
                    raise MiningRecoveryError("中断运行必须显式恢复")
                if status == "queued":
                    self._assert_snapshot(context_from_run(run))
                    self.store.transition_run(
                        run_id,
                        "generating",
                        event_key="run_generating",
                        stage="generating",
                    )
                elif status == "generating":
                    self._generate(run, cancellation)
                elif status == "screening":
                    self._screen(run, cancellation)
                elif status == "evaluating":
                    self._full_evaluate(run, cancellation)
                elif status == "holding_out":
                    self._holdout(run, cancellation)
                else:
                    raise MiningStateError(f"未知编排状态：{status}")
        except MiningBusyError:
            return self.store.get_run(run_id)
        except (_RunCancelled, WorkerCancelledError):
            return self.store.terminate_run(
                run_id,
                "cancelled",
                event_key="run_cancelled",
            )
        except SnapshotStaleError as exc:
            _LOGGER.warning(
                format_event(
                    {
                        "operation": "snapshot_stale",
                        "run_id": run_id,
                        "error_code": exc.code,
                    }
                )
            )
            return self.store.terminate_run(
                run_id,
                "stale",
                event_key="run_stale",
                error_code=exc.code,
                error_message=exc.message,
            )
        except Exception as exc:
            code = getattr(exc, "code", "MINING-100")
            _LOGGER.error(
                format_event(
                    {
                        "operation": "run_failed",
                        "run_id": run_id,
                        "error_code": code,
                    }
                )
            )
            return self.store.terminate_run(
                run_id,
                "failed",
                event_key=f"run_failed_{code}",
                error_code=code,
                error_message=str(exc)[:2000],
            )

    def _evaluation_input(
        self,
        run: Mapping[str, Any],
        candidate: Mapping[str, Any],
        context: Any,
    ) -> CandidateEvaluationInput:
        """构造单候选评估输入。"""

        return CandidateEvaluationInput(
            run_id=run["run_id"],
            candidate_id=candidate["candidate_id"],
            candidate=candidate_from_record(candidate),
            context=context,
            candidate_root=self.store.candidate_dir(
                run["run_id"], candidate["candidate_id"]
            ),
        )

    def _append_stage_started(
        self,
        run_id: str,
        candidate_id: str,
        stage: str,
        event_type: str,
    ) -> None:
        """记录质量预算开始事件。"""

        self.store.append_event(
            run_id,
            event_key=f"{event_type}_{candidate_id}",
            stage=stage,
            event_type=event_type,
            payload={"candidate_id": candidate_id, "stage": stage},
        )

    def _transition_if_needed(
        self,
        run_id: str,
        object_id: str,
        target: str,
        *,
        stage: str,
        failure_stage: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """幂等推进候选状态。"""

        current = self.store.get_candidate(run_id, object_id)
        if current["status"] == target:
            return current
        return self.store.transition_candidate(
            run_id,
            object_id,
            target,
            event_key=f"candidate_{object_id}_{target}",
            stage=stage,
            failure_stage=failure_stage,
            error_code=error_code,
            error_message=error_message,
        )

    def _assert_snapshot(self, context: Any) -> None:
        """检查冻结上下文是否仍与当前数据一致。"""

        assert_context_snapshot(
            context,
            db_file=self.db_file,
            factor_dir=self.factor_dir,
            manifest_file=self.manifest_file,
        )

    def _check_cancelled(self, run_id: str, cancellation: Event) -> None:
        """在安全点响应取消。"""

        if cancellation.is_set() or self.store.get_run(run_id)["status"] == "cancelled":
            raise _RunCancelled()

    def _is_cancelled(self, run_id: str, cancellation: Event) -> bool:
        """供 Worker 轮询调用的取消判断。"""

        try:
            return cancellation.is_set() or (
                self.store.get_run(run_id)["status"] == "cancelled"
            )
        except MiningError:
            return True

    def _latest_worker_job(self, run_id: str) -> str | None:
        """从事件恢复最近的 Worker 作业 ID。"""

        for event in reversed(self.store.list_events(run_id)):
            if event["event_type"] == "worker_job_submitted":
                return str(event["payload"]["job_id"])
        return None

    def _worker_client(self) -> WorkerClient:
        """延迟创建 Worker，生产环境不做 Mock 回退。"""

        if self.worker is None:
            self.worker = HttpWorkerClient.from_environment()
        return self.worker
