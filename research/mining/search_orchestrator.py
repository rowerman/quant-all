"""证据驱动 V2 搜索的同步执行入口。"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping

from common.paths import DB_FILE, FACTOR_DIR
from mining.candidate_batch import materialize_candidate_batch
from mining.errors import MiningStateError
from mining.logging_conf import format_event, get_mining_logger
from mining.search_context import FrozenSearchContext
from mining.search_evaluation_adapter import (
    development_context,
    evaluate_development,
    evaluate_final,
    evaluate_holdout,
    evaluate_selection,
)
from mining.search_orchestrator_pipeline import SearchPipelineMixin
from mining.search_orchestrator_turn import SearchTurnMixin
from mining.search_run import SearchRunRequest
from mining.search_worker_client import (
    HttpSearchWorkerClient,
    SearchWorkerClient,
)
from mining.snapshot import DEFAULT_DATA_MANIFEST, assert_context_snapshot
from mining.storage_v2 import MiningV2Store

_LOGGER = get_mining_logger("mining.search.orchestrator")


class MiningSearchOrchestrator(SearchTurnMixin, SearchPipelineMixin):
    """执行可恢复的 V2 证据驱动搜索阶段。"""

    def __init__(
        self,
        *,
        store: MiningV2Store | None = None,
        worker: SearchWorkerClient | None = None,
        materializer: Callable[..., Any] = materialize_candidate_batch,
        development_evaluator: Callable[..., Any] = evaluate_development,
        selection_evaluator: Callable[..., Any] = evaluate_selection,
        final_evaluator: Callable[..., Any] = evaluate_final,
        holdout_evaluator: Callable[..., Any] = evaluate_holdout,
        snapshot_check: Callable[..., None] = assert_context_snapshot,
        db_file: Path | str = DB_FILE,
        factor_dir: Path | str = FACTOR_DIR,
        manifest_file: Path | str = DEFAULT_DATA_MANIFEST,
    ) -> None:
        self.store = store or MiningV2Store()
        self._worker_client_value = worker
        self._materializer = materializer
        self._development_evaluator = development_evaluator
        self._selection_evaluator = selection_evaluator
        self._final_evaluator = final_evaluator
        self._holdout_evaluator = holdout_evaluator
        self._snapshot_check = snapshot_check
        self.db_file = Path(db_file)
        self.factor_dir = Path(factor_dir)
        self.manifest_file = Path(manifest_file)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mining-search-v2",
        )
        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._lock = RLock()

    def create_run(
        self,
        request: SearchRunRequest,
        context: FrozenSearchContext,
        *,
        background: bool = True,
    ) -> dict[str, Any]:
        """创建 V2 运行并按需后台执行。"""

        worker = self._worker()
        health = worker.health()
        if health.get("status") != "ready":
            raise MiningStateError("Worker v2 尚未就绪")
        run = self.store.create_run(request, context)
        if background and run["status"] == "queued":
            self._schedule(run["run_id"])
        return run

    def execute(self, run_id: str) -> dict[str, Any]:
        """同步推进到当前状态对应的阶段结束。"""

        context = self._context(run_id)
        self._check_snapshot(context)
        try:
            while True:
                run = self.store.get_run(run_id)
                status = run["status"]
                if status in {"review_ready", "failed", "cancelled", "stale"}:
                    return run
                if status == "interrupted":
                    raise MiningStateError("中断的 V2 运行必须显式恢复")
                if status == "queued":
                    self.store.transition_search_run(
                        run_id,
                        "searching",
                        event_key="run_searching",
                        stage="searching",
                    )
                elif status == "searching":
                    if self.store.cancel_requested(run_id):
                        self.store.transition_search_run(
                            run_id,
                            "cancelled",
                            event_key="run_cancelled",
                            stage="searching",
                            stop_reason="用户取消",
                        )
                        return self.store.get_run(run_id)
                    self._search(run, context)
                elif status == "selecting":
                    self._select(run, context)
                elif status == "evaluating":
                    self._evaluate(run, context)
                elif status == "holding_out":
                    self._holdout(run, context)
                else:
                    raise MiningStateError(f"未知 V2 搜索状态：{status}")
        except Exception as exc:
            self.store.append_event(
                run_id,
                event_key=f"run_failed_{getattr(exc, 'code', 'MINING-100')}",
                stage=self.store.search_control_state(run_id)["status"],
                event_type="search_run_failed",
                payload={
                    "error_code": getattr(exc, "code", "MINING-100"),
                    "error_message": str(exc)[:500],
                },
            )
            self.store.transition_search_run(
                run_id,
                "failed",
                event_key="run_failed",
                stage=self.store.search_control_state(run_id)["status"],
                stop_reason=str(exc)[:500],
            )
            raise

    def resume(self, run_id: str, *, background: bool = True) -> dict[str, Any]:
        """显式恢复中断运行。"""

        state = self.store.search_control_state(run_id)
        if state["status"] != "interrupted" or not state["interrupted_from"]:
            raise MiningStateError("只有中断的 V2 运行可以恢复")
        resumed = self.store.transition_search_run(
            run_id,
            state["interrupted_from"],
            event_key="run_resumed",
            stage=state["interrupted_from"],
        )
        if background:
            self._schedule(run_id)
        return resumed

    def cancel(self, run_id: str) -> dict[str, Any]:
        """请求在当前安全边界取消。"""

        requested = self.store.request_search_cancel(run_id)
        if requested["status"] not in {
            "searching",
            "selecting",
            "evaluating",
            "holding_out",
        }:
            return requested
        return requested

    def close(self) -> None:
        """关闭后台执行器。"""

        self._executor.shutdown(wait=True, cancel_futures=True)

    def _schedule(self, run_id: str) -> None:
        with self._lock:
            future = self._futures.get(run_id)
            if future is not None and not future.done():
                return
            self._futures[run_id] = self._executor.submit(self.execute, run_id)

    def _worker(self) -> SearchWorkerClient:
        if self._worker_client_value is None:
            self._worker_client_value = HttpSearchWorkerClient.from_environment()
        return self._worker_client_value

    def _context(self, run_id: str) -> FrozenSearchContext:
        return FrozenSearchContext.from_dict(self.store.get_run(run_id)["context"])

    def _check_snapshot(self, context: FrozenSearchContext) -> None:
        self._snapshot_check(
            development_context(context),
            db_file=self.db_file,
            factor_dir=self.factor_dir,
            manifest_file=self.manifest_file,
        )
