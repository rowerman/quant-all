"""独立 Worker 的本地 HTTP 作业服务。"""

from __future__ import annotations

import hmac
import os
from pathlib import Path
import threading
from typing import Mapping, Protocol

from flask import Flask, jsonify, request

from mining_worker.generator import (
    SubprocessGenerator,
    WorkerGenerationFailed,
    WorkerJobCancelled,
    WorkerJobTimeout,
)
from mining_worker.job_store import JobBusyError, JobConflictError, JobStore
from mining_worker.logging_conf import get_worker_logger, log_worker_event
from mining_worker.protocol_code import (
    WorkerCodeProposalRequest,
    WorkerCodeProposalResponse,
)
from mining_worker.protocol import (
    WorkerGenerateRequest,
    WorkerGenerationResponse,
    WorkerJobView,
    WorkerProtocolError,
)
from mining_worker.protocol_v2 import WorkerSearchRequest, WorkerSearchResponse
from mining_worker.protocol_v3 import WorkerSearchTurnRequest, WorkerSearchTurnResponse
from mining_worker.protocol_research import (
    WorkerResearchExtractionRequest,
    WorkerResearchExtractionResponse,
)
from mining_worker.protocol_research_conversion import (
    WorkerResearchConversionRequest,
    WorkerResearchConversionResponse,
)
from mining_worker.search_generator import SubprocessSearchGenerator
from mining_worker.search_generator_v3 import SubprocessSearchTurnGenerator
from mining_worker.code_generator import SubprocessCodeGenerator
from mining_worker.research_generator import SubprocessResearchGenerator
from mining_worker.research_conversion_generator import (
    SubprocessResearchConversionGenerator,
)
from mining_worker.worker_config import load_worker_config


class CandidateGenerator(Protocol):
    """Worker 服务依赖的候选生成器。"""

    def health(self) -> Mapping[str, object]:
        """返回健康状态。"""

    def generate(
        self,
        request: WorkerGenerateRequest,
        *,
        timeout_seconds: float,
    ) -> WorkerGenerationResponse:
        """同步执行一轮生成。"""

    def cancel(self, request_id: str) -> None:
        """取消运行中的生成。"""


class SearchGenerator(Protocol):
    """Worker 服务依赖的 V2 搜索生成器。"""

    def health(self) -> Mapping[str, object]:
        """返回健康状态。"""

    def generate(
        self,
        request: WorkerSearchRequest,
        *,
        timeout_seconds: float,
    ) -> WorkerSearchResponse:
        """同步执行一轮搜索建议生成。"""

    def cancel(self, request_id: str) -> None:
        """取消运行中的生成。"""


class SearchTurnGenerator(Protocol):
    """Worker 服务依赖的工具感知搜索生成器。"""

    def health(self) -> Mapping[str, object]:
        """返回健康状态。"""

    def generate(
        self,
        request: WorkerSearchTurnRequest,
        *,
        timeout_seconds: float,
    ) -> WorkerSearchTurnResponse:
        """执行一个 discover 或 finalize 阶段。"""

    def cancel(self, request_id: str) -> None:
        """取消运行中的生成。"""


class CodeGenerator(Protocol):
    """Worker 服务依赖的代码提案生成器。"""

    def health(self) -> Mapping[str, object]:
        """返回健康状态。"""

    def generate(
        self,
        request: WorkerCodeProposalRequest,
        *,
        timeout_seconds: float,
    ) -> WorkerCodeProposalResponse:
        """生成一个 Qlib 代码提案。"""

    def cancel(self, request_id: str) -> None:
        """取消运行中的生成。"""


class ResearchGenerator(Protocol):
    """Worker 服务依赖的研报字段抽取生成器。"""

    def health(self) -> Mapping[str, object]:
        """返回健康状态。"""

    def generate(
        self,
        request: WorkerResearchExtractionRequest,
        *,
        timeout_seconds: float,
    ) -> WorkerResearchExtractionResponse:
        """执行一个研报字段抽取窗口。"""

    def cancel(self, request_id: str) -> None:
        """取消运行中的生成。"""


class ResearchConversionGenerator(Protocol):
    """Worker 服务依赖的研报 DSL 转换生成器。"""

    def health(self) -> Mapping[str, object]:
        """返回健康状态。"""

    def generate(
        self,
        request: WorkerResearchConversionRequest,
        *,
        timeout_seconds: float,
    ) -> WorkerResearchConversionResponse:
        """执行一个研报转换分块。"""

    def cancel(self, request_id: str) -> None:
        """取消运行中的生成。"""


def _load_token() -> str:
    direct = os.environ.get("MINING_WORKER_TOKEN", "").strip()
    if direct:
        return direct
    path = Path(
        os.environ.get("MINING_WORKER_TOKEN_FILE", "/run/secrets/mining_worker_token")
    )
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _error(code: str, message: str, status: int):
    return (
        jsonify(
            {
                "error_code": code,
                "error_message": str(message)[:500],
            }
        ),
        status,
    )


def create_app(
    *,
    generator: CandidateGenerator | None = None,
    search_generator: SearchGenerator | None = None,
    search_turn_generator: SearchTurnGenerator | None = None,
    code_generator: CodeGenerator | None = None,
    research_generator: ResearchGenerator | None = None,
    research_conversion_generator: ResearchConversionGenerator | None = None,
    state_dir: Path | str | None = None,
    token: str | None = None,
    job_timeout: float | None = None,
) -> Flask:
    """创建可注入生成器的 Worker Flask 应用。"""

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024
    state_root = Path(
        state_dir
        or os.environ.get("MINING_WORKER_STATE_DIR", "/state")
    )
    store = JobStore(state_root / "jobs.sqlite")
    worker_generator = generator or SubprocessGenerator()
    worker_search_generator = search_generator or SubprocessSearchGenerator()
    worker_search_turn_generator = (
        search_turn_generator or SubprocessSearchTurnGenerator()
    )
    worker_code_generator = code_generator or SubprocessCodeGenerator()
    worker_research_generator = research_generator or SubprocessResearchGenerator()
    worker_research_conversion_generator = (
        research_conversion_generator
        or SubprocessResearchConversionGenerator()
    )
    worker_token = _load_token() if token is None else token
    logger = get_worker_logger("mining_worker.service")
    timeout_seconds = float(
        job_timeout
        if job_timeout is not None
        else os.environ.get(
            "MINING_WORKER_JOB_TIMEOUT",
            load_worker_config().job_timeout_seconds,
        )
    )
    cancel_events: dict[str, threading.Event] = {}
    cancel_lock = threading.RLock()
    app.extensions["worker_store"] = store
    app.extensions["worker_generator"] = worker_generator
    app.extensions["worker_search_generator"] = worker_search_generator
    app.extensions["worker_search_turn_generator"] = worker_search_turn_generator
    app.extensions["worker_code_generator"] = worker_code_generator
    app.extensions["worker_research_generator"] = worker_research_generator
    app.extensions["worker_research_conversion_generator"] = (
        worker_research_conversion_generator
    )
    app.extensions["worker_cancel_events"] = cancel_events

    def authorized() -> bool:
        if len(worker_token) < 32:
            return False
        header = request.headers.get("Authorization", "")
        expected = f"Bearer {worker_token}"
        return hmac.compare_digest(header, expected)

    def require_auth():
        if len(worker_token) < 32:
            return _error("MINING-119", "Worker Token 未配置", 503)
        if not authorized():
            return _error("MINING-119", "Worker Token 无效", 401)
        return None

    def run_job(job_id: str, request_id: str) -> None:
        event = cancel_events[job_id]
        if event.is_set():
            with cancel_lock:
                cancel_events.pop(job_id, None)
            return
        if not store.mark_running(job_id):
            with cancel_lock:
                cancel_events.pop(job_id, None)
            return
        try:
            payload = store.request(job_id)
            if payload is None:
                raise WorkerGenerationFailed("作业请求已丢失")
            generation_request = WorkerGenerateRequest.from_dict(payload)
            if event.is_set():
                raise WorkerJobCancelled("作业已取消")
            response = worker_generator.generate(
                generation_request,
                timeout_seconds=timeout_seconds,
            )
            if not event.is_set():
                store.finish_success(job_id, response.to_dict())
                log_worker_event(
                    logger,
                    "worker_job_succeeded",
                    job_id=job_id,
                    candidates=len(response.candidates),
                )
        except WorkerJobTimeout:
            store.finish_failure(job_id, "MINING-116", "Worker 生成超时")
            log_worker_event(
                logger, "worker_job_failed", job_id=job_id, error_code="MINING-116"
            )
        except WorkerJobCancelled:
            store.cancel(job_id, "作业已取消")
            log_worker_event(
                logger, "worker_job_cancelled", job_id=job_id, error_code="MINING-117"
            )
        except Exception as exc:
            store.finish_failure(
                job_id,
                "MINING-118",
                f"Worker 生成失败：{type(exc).__name__}: {str(exc)[:400]}",
            )
            log_worker_event(
                logger, "worker_job_failed", job_id=job_id, error_code="MINING-118"
            )
        finally:
            with cancel_lock:
                cancel_events.pop(job_id, None)

    def run_search_job(job_id: str, request_id: str) -> None:
        event = cancel_events[job_id]
        if event.is_set():
            with cancel_lock:
                cancel_events.pop(job_id, None)
            return
        if not store.mark_running(job_id):
            with cancel_lock:
                cancel_events.pop(job_id, None)
            return
        try:
            payload = store.request(job_id)
            if payload is None:
                raise WorkerGenerationFailed("搜索作业请求已丢失")
            search_request = WorkerSearchRequest.from_dict(payload)
            if event.is_set():
                raise WorkerJobCancelled("作业已取消")
            response = worker_search_generator.generate(
                search_request,
                timeout_seconds=timeout_seconds,
            )
            if not event.is_set():
                store.finish_success(job_id, response.to_dict())
                log_worker_event(
                    logger,
                    "worker_search_succeeded",
                    job_id=job_id,
                    candidates=len(response.candidates),
                    action=response.decision.action,
                )
        except WorkerJobTimeout:
            store.finish_failure(job_id, "MINING-116", "Worker v2 生成超时")
            log_worker_event(
                logger,
                "worker_search_failed",
                job_id=job_id,
                error_code="MINING-116",
            )
        except WorkerJobCancelled:
            store.cancel(job_id, "作业已取消")
            log_worker_event(
                logger,
                "worker_search_cancelled",
                job_id=job_id,
                error_code="MINING-117",
            )
        except Exception as exc:
            store.finish_failure(
                job_id,
                "MINING-118",
                f"Worker v2 生成失败：{type(exc).__name__}: {str(exc)[:400]}",
            )
            log_worker_event(
                logger,
                "worker_search_failed",
                job_id=job_id,
                error_code="MINING-118",
            )
        finally:
            with cancel_lock:
                cancel_events.pop(job_id, None)

    def run_search_turn_job(job_id: str, request_id: str) -> None:
        event = cancel_events[job_id]
        if event.is_set():
            with cancel_lock:
                cancel_events.pop(job_id, None)
            return
        if not store.mark_running(job_id):
            with cancel_lock:
                cancel_events.pop(job_id, None)
            return
        try:
            payload = store.request(job_id)
            if payload is None:
                raise WorkerGenerationFailed("搜索轮次请求已丢失")
            search_request = WorkerSearchTurnRequest.from_dict(payload)
            if event.is_set():
                raise WorkerJobCancelled("作业已取消")
            response = worker_search_turn_generator.generate(
                search_request,
                timeout_seconds=timeout_seconds,
            )
            if not event.is_set():
                store.finish_success(job_id, response.to_dict())
                log_worker_event(
                    logger,
                    "worker_search_turn_succeeded",
                    job_id=job_id,
                    phase=search_request.phase,
                    kind=response.kind,
                )
        except WorkerJobTimeout:
            store.finish_failure(job_id, "MINING-116", "Worker v3 生成超时")
            log_worker_event(
                logger,
                "worker_search_turn_failed",
                job_id=job_id,
                error_code="MINING-116",
            )
        except WorkerJobCancelled:
            store.cancel(job_id, "作业已取消")
            log_worker_event(
                logger,
                "worker_search_turn_cancelled",
                job_id=job_id,
                error_code="MINING-117",
            )
        except Exception as exc:
            store.finish_failure(
                job_id,
                "MINING-118",
                f"Worker v3 生成失败：{type(exc).__name__}: {str(exc)[:400]}",
            )
            log_worker_event(
                logger,
                "worker_search_turn_failed",
                job_id=job_id,
                error_code="MINING-118",
            )
        finally:
            with cancel_lock:
                cancel_events.pop(job_id, None)

    def run_code_job(job_id: str, request_id: str) -> None:
        event = cancel_events[job_id]
        if event.is_set():
            with cancel_lock:
                cancel_events.pop(job_id, None)
            return
        if not store.mark_running(job_id):
            with cancel_lock:
                cancel_events.pop(job_id, None)
            return
        try:
            payload = store.request(job_id)
            if payload is None:
                raise WorkerGenerationFailed("代码提案请求已丢失")
            code_request = WorkerCodeProposalRequest.from_dict(payload)
            if event.is_set():
                raise WorkerJobCancelled("作业已取消")
            response = worker_code_generator.generate(
                code_request,
                timeout_seconds=timeout_seconds,
            )
            if not event.is_set():
                store.finish_success(job_id, response.to_dict())
                log_worker_event(
                    logger,
                    "worker_code_succeeded",
                    job_id=job_id,
                    iteration=code_request.iteration,
                )
        except WorkerJobTimeout:
            store.finish_failure(job_id, "MINING-116", "Worker 代码提案超时")
            log_worker_event(
                logger,
                "worker_code_failed",
                job_id=job_id,
                error_code="MINING-116",
            )
        except WorkerJobCancelled:
            store.cancel(job_id, "作业已取消")
            log_worker_event(
                logger,
                "worker_code_cancelled",
                job_id=job_id,
                error_code="MINING-117",
            )
        except Exception as exc:
            store.finish_failure(
                job_id,
                "MINING-118",
                f"Worker 代码提案失败：{type(exc).__name__}: {str(exc)[:400]}",
            )
            log_worker_event(
                logger,
                "worker_code_failed",
                job_id=job_id,
                error_code="MINING-118",
            )
        finally:
            with cancel_lock:
                cancel_events.pop(job_id, None)

    def run_research_job(job_id: str, request_id: str) -> None:
        event = cancel_events[job_id]
        if event.is_set():
            with cancel_lock:
                cancel_events.pop(job_id, None)
            return
        if not store.mark_running(job_id):
            with cancel_lock:
                cancel_events.pop(job_id, None)
            return
        try:
            payload = store.request(job_id)
            if payload is None:
                raise WorkerGenerationFailed("研报抽取请求已丢失")
            research_request = WorkerResearchExtractionRequest.from_dict(payload)
            if event.is_set():
                raise WorkerJobCancelled("作业已取消")
            response = worker_research_generator.generate(
                research_request,
                timeout_seconds=timeout_seconds,
            )
            if not event.is_set():
                store.finish_success(job_id, response.to_dict())
                log_worker_event(
                    logger,
                    "worker_research_extraction_succeeded",
                    job_id=job_id,
                    window_index=research_request.window_index,
                    factor_count=len(response.factors),
                )
        except WorkerJobTimeout:
            store.finish_failure(job_id, "MINING-116", "Worker v5 研报抽取超时")
            log_worker_event(
                logger,
                "worker_research_extraction_failed",
                job_id=job_id,
                error_code="MINING-116",
            )
        except WorkerJobCancelled:
            store.cancel(job_id, "作业已取消")
            log_worker_event(
                logger,
                "worker_research_extraction_cancelled",
                job_id=job_id,
                error_code="MINING-117",
            )
        except Exception as exc:
            store.finish_failure(
                job_id,
                "MINING-118",
                f"Worker v5 研报抽取失败：{type(exc).__name__}: {str(exc)[:400]}",
            )
            log_worker_event(
                logger,
                "worker_research_extraction_failed",
                job_id=job_id,
                error_code="MINING-118",
            )
        finally:
            with cancel_lock:
                cancel_events.pop(job_id, None)

    def run_research_conversion_job(job_id: str, request_id: str) -> None:
        event = cancel_events[job_id]
        if event.is_set():
            with cancel_lock:
                cancel_events.pop(job_id, None)
            return
        if not store.mark_running(job_id):
            with cancel_lock:
                cancel_events.pop(job_id, None)
            return
        try:
            payload = store.request(job_id)
            if payload is None:
                raise WorkerGenerationFailed("研报转换请求已丢失")
            conversion_request = WorkerResearchConversionRequest.from_dict(payload)
            if event.is_set():
                raise WorkerJobCancelled("作业已取消")
            response = worker_research_conversion_generator.generate(
                conversion_request,
                timeout_seconds=timeout_seconds,
            )
            if not event.is_set():
                store.finish_success(job_id, response.to_dict())
                log_worker_event(
                    logger,
                    "worker_research_conversion_succeeded",
                    job_id=job_id,
                    chunk_index=conversion_request.chunk_index,
                    draft_count=len(conversion_request.drafts),
                    converted=sum(
                        item.status == "converted"
                        for item in response.outcomes
                    ),
                )
        except WorkerJobTimeout:
            store.finish_failure(job_id, "MINING-116", "Worker v6 研报转换超时")
            log_worker_event(
                logger,
                "worker_research_conversion_failed",
                job_id=job_id,
                error_code="MINING-116",
            )
        except WorkerJobCancelled:
            store.cancel(job_id, "作业已取消")
            log_worker_event(
                logger,
                "worker_research_conversion_cancelled",
                job_id=job_id,
                error_code="MINING-117",
            )
        except Exception as exc:
            store.finish_failure(
                job_id,
                "MINING-118",
                f"Worker v6 研报转换失败：{type(exc).__name__}: {str(exc)[:400]}",
            )
            log_worker_event(
                logger,
                "worker_research_conversion_failed",
                job_id=job_id,
                error_code="MINING-118",
            )
        finally:
            with cancel_lock:
                cancel_events.pop(job_id, None)

    @app.get("/health")
    def health():
        if len(worker_token) < 32:
            return _error("MINING-119", "Worker Token 未配置", 503)
        payload = dict(worker_generator.health())
        if payload.get("status") != "ready":
            return jsonify(payload), 503
        payload.update(
            {
                "status": "ready",
                "state_store": "ready",
                "protocol_versions": [
                    "mining_worker_v1",
                    "mining_worker_v2",
                    "mining_worker_v3",
                    "mining_worker_code_v1",
                    "mining_worker_code_v2",
                    "mining_worker_research_extraction_v1",
                    "mining_worker_research_conversion_v1",
                ],
            }
        )
        return jsonify(payload)

    @app.post("/v1/jobs")
    def create_job():
        denied = require_auth()
        if denied is not None:
            return denied
        if not request.is_json:
            return _error("MINING-118", "请求必须是 JSON", 400)
        try:
            generation_request = WorkerGenerateRequest.from_dict(request.get_json())
        except (WorkerProtocolError, TypeError) as exc:
            return _error("MINING-118", f"请求不符合 Worker 协议：{exc}", 400)
        try:
            job, created = store.create(
                job_id=generation_request.job_id,
                request_id=generation_request.request_id,
                payload_hash=generation_request.payload_hash,
                request=generation_request.to_dict(),
            )
        except JobConflictError as exc:
            return _error("MINING-105", str(exc), 409)
        except JobBusyError as exc:
            return _error("MINING-102", str(exc), 429)
        if created:
            log_worker_event(
                logger,
                "worker_job_created",
                job_id=job.job_id,
                round_index=generation_request.round_index,
                max_candidates=generation_request.max_candidates,
            )
            with cancel_lock:
                cancel_events[job.job_id] = threading.Event()
            threading.Thread(
                target=run_job,
                args=(job.job_id, generation_request.request_id),
                name=f"mining-worker-{job.job_id}",
                daemon=True,
            ).start()
        return jsonify(job.to_dict()), 202

    @app.post("/v2/jobs")
    def create_search_job():
        denied = require_auth()
        if denied is not None:
            return denied
        if not request.is_json:
            return _error("MINING-118", "请求必须是 JSON", 400)
        try:
            search_request = WorkerSearchRequest.from_dict(request.get_json())
        except (WorkerProtocolError, TypeError) as exc:
            return _error("MINING-118", f"请求不符合 Worker v2 协议：{exc}", 400)
        try:
            job, created = store.create(
                job_id=search_request.job_id,
                request_id=search_request.request_id,
                payload_hash=search_request.payload_hash,
                request=search_request.to_dict(),
            )
        except JobConflictError as exc:
            return _error("MINING-105", str(exc), 409)
        except JobBusyError as exc:
            return _error("MINING-102", str(exc), 429)
        if created:
            log_worker_event(
                logger,
                "worker_search_created",
                job_id=job.job_id,
                turn_index=search_request.turn_index,
                max_candidates=search_request.max_candidates,
            )
            with cancel_lock:
                cancel_events[job.job_id] = threading.Event()
            threading.Thread(
                target=run_search_job,
                args=(job.job_id, search_request.request_id),
                name=f"mining-worker-search-{job.job_id}",
                daemon=True,
            ).start()
        return jsonify(job.to_dict()), 202

    @app.post("/v3/jobs")
    def create_search_turn_job():
        denied = require_auth()
        if denied is not None:
            return denied
        if not request.is_json:
            return _error("MINING-118", "请求必须是 JSON", 400)
        try:
            search_request = WorkerSearchTurnRequest.from_dict(request.get_json())
        except (WorkerProtocolError, TypeError) as exc:
            return _error("MINING-118", f"请求不符合 Worker v3 协议：{exc}", 400)
        try:
            job, created = store.create(
                job_id=search_request.job_id,
                request_id=search_request.request_id,
                payload_hash=search_request.payload_hash,
                request=search_request.to_dict(),
            )
        except JobConflictError as exc:
            return _error("MINING-105", str(exc), 409)
        except JobBusyError as exc:
            return _error("MINING-102", str(exc), 429)
        if created:
            log_worker_event(
                logger,
                "worker_search_turn_created",
                job_id=job.job_id,
                turn_index=search_request.base_request.turn_index,
                phase=search_request.phase,
            )
            with cancel_lock:
                cancel_events[job.job_id] = threading.Event()
            threading.Thread(
                target=run_search_turn_job,
                args=(job.job_id, search_request.request_id),
                name=f"mining-worker-search-turn-{job.job_id}",
                daemon=True,
            ).start()
        return jsonify(job.to_dict()), 202

    @app.post("/v4/jobs")
    def create_code_job():
        denied = require_auth()
        if denied is not None:
            return denied
        if not request.is_json:
            return _error("MINING-118", "请求必须是 JSON", 400)
        try:
            code_request = WorkerCodeProposalRequest.from_dict(request.get_json())
        except (WorkerProtocolError, TypeError) as exc:
            return _error("MINING-118", f"请求不符合代码提案协议：{exc}", 400)
        try:
            job, created = store.create(
                job_id=code_request.job_id,
                request_id=code_request.request_id,
                payload_hash=code_request.payload_hash,
                request=code_request.to_dict(),
            )
        except JobConflictError as exc:
            return _error("MINING-105", str(exc), 409)
        except JobBusyError as exc:
            return _error("MINING-102", str(exc), 429)
        if created:
            log_worker_event(
                logger,
                "worker_code_created",
                job_id=job.job_id,
                iteration=code_request.iteration,
            )
            with cancel_lock:
                cancel_events[job.job_id] = threading.Event()
            threading.Thread(
                target=run_code_job,
                args=(job.job_id, code_request.request_id),
                name=f"mining-worker-code-{job.job_id}",
                daemon=True,
            ).start()
        return jsonify(job.to_dict()), 202

    @app.post("/v5/jobs")
    def create_research_job():
        denied = require_auth()
        if denied is not None:
            return denied
        if not request.is_json:
            return _error("MINING-118", "请求必须是 JSON", 400)
        try:
            research_request = WorkerResearchExtractionRequest.from_dict(
                request.get_json()
            )
        except (WorkerProtocolError, TypeError) as exc:
            return _error(
                "MINING-118",
                f"请求不符合研报抽取协议：{exc}",
                400,
            )
        try:
            job, created = store.create(
                job_id=research_request.job_id,
                request_id=research_request.request_id,
                payload_hash=research_request.payload_hash,
                request=research_request.to_dict(),
            )
        except JobConflictError as exc:
            return _error("MINING-105", str(exc), 409)
        except JobBusyError as exc:
            return _error("MINING-102", str(exc), 429)
        if created:
            log_worker_event(
                logger,
                "worker_research_extraction_created",
                job_id=job.job_id,
                window_index=research_request.window_index,
                max_factors=research_request.max_factors,
            )
            with cancel_lock:
                cancel_events[job.job_id] = threading.Event()
            threading.Thread(
                target=run_research_job,
                args=(job.job_id, research_request.request_id),
                name=f"mining-worker-research-{job.job_id}",
                daemon=True,
            ).start()
        return jsonify(job.to_dict()), 202

    @app.post("/v6/jobs")
    def create_research_conversion_job():
        denied = require_auth()
        if denied is not None:
            return denied
        if not request.is_json:
            return _error("MINING-118", "请求必须是 JSON", 400)
        try:
            conversion_request = WorkerResearchConversionRequest.from_dict(
                request.get_json()
            )
        except (WorkerProtocolError, TypeError) as exc:
            return _error(
                "MINING-118",
                f"请求不符合研报转换协议：{exc}",
                400,
            )
        try:
            job, created = store.create(
                job_id=conversion_request.job_id,
                request_id=conversion_request.request_id,
                payload_hash=conversion_request.payload_hash,
                request=conversion_request.to_dict(),
            )
        except JobConflictError as exc:
            return _error("MINING-105", str(exc), 409)
        except JobBusyError as exc:
            return _error("MINING-102", str(exc), 429)
        if created:
            log_worker_event(
                logger,
                "worker_research_conversion_created",
                job_id=job.job_id,
                chunk_index=conversion_request.chunk_index,
                draft_count=len(conversion_request.drafts),
            )
            with cancel_lock:
                cancel_events[job.job_id] = threading.Event()
            threading.Thread(
                target=run_research_conversion_job,
                args=(job.job_id, conversion_request.request_id),
                name=f"mining-worker-research-conversion-{job.job_id}",
                daemon=True,
            ).start()
        return jsonify(job.to_dict()), 202

    @app.get("/v1/jobs/<job_id>")
    def get_job(job_id: str):
        denied = require_auth()
        if denied is not None:
            return denied
        job = store.get(job_id)
        if job is None:
            return _error("MINING-110", "Worker 作业不存在", 404)
        return jsonify(job.to_dict())

    @app.post("/v1/jobs/<job_id>/cancel")
    def cancel_job(job_id: str):
        denied = require_auth()
        if denied is not None:
            return denied
        job = store.get(job_id)
        if job is None:
            return _error("MINING-110", "Worker 作业不存在", 404)
        if job.status in {"succeeded", "failed", "cancelled", "interrupted"}:
            return jsonify(job.to_dict())
        with cancel_lock:
            event = cancel_events.get(job_id)
            if event is not None:
                event.set()
        payload = store.request(job_id)
        request_id = (
            str(payload.get("request_id", "")) if payload is not None else ""
        )
        if request_id:
            worker_generator.cancel(request_id)
            worker_search_generator.cancel(request_id)
            worker_search_turn_generator.cancel(request_id)
            worker_code_generator.cancel(request_id)
            worker_research_generator.cancel(request_id)
            worker_research_conversion_generator.cancel(request_id)
        cancelled = store.cancel(job_id, "调用方取消作业")
        log_worker_event(
            logger, "worker_job_cancel_requested", job_id=job_id
        )
        return jsonify(cancelled.to_dict() if cancelled else job.to_dict()), 202

    return app


def app_from_environment() -> Flask:
    """按容器环境创建正式 Worker 应用。"""

    return create_app()
