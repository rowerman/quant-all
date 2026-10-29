"""主服务侧 Worker v2 搜索客户端和候选二次校验。"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Mapping, Protocol

from mining.contract_candidates import CandidateDefinition
from mining.errors import (
    ExpressionValidationError,
    MiningWorkerError,
    WorkerCancelledError,
    WorkerResponseError,
    WorkerTimeoutError,
)
from mining.expression_validation import validate_expression
from mining.logging_conf import format_event, get_mining_logger
from mining.worker_client import HttpWorkerClient, WorkerRejection, _ERROR_TYPES
from mining_worker.protocol import WorkerJobView, WorkerProtocolError
from mining_worker.protocol_v2 import (
    WorkerDecisionProposal,
    WorkerSearchHypothesis,
    WorkerSearchRequest,
    WorkerSearchResponse,
)

_LOGGER = get_mining_logger("mining.search_worker.client")


@dataclass(frozen=True)
class NormalizedSearchWorkerResult:
    """Worker v2 响应经主服务二次校验后的结果。"""

    job_id: str
    decision: WorkerDecisionProposal
    next_hypothesis: WorkerSearchHypothesis | None
    accepted: tuple[CandidateDefinition, ...]
    rejected: tuple[WorkerRejection, ...]
    usage: Mapping[str, object]


class SearchWorkerClient(Protocol):
    """V2 搜索编排依赖的 Worker 作业接口。"""

    def health(self) -> Mapping[str, object]:
        """读取 Worker 健康状态。"""

    def submit(self, request: WorkerSearchRequest) -> WorkerJobView:
        """提交一轮搜索请求。"""

    def get_job(self, job_id: str) -> WorkerJobView:
        """读取搜索作业状态。"""

    def cancel(self, job_id: str) -> WorkerJobView:
        """请求取消搜索作业。"""


class HttpSearchWorkerClient(HttpWorkerClient):
    """通过主服务现有鉴权和连接配置访问 Worker v2。"""

    def submit(self, request: WorkerSearchRequest) -> WorkerJobView:
        """提交搜索作业。"""

        try:
            return WorkerJobView.from_dict(
                self._request("POST", "/v2/jobs", payload=request.to_dict())
            )
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Worker v2 作业响应非法：{exc}") from exc

    def get_job(self, job_id: str) -> WorkerJobView:
        """读取搜索作业。"""

        try:
            return WorkerJobView.from_dict(
                self._request("GET", f"/v1/jobs/{job_id}")
            )
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Worker v2 作业响应非法：{exc}") from exc

    def cancel(self, job_id: str) -> WorkerJobView:
        """取消搜索作业。"""

        try:
            return WorkerJobView.from_dict(
                self._request("POST", f"/v1/jobs/{job_id}/cancel")
            )
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Worker v2 作业响应非法：{exc}") from exc

    def wait(
        self,
        job_id: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> WorkerSearchResponse:
        """轮询到搜索作业终态并解析严格响应。"""

        deadline = time.monotonic() + self.request_timeout
        while True:
            if should_cancel is not None and should_cancel():
                try:
                    self.cancel(job_id)
                finally:
                    raise WorkerCancelledError("主服务已取消 Worker v2 作业")
            job = self.get_job(job_id)
            if job.status == "succeeded":
                if job.result is None:
                    raise WorkerResponseError("Worker v2 成功状态缺少结果")
                try:
                    return WorkerSearchResponse.from_dict(job.result)
                except WorkerProtocolError as exc:
                    raise WorkerResponseError(f"Worker v2 响应非法：{exc}") from exc
            if job.status in {"cancelled", "interrupted"}:
                raise WorkerCancelledError(
                    job.error_message or f"Worker v2 作业状态为 {job.status}"
                )
            if job.status == "failed":
                code = job.error_code or "MINING-103"
                error_type = _ERROR_TYPES.get(code, MiningWorkerError)
                raise error_type((job.error_message or "Worker v2 作业失败")[:500])
            if time.monotonic() >= deadline:
                try:
                    self.cancel(job_id)
                finally:
                    raise WorkerTimeoutError("等待 Worker v2 作业超时")
            time.sleep(self.poll_interval)


class MockSearchWorkerClient:
    """仅用于测试或显式注入的确定性搜索 Worker。"""

    def __init__(self, response: WorkerSearchResponse) -> None:
        self.response = response
        self.requests: list[WorkerSearchRequest] = []

    def health(self) -> Mapping[str, object]:
        return {
            "status": "ready",
            "mode": "mock",
            "protocol_versions": ["mining_worker_v2"],
        }

    def submit(self, request: WorkerSearchRequest) -> WorkerJobView:
        self.requests.append(request)
        return WorkerJobView(
            job_id=request.job_id,
            request_id=request.request_id,
            status="succeeded",
            result=self.response.to_dict(),
        )

    def get_job(self, job_id: str) -> WorkerJobView:
        for request in reversed(self.requests):
            if request.job_id == job_id:
                return self.submit_without_record(request)
        raise WorkerResponseError("Mock Search Worker 作业不存在")

    def submit_without_record(self, request: WorkerSearchRequest) -> WorkerJobView:
        return WorkerJobView(
            job_id=request.job_id,
            request_id=request.request_id,
            status="succeeded",
            result=self.response.to_dict(),
        )

    def cancel(self, job_id: str) -> WorkerJobView:
        job = self.get_job(job_id)
        return WorkerJobView(
            job_id=job.job_id,
            request_id=job.request_id,
            status="cancelled",
        )


def _draft_to_candidate(
    draft,
    request: WorkerSearchRequest,
    discovery_index: int,
) -> CandidateDefinition:
    validated = validate_expression(
        draft.expression,
        direction=draft.direction,
        max_depth=request.max_expression_depth,
        excluded_fields=request.excluded_fields,
    )
    return validated.to_candidate(
        mechanism=draft.mechanism,
        description=draft.description,
        steps=draft.steps,
        discovery_round=request.turn_index,
        discovery_index=discovery_index,
    )


def generate_search_candidates(
    client: SearchWorkerClient,
    request: WorkerSearchRequest,
    *,
    should_cancel: Callable[[], bool] | None = None,
    on_submitted: Callable[[WorkerJobView], None] | None = None,
) -> NormalizedSearchWorkerResult:
    """执行一轮搜索建议，并只保留通过主服务 DSL 校验的候选。"""

    job = client.submit(request)
    if on_submitted is not None:
        on_submitted(job)
    if isinstance(client, HttpSearchWorkerClient):
        response = client.wait(job.job_id, should_cancel=should_cancel)
    else:
        current = client.get_job(job.job_id)
        if current.status != "succeeded" or current.result is None:
            raise WorkerResponseError("Mock Search Worker 未返回成功结果")
        try:
            response = WorkerSearchResponse.from_dict(current.result)
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Mock Search Worker 响应非法：{exc}") from exc
    if response.decision.action not in request.allowed_actions:
        raise WorkerResponseError("Worker 返回了本轮不允许的搜索动作")
    if len(response.candidates) > request.max_candidates:
        raise WorkerResponseError(
            f"Worker 返回 {len(response.candidates)} 个候选，超过预算 "
            f"{request.max_candidates}"
        )
    accepted: list[CandidateDefinition] = []
    rejected: list[WorkerRejection] = []
    seen_content: set[str] = set()
    seen_semantic: set[str] = set()
    for index, draft in enumerate(response.candidates, start=1):
        try:
            candidate = _draft_to_candidate(draft, request, index)
            if (
                candidate.content_hash in seen_content
                or candidate.semantic_hash in seen_semantic
            ):
                raise ExpressionValidationError(
                    "本轮存在重复候选",
                    reason="duplicate",
                )
            seen_content.add(candidate.content_hash)
            seen_semantic.add(candidate.semantic_hash)
            accepted.append(candidate)
        except (ExpressionValidationError, ValueError) as exc:
            rejected.append(
                WorkerRejection(
                    index=index,
                    error_code=getattr(exc, "code", "MINING-114"),
                    message=str(exc)[:500],
                )
            )
    _LOGGER.info(
        format_event(
            {
                "event": "search_worker_candidates_validated",
                "job_id": job.job_id,
                "run_id": request.run_id,
                "turn_index": request.turn_index,
                "accepted": len(accepted),
                "rejected": len(rejected),
                "model": request.llm_model,
            }
        )
    )
    return NormalizedSearchWorkerResult(
        job_id=job.job_id,
        decision=response.decision,
        next_hypothesis=response.next_hypothesis,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        usage=response.usage,
    )
