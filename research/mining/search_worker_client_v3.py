"""工具感知的 Worker v3 客户端与一次工具补充回合。"""

from __future__ import annotations

import time
from typing import Callable, Mapping, Protocol

from mining.analysis_tool_contracts import AnalysisToolRequest
from mining.analysis_tool_gateway import AnalysisToolGateway
from mining.errors import WorkerCancelledError, WorkerResponseError, WorkerTimeoutError
from mining.search_worker_client import (
    HttpSearchWorkerClient,
    NormalizedSearchWorkerResult,
    normalize_search_response,
)
from mining_worker.protocol import WorkerJobView, WorkerProtocolError
from mining_worker.protocol_v2 import WorkerSearchRequest, WorkerSearchResponse
from mining_worker.protocol_v3 import (
    WorkerSearchTurnRequest,
    WorkerSearchTurnResponse,
)


class SearchTurnWorkerClient(Protocol):
    """Worker v3 作业接口。"""

    def health(self) -> Mapping[str, object]:
        """读取健康状态。"""

    def submit(self, request: WorkerSearchTurnRequest) -> WorkerJobView:
        """提交工具感知搜索作业。"""

    def get_job(self, job_id: str) -> WorkerJobView:
        """读取作业。"""

    def cancel(self, job_id: str) -> WorkerJobView:
        """取消作业。"""


class HttpSearchTurnWorkerClient(HttpSearchWorkerClient):
    """通过 Worker v3 接口执行 discover/finalize 两步调用。"""

    def submit(self, request: WorkerSearchTurnRequest) -> WorkerJobView:
        """提交工具感知搜索请求。"""

        try:
            return WorkerJobView.from_dict(
                self._request("POST", "/v3/jobs", payload=request.to_dict())
            )
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Worker v3 作业响应非法：{exc}") from exc

    def wait(
        self,
        job_id: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> WorkerSearchTurnResponse:
        """轮询并解析 Worker v3 响应。"""

        deadline = time.monotonic() + self.request_timeout
        while True:
            if should_cancel is not None and should_cancel():
                try:
                    self.cancel(job_id)
                finally:
                    raise WorkerCancelledError("主服务已取消 Worker v3 作业")
            job = self.get_job(job_id)
            if job.status == "succeeded":
                if job.result is None:
                    raise WorkerResponseError("Worker v3 成功状态缺少结果")
                try:
                    return WorkerSearchTurnResponse.from_dict(job.result)
                except WorkerProtocolError as exc:
                    raise WorkerResponseError(
                        f"Worker v3 响应非法：{exc}"
                    ) from exc
            if job.status in {"cancelled", "interrupted"}:
                raise WorkerCancelledError(
                    job.error_message or f"Worker v3 作业状态为 {job.status}"
                )
            if job.status == "failed":
                raise WorkerResponseError(
                    (job.error_message or "Worker v3 作业失败")[:500]
                )
            if time.monotonic() >= deadline:
                try:
                    self.cancel(job_id)
                finally:
                    raise WorkerTimeoutError("等待 Worker v3 作业超时")
            time.sleep(self.poll_interval)


class MockSearchTurnWorkerClient:
    """按请求阶段返回确定性工具批次或最终决策。"""

    def __init__(
        self,
        *,
        tool_response: WorkerSearchTurnResponse,
        final_response: WorkerSearchTurnResponse,
    ) -> None:
        self.tool_response = tool_response
        self.final_response = final_response
        self.requests: list[WorkerSearchTurnRequest] = []

    def health(self) -> Mapping[str, object]:
        return {
            "status": "ready",
            "mode": "mock",
            "protocol_versions": ["mining_worker_v3"],
        }

    def submit(self, request: WorkerSearchTurnRequest) -> WorkerJobView:
        self.requests.append(request)
        response = (
            self.tool_response
            if request.phase == "discover"
            else self.final_response
        )
        return WorkerJobView(
            job_id=request.job_id,
            request_id=request.request_id,
            status="succeeded",
            result=response.to_dict(),
        )

    def get_job(self, job_id: str) -> WorkerJobView:
        for request in reversed(self.requests):
            if request.job_id == job_id:
                return self.submit_without_record(request)
        raise WorkerResponseError("Mock Worker v3 作业不存在")

    def submit_without_record(
        self,
        request: WorkerSearchTurnRequest,
    ) -> WorkerJobView:
        response = (
            self.tool_response
            if request.phase == "discover"
            else self.final_response
        )
        return WorkerJobView(
            job_id=request.job_id,
            request_id=request.request_id,
            status="succeeded",
            result=response.to_dict(),
        )

    def cancel(self, job_id: str) -> WorkerJobView:
        job = self.get_job(job_id)
        return WorkerJobView(
            job_id=job.job_id,
            request_id=job.request_id,
            status="cancelled",
        )


def _wait_response(
    client: SearchTurnWorkerClient,
    job_id: str,
    *,
    should_cancel: Callable[[], bool] | None,
) -> WorkerSearchTurnResponse:
    if isinstance(client, HttpSearchTurnWorkerClient):
        return client.wait(job_id, should_cancel=should_cancel)
    current = client.get_job(job_id)
    if current.status != "succeeded" or current.result is None:
        raise WorkerResponseError("Mock Worker v3 未返回成功结果")
    try:
        return WorkerSearchTurnResponse.from_dict(current.result)
    except WorkerProtocolError as exc:
        raise WorkerResponseError("Mock Worker v3 响应非法") from exc


def wait_search_turn_response(
    client: SearchTurnWorkerClient,
    job_id: str,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> WorkerSearchTurnResponse:
    """等待一个已提交的 Worker v3 作业。"""

    return _wait_response(client, job_id, should_cancel=should_cancel)


def resume_search_candidates_with_tools(
    client: SearchTurnWorkerClient,
    request: WorkerSearchRequest,
    gateway: AnalysisToolGateway,
    *,
    context,
    discover_job_id: str,
    should_cancel: Callable[[], bool] | None = None,
    on_submitted: Callable[[WorkerJobView, str], None] | None = None,
) -> NormalizedSearchWorkerResult:
    """从已提交的 discover 作业恢复工具补充回合。"""

    first = _wait_response(
        client,
        discover_job_id,
        should_cancel=should_cancel,
    )
    if first.kind == "decision":
        assert first.response is not None
        return normalize_search_response(request, first.response, discover_job_id)
    tool_requests = tuple(
        AnalysisToolRequest(
            call_id=item.call_id,
            tool_name=item.tool_name,
            arguments=item.arguments,
        )
        for item in first.tool_calls
    )
    results = gateway.execute_batch(
        run_id=request.run_id,
        context=context,
        turn_index=request.turn_index,
        turn_request_hash=request.payload_hash,
        requests=tool_requests,
        should_cancel=should_cancel,
    )
    finalize = WorkerSearchTurnRequest(
        request_id=f"{request.request_id}.final",
        phase="finalize",
        continuation_index=1,
        base_request=request,
        tool_catalog=gateway.catalog(),
        tool_results=tuple(item.to_dict() for item in results),
    )
    final_job = client.submit(finalize)
    if on_submitted is not None:
        on_submitted(final_job, "finalize")
    final = _wait_response(
        client,
        final_job.job_id,
        should_cancel=should_cancel,
    )
    if final.kind != "decision" or final.response is None:
        raise WorkerResponseError("Worker v3 finalize 阶段仍请求工具")
    return normalize_search_response(request, final.response, final_job.job_id)


def generate_search_candidates_with_tools(
    client: SearchTurnWorkerClient,
    request: WorkerSearchRequest,
    gateway: AnalysisToolGateway,
    *,
    context,
    should_cancel: Callable[[], bool] | None = None,
    on_submitted: Callable[[WorkerJobView, str], None] | None = None,
) -> NormalizedSearchWorkerResult:
    """执行至多一次工具补充，再校验最终候选和决策。"""

    discover = WorkerSearchTurnRequest(
        request_id=f"{request.request_id}.tools",
        phase="discover",
        continuation_index=0,
        base_request=request,
        tool_catalog=gateway.catalog(),
    )
    job = client.submit(discover)
    if on_submitted is not None:
        on_submitted(job, "discover")
    first = _wait_response(
        client,
        job.job_id,
        should_cancel=should_cancel,
    )
    if first.kind == "decision":
        assert first.response is not None
        return normalize_search_response(request, first.response, job.job_id)
    tool_requests = tuple(
        AnalysisToolRequest(
            call_id=item.call_id,
            tool_name=item.tool_name,
            arguments=item.arguments,
        )
        for item in first.tool_calls
    )
    results = gateway.execute_batch(
        run_id=request.run_id,
        context=context,
        turn_index=request.turn_index,
        turn_request_hash=request.payload_hash,
        requests=tool_requests,
        should_cancel=should_cancel,
    )
    finalize = WorkerSearchTurnRequest(
        request_id=f"{request.request_id}.final",
        phase="finalize",
        continuation_index=1,
        base_request=request,
        tool_catalog=gateway.catalog(),
        tool_results=tuple(item.to_dict() for item in results),
    )
    final_job = client.submit(finalize)
    if on_submitted is not None:
        on_submitted(final_job, "finalize")
    final = _wait_response(
        client,
        final_job.job_id,
        should_cancel=should_cancel,
    )
    if final.kind != "decision" or final.response is None:
        raise WorkerResponseError("Worker v3 finalize 阶段仍请求工具")
    return normalize_search_response(request, final.response, final_job.job_id)


__all__ = [
    "HttpSearchTurnWorkerClient",
    "MockSearchTurnWorkerClient",
    "SearchTurnWorkerClient",
    "generate_search_candidates_with_tools",
    "resume_search_candidates_with_tools",
    "wait_search_turn_response",
]
