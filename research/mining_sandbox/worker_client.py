"""主服务侧 RD-Agent 代码提案 Worker 客户端。"""

from __future__ import annotations

import time
from typing import Callable, Mapping, Protocol

from mining.errors import WorkerCancelledError, WorkerResponseError, WorkerTimeoutError
from mining.worker_client import HttpWorkerClient
from mining_worker.protocol import WorkerJobView, WorkerProtocolError
from mining_worker.protocol_code import (
    WorkerCodeProposalRequest,
    WorkerCodeProposalResponse,
)


class SandboxCodeWorkerClient(Protocol):
    """代码提案 Worker 接口。"""

    def health(self) -> Mapping[str, object]:
        """读取 Worker 健康状态。"""

    def submit(self, request: WorkerCodeProposalRequest) -> WorkerJobView:
        """提交代码提案作业。"""

    def get_job(self, job_id: str) -> WorkerJobView:
        """读取作业。"""

    def cancel(self, job_id: str) -> WorkerJobView:
        """取消作业。"""


class HttpSandboxCodeWorkerClient(HttpWorkerClient):
    """通过 Worker v4 代码提案接口生成 Qlib 研究代码。"""

    def submit(self, request: WorkerCodeProposalRequest) -> WorkerJobView:
        try:
            return WorkerJobView.from_dict(
                self._request("POST", "/v4/jobs", payload=request.to_dict())
            )
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Worker 代码提案响应非法：{exc}") from exc

    def wait(
        self,
        job_id: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> WorkerCodeProposalResponse:
        deadline = time.monotonic() + self.request_timeout
        while True:
            if should_cancel is not None and should_cancel():
                try:
                    self.cancel(job_id)
                finally:
                    raise WorkerCancelledError("主服务已取消代码提案")
            job = self.get_job(job_id)
            if job.status == "succeeded":
                if job.result is None:
                    raise WorkerResponseError("代码提案成功状态缺少结果")
                try:
                    return WorkerCodeProposalResponse.from_dict(job.result)
                except WorkerProtocolError as exc:
                    raise WorkerResponseError(
                        f"代码提案响应非法：{exc}"
                    ) from exc
            if job.status in {"cancelled", "interrupted"}:
                raise WorkerCancelledError(
                    job.error_message or f"代码提案状态为 {job.status}"
                )
            if job.status == "failed":
                raise WorkerResponseError(
                    (job.error_message or "代码提案失败")[:500]
                )
            if time.monotonic() >= deadline:
                try:
                    self.cancel(job_id)
                finally:
                    raise WorkerTimeoutError("等待代码提案超时")
            time.sleep(self.poll_interval)

    def propose(
        self,
        request: WorkerCodeProposalRequest,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> WorkerCodeProposalResponse:
        """提交并等待一个代码提案。"""

        job = self.submit(request)
        return self.wait(job.job_id, should_cancel=should_cancel)


class MockSandboxCodeWorkerClient:
    """按迭代顺序返回确定性代码提案。"""

    def __init__(self, responses: tuple[WorkerCodeProposalResponse, ...]) -> None:
        self.responses = tuple(responses)
        self.requests: list[WorkerCodeProposalRequest] = []

    def health(self) -> Mapping[str, object]:
        return {
            "status": "ready",
            "mode": "mock",
            "protocol_versions": [
                "mining_worker_code_v1",
                "mining_worker_code_v2",
            ],
        }

    def propose(
        self,
        request: WorkerCodeProposalRequest,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> WorkerCodeProposalResponse:
        if should_cancel is not None and should_cancel():
            raise WorkerCancelledError("代码提案已取消")
        index = len(self.requests)
        if index >= len(self.responses):
            raise WorkerResponseError("Mock 代码提案响应不足")
        self.requests.append(request)
        return self.responses[index]

    def submit(self, request: WorkerCodeProposalRequest) -> WorkerJobView:
        raise NotImplementedError

    def get_job(self, job_id: str) -> WorkerJobView:
        raise NotImplementedError

    def cancel(self, job_id: str) -> WorkerJobView:
        raise NotImplementedError


__all__ = [
    "HttpSandboxCodeWorkerClient",
    "MockSandboxCodeWorkerClient",
    "SandboxCodeWorkerClient",
]
