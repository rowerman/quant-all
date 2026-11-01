"""主服务侧 Worker v6 研报 DSL 转换客户端。"""

from __future__ import annotations

from typing import Protocol
import time

from mining.errors import (
    MiningWorkerError,
    WorkerCancelledError,
    WorkerResponseError,
    WorkerTimeoutError,
)
from mining.worker_client import HttpWorkerClient, _ERROR_TYPES
from mining_worker.protocol import WorkerJobView, WorkerProtocolError
from mining_worker.protocol_research_conversion import (
    WorkerResearchConversionRequest,
    WorkerResearchConversionResponse,
)


class ResearchFactorConverter(Protocol):
    """研报筛选服务依赖的同步 DSL 转换接口。"""

    def generate(
        self,
        request: WorkerResearchConversionRequest,
    ) -> WorkerResearchConversionResponse:
        """提交并等待一个转换分块完成。"""


class HttpResearchFactorConverter(HttpWorkerClient):
    """通过 Worker v6 作业接口执行研报 DSL 转换。"""

    def submit(self, request: WorkerResearchConversionRequest):
        """提交一个转换分块。"""

        try:
            return WorkerJobView.from_dict(
                self._request("POST", "/v6/jobs", payload=request.to_dict())
            )
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Worker v6 作业响应非法：{exc}") from exc

    def get_job(self, job_id: str):
        """读取 Worker 作业状态。"""

        try:
            return WorkerJobView.from_dict(
                self._request("GET", f"/v1/jobs/{job_id}")
            )
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Worker v6 作业响应非法：{exc}") from exc

    def cancel(self, job_id: str):
        """取消 Worker v6 作业。"""

        try:
            return WorkerJobView.from_dict(
                self._request("POST", f"/v1/jobs/{job_id}/cancel")
            )
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Worker v6 作业响应非法：{exc}") from exc

    def wait(self, job_id: str) -> WorkerResearchConversionResponse:
        """轮询到转换作业终态并解析严格响应。"""

        deadline = time.monotonic() + self.request_timeout
        while True:
            job = self.get_job(job_id)
            if job.status == "succeeded":
                if job.result is None:
                    raise WorkerResponseError("Worker v6 成功状态缺少结果")
                try:
                    response = WorkerResearchConversionResponse.from_dict(
                        job.result
                    )
                except WorkerProtocolError as exc:
                    raise WorkerResponseError(
                        f"Worker v6 研报转换响应非法：{exc}"
                    ) from exc
                if response.request_id != job.request_id:
                    raise WorkerResponseError(
                        "Worker v6 响应 request_id 与作业请求不一致"
                    )
                return response
            if job.status in {"cancelled", "interrupted"}:
                raise WorkerCancelledError(
                    job.error_message or f"Worker v6 作业状态为 {job.status}"
                )
            if job.status == "failed":
                code = job.error_code or "MINING-103"
                error_type = _ERROR_TYPES.get(code, MiningWorkerError)
                raise error_type(
                    (job.error_message or "Worker v6 研报转换失败")[:500]
                )
            if time.monotonic() >= deadline:
                try:
                    self.cancel(job_id)
                finally:
                    raise WorkerTimeoutError("等待 Worker v6 研报转换超时")
            time.sleep(self.poll_interval)

    def generate(
        self,
        request: WorkerResearchConversionRequest,
    ) -> WorkerResearchConversionResponse:
        """提交并等待一个转换分块。"""

        job = self.submit(request)
        return self.wait(job.job_id)


__all__ = ["HttpResearchFactorConverter", "ResearchFactorConverter"]
