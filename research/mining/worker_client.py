"""主服务侧 Worker 客户端及候选 S3 二次校验。"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import secrets
import time
from typing import Callable, Mapping, Protocol

import requests

from common.paths import DATA_DIR, MINING_WORKER_CONFIG_FILE
from mining.contract_candidates import CandidateDefinition
from mining.errors import (
    ExpressionValidationError,
    MiningWorkerError,
    WorkerCancelledError,
    WorkerConfigurationError,
    WorkerResponseError,
    WorkerTimeoutError,
)
from mining.expression_validation import validate_expression
from mining.logging_conf import format_event, get_mining_logger
from mining_worker.protocol import (
    WorkerCandidateDraft,
    WorkerGenerateRequest,
    WorkerGenerationResponse,
    WorkerJobView,
    WorkerProtocolError,
)
from mining_worker.worker_config import load_worker_config

_LOGGER = get_mining_logger("mining.worker.client")
_ERROR_TYPES = {
    "MINING-116": WorkerTimeoutError,
    "MINING-117": WorkerCancelledError,
    "MINING-118": WorkerResponseError,
    "MINING-119": WorkerConfigurationError,
}


@dataclass(frozen=True)
class WorkerRejection:
    """单个候选草稿未通过主服务校验的脱敏原因。"""

    index: int
    error_code: str
    message: str


@dataclass(frozen=True)
class NormalizedWorkerResult:
    """ Worker 响应经 S3 校验后的结果。"""

    job_id: str
    accepted: tuple[CandidateDefinition, ...]
    rejected: tuple[WorkerRejection, ...]
    usage: Mapping[str, object] = field(default_factory=dict)


class WorkerClient(Protocol):
    """主服务依赖的 Worker 作业接口。"""

    def health(self) -> Mapping[str, object]:
        """读取 Worker 健康状态。"""

    def submit(self, request: WorkerGenerateRequest) -> WorkerJobView:
        """提交一轮生成请求。"""

    def get_job(self, job_id: str) -> WorkerJobView:
        """读取作业状态。"""

    def cancel(self, job_id: str) -> WorkerJobView:
        """请求取消作业。"""


def ensure_worker_token(path: Path | str) -> str:
    """读取本地 Worker Token；不存在时生成仅当前用户可读的随机值。"""

    target = Path(path)
    if target.exists():
        token = target.read_text(encoding="utf-8").strip()
        if len(token) < 32:
            raise WorkerConfigurationError("Worker Token 文件内容无效")
        return token
    target.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(48)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = target.read_text(encoding="utf-8").strip()
        if len(existing) < 32:
            raise WorkerConfigurationError("Worker Token 文件内容无效")
        return existing
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(token + "\n")
    return token


class HttpWorkerClient:
    """通过仅本机 HTTP 作业接口调用独立 Worker。"""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        request_timeout: float | None = None,
        poll_interval: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if not base_url.strip():
            raise WorkerConfigurationError("MINING_WORKER_URL 不能为空")
        if len(token) < 32:
            raise WorkerConfigurationError("Worker Token 长度不足")
        config = load_worker_config(MINING_WORKER_CONFIG_FILE)
        request_timeout = (
            config.client_timeout_seconds
            if request_timeout is None
            else request_timeout
        )
        poll_interval = (
            config.poll_interval_seconds
            if poll_interval is None
            else poll_interval
        )
        if request_timeout <= 0 or poll_interval <= 0:
            raise WorkerConfigurationError("Worker 超时和轮询间隔必须大于 0")
        self.base_url = base_url.rstrip("/")
        self.request_timeout = float(request_timeout)
        self.poll_interval = float(poll_interval)
        self._headers = {"Authorization": f"Bearer {token}"}
        self._session = session or requests.Session()

    @classmethod
    def from_environment(cls) -> HttpWorkerClient:
        """从主服务环境构造客户端，不提供静默 Mock 回退。"""

        base_url = os.environ.get("MINING_WORKER_URL", "").strip()
        if not base_url:
            raise WorkerConfigurationError("未配置 MINING_WORKER_URL")
        token_path = Path(
            os.environ.get(
                "MINING_WORKER_TOKEN_FILE",
                DATA_DIR / "mining_worker" / "token",
            )
        )
        config = load_worker_config(MINING_WORKER_CONFIG_FILE)
        timeout = float(
            os.environ.get(
                "MINING_WORKER_TIMEOUT",
                config.client_timeout_seconds,
            )
        )
        return cls(base_url, ensure_worker_token(token_path), request_timeout=timeout)

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        try:
            response = self._session.request(
                method,
                f"{self.base_url}{path}",
                json=dict(payload) if payload is not None else None,
                headers=self._headers,
                timeout=10,
            )
        except requests.RequestException as exc:
            raise MiningWorkerError(f"Worker 不可达：{exc}") from exc
        if response.status_code in {401, 403}:
            raise WorkerConfigurationError("Worker 拒绝本地 Token")
        if response.status_code == 429:
            raise MiningWorkerError("Worker 已有作业正在运行")
        try:
            body = response.json()
        except ValueError as exc:
            raise WorkerResponseError("Worker 响应不是 JSON") from exc
        if not isinstance(body, dict):
            raise WorkerResponseError("Worker 响应根节点不是 JSON 对象")
        if response.status_code >= 400:
            code = str(body.get("error_code", "MINING-103"))
            message = str(body.get("error_message") or body.get("message") or "Worker 调用失败")
            error_type = _ERROR_TYPES.get(code, MiningWorkerError)
            raise error_type(message[:500])
        return body

    def health(self) -> Mapping[str, object]:
        """检查 Worker 是否完成模型和目录准备。"""

        body = self._request("GET", "/health")
        if body.get("status") != "ready":
            raise WorkerConfigurationError("Worker 尚未就绪")
        return body

    def submit(self, request: WorkerGenerateRequest) -> WorkerJobView:
        """提交作业并返回初始状态。"""

        try:
            return WorkerJobView.from_dict(
                self._request("POST", "/v1/jobs", payload=request.to_dict())
            )
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Worker 作业响应非法：{exc}") from exc

    def get_job(self, job_id: str) -> WorkerJobView:
        """读取作业状态。"""

        try:
            return WorkerJobView.from_dict(
                self._request("GET", f"/v1/jobs/{job_id}")
            )
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Worker 作业响应非法：{exc}") from exc

    def cancel(self, job_id: str) -> WorkerJobView:
        """请求取消作业。"""

        try:
            return WorkerJobView.from_dict(
                self._request("POST", f"/v1/jobs/{job_id}/cancel")
            )
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Worker 作业响应非法：{exc}") from exc

    def wait(
        self,
        job_id: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> WorkerGenerationResponse:
        """轮询到终态；客户端超时时主动取消 Worker 作业。"""

        deadline = time.monotonic() + self.request_timeout
        while True:
            if should_cancel is not None and should_cancel():
                try:
                    self.cancel(job_id)
                finally:
                    raise WorkerCancelledError("主服务已取消 Worker 作业")
            job = self.get_job(job_id)
            if job.status == "succeeded":
                if job.result is None:
                    raise WorkerResponseError("Worker 成功状态缺少结果")
                try:
                    return WorkerGenerationResponse.from_dict(job.result)
                except WorkerProtocolError as exc:
                    raise WorkerResponseError(f"Worker 候选响应非法：{exc}") from exc
            if job.status in {"cancelled", "interrupted"}:
                raise WorkerCancelledError(
                    job.error_message or f"Worker 作业状态为 {job.status}"
                )
            if job.status == "failed":
                code = job.error_code or "MINING-103"
                error_type = _ERROR_TYPES.get(code, MiningWorkerError)
                raise error_type((job.error_message or "Worker 作业失败")[:500])
            if time.monotonic() >= deadline:
                try:
                    self.cancel(job_id)
                finally:
                    raise WorkerTimeoutError("等待 Worker 作业超时")
            time.sleep(self.poll_interval)


class MockWorkerClient:
    """仅供测试或显式依赖注入的确定性 Worker。"""

    def __init__(self, response: WorkerGenerationResponse) -> None:
        self.response = response
        self.request: WorkerGenerateRequest | None = None

    def health(self) -> Mapping[str, object]:
        return {
            "status": "ready",
            "mode": "mock",
            "rdagent_version": "1.0.0",
        }

    def submit(self, request: WorkerGenerateRequest) -> WorkerJobView:
        self.request = request
        return WorkerJobView(
            job_id=request.job_id,
            request_id=request.request_id,
            status="succeeded",
            result=self.response.to_dict(),
        )

    def get_job(self, job_id: str) -> WorkerJobView:
        if self.request is None or self.request.job_id != job_id:
            raise WorkerResponseError("Mock Worker 作业不存在")
        return self.submit(self.request)

    def cancel(self, job_id: str) -> WorkerJobView:
        job = self.get_job(job_id)
        return WorkerJobView(
            job_id=job.job_id,
            request_id=job.request_id,
            status="cancelled",
        )


def _draft_to_candidate(
    draft: WorkerCandidateDraft,
    request: WorkerGenerateRequest,
    discovery_index: int,
) -> CandidateDefinition:
    desired = request.direction
    if desired == "positive" and draft.direction != 1:
        raise ExpressionValidationError("候选方向与冻结请求不一致", reason="direction")
    if desired == "negative" and draft.direction != -1:
        raise ExpressionValidationError("候选方向与冻结请求不一致", reason="direction")
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
        discovery_round=request.round_index,
        discovery_index=discovery_index,
    )


def generate_worker_candidates(
    client: WorkerClient,
    request: WorkerGenerateRequest,
    *,
    should_cancel: Callable[[], bool] | None = None,
    on_submitted: Callable[[WorkerJobView], None] | None = None,
) -> NormalizedWorkerResult:
    """执行一轮 Worker 生成，并只返回通过 S3 的候选。"""

    job = client.submit(request)
    if on_submitted is not None:
        on_submitted(job)
    if isinstance(client, HttpWorkerClient):
        response = client.wait(job.job_id, should_cancel=should_cancel)
    else:
        current = client.get_job(job.job_id)
        if current.status != "succeeded" or current.result is None:
            raise WorkerResponseError("Mock Worker 未返回成功结果")
        try:
            response = WorkerGenerationResponse.from_dict(current.result)
        except WorkerProtocolError as exc:
            raise WorkerResponseError(f"Mock Worker 响应非法：{exc}") from exc
    if len(response.candidates) > request.max_candidates:
        raise WorkerResponseError(
            f"Worker 返回 {len(response.candidates)} 个候选，超过预算 {request.max_candidates}"
        )
    accepted: list[CandidateDefinition] = []
    rejected: list[WorkerRejection] = []
    for index, draft in enumerate(response.candidates, start=1):
        try:
            accepted.append(_draft_to_candidate(draft, request, index))
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
                "event": "worker_candidates_validated",
                "job_id": job.job_id,
                "accepted": len(accepted),
                "rejected": len(rejected),
                "model": request.llm_model,
            }
        )
    )
    return NormalizedWorkerResult(
        job_id=job.job_id,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        usage=response.usage,
    )
