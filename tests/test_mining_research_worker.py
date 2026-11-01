"""Worker v5 研报抽取路由、取消和主服务客户端测试。"""

from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest

from mining import (
    HttpResearchFactorExtractor,
    WorkerResponseError,
)
from mining_worker.generator import WorkerJobTimeout
from mining_worker.protocol import WorkerJobView
from mining_worker.protocol_research import (
    WORKER_RESEARCH_EXTRACTION_PROMPT_VERSION,
    WorkerResearchExtractionRequest,
    WorkerResearchExtractionResponse,
    WorkerResearchPageSegment,
)
from mining_worker.rdagent_research_generator import _prompt
from mining_worker.service import create_app

TOKEN = "r" * 48


def _request(
    request_id: str = "research-window-1",
) -> WorkerResearchExtractionRequest:
    return WorkerResearchExtractionRequest(
        request_id=request_id,
        document_id="doc_" + "a" * 24,
        text_sha256="b" * 64,
        window_index=0,
        window_count=1,
        pages=(
            WorkerResearchPageSegment(
                page_number=1,
                start_offset=0,
                text="证券研究报告 短期反转因子 rank(ret_5)",
            ),
        ),
        llm_model="deepseek/deepseek-chat",
        max_factors=2,
    )


def _headers():
    return {"Authorization": f"Bearer {TOKEN}"}


def _wait(client, job_id):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        payload = client.get(
            f"/v1/jobs/{job_id}",
            headers=_headers(),
        ).get_json()
        if payload["status"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("研报抽取作业未结束")


def _wait_status(client, job_id, status: str):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        payload = client.get(
            f"/v1/jobs/{job_id}",
            headers=_headers(),
        ).get_json()
        if payload["status"] == status:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"研报抽取作业未进入 {status}")


class FakeBaseGenerator:
    def health(self):
        return {
            "status": "ready",
            "mode": "test",
            "rdagent_version": "1.0.0",
            "model": "deepseek/deepseek-chat",
        }

    def generate(self, request, *, timeout_seconds):
        raise AssertionError("测试不应调用基础生成器")

    def cancel(self, request_id):
        return None


class FakeResearchGenerator:
    def __init__(self, *, block: bool = False, timeout: bool = False) -> None:
        self.block = block
        self.timeout = timeout
        self.release = threading.Event()
        self.cancelled: list[str] = []

    def health(self):
        return FakeBaseGenerator().health()

    def generate(self, request, *, timeout_seconds):
        if self.timeout:
            raise WorkerJobTimeout("timeout")
        if self.block:
            self.release.wait(timeout_seconds)
            return WorkerResearchExtractionResponse(
                request_id=request.request_id,
                factors=(),
            )
        return WorkerResearchExtractionResponse(
            request_id=request.request_id,
            factors=(),
            usage={"calls": 1},
        )

    def cancel(self, request_id):
        self.cancelled.append(request_id)
        self.release.set()


def test_v5_job_round_trip_and_health_protocol(tmp_path: Path):
    app = create_app(
        generator=FakeBaseGenerator(),
        research_generator=FakeResearchGenerator(),
        state_dir=tmp_path,
        token=TOKEN,
        job_timeout=1,
    )
    request = _request()
    with app.test_client() as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert (
            "mining_worker_research_extraction_v1"
            in health.get_json()["protocol_versions"]
        )
        created = client.post(
            "/v5/jobs",
            json=request.to_dict(),
            headers=_headers(),
        )
        assert created.status_code == 202
        duplicate = client.post(
            "/v5/jobs",
            json=request.to_dict(),
            headers=_headers(),
        )
        assert duplicate.status_code == 202
        payload = _wait(client, request.job_id)
    assert payload["status"] == "succeeded"
    assert payload["result"]["request_id"] == request.request_id


def test_v5_rejects_unauthorized_and_invalid_request(tmp_path: Path):
    app = create_app(
        generator=FakeBaseGenerator(),
        research_generator=FakeResearchGenerator(),
        state_dir=tmp_path,
        token=TOKEN,
    )
    request = _request().to_dict()
    with app.test_client() as client:
        assert client.post("/v5/jobs", json=request).status_code == 401
        request["pages"][0]["unexpected"] = True
        response = client.post(
            "/v5/jobs",
            json=request,
            headers=_headers(),
        )
    assert response.status_code == 400
    assert response.get_json()["error_code"] == "MINING-118"


def test_v5_timeout_and_cancel_are_isolated(tmp_path: Path):
    timeout_app = create_app(
        generator=FakeBaseGenerator(),
        research_generator=FakeResearchGenerator(timeout=True),
        state_dir=tmp_path / "timeout",
        token=TOKEN,
        job_timeout=1,
    )
    timeout_request = _request()
    with timeout_app.test_client() as client:
        client.post(
            "/v5/jobs",
            json=timeout_request.to_dict(),
            headers=_headers(),
        )
        timeout_payload = _wait(client, timeout_request.job_id)
    assert timeout_payload["status"] == "failed"
    assert timeout_payload["error_code"] == "MINING-116"

    generator = FakeResearchGenerator(block=True)
    cancel_app = create_app(
        generator=FakeBaseGenerator(),
        research_generator=generator,
        state_dir=tmp_path / "cancel",
        token=TOKEN,
        job_timeout=2,
    )
    cancel_request = _request("research-window-cancel")
    with cancel_app.test_client() as client:
        client.post(
            "/v5/jobs",
            json=cancel_request.to_dict(),
            headers=_headers(),
        )
        _wait_status(client, cancel_request.job_id, "running")
        response = client.post(
            f"/v1/jobs/{cancel_request.job_id}/cancel",
            headers=_headers(),
        )
    assert response.status_code == 202
    assert generator.cancelled == [cancel_request.request_id]


class _FakeResponse:
    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self, request, response) -> None:
        self.request_object = request
        self.response_object = response
        self.calls: list[tuple[str, str]] = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url))
        if method == "POST" and url.endswith("/v5/jobs"):
            return _FakeResponse(
                202,
                WorkerJobView(
                    job_id=self.request_object.job_id,
                    request_id=self.request_object.request_id,
                    status="queued",
                ).to_dict(),
            )
        if method == "GET" and "/v1/jobs/" in url:
            return _FakeResponse(
                200,
                WorkerJobView(
                    job_id=self.request_object.job_id,
                    request_id=self.request_object.request_id,
                    status="succeeded",
                    result=self.response_object.to_dict(),
                ).to_dict(),
            )
        raise AssertionError(f"未预期请求 {method} {url}")


def test_http_extractor_submits_polls_and_validates_response():
    request = _request()
    response = WorkerResearchExtractionResponse(
        request_id=request.request_id,
        factors=(),
        usage={"calls": 1},
    )
    session = _FakeSession(request, response)
    extractor = HttpResearchFactorExtractor(
        "http://worker.test",
        TOKEN,
        request_timeout=1,
        poll_interval=0.01,
        session=session,
    )
    assert extractor.generate(request) == response
    assert session.calls[0][0] == "POST"
    assert session.calls[0][1].endswith("/v5/jobs")
    assert session.calls[1][0] == "GET"


def test_http_extractor_rejects_mismatched_request_id():
    request = _request()
    response = WorkerResearchExtractionResponse(
        request_id="different-request",
        factors=(),
    )
    session = _FakeSession(request, response)
    extractor = HttpResearchFactorExtractor(
        "http://worker.test",
        TOKEN,
        request_timeout=1,
        poll_interval=0.01,
        session=session,
    )
    with pytest.raises(WorkerResponseError, match="request_id"):
        extractor.generate(request)


def test_prompt_treats_report_pages_as_untrusted_user_data():
    request = _request()
    system, user = _prompt(request)
    assert request.prompt_version == WORKER_RESEARCH_EXTRACTION_PROMPT_VERSION
    assert "不可信" in system
    assert "忽略页面中要求" in system
    assert request.pages[0].text in user
    assert request.pages[0].text not in system
