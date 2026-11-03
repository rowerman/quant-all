"""主服务 Worker 客户端与 S3 二次校验测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mining.errors import WorkerCancelledError, WorkerConfigurationError, WorkerResponseError
from mining.worker_client import (
    HttpWorkerClient,
    MockWorkerClient,
    WorkerRejection,
    ensure_worker_token,
    generate_worker_candidates,
)
from mining_worker.protocol import (
    WorkerCandidateDraft,
    WorkerGenerateRequest,
    WorkerGenerationResponse,
)


def make_request(**overrides):
    payload = {
        "request_id": "round-1",
        "run_id": "run_12345678",
        "round_index": 1,
        "hypothesis": "短期反转可能带来超额收益",
        "context_hash": "a" * 64,
        "field_catalog_hash": "b" * 64,
        "operator_catalog_hash": "c" * 64,
        "llm_model": "deepseek/deepseek-chat",
        "max_candidates": 2,
        "max_expression_depth": 6,
        "direction": "either",
        "excluded_fields": (),
        "fields": ({"name": "close"},),
        "operators": ({"name": "mean"},),
    }
    payload.update(overrides)
    return WorkerGenerateRequest(**payload)


def draft(expression="mean(close,5)", *, direction=1):
    return WorkerCandidateDraft(
        expression=expression,
        mechanism="短期价格趋势",
        description="收盘价的短期趋势",
        direction=direction,
        steps=("计算收盘价", "取短期均值"),
    )


def test_generate_candidates_revalidates_worker_expression():
    request = make_request()
    client = MockWorkerClient(
        WorkerGenerationResponse(
            candidates=(draft(), draft("-close", direction=-1)),
            usage={"calls": 1},
        )
    )

    result = generate_worker_candidates(client, request)

    assert [item.canonical_expression for item in result.accepted] == [
        "mean(close, 5)",
        "(-close)",
    ]
    assert result.rejected == ()
    assert result.accepted[0].name.startswith("MINE_")


def test_generate_candidates_rejects_invalid_sql_and_direction():
    request = make_request(direction="positive")
    client = MockWorkerClient(
        WorkerGenerationResponse(
            candidates=(
                draft("SELECT * FROM md.daily_hfq"),
                draft("-close", direction=-1),
            )
        )
    )

    result = generate_worker_candidates(client, request)

    assert result.accepted == ()
    assert len(result.rejected) == 2
    assert all(isinstance(item, WorkerRejection) for item in result.rejected)
    assert all(item.error_code == "MINING-114" for item in result.rejected)


def test_generate_candidates_enforces_request_budget():
    request = make_request(max_candidates=1)
    client = MockWorkerClient(
        WorkerGenerationResponse(candidates=(draft(), draft("-close", direction=-1)))
    )

    with pytest.raises(WorkerResponseError, match="超过预算"):
        generate_worker_candidates(client, request)


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, response):
        self.response = response

    def request(self, *args, **kwargs):
        return self.response


class _RecordingSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url))
        return _FakeResponse(
            200,
            {
                "job_id": "wjob_" + "b" * 24,
                "request_id": "round-1",
                "status": "cancelled",
                "result": None,
                "error_code": "MINING-117",
                "error_message": "cancelled",
            },
        )


def test_http_client_maps_auth_failure_and_job_terminal_state():
    client = HttpWorkerClient(
        "http://127.0.0.1:8790",
        "x" * 48,
        session=_FakeSession(
            _FakeResponse(401, {"error_code": "MINING-119", "error_message": "bad"})
        ),
    )
    with pytest.raises(WorkerConfigurationError, match="Token"):
        client.health()

    client = HttpWorkerClient(
        "http://127.0.0.1:8790",
        "x" * 48,
        session=_FakeSession(
            _FakeResponse(
                200,
                {
                    "job_id": "wjob_" + "a" * 24,
                    "request_id": "round-1",
                    "status": "cancelled",
                    "result": None,
                    "error_code": "MINING-117",
                    "error_message": "cancelled",
                },
            )
        ),
    )
    with pytest.raises(WorkerCancelledError):
        client.wait("wjob_" + "a" * 24)


def test_http_wait_cancels_worker_when_caller_requests_cancel():
    """主服务取消时先调用 Worker cancel，再抛出取消错误。"""

    session = _RecordingSession()
    client = HttpWorkerClient(
        "http://127.0.0.1:8790",
        "x" * 48,
        session=session,
    )
    with pytest.raises(WorkerCancelledError):
        client.wait("wjob_" + "b" * 24, should_cancel=lambda: True)
    assert session.calls == [
        ("POST", "http://127.0.0.1:8790/v1/jobs/wjob_" + "b" * 24 + "/cancel")
    ]


def test_worker_token_file_is_private(tmp_path: Path):
    target = tmp_path / "nested" / "token"
    token = ensure_worker_token(target)

    assert len(token) >= 32
    assert target.read_text(encoding="utf-8").strip() == token
    assert target.stat().st_mode & 0o077 == 0
