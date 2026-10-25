"""Worker 作业服务、取消、并发和重启恢复测试。"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from mining_worker.job_store import JobStore
from mining_worker.generator import WorkerJobTimeout
from mining_worker.protocol import (
    WorkerCandidateDraft,
    WorkerGenerateRequest,
    WorkerGenerationResponse,
)
from mining_worker.service import create_app

TOKEN = "t" * 48


def make_request(request_id="round-1", *, hypothesis="短期反转可能带来超额收益"):
    return WorkerGenerateRequest(
        request_id=request_id,
        run_id="run_12345678",
        round_index=1,
        hypothesis=hypothesis,
        context_hash="a" * 64,
        field_catalog_hash="b" * 64,
        operator_catalog_hash="c" * 64,
        llm_model="deepseek/deepseek-chat",
        max_candidates=2,
        max_expression_depth=6,
        direction="either",
        excluded_fields=(),
        fields=({"name": "close"},),
        operators=({"name": "mean"},),
    )


RESPONSE = WorkerGenerationResponse(
    candidates=(
        WorkerCandidateDraft(
            expression="mean(close,5)",
            mechanism="短期价格趋势",
            description="收盘价短期均值",
            direction=1,
            steps=("计算短期均值",),
        ),
    ),
    usage={"calls": 1},
)


class FakeGenerator:
    def __init__(self, response=RESPONSE, block=False):
        self.response = response
        self.block = block
        self.release = threading.Event()
        self.cancelled = []

    def health(self):
        return {
            "status": "ready",
            "mode": "test",
            "rdagent_version": "1.0.0",
            "model": "deepseek/deepseek-chat",
        }

    def generate(self, request, *, timeout_seconds):
        if self.block:
            self.release.wait(timeout_seconds)
        return self.response

    def cancel(self, request_id):
        self.cancelled.append(request_id)
        self.release.set()


class TimeoutGenerator(FakeGenerator):
    def generate(self, request, *, timeout_seconds):
        raise WorkerJobTimeout("timeout")


def headers():
    return {"Authorization": f"Bearer {TOKEN}"}


def wait_for(client, job_id, statuses, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/v1/jobs/{job_id}", headers=headers())
        payload = response.get_json()
        if payload["status"] in statuses:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"作业未进入状态 {statuses}")


def test_health_auth_and_successful_job(tmp_path: Path):
    app = create_app(
        generator=FakeGenerator(),
        state_dir=tmp_path,
        token=TOKEN,
        job_timeout=1,
    )
    with app.test_client() as client:
        assert client.get("/health").status_code == 200
        assert client.post("/v1/jobs", json=make_request().to_dict()).status_code == 401
        created = client.post(
            "/v1/jobs",
            json=make_request().to_dict(),
            headers=headers(),
        )
        assert created.status_code == 202
        job_id = created.get_json()["job_id"]
        payload = wait_for(client, job_id, {"succeeded"})
        assert payload["result"]["candidates"][0]["expression"] == "mean(close,5)"


def test_duplicate_conflict_and_busy_slot(tmp_path: Path):
    generator = FakeGenerator(block=True)
    app = create_app(
        generator=generator,
        state_dir=tmp_path,
        token=TOKEN,
        job_timeout=2,
    )
    with app.test_client() as client:
        first = make_request()
        assert client.post("/v1/jobs", json=first.to_dict(), headers=headers()).status_code == 202
        duplicate = client.post("/v1/jobs", json=first.to_dict(), headers=headers())
        assert duplicate.status_code == 202

        changed = make_request(hypothesis="另一个满足最小长度的研究假设")
        conflicted = client.post("/v1/jobs", json=changed.to_dict(), headers=headers())
        assert conflicted.status_code == 409
        assert conflicted.get_json()["error_code"] == "MINING-105"

        other = make_request("round-2")
        busy = client.post("/v1/jobs", json=other.to_dict(), headers=headers())
        assert busy.status_code == 429
        assert busy.get_json()["error_code"] == "MINING-102"
        generator.release.set()
        wait_for(client, first.job_id, {"succeeded"})


def test_cancel_marks_job_and_calls_generator(tmp_path: Path):
    generator = FakeGenerator(block=True)
    app = create_app(
        generator=generator,
        state_dir=tmp_path,
        token=TOKEN,
        job_timeout=2,
    )
    request = make_request()
    with app.test_client() as client:
        client.post("/v1/jobs", json=request.to_dict(), headers=headers())
        wait_for(client, request.job_id, {"running"})
        cancelled = client.post(
            f"/v1/jobs/{request.job_id}/cancel",
            headers=headers(),
        )
        assert cancelled.status_code == 202
        assert cancelled.get_json()["status"] == "cancelled"
        assert generator.cancelled == [request.request_id]


def test_generator_timeout_is_recorded_as_mining_116(tmp_path: Path):
    app = create_app(
        generator=TimeoutGenerator(),
        state_dir=tmp_path,
        token=TOKEN,
        job_timeout=1,
    )
    request = make_request()
    with app.test_client() as client:
        created = client.post(
            "/v1/jobs",
            json=request.to_dict(),
            headers=headers(),
        )
        assert created.status_code == 202
        payload = wait_for(client, request.job_id, {"failed"})
        assert payload["error_code"] == "MINING-116"


def test_reopen_marks_unfinished_jobs_interrupted(tmp_path: Path):
    path = tmp_path / "jobs.sqlite"
    store = JobStore(path)
    request = make_request()
    view, created = store.create(
        job_id=request.job_id,
        request_id=request.request_id,
        payload_hash=request.payload_hash,
        request=request.to_dict(),
    )
    assert created
    store.mark_running(view.job_id)
    store.close()

    reopened = JobStore(path)
    payload = reopened.get(view.job_id)
    assert payload is not None
    assert payload.status == "interrupted"
    assert payload.error_code == "MINING-117"
    reopened.close()
