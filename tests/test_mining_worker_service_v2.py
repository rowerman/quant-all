"""Worker v2 HTTP 作业入口测试。"""

from __future__ import annotations

from pathlib import Path
import time

from mining_worker.protocol import WorkerCandidateDraft
from mining_worker.protocol_code import (
    WorkerCodeProposalRequest,
    WorkerCodeProposalResponse,
)
from mining_worker.protocol_v2 import (
    WorkerDecisionProposal,
    WorkerSearchHypothesis,
    WorkerSearchRequest,
    WorkerSearchResponse,
)
from mining_worker.protocol_v3 import (
    WorkerAnalysisToolCall,
    WorkerSearchTurnRequest,
    WorkerSearchTurnResponse,
)
from mining_worker.service import create_app

TOKEN = "v" * 48


def _request(request_id: str = "search-round-1") -> WorkerSearchRequest:
    return WorkerSearchRequest(
        request_id=request_id,
        run_id="searchrun_12345678",
        turn_index=1,
        branch_id="branch-0001",
        context_hash="a" * 64,
        hypothesis=WorkerSearchHypothesis(
            statement="短期价格反转可能带来开发期超额收益",
            mechanism="短期过度反应后的价格修复",
            expected_direction="positive",
            falsification_criteria=("年度同号率低于 0.6",),
        ),
        lineage=(),
        evidence={
            "schema_version": "mining_v2",
            "run_id": "searchrun_12345678",
            "hypothesis_id": "hyp_12345678",
            "batch_id": "batch_12345678",
            "turn_index": 1,
            "development_start": "2012-01-01",
            "development_end": "2019-12-31",
            "metrics": {},
            "failures": [],
            "correlation_clusters": [],
            "trial_counts": {
                "proposed": 1,
                "rejected": 0,
                "duplicate": 0,
                "invalid": 0,
                "gate_failed": 0,
                "evaluated": 1,
                "full_evaluated": 0,
                "holdout_claimed": 0,
                "invalidated": 0,
            },
            "budget_state": {"turns_used": 1},
            "created_at": "2026-10-29T00:00:00+00:00",
        },
        budget={
            "max_search_turns": 6,
            "max_hypothesis_branches": 3,
            "max_refines_per_branch": 3,
            "max_candidates_per_turn": 4,
            "max_candidates_per_run": 24,
            "max_full_evaluations": 4,
            "stop_after_no_improvement_turns": 2,
            "max_tool_calls_per_turn": 4,
            "schema_version": "mining_search_v2",
        },
        allowed_actions=("refine", "branch", "stop", "promote"),
        llm_model="deepseek/deepseek-chat",
        max_candidates=2,
        max_expression_depth=6,
        excluded_fields=(),
        fields=({"name": "ret"},),
        operators=({"name": "rank"},),
    )


RESPONSE = WorkerSearchResponse(
    decision=WorkerDecisionProposal(
        action="refine",
        reason="开发期信号仍然偏弱，需要缩短窗口",
    ),
    next_hypothesis=WorkerSearchHypothesis(
        statement="更短窗口的短期反转可能提高开发期信号",
        mechanism="更快的价格修复",
        expected_direction="positive",
        falsification_criteria=("连续两轮没有改善",),
    ),
    candidates=(
        WorkerCandidateDraft(
            expression="rank(mean(ret,3))",
            mechanism="短期反转",
            description="三日收益排名",
            direction=1,
            steps=("计算三日收益",),
        ),
    ),
    usage={"calls": 1},
)


class FakeSearchGenerator:
    def health(self):
        return {"status": "ready", "mode": "test"}

    def generate(self, request, *, timeout_seconds):
        return RESPONSE

    def cancel(self, request_id):
        return None


class FakeV1Generator(FakeSearchGenerator):
    def generate(self, request, *, timeout_seconds):
        raise AssertionError("v2 测试不应调用 v1 生成器")


class FakeSearchTurnGenerator:
    def health(self):
        return {"status": "ready", "mode": "test"}

    def generate(self, request, *, timeout_seconds):
        return WorkerSearchTurnResponse(
            request_id=request.request_id,
            kind="tool_batch",
            tool_calls=(
                WorkerAnalysisToolCall(
                    call_id="field-1",
                    tool_name="lookup_fields",
                    arguments={"query": "ret"},
                ),
            ),
        )

    def cancel(self, request_id):
        return None


class FakeCodeGenerator:
    def health(self):
        return {"status": "ready", "mode": "test"}

    def generate(self, request, *, timeout_seconds):
        return WorkerCodeProposalResponse(
            proposal_id="proposal_service",
            code="print('sandbox')",
            rationale="测试代码提案",
            model=request.llm_model,
        )

    def cancel(self, request_id):
        return None


def _headers():
    return {"Authorization": f"Bearer {TOKEN}"}


def _wait(client, job_id):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        payload = client.get(f"/v1/jobs/{job_id}", headers=_headers()).get_json()
        if payload["status"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("v2 作业未结束")


def test_v2_job_round_trip_is_idempotent_and_isolated_from_v1(tmp_path: Path):
    app = create_app(
        generator=FakeV1Generator(),
        search_generator=FakeSearchGenerator(),
        state_dir=tmp_path,
        token=TOKEN,
        job_timeout=1,
    )
    request = _request()
    with app.test_client() as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert "mining_worker_v2" in health.get_json()["protocol_versions"]

        first = client.post(
            "/v2/jobs",
            json=request.to_dict(),
            headers=_headers(),
        )
        assert first.status_code == 202
        duplicate = client.post(
            "/v2/jobs",
            json=request.to_dict(),
            headers=_headers(),
        )
        assert duplicate.status_code == 202
        payload = _wait(client, request.job_id)
        assert payload["status"] == "succeeded"
        assert payload["result"]["decision"]["action"] == "refine"
        assert payload["result"]["next_hypothesis"]["statement"].startswith(
            "更短窗口"
        )


def test_v2_job_rejects_private_window_before_persisting(tmp_path: Path):
    app = create_app(
        search_generator=FakeSearchGenerator(),
        state_dir=tmp_path,
        token=TOKEN,
    )
    payload = _request().to_dict()
    payload["evidence"]["development_start"] = "2020-01-01"
    with app.test_client() as client:
        response = client.post(
            "/v2/jobs",
            json=payload,
            headers=_headers(),
        )
    assert response.status_code == 400
    assert response.get_json()["error_code"] == "MINING-118"


def test_v3_job_returns_tool_batch_without_v1_or_v2_fallback(tmp_path: Path):
    app = create_app(
        generator=FakeV1Generator(),
        search_generator=FakeSearchGenerator(),
        search_turn_generator=FakeSearchTurnGenerator(),
        state_dir=tmp_path,
        token=TOKEN,
        job_timeout=1,
    )
    request = WorkerSearchTurnRequest(
        request_id="search-round-1.tools",
        phase="discover",
        continuation_index=0,
        base_request=_request(),
        tool_catalog=(
            {
                "name": "lookup_fields",
                "version": "1",
                "description": "字段查询",
                "required_arguments": ["query"],
                "optional_arguments": ["limit"],
                "maximum_result_bytes": 65536,
            },
        ),
    )
    with app.test_client() as client:
        health = client.get("/health")
        assert "mining_worker_v3" in health.get_json()["protocol_versions"]
        created = client.post(
            "/v3/jobs",
            json=request.to_dict(),
            headers=_headers(),
        )
        assert created.status_code == 202
        payload = _wait(client, request.job_id)

    assert payload["status"] == "succeeded"
    assert payload["result"]["kind"] == "tool_batch"
    assert payload["result"]["tool_calls"][0]["tool_name"] == "lookup_fields"


def test_v4_code_job_returns_strict_qlib_proposal(tmp_path: Path):
    app = create_app(
        generator=FakeV1Generator(),
        search_generator=FakeSearchGenerator(),
        code_generator=FakeCodeGenerator(),
        state_dir=tmp_path,
        token=TOKEN,
        job_timeout=1,
    )
    request = WorkerCodeProposalRequest(
        request_id="code-proposal-1",
        run_id="sandboxrun_12345678",
        iteration=1,
        hypothesis="短期价格变化与未来收益可能相关",
        snapshot={
            "context_hash": "a" * 64,
            "manifest_hash": "b" * 64,
            "start_date": "2012-01-01",
            "end_date": "2019-12-31",
            "fields": ["close", "label"],
            "image_id": "quant-mining-sandbox:s5",
        },
        previous_attempt=None,
        max_code_bytes=32768,
        llm_model="deepseek/deepseek-chat",
    )
    with app.test_client() as client:
        created = client.post(
            "/v4/jobs",
            json=request.to_dict(),
            headers=_headers(),
        )
        assert created.status_code == 202
        payload = _wait(client, request.job_id)

    assert payload["status"] == "succeeded"
    assert payload["result"]["proposal_id"] == "proposal_service"
    assert payload["result"]["model"] == request.llm_model
