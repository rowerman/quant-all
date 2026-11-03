"""Worker 纯协议契约测试。"""

from __future__ import annotations

import pytest

from mining_worker.protocol import (
    WorkerCandidateDraft,
    WorkerGenerateRequest,
    WorkerGenerationResponse,
    WorkerProtocolError,
)
from mining_worker.worker_config import load_worker_config


def request_payload(**overrides):
    payload = {
        "schema_version": "mining_worker_v1",
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
        "excluded_fields": [],
        "fields": [{"name": "close", "dtype": "number"}],
        "operators": [{"name": "mean", "arity": 2}],
        "feedback": [],
    }
    payload.update(overrides)
    return payload


def test_worker_request_round_trip_and_stable_job_id():
    request = WorkerGenerateRequest.from_dict(request_payload())

    assert WorkerGenerateRequest.from_dict(request.to_dict()) == request
    assert request.job_id == WorkerGenerateRequest.from_dict(request.to_dict()).job_id
    assert request.payload_hash == WorkerGenerateRequest.from_dict(request.to_dict()).payload_hash


def test_worker_request_rejects_unknown_or_unsafe_input():
    payload = request_payload()
    payload["unknown"] = True
    with pytest.raises(WorkerProtocolError, match="请求字段不匹配"):
        WorkerGenerateRequest.from_dict(payload)

    with pytest.raises(WorkerProtocolError, match="fields"):
        WorkerGenerateRequest.from_dict(request_payload(fields=[]))
    with pytest.raises(WorkerProtocolError, match="max_candidates"):
        WorkerGenerateRequest.from_dict(request_payload(max_candidates=9))


def test_worker_candidate_rejects_sql_and_requires_steps():
    with pytest.raises(WorkerProtocolError, match="steps"):
        WorkerCandidateDraft.from_dict(
            {
                "expression": "SELECT * FROM md.daily_hfq",
                "mechanism": "非法 SQL",
                "description": "不应执行",
                "direction": 1,
                "steps": [],
            }
        )


def test_worker_response_is_strict_and_non_empty():
    candidate = {
        "expression": "mean(close,5)",
        "mechanism": "短期价格趋势",
        "description": "五日均价",
        "direction": 1,
        "steps": ["取五日收盘均价"],
    }
    response = WorkerGenerationResponse.from_dict(
        {
            "schema_version": "mining_worker_v1",
            "candidates": [candidate],
            "usage": {"calls": 1},
        }
    )
    assert response.candidates[0].expression == "mean(close,5)"

    with pytest.raises(WorkerProtocolError, match="没有返回候选"):
        WorkerGenerationResponse.from_dict(
            {
                "schema_version": "mining_worker_v1",
                "candidates": [],
                "usage": {},
            }
        )


def test_worker_config_pins_versions_and_timeouts():
    config = load_worker_config()

    assert config.rdagent_version == "1.0.0"
    assert config.python_version == "3.11"
    assert config.job_timeout_seconds <= config.client_timeout_seconds
