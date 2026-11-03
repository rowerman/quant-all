"""Worker v4 代码提案与知识证据协议测试。"""

from __future__ import annotations

import json

import pytest

from mining.contract_types import sha256_json
from mining_worker.protocol import WorkerProtocolError
from mining_worker.protocol_code import (
    CODE_PROMPT_VERSION,
    LEGACY_CODE_PROMPT_VERSION,
    LEGACY_WORKER_CODE_SCHEMA_VERSION,
    WORKER_CODE_SCHEMA_VERSION,
    WorkerCodeFeedbackEvidence,
    WorkerCodeKnowledgeEntry,
    WorkerCodeKnowledgeEvidence,
    WorkerCodeProposalRequest,
    WorkerCodeProposalResponse,
)
from mining_worker.rdagent_code_generator import _prompt


def _feedback() -> WorkerCodeFeedbackEvidence:
    payload = {
        "schema_version": "mining_sandbox_feedback_v1",
        "error_code": "MINING-139",
        "category": "execution",
        "stage": "execution",
        "summary": "沙箱执行以非零状态退出",
        "repair_hint": "修复运行时异常",
        "retryable": True,
    }
    return WorkerCodeFeedbackEvidence(
        **payload,
        feedback_hash=sha256_json(payload),
    )


def _failure_entry() -> WorkerCodeKnowledgeEntry:
    return WorkerCodeKnowledgeEntry(
        kind="failure",
        iteration=1,
        source_iterations=(1,),
        code_hash="a" * 64,
        feedback=_feedback(),
        metrics=None,
        evaluation_hash=None,
        summary="第 1 轮失败：MINING-139（execution）",
        entry_hash="b" * 64,
    )


def _evidence() -> WorkerCodeKnowledgeEvidence:
    return WorkerCodeKnowledgeEvidence(
        task_hash="c" * 64,
        index_hash="d" * 64,
        retrieval_hash="e" * 64,
        entries=(_failure_entry(),),
    )


def _request() -> WorkerCodeProposalRequest:
    return WorkerCodeProposalRequest(
        request_id="sandboxrun_protocol:code:1",
        run_id="sandboxrun_protocol",
        iteration=1,
        hypothesis="短期价格修复可能形成可检验因子",
        snapshot={
            "context_hash": "f" * 64,
            "manifest_hash": "a" * 64,
            "start_date": "2012-01-01",
            "end_date": "2019-12-31",
            "fields": ["close", "label"],
            "image_id": "quant-mining-sandbox:s5",
        },
        previous_attempt=None,
        max_code_bytes=32768,
        llm_model="deepseek/deepseek-chat",
        task_hash="c" * 64,
    )


def _previous_attempt() -> dict:
    feedback = _feedback()
    return {
        "iteration": 1,
        "code_hash": "a" * 64,
        "status": "failed",
        "metrics": None,
        "error_code": feedback.error_code,
        "feedback": feedback.to_dict(),
        "feedback_hash": feedback.feedback_hash,
    }


def test_v2_request_and_knowledge_evidence_round_trip():
    request = WorkerCodeProposalRequest(
        **_request().to_dict()
        | {
            "iteration": 2,
            "previous_attempt": _previous_attempt(),
            "knowledge_evidence": _evidence(),
        }
    )

    parsed = WorkerCodeProposalRequest.from_dict(request.to_dict())

    assert parsed == request
    assert parsed.schema_version == WORKER_CODE_SCHEMA_VERSION
    assert parsed.prompt_version == CODE_PROMPT_VERSION
    assert parsed.knowledge_evidence.task_hash == parsed.task_hash
    assert parsed.previous_attempt["feedback"]["error_code"] == "MINING-139"


def test_legacy_v1_request_remains_readable():
    payload = _request().to_dict()
    payload["schema_version"] = LEGACY_WORKER_CODE_SCHEMA_VERSION
    payload["prompt_version"] = LEGACY_CODE_PROMPT_VERSION
    payload.pop("task_hash")
    payload.pop("knowledge_evidence")

    request = WorkerCodeProposalRequest.from_dict(payload)

    assert request.schema_version == LEGACY_WORKER_CODE_SCHEMA_VERSION
    assert request.to_dict() == payload


def test_legacy_v2_response_remains_readable():
    response = WorkerCodeProposalResponse(
        proposal_id="proposal_legacy",
        code="print('legacy')",
        rationale="旧提示响应",
        model="deepseek/deepseek-chat",
        prompt_version=LEGACY_CODE_PROMPT_VERSION,
    )

    assert WorkerCodeProposalResponse.from_dict(response.to_dict()) == response


def test_request_rejects_cross_task_unknown_and_future_evidence():
    request = _request()
    evidence = _evidence()

    with pytest.raises(WorkerProtocolError, match="第一轮"):
        WorkerCodeProposalRequest(
            **request.to_dict() | {"knowledge_evidence": evidence}
        )

    with pytest.raises(WorkerProtocolError, match="不同任务"):
        WorkerCodeProposalRequest(
            **request.to_dict()
            | {
                "iteration": 2,
                "previous_attempt": _previous_attempt(),
                "knowledge_evidence": WorkerCodeKnowledgeEvidence(
                    task_hash="f" * 64,
                    index_hash=evidence.index_hash,
                    retrieval_hash=evidence.retrieval_hash,
                    entries=evidence.entries,
                ),
            }
        )

    with pytest.raises(WorkerProtocolError, match="当前或未来"):
        WorkerCodeProposalRequest(
            **request.to_dict()
            | {
                "iteration": 2,
                "previous_attempt": _previous_attempt(),
                "knowledge_evidence": WorkerCodeKnowledgeEvidence(
                    task_hash=request.task_hash,
                    index_hash=evidence.index_hash,
                    retrieval_hash=evidence.retrieval_hash,
                    entries=(
                        WorkerCodeKnowledgeEntry.from_dict(
                            {
                                **_failure_entry().to_dict(),
                                "iteration": 2,
                                "source_iterations": [2],
                            }
                        ),
                    ),
                ),
            }
        )


def test_nested_unknown_fields_and_tampered_hashes_are_rejected():
    payload = _evidence().to_dict()
    payload["entries"][0]["sources"] = []
    with pytest.raises(WorkerProtocolError, match="字段不匹配"):
        WorkerCodeKnowledgeEvidence.from_dict(payload)

    payload = _evidence().to_dict()
    payload["entries"][0]["feedback"]["summary"] = "被篡改"
    with pytest.raises(WorkerProtocolError, match="哈希不匹配"):
        WorkerCodeKnowledgeEvidence.from_dict(payload)

    entry = _failure_entry().to_dict()
    entry["metrics"] = {
        "rank_ic": 0.1,
        "icir": 0.2,
        "coverage": 0.8,
        "valid_days": 3,
        "unknown": 1,
    }
    with pytest.raises(WorkerProtocolError, match="必须恰好包含"):
        WorkerCodeKnowledgeEntry.from_dict(entry)


def test_prompt_contains_previous_feedback_and_retrieved_evidence():
    request = WorkerCodeProposalRequest(
        **_request().to_dict()
        | {
            "iteration": 2,
            "previous_attempt": _previous_attempt(),
            "knowledge_evidence": _evidence(),
        }
    )

    _, user = _prompt(request)
    payload = json.loads(user)

    assert payload["previous_attempt"]["feedback"]["repair_hint"] == (
        "修复运行时异常"
    )
    assert payload["knowledge_evidence"]["entries"][0]["kind"] == "failure"
    assert "sources" not in payload["knowledge_evidence"]["entries"][0]
