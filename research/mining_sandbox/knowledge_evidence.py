"""把当前运行知识转换为 Worker 可见的白名单证据。"""

from __future__ import annotations

from typing import Any, Mapping

from mining_sandbox.errors import SandboxKnowledgeError
from mining_sandbox.feedback import SandboxCodeFeedback
from mining_sandbox.knowledge import (
    SandboxKnowledgeBase,
    SandboxKnowledgeEntry,
)
from mining_worker.protocol import WorkerProtocolError
from mining_worker.protocol_code import (
    WorkerCodeFeedbackEvidence,
    WorkerCodeKnowledgeEntry,
    WorkerCodeKnowledgeEvidence,
)


def worker_feedback_evidence(
    value: SandboxCodeFeedback | Mapping[str, Any],
) -> WorkerCodeFeedbackEvidence:
    """把已脱敏反馈转换为 Worker 证据。"""

    try:
        feedback = (
            value
            if isinstance(value, SandboxCodeFeedback)
            else SandboxCodeFeedback.from_dict(value)
        )
        return WorkerCodeFeedbackEvidence(
            **feedback.to_dict(),
            feedback_hash=feedback.feedback_hash,
        )
    except Exception as exc:
        raise SandboxKnowledgeError(f"知识反馈无法转换为 Worker 证据：{exc}") from exc


def _worker_entry(
    entry: SandboxKnowledgeEntry,
) -> WorkerCodeKnowledgeEntry:
    feedback = (
        worker_feedback_evidence(entry.feedback)
        if entry.feedback is not None
        else None
    )
    try:
        return WorkerCodeKnowledgeEntry(
            kind=entry.kind,
            iteration=entry.iteration,
            source_iterations=entry.source_iterations,
            code_hash=entry.code_hash,
            feedback=feedback,
            metrics=entry.metrics,
            evaluation_hash=entry.evaluation_hash,
            summary=entry.summary,
            entry_hash=entry.entry_hash,
        )
    except WorkerProtocolError as exc:
        raise SandboxKnowledgeError(
            f"知识条目无法转换为 Worker 证据：{exc}"
        ) from exc


def build_worker_knowledge_evidence(
    knowledge: SandboxKnowledgeBase,
) -> WorkerCodeKnowledgeEvidence:
    """从当前运行知识生成确定性 Worker 证据。"""

    if not isinstance(knowledge, SandboxKnowledgeBase):
        raise SandboxKnowledgeError("knowledge 必须是 SandboxKnowledgeBase")
    selection = knowledge.retrieve(knowledge.task_hash)
    entries = (
        selection.failures
        + selection.successes
        + selection.repairs
    )
    if not 1 <= len(entries) <= 4:
        raise SandboxKnowledgeError("Worker 知识证据必须包含 1～4 条")
    try:
        return WorkerCodeKnowledgeEvidence(
            task_hash=knowledge.task_hash,
            index_hash=knowledge.index_hash,
            retrieval_hash=selection.selection_hash,
            entries=tuple(_worker_entry(item) for item in entries),
        )
    except WorkerProtocolError as exc:
        raise SandboxKnowledgeError(
            f"知识索引无法转换为 Worker 证据：{exc}"
        ) from exc


def worker_previous_attempt(
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    """生成下一轮请求中的上一尝试摘要。"""

    feedback = (
        worker_feedback_evidence(attempt["feedback"])
        if attempt.get("feedback") is not None
        else None
    )
    return {
        "iteration": int(attempt["iteration"]),
        "code_hash": str(attempt["code_hash"]),
        "status": str(attempt["status"]),
        "metrics": (
            dict(attempt["metrics"])
            if attempt.get("metrics") is not None
            else None
        ),
        "error_code": attempt.get("error_code"),
        "feedback": feedback.to_dict() if feedback is not None else None,
        "feedback_hash": feedback.feedback_hash if feedback is not None else None,
    }


__all__ = [
    "build_worker_knowledge_evidence",
    "worker_feedback_evidence",
    "worker_previous_attempt",
]
