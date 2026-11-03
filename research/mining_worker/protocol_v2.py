"""Worker v2 搜索请求、响应和主服务可见证据摘要。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Mapping

from mining_worker.protocol import WorkerCandidateDraft, WorkerProtocolError

WORKER_SEARCH_SCHEMA_VERSION = "mining_worker_v2"
SEARCH_ACTIONS = frozenset({"refine", "branch", "stop", "promote"})
SEARCH_DIRECTIONS = frozenset({"positive", "negative", "non_monotonic"})
FORBIDDEN_AGENT_KEY = re.compile(
    r"(?<![a-z0-9])(selection|final|holdout|test_segment|module_test)"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)
FORBIDDEN_DATES = (
    "2020-01-01",
    "2022-12-31",
    "2023-01-01",
    "2025-10-15",
    "2025-10-16",
    "2026-10-15",
)
_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_IDENTIFIER = re.compile(
    r"^(selection|final|holdout|test_segment|module_test)([._:-]|$)",
    re.IGNORECASE,
)


def _text(value: Any, name: str, *, minimum: int = 1, maximum: int = 4000) -> str:
    if not isinstance(value, str):
        raise WorkerProtocolError(f"{name} 必须是字符串")
    result = value.strip()
    if not minimum <= len(result) <= maximum:
        raise WorkerProtocolError(
            f"{name} 长度必须在 {minimum}～{maximum} 个字符"
        )
    return result


def _identifier(value: Any, name: str) -> str:
    result = _text(value, name, maximum=160)
    if not _ID_PATTERN.fullmatch(result):
        raise WorkerProtocolError(f"{name} 包含非法字符")
    return result


def _nullable_identifier(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, name)


def _hash(value: Any, name: str) -> str:
    result = _text(value, name, minimum=64, maximum=64)
    if not _HASH_PATTERN.fullmatch(result):
        raise WorkerProtocolError(f"{name} 必须是 64 位小写十六进制 SHA-256")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerProtocolError(f"{name} 必须是整数")
    if not minimum <= value <= maximum:
        raise WorkerProtocolError(f"{name} 必须在 {minimum}～{maximum}")
    return value


def _string_tuple(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = 500,
    item_maximum: int = 500,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise WorkerProtocolError(f"{name} 必须是字符串数组")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        current = _text(item, name, maximum=item_maximum)
        if current not in seen:
            result.append(current)
            seen.add(current)
    if not minimum <= len(result) <= maximum:
        raise WorkerProtocolError(f"{name} 数量必须在 {minimum}～{maximum}")
    return tuple(result)


def _mapping_tuple(value: Any, name: str, *, maximum: int) -> tuple[dict, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise WorkerProtocolError(f"{name} 必须是 JSON 对象数组")
    if len(value) > maximum:
        raise WorkerProtocolError(f"{name} 最多包含 {maximum} 项")
    result: list[dict] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise WorkerProtocolError(f"{name} 每项必须是 JSON 对象")
        current = dict(item)
        encoded = json.dumps(
            current,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(encoded.encode("utf-8")) > 12000:
            raise WorkerProtocolError(f"{name} 单项超过 12000 字节")
        result.append(json.loads(encoded))
    return tuple(result)


def _strict_mapping(
    value: Any,
    name: str,
    fields: set[str],
    *,
    maximum_bytes: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkerProtocolError(f"{name} 必须是 JSON 对象")
    actual = set(value)
    missing = fields - actual
    unknown = actual - fields
    if missing or unknown:
        raise WorkerProtocolError(
            f"{name} 字段不匹配：缺少 {sorted(missing)}，未知 {sorted(unknown)}"
        )
    result = dict(value)
    encoded = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > maximum_bytes:
        raise WorkerProtocolError(f"{name} 超过 {maximum_bytes} 字节")
    return result


def ensure_agent_payload(value: Any, *, path: str = "payload") -> None:
    """拒绝 Agent 载荷中的受控窗口字段和日期。"""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise WorkerProtocolError(f"{path} 的字段名必须是字符串")
            trial_count = (
                key == "holdout_claimed"
                and path.endswith("evidence.trial_counts")
                and isinstance(item, int)
            )
            if FORBIDDEN_AGENT_KEY.search(key) and not trial_count:
                raise WorkerProtocolError(f"{path} 不允许字段 {key!r}")
            ensure_agent_payload(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ensure_agent_payload(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if _FORBIDDEN_IDENTIFIER.match(value):
            raise WorkerProtocolError(f"{path} 包含受控标识 {value!r}")
        for forbidden in FORBIDDEN_DATES:
            if forbidden in value:
                raise WorkerProtocolError(f"{path} 包含非开发期日期 {forbidden}")


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:24]}"


@dataclass(frozen=True)
class WorkerSearchHypothesis:
    """Worker 可见的当前或下一研究假设。"""

    statement: str
    mechanism: str
    expected_direction: str
    falsification_criteria: tuple[str, ...]
    parent_hypothesis_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "statement",
            _text(self.statement, "statement", minimum=8, maximum=4000),
        )
        object.__setattr__(
            self,
            "mechanism",
            _text(self.mechanism, "mechanism", maximum=1000),
        )
        if self.expected_direction not in SEARCH_DIRECTIONS:
            raise WorkerProtocolError(
                "expected_direction 必须是 positive、negative 或 non_monotonic"
            )
        object.__setattr__(
            self,
            "falsification_criteria",
            _string_tuple(
                self.falsification_criteria,
                "falsification_criteria",
                minimum=1,
                maximum=12,
            ),
        )
        object.__setattr__(
            self,
            "parent_hypothesis_id",
            _nullable_identifier(
                self.parent_hypothesis_id,
                "parent_hypothesis_id",
            ),
        )
        ensure_agent_payload(self.to_dict(), path="hypothesis")

    @classmethod
    def from_dict(cls, value: Any) -> WorkerSearchHypothesis:
        """严格解析假设摘要。"""

        payload = _strict_mapping(
            value,
            "hypothesis",
            set(cls.__dataclass_fields__),
            maximum_bytes=16000,
        )
        payload["falsification_criteria"] = tuple(
            payload["falsification_criteria"]
        )
        ensure_agent_payload(payload, path="hypothesis")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回可传输字典。"""

        return {
            "statement": self.statement,
            "mechanism": self.mechanism,
            "expected_direction": self.expected_direction,
            "falsification_criteria": list(self.falsification_criteria),
            "parent_hypothesis_id": self.parent_hypothesis_id,
        }


@dataclass(frozen=True)
class WorkerSearchCandidate:
    """Worker 请求中的候选谱系摘要。"""

    candidate_id: str
    name: str
    expression: str
    status: str
    parent_candidate_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _identifier(self.candidate_id, "candidate_id")
        )
        object.__setattr__(self, "name", _text(self.name, "name", maximum=200))
        object.__setattr__(
            self, "expression", _text(self.expression, "expression", maximum=12000)
        )
        object.__setattr__(self, "status", _identifier(self.status, "status"))
        object.__setattr__(
            self,
            "parent_candidate_id",
            _nullable_identifier(
                self.parent_candidate_id,
                "parent_candidate_id",
            ),
        )
        ensure_agent_payload(self.to_dict(), path="candidate")

    @classmethod
    def from_dict(cls, value: Any) -> WorkerSearchCandidate:
        """严格解析候选谱系摘要。"""

        payload = _strict_mapping(
            value,
            "lineage",
            set(cls.__dataclass_fields__),
            maximum_bytes=16000,
        )
        ensure_agent_payload(payload, path="lineage")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回可传输字典。"""

        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "expression": self.expression,
            "status": self.status,
            "parent_candidate_id": self.parent_candidate_id,
        }


@dataclass(frozen=True)
class WorkerSearchRequest:
    """主服务发给 Worker v2 的一轮假设与候选修订请求。"""

    request_id: str
    run_id: str
    turn_index: int
    branch_id: str
    context_hash: str
    hypothesis: WorkerSearchHypothesis
    lineage: tuple[WorkerSearchCandidate, ...]
    evidence: Mapping[str, Any] | None
    budget: Mapping[str, Any]
    allowed_actions: tuple[str, ...]
    llm_model: str
    max_candidates: int
    max_expression_depth: int
    excluded_fields: tuple[str, ...]
    fields: tuple[Mapping[str, Any], ...]
    operators: tuple[Mapping[str, Any], ...]
    schema_version: str = WORKER_SEARCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_SEARCH_SCHEMA_VERSION:
            raise WorkerProtocolError(
                f"schema_version 必须为 {WORKER_SEARCH_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "request_id", _identifier(self.request_id, "request_id"))
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "turn_index",
            _integer(self.turn_index, "turn_index", 1, 1000),
        )
        object.__setattr__(self, "branch_id", _identifier(self.branch_id, "branch_id"))
        object.__setattr__(
            self, "context_hash", _hash(self.context_hash, "context_hash")
        )
        if not isinstance(self.hypothesis, WorkerSearchHypothesis):
            raise WorkerProtocolError("hypothesis 必须是 WorkerSearchHypothesis")
        if (
            isinstance(self.lineage, (str, bytes))
            or not isinstance(self.lineage, (list, tuple))
            or len(self.lineage) > 24
        ):
            raise WorkerProtocolError("lineage 最多包含 24 个候选摘要")
        lineage = tuple(self.lineage)
        if any(not isinstance(item, WorkerSearchCandidate) for item in lineage):
            raise WorkerProtocolError("lineage 每项必须是 WorkerSearchCandidate")
        object.__setattr__(self, "lineage", lineage)
        if self.evidence is not None:
            object.__setattr__(
                self,
                "evidence",
                _strict_mapping(
                    self.evidence,
                    "evidence",
                    {
                        "schema_version",
                        "run_id",
                        "hypothesis_id",
                        "batch_id",
                        "turn_index",
                        "development_start",
                        "development_end",
                        "metrics",
                        "failures",
                        "correlation_clusters",
                        "trial_counts",
                        "budget_state",
                        "created_at",
                    },
                    maximum_bytes=65536,
                ),
            )
        elif self.turn_index != 1:
            raise WorkerProtocolError("从第 2 轮起必须提供开发期证据")
        object.__setattr__(
            self,
            "budget",
            _strict_mapping(
                self.budget,
                "budget",
                {
                    "max_search_turns",
                    "max_hypothesis_branches",
                    "max_refines_per_branch",
                    "max_candidates_per_turn",
                    "max_candidates_per_run",
                    "max_full_evaluations",
                    "stop_after_no_improvement_turns",
                    "max_tool_calls_per_turn",
                    "schema_version",
                },
                maximum_bytes=4096,
            ),
        )
        actions = _string_tuple(
            self.allowed_actions,
            "allowed_actions",
            minimum=1,
            maximum=4,
            item_maximum=32,
        )
        if any(action not in SEARCH_ACTIONS for action in actions):
            raise WorkerProtocolError("allowed_actions 包含未知动作")
        object.__setattr__(self, "allowed_actions", actions)
        object.__setattr__(
            self, "llm_model", _text(self.llm_model, "llm_model", maximum=200)
        )
        object.__setattr__(
            self,
            "max_candidates",
            _integer(self.max_candidates, "max_candidates", 1, 4),
        )
        object.__setattr__(
            self,
            "max_expression_depth",
            _integer(
                self.max_expression_depth,
                "max_expression_depth",
                2,
                8,
            ),
        )
        object.__setattr__(
            self,
            "excluded_fields",
            _string_tuple(
                self.excluded_fields,
                "excluded_fields",
                maximum=300,
            ),
        )
        object.__setattr__(
            self,
            "fields",
            _mapping_tuple(self.fields, "fields", maximum=300),
        )
        object.__setattr__(
            self,
            "operators",
            _mapping_tuple(self.operators, "operators", maximum=100),
        )
        if not self.fields or not self.operators:
            raise WorkerProtocolError("fields 和 operators 不能为空")
        ensure_agent_payload(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> WorkerSearchRequest:
        """严格解析不可信 Worker v2 请求。"""

        payload = _strict_mapping(
            value,
            "worker_search_request",
            set(cls.__dataclass_fields__),
            maximum_bytes=262144,
        )
        payload["hypothesis"] = WorkerSearchHypothesis.from_dict(
            payload["hypothesis"]
        )
        raw_lineage = payload["lineage"]
        if not isinstance(raw_lineage, list):
            raise WorkerProtocolError("lineage 必须是数组")
        payload["lineage"] = tuple(
            WorkerSearchCandidate.from_dict(item) for item in raw_lineage
        )
        payload["allowed_actions"] = tuple(payload["allowed_actions"])
        payload["excluded_fields"] = tuple(payload["excluded_fields"])
        payload["fields"] = tuple(payload["fields"])
        payload["operators"] = tuple(payload["operators"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定可传输字典。"""

        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "turn_index": self.turn_index,
            "branch_id": self.branch_id,
            "context_hash": self.context_hash,
            "hypothesis": self.hypothesis.to_dict(),
            "lineage": [item.to_dict() for item in self.lineage],
            "evidence": (
                dict(self.evidence) if self.evidence is not None else None
            ),
            "budget": dict(self.budget),
            "allowed_actions": list(self.allowed_actions),
            "llm_model": self.llm_model,
            "max_candidates": self.max_candidates,
            "max_expression_depth": self.max_expression_depth,
            "excluded_fields": list(self.excluded_fields),
            "fields": list(self.fields),
            "operators": list(self.operators),
        }

    @property
    def payload_hash(self) -> str:
        """返回请求幂等哈希。"""

        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @property
    def job_id(self) -> str:
        """返回由请求 ID 确定的稳定作业 ID。"""

        return _stable_id("sjob", {"request_id": self.request_id})


@dataclass(frozen=True)
class WorkerDecisionProposal:
    """Worker 对下一搜索动作的建议，最终决定仍由主服务控制器完成。"""

    action: str
    reason: str

    def __post_init__(self) -> None:
        if self.action not in SEARCH_ACTIONS:
            raise WorkerProtocolError("decision.action 包含未知动作")
        object.__setattr__(
            self,
            "reason",
            _text(self.reason, "decision.reason", maximum=2000),
        )
        ensure_agent_payload(self.to_dict(), path="decision")

    @classmethod
    def from_dict(cls, value: Any) -> WorkerDecisionProposal:
        """严格解析决策建议。"""

        payload = _strict_mapping(
            value,
            "decision",
            set(cls.__dataclass_fields__),
            maximum_bytes=4096,
        )
        ensure_agent_payload(payload, path="decision")
        return cls(**payload)

    def to_dict(self) -> dict[str, str]:
        """返回可传输字典。"""

        return {"action": self.action, "reason": self.reason}


@dataclass(frozen=True)
class WorkerSearchResponse:
    """Worker v2 一轮决策建议、下一假设和未信任候选草稿。"""

    decision: WorkerDecisionProposal
    next_hypothesis: WorkerSearchHypothesis | None
    candidates: tuple[WorkerCandidateDraft, ...]
    usage: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = WORKER_SEARCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_SEARCH_SCHEMA_VERSION:
            raise WorkerProtocolError(
                f"schema_version 必须为 {WORKER_SEARCH_SCHEMA_VERSION}"
            )
        if not isinstance(self.decision, WorkerDecisionProposal):
            raise WorkerProtocolError("decision 必须是 WorkerDecisionProposal")
        if self.next_hypothesis is not None and not isinstance(
            self.next_hypothesis,
            WorkerSearchHypothesis,
        ):
            raise WorkerProtocolError(
                "next_hypothesis 必须是 WorkerSearchHypothesis 或 null"
            )
        if (
            isinstance(self.candidates, (str, bytes))
            or not isinstance(self.candidates, (list, tuple))
            or len(self.candidates) > 4
        ):
            raise WorkerProtocolError("candidates 最多包含 4 项")
        candidates = tuple(self.candidates)
        if any(not isinstance(item, WorkerCandidateDraft) for item in candidates):
            raise WorkerProtocolError("candidates 每项必须是 WorkerCandidateDraft")
        object.__setattr__(self, "candidates", candidates)
        if not isinstance(self.usage, Mapping):
            raise WorkerProtocolError("usage 必须是 JSON 对象")
        usage = dict(self.usage)
        if len(json.dumps(usage, ensure_ascii=False).encode("utf-8")) > 16000:
            raise WorkerProtocolError("usage 超过 16000 字节")
        for key, value in usage.items():
            if not isinstance(key, str):
                raise WorkerProtocolError("usage 键必须是字符串")
            if isinstance(value, float) and not math.isfinite(value):
                raise WorkerProtocolError("usage 不能包含非有限数值")
        object.__setattr__(self, "usage", usage)
        if self.decision.action in {"refine", "branch"}:
            if self.next_hypothesis is None:
                raise WorkerProtocolError("refine/branch 必须提供下一假设")
            if not candidates:
                raise WorkerProtocolError("refine/branch 必须提供候选")
        elif self.next_hypothesis is not None or candidates:
            raise WorkerProtocolError("stop/promote 不得提供下一假设或候选")
        ensure_agent_payload(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> WorkerSearchResponse:
        """严格解析不可信 Worker v2 响应。"""

        payload = _strict_mapping(
            value,
            "worker_search_response",
            set(cls.__dataclass_fields__),
            maximum_bytes=262144,
        )
        payload["decision"] = WorkerDecisionProposal.from_dict(payload["decision"])
        if payload["next_hypothesis"] is not None:
            payload["next_hypothesis"] = WorkerSearchHypothesis.from_dict(
                payload["next_hypothesis"]
            )
        raw_candidates = payload["candidates"]
        if not isinstance(raw_candidates, list):
            raise WorkerProtocolError("candidates 必须是数组")
        payload["candidates"] = tuple(
            WorkerCandidateDraft.from_dict(item) for item in raw_candidates
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定可传输字典。"""

        return {
            "schema_version": self.schema_version,
            "decision": self.decision.to_dict(),
            "next_hypothesis": (
                self.next_hypothesis.to_dict()
                if self.next_hypothesis is not None
                else None
            ),
            "candidates": [item.to_dict() for item in self.candidates],
            "usage": dict(self.usage),
        }
