"""V3 因子研究闭环的版本、来源、候选和证据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mining.contract_candidates import CandidateDefinition
from mining.contract_types import (
    IDEMPOTENCY_PATTERN,
    identifier,
    sha256_json,
)
from mining.errors import MiningContractError
from mining.search_contract_types import choice, hash_text, strict_fields

MINING_V3_SCHEMA_VERSION = "mining_v3"
V3_RESEARCH_INPUT_SCHEMA_VERSION = "mining_research_input_v3"
V3_RESEARCH_SOURCE_SCHEMA_VERSION = "mining_research_source_v3"
V3_RESEARCH_CANDIDATE_SCHEMA_VERSION = "mining_research_candidate_v3"
V3_RESEARCH_EVIDENCE_SCHEMA_VERSION = "mining_research_evidence_v3"
V3_RUN_ID_PREFIX = "researchrun"
V3_PUBLICATION_SOURCE_KIND = "v3_research"
V3_RESEARCH_INPUT_MODES = (
    "hypothesis",
    "free_exploration",
    "research_report",
    "costeer",
)
V3_RESEARCH_SOURCE_KINDS = V3_RESEARCH_INPUT_MODES
V3_REQUIRED_EVIDENCE_STAGES = (
    "technical",
    "development",
    "selection",
    "full",
    "holdout",
)
V3_EVIDENCE_STATUSES = ("running", "passed", "failed", "error")


def _candidate_payload(candidate: CandidateDefinition) -> dict[str, Any]:
    """把候选定义转换为稳定 JSON 载荷。"""

    return {
        "name": candidate.name,
        "mechanism": candidate.mechanism,
        "description": candidate.description,
        "canonical_expression": candidate.canonical_expression,
        "direction": candidate.direction,
        "lookback": candidate.lookback,
        "dependencies": list(candidate.dependencies),
        "output_type": candidate.output_type,
        "steps": list(candidate.steps),
        "discovery_round": candidate.discovery_round,
        "discovery_index": candidate.discovery_index,
        "parameters": dict(candidate.parameters),
    }


def _candidate_from_dict(value: Any) -> CandidateDefinition:
    """严格解析候选定义，不改动既有候选哈希口径。"""

    payload = strict_fields(
        value,
        "candidate_definition",
        set(CandidateDefinition.__dataclass_fields__),
    )
    for name in ("dependencies", "steps"):
        payload[name] = tuple(payload[name])
    return CandidateDefinition(**payload)


@dataclass(frozen=True)
class ResearchSourceReference:
    """V3 候选允许持久化的来源身份和审计引用。"""

    source_kind: str
    source_id: str
    source_hash: str
    artifact_manifest_hash: str | None = None
    promotion_hash: str | None = None
    schema_version: str = V3_RESEARCH_SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != V3_RESEARCH_SOURCE_SCHEMA_VERSION:
            raise MiningContractError("V3 来源 schema_version 不匹配")
        object.__setattr__(
            self,
            "source_kind",
            choice(self.source_kind, "source_kind", V3_RESEARCH_SOURCE_KINDS),
        )
        object.__setattr__(self, "source_id", identifier(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "source_hash",
            hash_text(self.source_hash, "source_hash"),
        )
        for name in ("artifact_manifest_hash", "promotion_hash"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, hash_text(value, name))
        sandbox_refs = (
            self.artifact_manifest_hash is not None,
            self.promotion_hash is not None,
        )
        if self.source_kind == "costeer" and not all(sandbox_refs):
            raise MiningContractError(
                "costeer 来源必须携带产物清单和人工提升记录哈希"
            )
        if self.source_kind != "costeer" and any(sandbox_refs):
            raise MiningContractError(
                "非 costeer 来源不得携带沙箱产物或人工提升记录哈希"
            )

    def to_dict(self) -> dict[str, Any]:
        """返回严格持久化字典。"""

        return {
            "schema_version": self.schema_version,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "source_hash": self.source_hash,
            "artifact_manifest_hash": self.artifact_manifest_hash,
            "promotion_hash": self.promotion_hash,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ResearchSourceReference":
        """严格解析 V3 来源引用。"""

        return cls(
            **strict_fields(
                value,
                "v3_research_source",
                set(cls.__dataclass_fields__),
            )
        )

    @property
    def reference_hash(self) -> str:
        """返回来源引用稳定哈希。"""

        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class ResearchInputEnvelope:
    """四类 V3 研究输入归一后的不可变包络。"""

    run_id: str
    idempotency_key: str
    input_mode: str
    context_hash: str
    request_hash: str
    source: ResearchSourceReference
    schema_version: str = V3_RESEARCH_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != V3_RESEARCH_INPUT_SCHEMA_VERSION:
            raise MiningContractError("V3 输入包络 schema_version 不匹配")
        run_id = identifier(self.run_id, "run_id")
        if not run_id.startswith(f"{V3_RUN_ID_PREFIX}_"):
            raise MiningContractError(
                f"V3 运行 ID 必须以 {V3_RUN_ID_PREFIX}_ 开头"
            )
        object.__setattr__(self, "run_id", run_id)
        if (
            not isinstance(self.idempotency_key, str)
            or not IDEMPOTENCY_PATTERN.fullmatch(self.idempotency_key)
        ):
            raise MiningContractError(
                "idempotency_key 长度为 8～128，只能包含字母、数字、点、下划线、"
                "冒号和连字符"
            )
        input_mode = choice(
            self.input_mode,
            "input_mode",
            V3_RESEARCH_INPUT_MODES,
        )
        object.__setattr__(self, "input_mode", input_mode)
        for name in ("context_hash", "request_hash"):
            object.__setattr__(
                self,
                name,
                hash_text(getattr(self, name), name),
            )
        if not isinstance(self.source, ResearchSourceReference):
            raise MiningContractError("source 必须是 ResearchSourceReference")
        if self.source.source_kind != input_mode:
            raise MiningContractError("输入模式必须与来源类型一致")

    def to_dict(self) -> dict[str, Any]:
        """返回稳定输入包络。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "idempotency_key": self.idempotency_key,
            "input_mode": self.input_mode,
            "context_hash": self.context_hash,
            "request_hash": self.request_hash,
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ResearchInputEnvelope":
        """严格解析 V3 输入包络。"""

        payload = strict_fields(
            value,
            "v3_research_input",
            set(cls.__dataclass_fields__),
        )
        payload["source"] = ResearchSourceReference.from_dict(payload["source"])
        return cls(**payload)

    @property
    def input_hash(self) -> str:
        """返回不包含数据库时间的输入包络哈希。"""

        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class EvidenceReference:
    """V3 单阶段评估证据的不可变引用。"""

    candidate_id: str
    stage: str
    evaluation_id: str
    input_hash: str
    evidence_hash: str
    status: str
    schema_version: str = V3_RESEARCH_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != V3_RESEARCH_EVIDENCE_SCHEMA_VERSION:
            raise MiningContractError("V3 证据 schema_version 不匹配")
        for name in ("candidate_id", "evaluation_id"):
            object.__setattr__(
                self,
                name,
                identifier(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "stage",
            choice(self.stage, "stage", V3_REQUIRED_EVIDENCE_STAGES),
        )
        object.__setattr__(
            self,
            "status",
            choice(self.status, "status", V3_EVIDENCE_STATUSES),
        )
        for name in ("input_hash", "evidence_hash"):
            object.__setattr__(
                self,
                name,
                hash_text(getattr(self, name), name),
            )

    def to_dict(self) -> dict[str, Any]:
        """返回严格证据引用字典。"""

        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "evaluation_id": self.evaluation_id,
            "input_hash": self.input_hash,
            "evidence_hash": self.evidence_hash,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EvidenceReference":
        """严格解析 V3 证据引用。"""

        return cls(
            **strict_fields(
                value,
                "v3_research_evidence",
                set(cls.__dataclass_fields__),
            )
        )

    @property
    def reference_hash(self) -> str:
        """返回证据引用稳定哈希。"""

        return sha256_json(self.to_dict())


def _evidence_tuple(value: Any) -> tuple[EvidenceReference, ...]:
    """校验证据引用并规范为确定顺序。"""

    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise MiningContractError("evidence 必须是证据引用序列")
    rows = tuple(value)
    if any(not isinstance(item, EvidenceReference) for item in rows):
        raise MiningContractError("evidence 包含未知对象")
    evaluation_ids = tuple(item.evaluation_id for item in rows)
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise MiningContractError("evidence 不能包含重复评估 ID")
    return tuple(
        sorted(
            rows,
            key=lambda item: (
                V3_REQUIRED_EVIDENCE_STAGES.index(item.stage),
                item.evaluation_id,
            ),
        )
    )


@dataclass(frozen=True)
class CandidateEnvelope:
    """候选内容、来源和候选级证据的统一 V3 审核视图。"""

    candidate: CandidateDefinition
    source: ResearchSourceReference
    evidence: tuple[EvidenceReference, ...] = field(default_factory=tuple)
    schema_version: str = V3_RESEARCH_CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != V3_RESEARCH_CANDIDATE_SCHEMA_VERSION:
            raise MiningContractError("V3 候选包络 schema_version 不匹配")
        if not isinstance(self.candidate, CandidateDefinition):
            raise MiningContractError("candidate 必须是 CandidateDefinition")
        if not isinstance(self.source, ResearchSourceReference):
            raise MiningContractError("source 必须是 ResearchSourceReference")
        evidence = _evidence_tuple(self.evidence)
        if any(item.candidate_id != self.candidate.candidate_id for item in evidence):
            raise MiningContractError("证据引用没有绑定当前候选")
        object.__setattr__(self, "evidence", evidence)

    def to_dict(self) -> dict[str, Any]:
        """返回不包含额外来源正文的候选包络。"""

        return {
            "schema_version": self.schema_version,
            "candidate": _candidate_payload(self.candidate),
            "source": self.source.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "CandidateEnvelope":
        """严格解析 V3 候选包络。"""

        payload = strict_fields(
            value,
            "v3_research_candidate",
            set(cls.__dataclass_fields__),
        )
        payload["candidate"] = _candidate_from_dict(payload["candidate"])
        payload["source"] = ResearchSourceReference.from_dict(payload["source"])
        if isinstance(payload["evidence"], (str, bytes)) or not isinstance(
            payload["evidence"],
            (list, tuple),
        ):
            raise MiningContractError("V3 candidate evidence 必须是证据引用序列")
        payload["evidence"] = tuple(
            EvidenceReference.from_dict(item)
            for item in payload["evidence"]
        )
        return cls(**payload)

    @property
    def envelope_hash(self) -> str:
        """返回候选、来源和全部证据绑定的稳定哈希。"""

        return sha256_json(self.to_dict())

    def approval_material(self, context_hash: str) -> dict[str, Any]:
        """构造后续审批哈希使用的稳定材料。"""

        context_hash = hash_text(context_hash, "context_hash")
        passed_stages = {
            item.stage for item in self.evidence if item.status == "passed"
        }
        missing = [
            stage
            for stage in V3_REQUIRED_EVIDENCE_STAGES
            if stage not in passed_stages
        ]
        if missing:
            raise MiningContractError(f"审批证据缺少通过阶段：{missing}")
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate.candidate_id,
            "candidate_content_hash": self.candidate.content_hash,
            "semantic_hash": self.candidate.semantic_hash,
            "context_hash": context_hash,
            "source_hash": self.source.reference_hash,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    def approval_evidence_hash(self, context_hash: str) -> str:
        """返回覆盖候选、来源、上下文和分阶段证据的审批哈希。"""

        return sha256_json(self.approval_material(context_hash))


__all__ = [
    "MINING_V3_SCHEMA_VERSION",
    "V3_EVIDENCE_STATUSES",
    "V3_PUBLICATION_SOURCE_KIND",
    "V3_REQUIRED_EVIDENCE_STAGES",
    "V3_RESEARCH_CANDIDATE_SCHEMA_VERSION",
    "V3_RESEARCH_EVIDENCE_SCHEMA_VERSION",
    "V3_RESEARCH_INPUT_MODES",
    "V3_RESEARCH_INPUT_SCHEMA_VERSION",
    "V3_RESEARCH_SOURCE_KINDS",
    "V3_RESEARCH_SOURCE_SCHEMA_VERSION",
    "V3_RUN_ID_PREFIX",
    "CandidateEnvelope",
    "EvidenceReference",
    "ResearchInputEnvelope",
    "ResearchSourceReference",
]
