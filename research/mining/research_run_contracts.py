"""资料驱动 V2 运行的输入、来源和摘要契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping

from mining.contract_types import (
    ensure_string_tuple,
    identifier,
    sha256_json,
    text,
    utc_now_iso,
    validate_utc_timestamp,
)
from mining.errors import ResearchRunContractError
from mining.research_extraction_contracts import (
    RESEARCH_FACTOR_FIELDS,
    ResearchFieldClaim,
)
from mining.contract_candidates import CandidateDefinition
from mining.research_factor_screening_contracts import (
    ResearchConversionLimits,
    ResearchScreeningDecision,
    RESEARCH_SCREENING_STATUSES,
)
from mining.research_material_contracts import (
    ResearchMaterialFailure,
    ResearchMaterialInput,
    ResearchMaterialManifest,
)
from mining.research_material_parse_contracts import (
    ResearchMaterialClassification,
)
from mining.research_material_safety import validate_relative_material_path
from mining.search_cards import HypothesisCard
from mining.search_contract_types import mapping_with_size, strict_fields

RESEARCH_RUN_SCHEMA_VERSION = "mining_research_run_v1"
RESEARCH_RUN_STATUSES = ("processing", "ready", "failed")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _strict_fields(value: Any, name: str, expected: set[str]) -> dict[str, Any]:
    """校验 JSON 对象字段集合并统一错误类型。"""

    try:
        return strict_fields(value, name, expected)
    except Exception as error:
        raise ResearchRunContractError(str(error)) from error


def _hash(value: Any, name: str) -> str:
    """校验 SHA-256。"""

    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ResearchRunContractError(f"{name} 必须是 64 位小写 SHA-256")
    return value


def _bounded_int(value: Any, name: str, *, minimum: int = 0) -> int:
    """校验非布尔整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ResearchRunContractError(f"{name} 必须是大于等于 {minimum} 的整数")
    return value


def _material_target(value: Any) -> str:
    """校验资料根目录内的相对文件或目录。"""

    if value == ".":
        return "."
    try:
        return validate_relative_material_path(value)
    except Exception as error:
        raise ResearchRunContractError(str(error)) from error


@dataclass(frozen=True)
class ResearchRunInput:
    """一次资料驱动运行在创建时冻结的输入。"""

    run_id: str
    idempotency_key: str
    material_path: str
    manifest: ResearchMaterialManifest
    limits: ResearchConversionLimits
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = RESEARCH_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_RUN_SCHEMA_VERSION:
            raise ResearchRunContractError(
                f"schema_version 必须为 {RESEARCH_RUN_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "idempotency_key",
            text(self.idempotency_key, "idempotency_key", maximum=128),
        )
        object.__setattr__(
            self,
            "material_path",
            _material_target(self.material_path),
        )
        if not isinstance(self.manifest, ResearchMaterialManifest):
            raise ResearchRunContractError(
                "manifest 必须是 ResearchMaterialManifest"
            )
        if not isinstance(self.limits, ResearchConversionLimits):
            raise ResearchRunContractError(
                "limits 必须是 ResearchConversionLimits"
            )
        if len(self.manifest.materials) > self.limits.max_documents:
            raise ResearchRunContractError(
                "资料清单数量超过本次转换上限"
            )
        object.__setattr__(
            self,
            "created_at",
            validate_utc_timestamp(self.created_at, "created_at"),
        )

    def material(self) -> dict[str, Any]:
        """返回稳定的输入哈希字段。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "idempotency_key": self.idempotency_key,
            "material_path": self.material_path,
            "manifest": self.manifest.material(),
            "limits": self.limits.to_dict(),
        }

    @property
    def input_hash(self) -> str:
        """返回资料运行输入哈希。"""

        return sha256_json(self.material())

    @classmethod
    def from_dict(cls, value: Any) -> "ResearchRunInput":
        """严格解析持久化输入。"""

        payload = _strict_fields(
            value,
            "research_run_input",
            set(cls.__dataclass_fields__),
        )
        payload["manifest"] = ResearchMaterialManifest.from_dict(
            payload["manifest"]
        )
        payload["limits"] = ResearchConversionLimits.from_dict(payload["limits"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return {
            **self.material(),
            "manifest": self.manifest.to_dict(),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ResearchMaterialView:
    """运行摘要中的单份资料状态，不包含正文。"""

    document_id: str
    relative_path: str
    status: str
    classification: str | None = None
    page_count: int = 0
    error_code: str | None = None
    failure_category: str | None = None
    failure_message: str | None = None
    schema_version: str = RESEARCH_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_RUN_SCHEMA_VERSION:
            raise ResearchRunContractError("资料摘要 schema_version 不匹配")
        object.__setattr__(self, "document_id", identifier(self.document_id, "document_id"))
        object.__setattr__(
            self,
            "relative_path",
            _material_target(self.relative_path),
        )
        if self.status not in {"parsed", "filtered", "failed"}:
            raise ResearchRunContractError("资料摘要 status 不合法")
        if self.classification is not None:
            object.__setattr__(
                self,
                "classification",
                identifier(self.classification, "classification"),
            )
        object.__setattr__(
            self,
            "page_count",
            _bounded_int(self.page_count, "page_count"),
        )
        if self.error_code is not None:
            object.__setattr__(
                self,
                "error_code",
                identifier(self.error_code, "error_code"),
            )
        if self.failure_category is not None:
            object.__setattr__(
                self,
                "failure_category",
                identifier(self.failure_category, "failure_category"),
            )
        if self.failure_message is not None:
            object.__setattr__(
                self,
                "failure_message",
                text(self.failure_message, "failure_message", maximum=1000),
            )

    def to_dict(self) -> dict[str, Any]:
        """返回页面使用的字典。"""

        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ResearchMaterialView":
        """严格解析资料摘要。"""

        return cls(
            **_strict_fields(
                value,
                "research_material_view",
                set(cls.__dataclass_fields__),
            )
        )


@dataclass(frozen=True)
class ResearchRunDecision:
    """运行摘要中的一条草稿级筛选结论。"""

    document_id: str
    draft_id: str
    factor_hash: str
    status: str
    reason_code: str | None
    checks: tuple[str, ...]
    candidate_id: str | None = None
    admitted: bool = False
    schema_version: str = RESEARCH_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_RUN_SCHEMA_VERSION:
            raise ResearchRunContractError("筛选结论摘要 schema_version 不匹配")
        for name in ("document_id", "draft_id"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        object.__setattr__(self, "factor_hash", _hash(self.factor_hash, "factor_hash"))
        if self.status not in RESEARCH_SCREENING_STATUSES:
            raise ResearchRunContractError("筛选结论 status 不合法")
        if self.reason_code is not None:
            object.__setattr__(
                self,
                "reason_code",
                identifier(self.reason_code, "reason_code"),
            )
        object.__setattr__(
            self,
            "checks",
            tuple(
                sorted(
                    set(
                        ensure_string_tuple(
                            self.checks,
                            "checks",
                            maximum=16,
                        )
                    )
                )
            ),
        )
        if self.candidate_id is not None:
            object.__setattr__(
                self,
                "candidate_id",
                identifier(self.candidate_id, "candidate_id"),
            )
        if not isinstance(self.admitted, bool):
            raise ResearchRunContractError("admitted 必须是布尔值")

    @classmethod
    def from_screening(
        cls,
        decision: ResearchScreeningDecision,
        *,
        candidate_id: str | None,
        admitted: bool,
    ) -> "ResearchRunDecision":
        """从 P2-04 结论构造运行摘要。"""

        if not isinstance(decision, ResearchScreeningDecision):
            raise ResearchRunContractError(
                "decision 必须是 ResearchScreeningDecision"
            )
        return cls(
            document_id=decision.document_id,
            draft_id=decision.draft_id,
            factor_hash=decision.factor_hash,
            status=decision.status,
            reason_code=decision.reason_code,
            checks=decision.checks,
            candidate_id=candidate_id,
            admitted=admitted,
        )

    def to_dict(self) -> dict[str, Any]:
        """返回页面摘要字典。"""

        result = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }
        result["checks"] = list(self.checks)
        return result

    @classmethod
    def from_dict(cls, value: Any) -> "ResearchRunDecision":
        """严格解析结论摘要。"""

        payload = _strict_fields(
            value,
            "research_run_decision",
            set(cls.__dataclass_fields__),
        )
        payload["checks"] = tuple(payload["checks"])
        return cls(**payload)


@dataclass(frozen=True)
class ResearchRunSummary:
    """一次资料驱动运行可公开的来源和筛选摘要。"""

    input: ResearchRunInput
    status: str
    extraction_ids: tuple[str, ...]
    conversion_id: str | None
    conversion_hash: str | None
    model: str
    catalog_hash: str
    materials: tuple[ResearchMaterialView, ...]
    decisions: tuple[ResearchRunDecision, ...]
    extracted_factor_count: int
    accepted_count: int
    admitted_count: int
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    schema_version: str = RESEARCH_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_RUN_SCHEMA_VERSION:
            raise ResearchRunContractError("运行摘要 schema_version 不匹配")
        if not isinstance(self.input, ResearchRunInput):
            raise ResearchRunContractError("input 必须是 ResearchRunInput")
        if self.status not in RESEARCH_RUN_STATUSES:
            raise ResearchRunContractError("运行摘要 status 不合法")
        object.__setattr__(
            self,
            "extraction_ids",
            tuple(
                sorted(
                    set(
                        ensure_string_tuple(
                            self.extraction_ids,
                            "extraction_ids",
                            maximum=64,
                        )
                    )
                )
            ),
        )
        if self.conversion_id is not None:
            object.__setattr__(
                self,
                "conversion_id",
                identifier(self.conversion_id, "conversion_id"),
            )
        for name in ("conversion_hash", "catalog_hash"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _hash(value, name))
        object.__setattr__(self, "model", text(self.model, "model", maximum=200))
        materials = tuple(self.materials)
        if any(not isinstance(item, ResearchMaterialView) for item in materials):
            raise ResearchRunContractError("materials 包含未知对象")
        object.__setattr__(self, "materials", materials)
        decisions = tuple(self.decisions)
        if any(not isinstance(item, ResearchRunDecision) for item in decisions):
            raise ResearchRunContractError("decisions 包含未知对象")
        object.__setattr__(self, "decisions", decisions)
        for name in ("extracted_factor_count", "accepted_count", "admitted_count"):
            object.__setattr__(
                self,
                name,
                _bounded_int(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "created_at",
            validate_utc_timestamp(self.created_at, "created_at"),
        )
        object.__setattr__(
            self,
            "updated_at",
            validate_utc_timestamp(self.updated_at, "updated_at"),
        )

    def material(self) -> dict[str, Any]:
        """返回不包含时间的持久化字段。"""

        return {
            "schema_version": self.schema_version,
            "input": self.input.material(),
            "status": self.status,
            "extraction_ids": list(self.extraction_ids),
            "conversion_id": self.conversion_id,
            "conversion_hash": self.conversion_hash,
            "model": self.model,
            "catalog_hash": self.catalog_hash,
            "materials": [item.to_dict() for item in self.materials],
            "decisions": [item.to_dict() for item in self.decisions],
            "extracted_factor_count": self.extracted_factor_count,
            "accepted_count": self.accepted_count,
            "admitted_count": self.admitted_count,
        }

    @property
    def summary_hash(self) -> str:
        """返回不含时间的摘要哈希。"""

        return sha256_json(self.material())

    def to_dict(self) -> dict[str, Any]:
        """返回页面和持久化字典。"""

        return {
            **self.material(),
            "input": self.input.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary_hash": self.summary_hash,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ResearchRunSummary":
        """严格解析持久化摘要。"""

        if isinstance(value, Mapping) and "summary_hash" in value:
            value = {key: item for key, item in value.items() if key != "summary_hash"}
        payload = _strict_fields(
            value,
            "research_run_summary",
            set(cls.__dataclass_fields__),
        )
        payload["input"] = ResearchRunInput.from_dict(payload["input"])
        payload["materials"] = tuple(
            ResearchMaterialView.from_dict(item) for item in payload["materials"]
        )
        payload["decisions"] = tuple(
            ResearchRunDecision.from_dict(item) for item in payload["decisions"]
        )
        payload["extraction_ids"] = tuple(payload["extraction_ids"])
        return cls(**payload)


@dataclass(frozen=True, kw_only=True)
class ResearchCandidateProvenance:
    """一个资料候选与页码原文证据的冻结绑定。"""

    run_id: str
    candidate_id: str
    document_id: str
    extraction_id: str
    extraction_hash: str
    draft_id: str
    factor_hash: str
    material: ResearchMaterialInput
    classification: ResearchMaterialClassification
    fields: tuple[ResearchFieldClaim, ...]
    checks: tuple[str, ...]
    field_mapping: Mapping[str, str]
    catalog_hash: str
    converter_model: str
    conversion_prompt_version: str
    extraction_model: str
    extraction_prompt_version: str
    hypothesis: HypothesisCard
    candidate: CandidateDefinition
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = RESEARCH_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_RUN_SCHEMA_VERSION:
            raise ResearchRunContractError("候选来源 schema_version 不匹配")
        for name in ("run_id", "candidate_id", "document_id", "extraction_id", "draft_id"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        for name in ("extraction_hash", "factor_hash", "catalog_hash"):
            object.__setattr__(self, name, _hash(getattr(self, name), name))
        if not isinstance(self.material, ResearchMaterialInput):
            raise ResearchRunContractError("material 必须是 ResearchMaterialInput")
        if not isinstance(self.classification, ResearchMaterialClassification):
            raise ResearchRunContractError(
                "classification 必须是 ResearchMaterialClassification"
            )
        if (
            self.material.document_id != self.document_id
            or self.classification.document_id != self.document_id
        ):
            raise ResearchRunContractError("资料、分类和候选来源的 document_id 不一致")
        fields = tuple(self.fields)
        if any(not isinstance(item, ResearchFieldClaim) for item in fields):
            raise ResearchRunContractError("fields 包含未知对象")
        if len(fields) != 6:
            raise ResearchRunContractError("候选来源必须保存六类字段")
        if tuple(item.field_name for item in fields) != RESEARCH_FACTOR_FIELDS:
            raise ResearchRunContractError("候选来源字段必须按固定顺序完整保存")
        if any(
            citation.document_id != self.document_id
            for claim in fields
            for citation in claim.citations
        ):
            raise ResearchRunContractError("候选引用必须属于当前资料")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(
            self,
            "checks",
            tuple(
                sorted(
                    set(
                        ensure_string_tuple(
                            self.checks,
                            "checks",
                            maximum=16,
                        )
                    )
                )
            ),
        )
        mapping = mapping_with_size(
            self.field_mapping,
            "field_mapping",
            maximum_bytes=8192,
        )
        if not mapping or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in mapping.items()
        ):
            raise ResearchRunContractError(
                "field_mapping 必须是非空字符串对象"
            )
        object.__setattr__(self, "field_mapping", mapping)
        for name in (
            "converter_model",
            "conversion_prompt_version",
            "extraction_model",
            "extraction_prompt_version",
        ):
            object.__setattr__(
                self,
                name,
                text(getattr(self, name), name, maximum=200),
            )
        if not isinstance(self.hypothesis, HypothesisCard):
            raise ResearchRunContractError("hypothesis 必须是 HypothesisCard")
        if self.hypothesis.run_id != self.run_id:
            raise ResearchRunContractError("来源假设必须属于当前 V2 运行")
        if not isinstance(self.candidate, CandidateDefinition):
            raise ResearchRunContractError("candidate 必须是 CandidateDefinition")
        object.__setattr__(
            self,
            "created_at",
            validate_utc_timestamp(self.created_at, "created_at"),
        )

    def stable_material(self) -> dict[str, Any]:
        """返回候选来源稳定哈希字段。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "document_id": self.document_id,
            "extraction_id": self.extraction_id,
            "extraction_hash": self.extraction_hash,
            "draft_id": self.draft_id,
            "factor_hash": self.factor_hash,
            "material": self.material.material(),
            "classification": self.classification.material(),
            "fields": [item.material() for item in self.fields],
            "checks": list(self.checks),
            "field_mapping": dict(self.field_mapping),
            "catalog_hash": self.catalog_hash,
            "converter_model": self.converter_model,
            "conversion_prompt_version": self.conversion_prompt_version,
            "extraction_model": self.extraction_model,
            "extraction_prompt_version": self.extraction_prompt_version,
            "hypothesis": self.hypothesis.material(),
        }

    @property
    def provenance_hash(self) -> str:
        """返回候选来源哈希。"""

        return sha256_json(self.stable_material())

    def to_dict(self) -> dict[str, Any]:
        """返回可展示和持久化的来源字典。"""

        return {
            **self.stable_material(),
            "candidate_id": self.candidate_id,
            "material": self.material.to_dict(),
            "classification": self.classification.to_dict(),
            "fields": [item.to_dict() for item in self.fields],
            "hypothesis": self.hypothesis.to_dict(),
            "candidate": {
                "name": self.candidate.name,
                "mechanism": self.candidate.mechanism,
                "description": self.candidate.description,
                "canonical_expression": self.candidate.canonical_expression,
                "direction": self.candidate.direction,
                "lookback": self.candidate.lookback,
                "dependencies": list(self.candidate.dependencies),
                "output_type": self.candidate.output_type,
                "steps": list(self.candidate.steps),
                "discovery_round": self.candidate.discovery_round,
                "discovery_index": self.candidate.discovery_index,
                "parameters": dict(self.candidate.parameters),
            },
            "created_at": self.created_at,
            "provenance_hash": self.provenance_hash,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ResearchCandidateProvenance":
        """严格解析候选来源。"""

        if isinstance(value, Mapping) and "provenance_hash" in value:
            value = {
                key: item
                for key, item in value.items()
                if key != "provenance_hash"
            }
        payload = _strict_fields(
            value,
            "research_candidate_provenance",
            set(cls.__dataclass_fields__),
        )
        payload["material"] = ResearchMaterialInput.from_dict(payload["material"])
        payload["classification"] = ResearchMaterialClassification.from_dict(
            payload["classification"]
        )
        payload["fields"] = tuple(
            ResearchFieldClaim.from_dict(item) for item in payload["fields"]
        )
        payload["checks"] = tuple(payload["checks"])
        payload["hypothesis"] = HypothesisCard.from_dict(payload["hypothesis"])
        payload["candidate"] = CandidateDefinition(**payload["candidate"])
        return cls(**payload)


def material_view_from_result(
    *,
    material: ResearchMaterialInput,
    status: str,
    classification: str | None,
    page_count: int,
    failure: ResearchMaterialFailure | None,
) -> ResearchMaterialView:
    """从解析结果构造不含正文的资料摘要。"""

    return ResearchMaterialView(
        document_id=material.document_id,
        relative_path=material.relative_path,
        status=status,
        classification=classification,
        page_count=page_count,
        error_code=None if failure is None else failure.error_code,
        failure_category=None if failure is None else failure.category,
        failure_message=None if failure is None else failure.message,
    )


__all__ = [
    "RESEARCH_RUN_SCHEMA_VERSION",
    "RESEARCH_RUN_STATUSES",
    "ResearchCandidateProvenance",
    "ResearchMaterialView",
    "ResearchRunDecision",
    "ResearchRunInput",
    "ResearchRunSummary",
    "material_view_from_result",
]
