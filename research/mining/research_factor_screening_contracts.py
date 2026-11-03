"""研报因子草稿筛选、DSL 转换和尝试台账契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from mining.contract_candidates import CandidateDefinition
from mining.contract_types import (
    ensure_string_tuple,
    identifier,
    sha256_json,
    stable_id,
    text,
    utc_now_iso,
    validate_utc_timestamp,
)
from mining.errors import ResearchScreeningContractError
from mining.search_cards import HypothesisCard
from mining.search_contract_types import (
    bounded_integer,
    mapping_with_size,
    strict_fields,
)
from mining.search_trials import TrialRecord

RESEARCH_SCREENING_SCHEMA_VERSION = "mining_research_screening_v1"
RESEARCH_CONVERSION_SCHEMA_VERSION = "mining_research_conversion_v1"
RESEARCH_CONVERSION_PROMPT_VERSION = "mining_research_dsl_conversion_prompt_v1"
RESEARCH_SCREENING_STATUSES = (
    "accepted",
    "rejected",
    "invalid",
    "duplicate",
)


def _strict_fields(value: Any, name: str, expected: set[str]) -> dict[str, Any]:
    """校验 JSON 对象字段集合。"""

    try:
        return strict_fields(value, name, expected)
    except Exception as error:
        raise ResearchScreeningContractError(str(error)) from error


def _mapping(value: Any, name: str, *, maximum_bytes: int) -> dict[str, Any]:
    """校验并复制有大小上限的 JSON 对象。"""

    try:
        return mapping_with_size(value, name, maximum_bytes=maximum_bytes)
    except Exception as error:
        raise ResearchScreeningContractError(str(error)) from error


@dataclass(frozen=True)
class ResearchConversionContext:
    """把研报草稿绑定到既有 V2 运行、轮次和分支。"""

    run_id: str
    turn_index: int
    branch_id: str
    parent_hypothesis_id: str | None = None
    excluded_fields: tuple[str, ...] = ()
    schema_version: str = RESEARCH_SCREENING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_SCREENING_SCHEMA_VERSION:
            raise ResearchScreeningContractError(
                f"schema_version 必须为 {RESEARCH_SCREENING_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "turn_index",
            bounded_integer(
                self.turn_index,
                "turn_index",
                minimum=1,
                maximum=1000,
            ),
        )
        object.__setattr__(self, "branch_id", identifier(self.branch_id, "branch_id"))
        if self.parent_hypothesis_id is not None:
            object.__setattr__(
                self,
                "parent_hypothesis_id",
                identifier(self.parent_hypothesis_id, "parent_hypothesis_id"),
            )
        object.__setattr__(
            self,
            "excluded_fields",
            tuple(
                sorted(
                    ensure_string_tuple(
                        self.excluded_fields,
                        "excluded_fields",
                        maximum=500,
                    )
                )
            ),
        )

    def material(self) -> dict[str, Any]:
        """返回稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "turn_index": self.turn_index,
            "branch_id": self.branch_id,
            "parent_hypothesis_id": self.parent_hypothesis_id,
            "excluded_fields": list(self.excluded_fields),
        }

    @property
    def context_hash(self) -> str:
        """返回转换上下文哈希。"""

        return sha256_json(self.material())

    @classmethod
    def from_dict(cls, value: Any) -> ResearchConversionContext:
        """严格解析转换上下文。"""

        payload = _strict_fields(
            value,
            "research_conversion_context",
            set(cls.__dataclass_fields__),
        )
        payload["excluded_fields"] = tuple(payload["excluded_fields"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return self.material()


@dataclass(frozen=True)
class ResearchConversionLimits:
    """一次研报转换批次不可扩大的预算。"""

    max_documents: int = 16
    max_factors: int = 128
    max_factors_per_job: int = 8
    max_expression_depth: int = 6

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_documents",
            bounded_integer(
                self.max_documents,
                "max_documents",
                minimum=1,
                maximum=64,
            ),
        )
        object.__setattr__(
            self,
            "max_factors",
            bounded_integer(
                self.max_factors,
                "max_factors",
                minimum=1,
                maximum=512,
            ),
        )
        object.__setattr__(
            self,
            "max_factors_per_job",
            bounded_integer(
                self.max_factors_per_job,
                "max_factors_per_job",
                minimum=1,
                maximum=8,
            ),
        )
        object.__setattr__(
            self,
            "max_expression_depth",
            bounded_integer(
                self.max_expression_depth,
                "max_expression_depth",
                minimum=2,
                maximum=8,
            ),
        )
        if self.max_factors_per_job > self.max_factors:
            raise ResearchScreeningContractError(
                "max_factors_per_job 不得超过 max_factors"
            )

    def to_dict(self) -> dict[str, int]:
        """返回稳定字典。"""

        return {
            "max_documents": self.max_documents,
            "max_factors": self.max_factors,
            "max_factors_per_job": self.max_factors_per_job,
            "max_expression_depth": self.max_expression_depth,
        }

    @property
    def limits_hash(self) -> str:
        """返回预算哈希。"""

        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> ResearchConversionLimits:
        """严格解析预算。"""

        return cls(
            **_strict_fields(
                value,
                "research_conversion_limits",
                set(cls.__dataclass_fields__),
            )
        )


@dataclass(frozen=True)
class ResearchConvertedCandidate:
    """一个通过主服务校验的研报候选草稿。"""

    document_id: str
    extraction_id: str
    draft_id: str
    factor_hash: str
    checks: tuple[str, ...]
    field_mapping: Mapping[str, str]
    hypothesis: HypothesisCard
    candidate: CandidateDefinition
    schema_version: str = RESEARCH_CONVERSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_CONVERSION_SCHEMA_VERSION:
            raise ResearchScreeningContractError(
                f"schema_version 必须为 {RESEARCH_CONVERSION_SCHEMA_VERSION}"
            )
        for name in ("document_id", "extraction_id", "draft_id"):
            object.__setattr__(
                self,
                name,
                identifier(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "factor_hash",
            text(self.factor_hash, "factor_hash", minimum=64, maximum=64),
        )
        object.__setattr__(
            self,
            "checks",
            tuple(
                ensure_string_tuple(
                    self.checks,
                    "checks",
                    minimum=1,
                    maximum=16,
                )
            ),
        )
        field_mapping = _mapping(
            self.field_mapping,
            "field_mapping",
            maximum_bytes=8192,
        )
        if not field_mapping or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in field_mapping.items()
        ):
            raise ResearchScreeningContractError(
                "field_mapping 必须是非空字符串对象"
            )
        object.__setattr__(self, "field_mapping", field_mapping)
        if not isinstance(self.hypothesis, HypothesisCard):
            raise ResearchScreeningContractError("hypothesis 必须是 HypothesisCard")
        if not isinstance(self.candidate, CandidateDefinition):
            raise ResearchScreeningContractError(
                "candidate 必须是 CandidateDefinition"
            )

    def material(self) -> dict[str, Any]:
        """返回稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "extraction_id": self.extraction_id,
            "draft_id": self.draft_id,
            "factor_hash": self.factor_hash,
            "checks": list(self.checks),
            "field_mapping": dict(self.field_mapping),
            "hypothesis": self.hypothesis.material(),
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
        }

    @property
    def conversion_hash(self) -> str:
        """返回转换结果哈希。"""

        return sha256_json(self.material())


@dataclass(frozen=True)
class ResearchScreeningDecision:
    """一个研报因子草稿的确定结论。"""

    document_id: str
    draft_id: str
    factor_hash: str
    status: str
    reason_code: str | None
    checks: tuple[str, ...]
    candidate_id: str | None = None
    trial_id: str | None = None
    schema_version: str = RESEARCH_SCREENING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_SCREENING_SCHEMA_VERSION:
            raise ResearchScreeningContractError(
                f"schema_version 必须为 {RESEARCH_SCREENING_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "document_id", identifier(self.document_id, "document_id"))
        object.__setattr__(self, "draft_id", identifier(self.draft_id, "draft_id"))
        object.__setattr__(
            self,
            "factor_hash",
            text(self.factor_hash, "factor_hash", minimum=64, maximum=64),
        )
        if self.status not in RESEARCH_SCREENING_STATUSES:
            raise ResearchScreeningContractError(
                f"status 必须是 {', '.join(RESEARCH_SCREENING_STATUSES)} 之一"
            )
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
                    ensure_string_tuple(
                        self.checks,
                        "checks",
                        maximum=16,
                    )
                )
            ),
        )
        for name in ("candidate_id", "trial_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, identifier(value, name))
        if self.status == "accepted" and (self.candidate_id is None or self.reason_code):
            raise ResearchScreeningContractError(
                "accepted 结论必须包含 candidate_id 且没有 reason_code"
            )
        if self.status != "accepted" and self.reason_code is None:
            raise ResearchScreeningContractError(
                f"{self.status} 结论必须包含 reason_code"
            )
        if self.trial_id is None:
            raise ResearchScreeningContractError("结论必须包含 trial_id")

    def material(self) -> dict[str, Any]:
        """返回稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "draft_id": self.draft_id,
            "factor_hash": self.factor_hash,
            "status": self.status,
            "reason_code": self.reason_code,
            "checks": list(self.checks),
            "candidate_id": self.candidate_id,
            "trial_id": self.trial_id,
        }

    @property
    def decision_hash(self) -> str:
        """返回结论哈希。"""

        return sha256_json(self.material())

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return {**self.material(), "decision_hash": self.decision_hash}


@dataclass(frozen=True)
class ResearchConversionResult:
    """一次跨研报筛选、转换和去重后的完整结果。"""

    extraction_ids: tuple[str, ...]
    model: str
    catalog_hash: str
    context: ResearchConversionContext
    limits: ResearchConversionLimits
    candidates: tuple[ResearchConvertedCandidate, ...]
    decisions: tuple[ResearchScreeningDecision, ...]
    trials: tuple[TrialRecord, ...]
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = RESEARCH_SCREENING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_SCREENING_SCHEMA_VERSION:
            raise ResearchScreeningContractError(
                f"schema_version 必须为 {RESEARCH_SCREENING_SCHEMA_VERSION}"
            )
        object.__setattr__(
            self,
            "extraction_ids",
            tuple(
                sorted(
                    ensure_string_tuple(
                        self.extraction_ids,
                        "extraction_ids",
                        maximum=64,
                    )
                )
            ),
        )
        object.__setattr__(self, "model", text(self.model, "model", maximum=200))
        object.__setattr__(
            self,
            "catalog_hash",
            text(self.catalog_hash, "catalog_hash", minimum=64, maximum=64),
        )
        if not isinstance(self.context, ResearchConversionContext):
            raise ResearchScreeningContractError(
                "context 必须是 ResearchConversionContext"
            )
        if not isinstance(self.limits, ResearchConversionLimits):
            raise ResearchScreeningContractError(
                "limits 必须是 ResearchConversionLimits"
            )
        if not isinstance(self.candidates, (list, tuple)):
            raise ResearchScreeningContractError("candidates 必须是数组")
        candidates = tuple(self.candidates)
        if any(
            not isinstance(item, ResearchConvertedCandidate)
            for item in candidates
        ):
            raise ResearchScreeningContractError(
                "candidates 包含非法转换结果"
            )
        if len({item.candidate.candidate_id for item in candidates}) != len(candidates):
            raise ResearchScreeningContractError("candidates 不能包含重复候选")
        if not isinstance(self.decisions, (list, tuple)):
            raise ResearchScreeningContractError("decisions 必须是数组")
        decisions = tuple(self.decisions)
        if any(
            not isinstance(item, ResearchScreeningDecision)
            for item in decisions
        ):
            raise ResearchScreeningContractError(
                "decisions 包含非法筛选结论"
            )
        if len({item.trial_id for item in decisions}) != len(decisions):
            raise ResearchScreeningContractError("每个草稿必须对应唯一尝试记录")
        if not isinstance(self.trials, (list, tuple)):
            raise ResearchScreeningContractError("trials 必须是数组")
        trials = tuple(self.trials)
        if any(not isinstance(item, TrialRecord) for item in trials):
            raise ResearchScreeningContractError("trials 包含非法尝试记录")
        if {item.trial_id for item in trials} != {
            item.trial_id for item in decisions
        }:
            raise ResearchScreeningContractError("结论和尝试记录必须一一对应")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "trials", trials)
        object.__setattr__(
            self,
            "created_at",
            validate_utc_timestamp(self.created_at, "created_at"),
        )

    def material(self) -> dict[str, Any]:
        """返回不包含创建时间的稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "extraction_ids": list(self.extraction_ids),
            "model": self.model,
            "catalog_hash": self.catalog_hash,
            "context": self.context.to_dict(),
            "limits": self.limits.to_dict(),
            "candidates": [item.material() for item in self.candidates],
            "decisions": [item.material() for item in self.decisions],
            "trials": [item.material() for item in self.trials],
        }

    @property
    def conversion_hash(self) -> str:
        """返回批次稳定哈希。"""

        return sha256_json(self.material())

    @property
    def conversion_id(self) -> str:
        """返回批次稳定 ID。"""

        return stable_id("researchconv", {"conversion_hash": self.conversion_hash})

    def to_dict(self) -> dict[str, Any]:
        """返回页面和持久化使用的字典。"""

        return {
            **self.material(),
            "conversion_hash": self.conversion_hash,
            "conversion_id": self.conversion_id,
            "created_at": self.created_at,
        }


__all__ = [
    "RESEARCH_CONVERSION_PROMPT_VERSION",
    "RESEARCH_CONVERSION_SCHEMA_VERSION",
    "RESEARCH_SCREENING_SCHEMA_VERSION",
    "RESEARCH_SCREENING_STATUSES",
    "ResearchConversionContext",
    "ResearchConversionLimits",
    "ResearchConversionResult",
    "ResearchConvertedCandidate",
    "ResearchScreeningDecision",
]
