"""V2 搜索候选进入现有审批和发布链路前的只读映射契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mining.contract_candidates import CandidateDefinition
from mining.contract_requests import FrozenContext, MiningConstraints
from mining.contract_types import identifier, sha256_json
from mining.errors import MiningContractError
from mining.search_context import FrozenSearchContext
from mining.search_contract_types import hash_text

SEARCH_PUBLICATION_BRIDGE_SCHEMA_VERSION = "mining_search_publication_bridge_v1"
SEARCH_PUBLICATION_CANDIDATE_STATUS = "review_ready"


@dataclass(frozen=True)
class SearchEvidenceReference:
    """发布桥接保留的不可变评估证据身份。"""

    stage: str
    evaluation_id: str
    input_hash: str
    evidence_hash: str
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", identifier(self.stage, "stage"))
        object.__setattr__(
            self,
            "evaluation_id",
            identifier(self.evaluation_id, "evaluation_id"),
        )
        object.__setattr__(
            self,
            "input_hash",
            hash_text(self.input_hash, "input_hash"),
        )
        object.__setattr__(
            self,
            "evidence_hash",
            hash_text(self.evidence_hash, "evidence_hash"),
        )
        object.__setattr__(self, "status", identifier(self.status, "status"))

    def to_dict(self) -> dict[str, str]:
        """返回稳定证据引用字典。"""

        return {
            "stage": self.stage,
            "evaluation_id": self.evaluation_id,
            "input_hash": self.input_hash,
            "evidence_hash": self.evidence_hash,
            "status": self.status,
        }


@dataclass(frozen=True)
class SearchPublicationCandidate:
    """经过 V2 轨迹校验的候选发布输入视图。"""

    run_id: str
    candidate_id: str
    content_hash: str
    semantic_hash: str
    name: str
    definition: CandidateDefinition
    status: str
    hypothesis_id: str
    batch_id: str
    source_context_hash: str
    source_trace_hash: str
    scorecard_hash: str
    evidence: tuple[SearchEvidenceReference, ...]
    schema_version: str = SEARCH_PUBLICATION_BRIDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_PUBLICATION_BRIDGE_SCHEMA_VERSION:
            raise MiningContractError("V2 发布桥接 schema_version 不匹配")
        for name in ("run_id", "candidate_id", "hypothesis_id", "batch_id"):
            object.__setattr__(
                self,
                name,
                identifier(getattr(self, name), name),
            )
        for name in (
            "content_hash",
            "semantic_hash",
            "source_context_hash",
            "source_trace_hash",
            "scorecard_hash",
        ):
            object.__setattr__(
                self,
                name,
                hash_text(getattr(self, name), name),
            )
        if not isinstance(self.definition, CandidateDefinition):
            raise MiningContractError("definition 必须是 CandidateDefinition")
        if self.status != SEARCH_PUBLICATION_CANDIDATE_STATUS:
            raise MiningContractError(
                f"候选状态 {self.status} 不能进入发布桥接"
            )
        if self.candidate_id != self.definition.candidate_id:
            raise MiningContractError("候选 ID 与冻结定义不一致")
        if self.content_hash != self.definition.content_hash:
            raise MiningContractError("候选内容哈希与冻结定义不一致")
        if self.semantic_hash != self.definition.semantic_hash:
            raise MiningContractError("候选语义哈希与冻结定义不一致")
        if self.name != self.definition.name:
            raise MiningContractError("候选名称与冻结定义不一致")
        if isinstance(self.evidence, str) or not isinstance(
            self.evidence,
            (list, tuple),
        ):
            raise MiningContractError("evidence 必须是证据引用序列")
        evidence = tuple(self.evidence)
        if not evidence:
            raise MiningContractError("进入发布桥接的候选必须有评估证据")
        if any(not isinstance(item, SearchEvidenceReference) for item in evidence):
            raise MiningContractError("evidence 包含未知对象")
        evidence_ids = tuple(item.evaluation_id for item in evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise MiningContractError("evidence 不能包含重复评估 ID")
        object.__setattr__(
            self,
            "evidence",
            tuple(
                sorted(
                    evidence,
                    key=lambda item: (
                        item.stage,
                        item.input_hash,
                        item.evaluation_id,
                    ),
                )
            ),
        )

    @property
    def evidence_set_hash(self) -> str:
        """返回全部分阶段证据引用的内容哈希。"""

        return sha256_json([item.to_dict() for item in self.evidence])

    def publication_candidate(self) -> dict[str, Any]:
        """返回现有发布构建器使用的轻量字段。"""

        return {
            "candidate_id": self.candidate_id,
            "content_hash": self.content_hash,
            "semantic_hash": self.semantic_hash,
            "name": self.name,
            "definition": {
                "name": self.definition.name,
                "mechanism": self.definition.mechanism,
                "description": self.definition.description,
                "canonical_expression": self.definition.canonical_expression,
                "direction": self.definition.direction,
                "lookback": self.definition.lookback,
                "dependencies": list(self.definition.dependencies),
                "output_type": self.definition.output_type,
                "steps": list(self.definition.steps),
                "discovery_round": self.definition.discovery_round,
                "discovery_index": self.definition.discovery_index,
                "parameters": dict(self.definition.parameters),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """返回用于审计和页面确认的稳定字典。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "content_hash": self.content_hash,
            "semantic_hash": self.semantic_hash,
            "name": self.name,
            "definition": self.publication_candidate()["definition"],
            "status": self.status,
            "hypothesis_id": self.hypothesis_id,
            "batch_id": self.batch_id,
            "source_context_hash": self.source_context_hash,
            "source_trace_hash": self.source_trace_hash,
            "scorecard_hash": self.scorecard_hash,
            "evidence": [item.to_dict() for item in self.evidence],
            "evidence_set_hash": self.evidence_set_hash,
        }


@dataclass(frozen=True)
class SearchPublicationInputs:
    """一个已进入审核态 V2 运行的发布桥接输入。"""

    run_id: str
    source_context_hash: str
    source_trace_hash: str
    builder_context: FrozenContext
    candidates: tuple[SearchPublicationCandidate, ...]
    schema_version: str = SEARCH_PUBLICATION_BRIDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_PUBLICATION_BRIDGE_SCHEMA_VERSION:
            raise MiningContractError("V2 发布输入 schema_version 不匹配")
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "source_context_hash",
            hash_text(self.source_context_hash, "source_context_hash"),
        )
        object.__setattr__(
            self,
            "source_trace_hash",
            hash_text(self.source_trace_hash, "source_trace_hash"),
        )
        if not isinstance(self.builder_context, FrozenContext):
            raise MiningContractError("builder_context 必须是 FrozenContext")
        if isinstance(self.candidates, str) or not isinstance(
            self.candidates,
            (list, tuple),
        ):
            raise MiningContractError("candidates 必须是候选视图序列")
        candidates = tuple(self.candidates)
        if not candidates:
            raise MiningContractError("发布桥接至少需要一个候选")
        if any(
            not isinstance(item, SearchPublicationCandidate)
            for item in candidates
        ):
            raise MiningContractError("candidates 包含未知对象")
        candidate_ids = tuple(item.candidate_id for item in candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise MiningContractError("candidates 不能包含重复候选")
        if any(item.run_id != self.run_id for item in candidates):
            raise MiningContractError("候选与发布运行不一致")
        if any(
            item.source_context_hash != self.source_context_hash
            or item.source_trace_hash != self.source_trace_hash
            for item in candidates
        ):
            raise MiningContractError("候选来源上下文或轨迹与发布输入不一致")
        object.__setattr__(
            self,
            "candidates",
            tuple(sorted(candidates, key=lambda item: item.candidate_id)),
        )

    def publication_candidates(self) -> tuple[dict[str, Any], ...]:
        """返回按候选 ID 排序的发布构建器输入。"""

        return tuple(
            item.publication_candidate()
            for item in self.candidates
        )

    def to_dict(self) -> dict[str, Any]:
        """返回不持久化派生上下文的审计摘要。"""

        context = self.builder_context
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "source_context_hash": self.source_context_hash,
            "source_trace_hash": self.source_trace_hash,
            "derived_builder_context_hash": context.context_hash,
            "data_fingerprint": context.data_fingerprint,
            "research_start": context.research_start,
            "research_end": context.research_end,
            "full_eval_start": context.full_eval_start,
            "full_eval_end": context.full_eval_end,
            "holdout_start": context.holdout_start,
            "holdout_end": context.holdout_end,
            "candidates": [item.to_dict() for item in self.candidates],
        }


def publication_context(context: FrozenSearchContext) -> FrozenContext:
    """把 V2 冻结口径转换为仅供发布构建器使用的临时 V1 上下文。"""

    if not isinstance(context, FrozenSearchContext):
        raise MiningContractError("context 必须是 FrozenSearchContext")
    budget = context.budget
    policy = context.control_policy
    return FrozenContext(
        data_version=context.data_version,
        data_fingerprint=context.data_fingerprint,
        label=context.label,
        universe=context.universe,
        price_channel=context.price_channel,
        research_start=context.windows.development_start,
        research_end=context.windows.final_end,
        full_eval_start=context.windows.final_start,
        full_eval_end=context.windows.final_end,
        holdout_start=context.windows.holdout_start,
        holdout_end=context.windows.holdout_end,
        llm_profile=context.llm_profile,
        llm_model=context.llm_model,
        field_catalog=context.field_catalog,
        operator_catalog=context.operator_catalog,
        existing_factors=context.existing_factors,
        budget=MiningConstraints(
            max_rounds=budget.max_search_turns,
            max_candidates=budget.max_candidates_per_run,
            full_eval_top_k=budget.max_full_evaluations,
            max_llm_calls=budget.max_search_turns,
            max_expression_depth=6,
            min_coverage=policy.min_coverage,
            max_failure_rate=policy.max_failure_rate,
            min_quality_checks_for_failure_stop=min(
                policy.min_quality_checks_for_failure_stop,
                budget.max_candidates_per_run,
            ),
        ),
        created_at=context.created_at,
    )


__all__ = [
    "SEARCH_PUBLICATION_BRIDGE_SCHEMA_VERSION",
    "SEARCH_PUBLICATION_CANDIDATE_STATUS",
    "SearchEvidenceReference",
    "SearchPublicationCandidate",
    "SearchPublicationInputs",
    "publication_context",
]
