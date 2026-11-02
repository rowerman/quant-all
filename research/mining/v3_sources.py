"""V3 四类研究来源的统一适配。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from mining.contract_candidates import CandidateDefinition
from mining.contract_types import identifier, sha256_json, stable_id
from mining.errors import MiningContractError
from mining.free_exploration_contracts import ResearchDirection
from mining.research_run_contracts import (
    ResearchCandidateProvenance,
    ResearchRunInput,
)
from mining.search_contract_types import (
    FREE_EXPLORATION_INPUT_MODE,
    HYPOTHESIS_INPUT_MODE,
)
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_run import SearchRunRequest
from mining.v3_contracts import (
    V3_REQUIRED_EVIDENCE_STAGES,
    V3_RESEARCH_SOURCE_SCHEMA_VERSION,
    V3_RUN_ID_PREFIX,
    EvidenceReference,
    ResearchInputEnvelope,
    ResearchSourceReference,
)
from mining_sandbox.artifacts import PromotionRequest


def research_run_id(idempotency_key: str) -> str:
    """由幂等键派生 V3 运行 ID。"""

    return stable_id(
        V3_RUN_ID_PREFIX,
        {"idempotency_key": identifier(idempotency_key, "idempotency_key")},
    )


def _source_reference(
    *,
    source_kind: str,
    source_hash: str,
    artifact_manifest_hash: str | None = None,
    promotion_hash: str | None = None,
) -> ResearchSourceReference:
    """按来源类型和哈希构造不含原始内容的来源引用。"""

    return ResearchSourceReference(
        source_kind=source_kind,
        source_id=stable_id(
            "v3src",
            {
                "schema_version": V3_RESEARCH_SOURCE_SCHEMA_VERSION,
                "source_kind": source_kind,
                "source_hash": source_hash,
            },
        ),
        source_hash=source_hash,
        artifact_manifest_hash=artifact_manifest_hash,
        promotion_hash=promotion_hash,
    )


def _search_input_envelope(
    request: SearchRunRequest,
    context_hash: str,
    *,
    source_hash: str,
    source_kind: str,
) -> ResearchInputEnvelope:
    """构造手工假设或自由探索输入包络。"""

    run_id = (
        request.run_id
        if request.run_id.startswith(f"{V3_RUN_ID_PREFIX}_")
        else research_run_id(request.idempotency_key)
    )
    return ResearchInputEnvelope(
        run_id=run_id,
        idempotency_key=request.idempotency_key,
        input_mode=source_kind,
        context_hash=context_hash,
        request_hash=request.request_hash,
        source=_source_reference(
            source_kind=source_kind,
            source_hash=source_hash,
        ),
    )


def hypothesis_input_envelope(
    request: SearchRunRequest,
    *,
    context_hash: str,
) -> ResearchInputEnvelope:
    """把手工作假请求归一为 V3 输入。"""

    if not isinstance(request, SearchRunRequest):
        raise MiningContractError("request 必须是 SearchRunRequest")
    if request.input_mode != HYPOTHESIS_INPUT_MODE:
        raise MiningContractError(
            "只有 hypothesis 请求可以构造手工假设输入"
        )
    source_hash = sha256_json(
        {
            "source_kind": HYPOTHESIS_INPUT_MODE,
            "request_hash": request.request_hash,
            "idempotency_key": request.idempotency_key,
        }
    )
    return _search_input_envelope(
        request,
        context_hash,
        source_hash=source_hash,
        source_kind=HYPOTHESIS_INPUT_MODE,
    )


def free_exploration_input_envelope(
    request: SearchRunRequest,
    *,
    context_hash: str,
) -> ResearchInputEnvelope:
    """把自由探索请求归一为 V3 输入。"""

    if not isinstance(request, SearchRunRequest):
        raise MiningContractError("request 必须是 SearchRunRequest")
    if request.input_mode != FREE_EXPLORATION_INPUT_MODE:
        raise MiningContractError(
            "只有 free_exploration 请求可以构造自由探索输入"
        )
    source_hash = sha256_json(
        {
            "source_kind": FREE_EXPLORATION_INPUT_MODE,
            "request_hash": request.request_hash,
            "effective_seed": request.effective_exploration_seed,
            "seed_source": (
                "request"
                if request.exploration_seed is not None
                else "request_hash"
            ),
            "idempotency_key": request.idempotency_key,
        }
    )
    return _search_input_envelope(
        request,
        context_hash,
        source_hash=source_hash,
        source_kind=FREE_EXPLORATION_INPUT_MODE,
    )


def free_exploration_candidate_source(
    request: SearchRunRequest,
    direction: ResearchDirection,
) -> ResearchSourceReference:
    """绑定自由探索请求与已冻结研究方向。"""

    if not isinstance(request, SearchRunRequest):
        raise MiningContractError("request 必须是 SearchRunRequest")
    if request.input_mode != FREE_EXPLORATION_INPUT_MODE:
        raise MiningContractError("只有自由探索请求可以绑定研究方向")
    if not isinstance(direction, ResearchDirection):
        raise MiningContractError("direction 必须是 ResearchDirection")
    if direction.run_id != request.run_id:
        raise MiningContractError(
            "研究方向与自由探索请求不属于同一 V2 运行"
        )
    if (
        direction.request_hash != request.request_hash
        or direction.seed != request.effective_exploration_seed
    ):
        raise MiningContractError("研究方向没有绑定当前请求和有效种子")
    source_hash = sha256_json(
        {
            "source_kind": FREE_EXPLORATION_INPUT_MODE,
            "request_hash": request.request_hash,
            "effective_seed": request.effective_exploration_seed,
            "direction_hash": direction.direction_hash,
        }
    )
    return _source_reference(
        source_kind=FREE_EXPLORATION_INPUT_MODE,
        source_hash=source_hash,
    )


def research_report_input_envelope(
    research_input: ResearchRunInput,
    *,
    context_hash: str,
) -> ResearchInputEnvelope:
    """把资料运行输入归一为 V3 输入，不包含文件正文。"""

    if not isinstance(research_input, ResearchRunInput):
        raise MiningContractError("research_input 必须是 ResearchRunInput")
    source_hash = sha256_json(
        {
            "source_kind": "research_report",
            "input_hash": research_input.input_hash,
            "manifest_hash": research_input.manifest.manifest_hash,
        }
    )
    run_id = (
        research_input.run_id
        if research_input.run_id.startswith(f"{V3_RUN_ID_PREFIX}_")
        else research_run_id(research_input.idempotency_key)
    )
    return ResearchInputEnvelope(
        run_id=run_id,
        idempotency_key=research_input.idempotency_key,
        input_mode="research_report",
        context_hash=context_hash,
        request_hash=research_input.input_hash,
        source=_source_reference(
            source_kind="research_report",
            source_hash=source_hash,
        ),
    )


def research_report_candidate_source(
    provenance: ResearchCandidateProvenance,
) -> ResearchSourceReference:
    """构造单个资料候选的页码和转换来源引用。"""

    if not isinstance(provenance, ResearchCandidateProvenance):
        raise MiningContractError("provenance 必须是 ResearchCandidateProvenance")
    return _source_reference(
        source_kind="research_report",
        source_hash=provenance.provenance_hash,
    )


def source_independent_research_candidate(
    provenance: ResearchCandidateProvenance,
) -> CandidateDefinition:
    """移除 V2 资料兼容参数，恢复 V3 的来源无关候选身份。"""

    if not isinstance(provenance, ResearchCandidateProvenance):
        raise MiningContractError("provenance 必须是 ResearchCandidateProvenance")
    parameters = dict(provenance.candidate.parameters)
    if (
        parameters.get("source_kind") != "research_report"
        or parameters.get("research_provenance_hash")
        != provenance.provenance_hash
    ):
        raise MiningContractError("资料候选没有绑定当前资料 provenance")
    parameters.pop("source_kind")
    parameters.pop("research_provenance_hash")
    return replace(provenance.candidate, parameters=parameters)


def costeer_promotion_hash(promotion: PromotionRequest) -> str:
    """返回人工提升记录的稳定哈希。"""

    if not isinstance(promotion, PromotionRequest):
        raise MiningContractError("promotion 必须是 PromotionRequest")
    return sha256_json(promotion.material())


def costeer_candidate_source(
    promotion: PromotionRequest,
) -> ResearchSourceReference:
    """把已校验的人工提升记录归一为 CoSTEER 来源。"""

    promotion_hash = costeer_promotion_hash(promotion)
    source_hash = sha256_json(
        {
            "source_kind": "costeer",
            "sandbox_run_id": promotion.sandbox_run_id,
            "artifact_manifest_hash": promotion.artifact_manifest_hash,
            "promotion_hash": promotion_hash,
        }
    )
    return ResearchSourceReference(
        source_kind="costeer",
        source_id=promotion.promotion_id,
        source_hash=source_hash,
        artifact_manifest_hash=promotion.artifact_manifest_hash,
        promotion_hash=promotion_hash,
    )


def costeer_input_envelope(
    promotion: PromotionRequest,
    *,
    context_hash: str,
    idempotency_key: str | None = None,
    run_id: str | None = None,
) -> ResearchInputEnvelope:
    """构造 CoSTEER 人工提升导入的 V3 输入。"""

    if not isinstance(promotion, PromotionRequest):
        raise MiningContractError("promotion 必须是 PromotionRequest")
    key = promotion.promotion_id if idempotency_key is None else identifier(
        idempotency_key,
        "idempotency_key",
    )
    source = costeer_candidate_source(promotion)
    request_hash = sha256_json(
        {
            "source_reference_hash": source.reference_hash,
            "target_type": promotion.target_type,
            "target_reference_hash": sha256_json(
                {"target_reference": promotion.target_reference}
            ),
        }
    )
    return ResearchInputEnvelope(
        run_id=run_id or research_run_id(key),
        idempotency_key=key,
        input_mode="costeer",
        context_hash=context_hash,
        request_hash=request_hash,
        source=source,
    )


def evidence_reference_from_search(
    evidence: SearchEvaluationEvidence,
) -> EvidenceReference:
    """把 V2 搜索评估证据映射为 V3 五阶段证据引用。"""

    if not isinstance(evidence, SearchEvaluationEvidence):
        raise MiningContractError("evidence 必须是 SearchEvaluationEvidence")
    if evidence.stage not in V3_REQUIRED_EVIDENCE_STAGES:
        raise MiningContractError(
            f"V2 证据阶段 {evidence.stage} 不能进入 V3 五阶段证据"
        )
    return EvidenceReference(
        candidate_id=evidence.candidate_id,
        stage=evidence.stage,
        evaluation_id=evidence.evaluation_id,
        input_hash=evidence.input_hash,
        evidence_hash=evidence.evidence_hash,
        status=evidence.status,
    )


__all__ = [
    "costeer_candidate_source",
    "costeer_input_envelope",
    "costeer_promotion_hash",
    "evidence_reference_from_search",
    "free_exploration_candidate_source",
    "free_exploration_input_envelope",
    "hypothesis_input_envelope",
    "research_report_candidate_source",
    "research_report_input_envelope",
    "research_run_id",
    "source_independent_research_candidate",
]
