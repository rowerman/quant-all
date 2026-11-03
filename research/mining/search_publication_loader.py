"""只读加载 V2 搜索运行并构造发布桥接输入。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from mining.contract_types import identifier
from mining.errors import MiningContractError
from mining.search_context import FrozenSearchContext
from mining.search_contract_types import hash_text
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_orchestrator_support import candidate_from_v2_record
from mining.search_publication_contracts import (
    SEARCH_PUBLICATION_CANDIDATE_STATUS,
    SearchEvidenceReference,
    SearchPublicationCandidate,
    SearchPublicationInputs,
    publication_context,
)
from mining.search_scorecard import build_candidate_scorecard
from mining.storage_v2 import MiningV2Store


def _evidence_reference(
    row: Mapping[str, Any],
    evidence: SearchEvaluationEvidence,
) -> SearchEvidenceReference:
    """校验存储列与证据正文后返回轻量引用。"""

    for name in (
        "run_id",
        "candidate_id",
        "stage",
        "input_hash",
        "status",
    ):
        if row.get(name) != getattr(evidence, name):
            raise MiningContractError(f"V2 评估证据 {name} 与正文不一致")
    if row.get("evaluation_id") != evidence.evaluation_id:
        raise MiningContractError("V2 评估证据 ID 与正文不一致")
    if row.get("evidence_hash") != evidence.evidence_hash:
        raise MiningContractError("V2 评估证据哈希与正文不一致")
    return SearchEvidenceReference(
        stage=evidence.stage,
        evaluation_id=evidence.evaluation_id,
        input_hash=evidence.input_hash,
        evidence_hash=evidence.evidence_hash,
        status=evidence.status,
    )


def load_search_publication_inputs(
    store: MiningV2Store,
    run_id: str,
    candidate_ids: Sequence[str],
) -> SearchPublicationInputs:
    """只读加载审核态 V2 运行的发布桥接输入。"""

    if not isinstance(store, MiningV2Store):
        raise MiningContractError("store 必须是 MiningV2Store")
    run_id = identifier(run_id, "run_id")
    if isinstance(candidate_ids, str) or not isinstance(
        candidate_ids,
        Sequence,
    ):
        raise MiningContractError("candidate_ids 必须是标识序列")
    selected = tuple(
        sorted(identifier(item, "candidate_id") for item in candidate_ids)
    )
    if not selected:
        raise MiningContractError("candidate_ids 不能为空")
    if len(set(selected)) != len(selected):
        raise MiningContractError("candidate_ids 不能重复")

    run = store.get_run(run_id)
    if run["status"] != SEARCH_PUBLICATION_CANDIDATE_STATUS:
        raise MiningContractError(
            f"V2 运行状态 {run['status']} 不能进入发布桥接"
        )
    store.validate_search_trace(run_id)
    context = FrozenSearchContext.from_dict(run["context"])
    source_context_hash = hash_text(run["context_hash"], "context_hash")
    if source_context_hash != context.context_hash:
        raise MiningContractError("V2 存储上下文哈希与正文不一致")
    trace_hash = hash_text(run["trace_hash"], "trace_hash")

    candidates = []
    for candidate_id in selected:
        record = store.get_candidate(run_id, candidate_id)
        if record["status"] != SEARCH_PUBLICATION_CANDIDATE_STATUS:
            raise MiningContractError(
                f"候选 {candidate_id} 状态为 {record['status']}，不能进入发布桥接"
            )
        definition = candidate_from_v2_record(record)
        if record["candidate_hash"] != definition.content_hash:
            raise MiningContractError(
                f"候选 {candidate_id} 内容哈希与冻结定义不一致"
            )
        if record["semantic_hash"] != definition.semantic_hash:
            raise MiningContractError(
                f"候选 {candidate_id} 语义哈希与冻结定义不一致"
            )
        if record["name"] != definition.name:
            raise MiningContractError(
                f"候选 {candidate_id} 名称与冻结定义不一致"
            )
        if definition.parameters.get("source_kind") == "research_report":
            store.require_research_candidate(
                run_id,
                candidate_id,
                candidate_content_hash=definition.content_hash,
            )
        evidence = tuple(
            _evidence_reference(
                row,
                SearchEvaluationEvidence.from_dict(row["evidence"]),
            )
            for row in store.list_evaluation_evidence(run_id, candidate_id)
        )
        scorecard = build_candidate_scorecard(store, run_id, candidate_id)
        lineage = record["lineage"]
        candidates.append(
            SearchPublicationCandidate(
                run_id=run_id,
                candidate_id=candidate_id,
                content_hash=definition.content_hash,
                semantic_hash=definition.semantic_hash,
                name=definition.name,
                definition=definition,
                status=record["status"],
                hypothesis_id=str(lineage["hypothesis_id"]),
                batch_id=str(lineage["batch_id"]),
                source_context_hash=source_context_hash,
                source_trace_hash=trace_hash,
                scorecard_hash=scorecard.scorecard_hash,
                evidence=evidence,
            )
        )
    return SearchPublicationInputs(
        run_id=run_id,
        source_context_hash=source_context_hash,
        source_trace_hash=trace_hash,
        builder_context=publication_context(context),
        candidates=tuple(candidates),
    )


__all__ = ["load_search_publication_inputs"]
