"""V3 审核和共享发布测试共用的确定性数据。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import duckdb

from mining.contract_candidates import CandidateDefinition
from mining.contract_evidence import ApprovalDecision
from mining.contract_types import canonical_json, sha256_json
from mining.expression_validation import validate_expression
from mining.search_context import FrozenSearchContext
from mining.search_run import SearchRunRequest
from mining.storage_v3 import MiningV3Store
from mining.storage_v3_runtime import MiningV3RuntimeStore
from mining.storage_lock import database_state_lock
from mining.v3_contracts import (
    V3_REQUIRED_EVIDENCE_STAGES,
    CandidateEnvelope,
    EvidenceReference,
    ResearchInputEnvelope,
    ResearchSourceReference,
)
from mining.v3_sources import hypothesis_input_envelope


CREATED_AT = "2026-11-02T00:00:00+00:00"


def v3_context() -> FrozenSearchContext:
    """返回满足冻结窗口和预算约束的测试上下文。"""

    return FrozenSearchContext(
        data_fingerprint="a" * 64,
        created_at=CREATED_AT,
    )


def v3_candidate(
    name: str = "V3_SHARED_TREND",
) -> CandidateDefinition:
    """返回可进入正式构建的受控 DSL 候选。"""

    candidate = validate_expression(
        "mean(close, 5)",
        direction=1,
    ).to_candidate(
        mechanism="价格趋势",
        description="验证 V3 审核和共享发布闭环",
        steps=("读取收盘价", "计算五日均值"),
        discovery_round=1,
        discovery_index=1,
    )
    return replace(candidate, name=name)


def v3_evidence(
    candidate_id: str,
    *,
    failed_stage: str | None = None,
) -> tuple[EvidenceReference, ...]:
    """构造五阶段证据，可指定一个失败阶段。"""

    return tuple(
        EvidenceReference(
            candidate_id=candidate_id,
            stage=stage,
            evaluation_id=f"eval_{stage}",
            input_hash=f"{index + 1:x}" * 64,
            evidence_hash=f"{index + 5:x}" * 64,
            status="failed" if stage == failed_stage else "passed",
        )
        for index, stage in enumerate(V3_REQUIRED_EVIDENCE_STAGES)
    )


def v3_source(
    source_kind: str,
    *,
    suffix: str = "primary",
) -> ResearchSourceReference:
    """返回四类来源之一的白名单身份。"""

    costeer = source_kind == "costeer"
    return ResearchSourceReference(
        source_kind=source_kind,
        source_id=f"source_{source_kind}_{suffix}",
        source_hash=sha256_json(
            {"source_kind": source_kind, "suffix": suffix}
        ),
        artifact_manifest_hash="1" * 64 if costeer else None,
        promotion_hash="2" * 64 if costeer else None,
    )


def v3_input(
    source_kind: str,
    context: FrozenSearchContext,
    *,
    suffix: str = "primary",
) -> ResearchInputEnvelope:
    """构造与指定来源类型一致的 V3 输入。"""

    if source_kind == "hypothesis":
        request = SearchRunRequest(
            f"v3-{suffix}-0001",
            "价格趋势可能带来未来超额收益",
        )
        return hypothesis_input_envelope(
            request,
            context_hash=context.context_hash,
        )
    return ResearchInputEnvelope(
        run_id=f"researchrun_{source_kind[:8]:0<24}",
        idempotency_key=f"v3-{source_kind}-{suffix}",
        input_mode=source_kind,
        context_hash=context.context_hash,
        request_hash=sha256_json(
            {"source_kind": source_kind, "suffix": suffix}
        ),
        source=v3_source(source_kind, suffix=suffix),
    )


def v3_candidate_envelope(
    input_envelope: ResearchInputEnvelope,
    *,
    name: str = "V3_SHARED_TREND",
    failed_stage: str | None = None,
) -> CandidateEnvelope:
    """构造绑定 V3 输入来源的五阶段候选包络。"""

    candidate = v3_candidate(name)
    return CandidateEnvelope(
        candidate=candidate,
        source=v3_source(
            input_envelope.input_mode,
            suffix=(
                "report"
                if input_envelope.input_mode == "research_report"
                else "promotion"
                if input_envelope.input_mode == "costeer"
                else "primary"
            ),
        ),
        evidence=v3_evidence(
            candidate.candidate_id,
            failed_stage=failed_stage,
        ),
    )


def seed_v3_runtime_state(
    store: MiningV3Store,
    input_envelope: ResearchInputEnvelope,
    context: FrozenSearchContext,
    *,
    run_status: str = "review_ready",
) -> MiningV3RuntimeStore:
    """为 contract 单元测试建立最小且一致的 V3 运行状态。"""

    runtime = MiningV3RuntimeStore(
        store.db_path,
        store.data_root,
        contract_store=store,
    )
    windows = context.windows
    now = CREATED_AT
    with database_state_lock(store.db_path):
        with duckdb.connect(str(store.db_path)) as conn:
            with conn:
                conn.execute(
                    "INSERT INTO mining_v3.runtime_context VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?)",
                    [
                        context.context_id,
                        context.context_hash,
                        context.data_version,
                        context.data_fingerprint,
                        context.label,
                        context.universe,
                        context.price_channel,
                        windows.development_start,
                        windows.development_end,
                        windows.selection_start,
                        windows.selection_end,
                        windows.final_start,
                        windows.final_end,
                        windows.holdout_start,
                        windows.holdout_end,
                        context.field_catalog_hash,
                        context.operator_catalog_hash,
                        sha256_json(list(context.existing_factors)),
                        context.budget.budget_hash,
                        context.windows_hash,
                        canonical_json(context.to_dict()),
                        context.created_at,
                    ],
                )
                conn.execute(
                    "INSERT INTO mining_v3.runtime_run VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        input_envelope.run_id,
                        context.context_id,
                        input_envelope.idempotency_key,
                        input_envelope.request_hash,
                        run_status,
                        input_envelope.input_mode,
                        "{}",
                        None,
                        now,
                        now,
                    ],
                )
                conn.execute(
                    "INSERT INTO mining_v3.runtime_run_control VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        input_envelope.run_id,
                        run_status,
                        None,
                        False,
                        None,
                        None,
                        0,
                        None,
                        now,
                    ],
                )
    return runtime


def make_v3_candidate_review_ready(
    store: MiningV3Store,
    run_id: str,
    envelope: CandidateEnvelope,
) -> None:
    """把候选沿规范状态机推进到人工审核态。"""

    store.initialize_candidate_state(
        run_id,
        envelope.candidate.candidate_id,
        envelope.source.source_id,
    )
    for target in (
        "validating",
        "screening",
        "evaluating",
        "evaluated",
        "holdout_running",
        "review_ready",
    ):
        store.sync_candidate_state(
            run_id,
            envelope.candidate.candidate_id,
            target,
        )


def approved_v3_run(
    tmp_path: Path,
    *,
    source_kind: str = "hypothesis",
    name: str = "V3_SHARED_TREND",
) -> tuple[MiningV3Store, ResearchInputEnvelope, CandidateEnvelope]:
    """创建已经完成人工批准的 V3 运行。"""

    store = MiningV3Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    context = v3_context()
    input_envelope = v3_input(source_kind, context)
    envelope = v3_candidate_envelope(input_envelope, name=name)
    store.register_input(input_envelope)
    store.register_context(input_envelope.run_id, context)
    store.record_candidate_envelope(input_envelope.run_id, envelope)
    seed_v3_runtime_state(store, input_envelope, context)
    make_v3_candidate_review_ready(store, input_envelope.run_id, envelope)
    review = store.candidate_review(
        input_envelope.run_id,
        envelope.candidate.candidate_id,
        envelope.source.source_id,
    )
    decision = ApprovalDecision(
        run_id=input_envelope.run_id,
        candidate_id=envelope.candidate.candidate_id,
        decision="approved",
        reviewer="reviewer-a",
        note="通过统一 V3 审核",
        candidate_content_hash=review.candidate_content_hash,
        semantic_hash=review.semantic_hash,
        context_hash=review.context_hash,
        evidence_hash=review.approval_evidence_hash,
        created_at=CREATED_AT,
    )
    store.record_approval(decision, envelope.source.source_id)
    return store, input_envelope, envelope


__all__ = [
    "CREATED_AT",
    "approved_v3_run",
    "make_v3_candidate_review_ready",
    "seed_v3_runtime_state",
    "v3_candidate",
    "v3_candidate_envelope",
    "v3_context",
    "v3_evidence",
    "v3_input",
    "v3_source",
]
