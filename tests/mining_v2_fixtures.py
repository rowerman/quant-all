"""V2 审批与发布测试共用夹具。"""

from __future__ import annotations

from pathlib import Path

from mining.contract_evidence import ApprovalDecision
from mining.expression_validation import validate_expression
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_context import FrozenSearchContext
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_orchestrator_support import candidate_payload
from mining.search_review import SEARCH_APPROVAL_REQUIRED_STAGES
from mining.search_run import SearchRunRequest
from mining.storage_v2 import MiningV2Store


def approved_search_run(
    tmp_path: Path,
) -> tuple[MiningV2Store, str, tuple[str, ...]]:
    """创建带两个已批准候选的 V2 审核态运行。"""

    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    run = store.create_run(
        SearchRunRequest(
            "search-publication-0001",
            "短期趋势可能带来开发期超额收益",
        ),
        FrozenSearchContext(data_fingerprint="a" * 64),
    )
    card = HypothesisCard(
        run_id=run["run_id"],
        turn_index=1,
        branch_id="branch-0001",
        statement="短期趋势可能带来开发期超额收益",
        mechanism="价格趋势延续",
        expected_direction="positive",
        falsification_criteria=("完整评估未通过",),
    )
    definitions = (
        validate_expression("mean(close, 5)", direction=1).to_candidate(
            mechanism="短期趋势",
            description="五日收盘均值",
            steps=("读取收盘价", "计算五日均值"),
            discovery_round=1,
            discovery_index=1,
        ),
        validate_expression("mean(close, 10)", direction=1).to_candidate(
            mechanism="中期趋势",
            description="十日收盘均值",
            steps=("读取收盘价", "计算十日均值"),
            discovery_round=1,
            discovery_index=2,
        ),
    )
    candidate_ids = tuple(item.candidate_id for item in definitions)
    batch = ExperimentBatch(
        run_id=run["run_id"],
        hypothesis_id=card.hypothesis_id,
        turn_index=1,
        branch_id=card.branch_id,
        candidate_ids=candidate_ids,
        parent_candidate_ids=(),
        development_start="2012-01-01",
        development_end="2019-12-31",
        input_hash="b" * 64,
    )
    store.begin_turn(run["run_id"], card, batch)
    for candidate_index, candidate in enumerate(definitions, start=1):
        store.record_candidate(
            run["run_id"],
            candidate_id=candidate.candidate_id,
            candidate_hash=candidate.content_hash,
            semantic_hash=candidate.semantic_hash,
            name=candidate.name,
            status="proposed",
            hypothesis_id=card.hypothesis_id,
            batch_id=batch.batch_id,
            payload=candidate_payload(candidate),
        )
        for target in (
            "validating",
            "screening",
            "evaluating",
            "evaluated",
            "holdout_running",
            "review_ready",
        ):
            store.transition_candidate(
                run["run_id"],
                candidate.candidate_id,
                target,
            )
        for stage_index, stage in enumerate(SEARCH_APPROVAL_REQUIRED_STAGES):
            suffix = f"{candidate_index:x}{stage_index + 1:x}" * 32
            store.record_evaluation_evidence(
                SearchEvaluationEvidence(
                    run_id=run["run_id"],
                    candidate_id=candidate.candidate_id,
                    stage=stage,
                    input_hash=suffix,
                    status="passed",
                    metrics={
                        "rank_ic_mean": 0.03,
                        "rank_ic_ir": 0.4,
                        "year_sign_consistency": 0.75,
                        "valid_days": 100,
                        "values": {"coverage": 0.8, "rows": 10000},
                    },
                )
            )
    for target in (
        "searching",
        "selecting",
        "evaluating",
        "holding_out",
        "review_ready",
    ):
        store.transition_search_run(
            run["run_id"],
            target,
            event_key=f"run_{target}",
            stage=target,
        )
    store.validate_search_trace(run["run_id"])
    for candidate_id in candidate_ids:
        eligibility = store.search_approval_eligibility(
            run["run_id"],
            candidate_id,
        )
        store.record_search_approval(
            ApprovalDecision(
                run_id=run["run_id"],
                candidate_id=candidate_id,
                decision="approved",
                reviewer="reviewer-1",
                note="证据完整，允许发布",
                candidate_content_hash=eligibility.candidate_content_hash,
                semantic_hash=eligibility.semantic_hash,
                context_hash=eligibility.context_hash,
                evidence_hash=eligibility.evidence_hash,
            )
        )
    return store, run["run_id"], candidate_ids
