"""V3 运行表、候选来源和五阶段证据同步测试。"""

from __future__ import annotations

from pathlib import Path

from mining.expression_validation import validate_expression
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_context import FrozenSearchContext
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_orchestrator_support import candidate_payload
from mining.search_run import SearchRunRequest
from mining.storage_v3 import MiningV3Store
from mining.storage_v3_runtime import MiningV3RuntimeStore
from mining.v3_sources import hypothesis_input_envelope
from mining.v3_sources import costeer_input_envelope
from mining.search_orchestrator_v3 import MiningV3SearchOrchestrator
from mining_sandbox.artifacts import PromotionRequest
from tests.test_mining_search_orchestrator import (
    FakeDevelopmentEvaluator,
    FakeFinalEvaluator,
    FakeHoldoutEvaluator,
    FakeMaterializer,
    FakePortfolioSelector,
    FakeSelectionEvaluator,
    SequenceWorker,
)


def test_v3_runtime_uses_researchrun_and_syncs_contract_evidence(tmp_path: Path):
    contract = MiningV3Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    runtime = MiningV3RuntimeStore(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
        contract_store=contract,
    )
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=({"name": "close"},),
        operator_catalog=({"name": "mean"},),
    )
    request = SearchRunRequest(
        "v3-runtime-0001",
        "短期反转可能带来开发期超额收益",
        run_id_prefix="researchrun",
    )
    envelope = hypothesis_input_envelope(
        request,
        context_hash=context.context_hash,
    )
    runtime.configure_v3_run(request, context, envelope)
    run = runtime.create_run(request, context)

    assert run["run_id"].startswith("researchrun_")
    assert contract.load_input(run["run_id"]) == envelope
    assert runtime.get_run(run["run_id"])["status"] == "queued"

    hypothesis = HypothesisCard(
        run_id=run["run_id"],
        turn_index=1,
        branch_id="branch-0001",
        statement="短期反转可能带来开发期超额收益",
        mechanism="短期价格修复",
        expected_direction="positive",
        falsification_criteria=("阶段证据未通过",),
    )
    candidate = validate_expression(
        "mean(close, 5)",
        direction=1,
    ).to_candidate(
        mechanism="短期价格修复",
        description="五日均值",
        steps=("读取收盘价", "计算五日均值"),
        discovery_round=1,
        discovery_index=1,
    )
    batch = ExperimentBatch(
        run_id=run["run_id"],
        hypothesis_id=hypothesis.hypothesis_id,
        turn_index=1,
        branch_id=hypothesis.branch_id,
        candidate_ids=(candidate.candidate_id,),
        parent_candidate_ids=(),
        development_start="2012-01-01",
        development_end="2019-12-31",
        input_hash="b" * 64,
    )
    runtime.begin_turn(run["run_id"], hypothesis, batch)
    runtime.record_candidate(
        run["run_id"],
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.content_hash,
        semantic_hash=candidate.semantic_hash,
        name=candidate.name,
        status="proposed",
        hypothesis_id=hypothesis.hypothesis_id,
        batch_id=batch.batch_id,
        payload=candidate_payload(candidate),
    )
    for index, stage in enumerate(
        ("technical", "development", "selection", "full", "holdout")
    ):
        runtime.record_evaluation_evidence(
            SearchEvaluationEvidence(
                run_id=run["run_id"],
                candidate_id=candidate.candidate_id,
                stage=stage,
                input_hash=f"{index + 1:x}" * 64,
                status="passed",
                metrics={"rank_ic_mean": 0.03, "rank_ic_ir": 0.4},
            )
        )

    source_id = contract.candidate_source_id(
        run["run_id"],
        candidate.candidate_id,
    )
    stored = contract.get_candidate_envelope(
        run["run_id"],
        candidate.candidate_id,
        source_id,
    )
    assert stored.source == envelope.source
    assert [item.stage for item in stored.evidence] == [
        "technical",
        "development",
        "selection",
        "full",
        "holdout",
    ]
    assert stored.candidate.candidate_id == candidate.candidate_id


def test_v3_orchestrator_completes_worker_search_into_review_ready(
    tmp_path: Path,
):
    contract = MiningV3Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    runtime = MiningV3RuntimeStore(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
        contract_store=contract,
    )
    orchestrator = MiningV3SearchOrchestrator(
        store=runtime,
        worker=SequenceWorker(),
        materializer=FakeMaterializer(),
        development_evaluator=FakeDevelopmentEvaluator(),
        selection_evaluator=FakeSelectionEvaluator(),
        final_evaluator=FakeFinalEvaluator(),
        holdout_evaluator=FakeHoldoutEvaluator(),
        portfolio_selector=FakePortfolioSelector(),
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=(
            {"name": "close"},
            {"name": "volume_share"},
        ),
        operator_catalog=({"name": "mean"},),
    )
    request = SearchRunRequest(
        "v3-orchestrator-0001",
        "短期反转可能带来开发期超额收益",
        run_id_prefix="researchrun",
    )
    input_envelope = hypothesis_input_envelope(
        request,
        context_hash=context.context_hash,
    )
    runtime.configure_v3_run(request, context, input_envelope)
    run = orchestrator.create_run(request, context, background=False)

    result = orchestrator.execute(run["run_id"])

    assert result["status"] == "review_ready"
    reviews = contract.list_candidate_reviews(run["run_id"])
    assert reviews
    assert all(review.eligible for review in reviews)
    assert {
        item.stage
        for review in reviews
        for item in review.evidence
    } == {"technical", "development", "selection", "full", "holdout"}
    orchestrator.close()


def test_v3_costeer_seed_skips_worker_and_reaches_review_ready(tmp_path: Path):
    contract = MiningV3Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    runtime = MiningV3RuntimeStore(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
        contract_store=contract,
    )
    worker = SequenceWorker()
    orchestrator = MiningV3SearchOrchestrator(
        store=runtime,
        worker=worker,
        materializer=FakeMaterializer(),
        development_evaluator=FakeDevelopmentEvaluator(),
        selection_evaluator=FakeSelectionEvaluator(),
        final_evaluator=FakeFinalEvaluator(),
        holdout_evaluator=FakeHoldoutEvaluator(),
        portfolio_selector=FakePortfolioSelector(),
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=({"name": "close"},),
        operator_catalog=({"name": "mean"},),
    )
    promotion = PromotionRequest(
        sandbox_run_id="sandbox_000000000000000000000001",
        artifact_manifest_hash="1" * 64,
        reviewer="reviewer-a",
        reason="人工转写为受控 DSL 并重新评估",
        target_type="dsl",
        target_reference="mean(close, 5)",
    )
    candidate = validate_expression(
        promotion.target_reference,
        direction=1,
    ).to_candidate(
        mechanism="沙箱代码研究人工转写",
        description="从 CoSTEER 沙箱人工转写的五日均值",
        steps=("读取收盘价", "计算五日均值"),
        discovery_round=1,
        discovery_index=1,
    )
    request = SearchRunRequest(
        "v3-costeer-0001",
        "CoSTEER 人工提升候选",
        run_id_prefix="researchrun",
    )
    input_envelope = costeer_input_envelope(
        promotion,
        context_hash=context.context_hash,
        idempotency_key=request.idempotency_key,
        run_id=request.run_id,
    )
    runtime.configure_v3_run(
        request,
        context,
        input_envelope,
        seed_candidate=candidate,
    )
    run = orchestrator.create_run(request, context, background=False)

    result = orchestrator.execute(run["run_id"])

    assert result["status"] == "review_ready"
    assert worker.requests == []
    reviews = contract.list_candidate_reviews(run["run_id"])
    assert len(reviews) == 1
    assert reviews[0].source_id == promotion.promotion_id
    assert reviews[0].eligible is True
    orchestrator.close()
