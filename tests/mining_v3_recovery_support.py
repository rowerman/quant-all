"""V3 重启恢复测试共用的运行环境和确定性依赖。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mining.contract_evidence import ApprovalDecision
from mining.expression_validation import validate_expression
from mining.research_run_contracts import ResearchRunInput
from mining.search_context import FrozenSearchContext
from mining.search_orchestrator_v3 import MiningV3SearchOrchestrator
from mining.search_run import SearchRunRequest
from mining.storage_v3 import MiningV3Store
from mining.storage_v3_runtime import MiningV3RuntimeStore
from mining.v3_sources import (
    costeer_input_envelope,
    free_exploration_input_envelope,
    hypothesis_input_envelope,
    research_report_input_envelope,
)
from mining_sandbox.artifacts import PromotionRequest
from tests.test_mining_free_exploration import (
    FreeBranchWorker,
    _free_request,
    _rich_context,
)
from tests.test_mining_research_run import (
    FakeEvaluator,
    FakeFinalEvaluator,
    FakeHoldoutEvaluator,
    FakeMaterializer,
    FakePortfolio,
    FakeResearchConverter,
    FakeResearchExtractor,
    FakeSearchWorker,
    InterruptingEvaluator,
    _input,
    _write_pdf,
)
from tests.test_mining_search_orchestrator import (
    FakeDevelopmentEvaluator,
    FakeFinalEvaluator as SearchFakeFinalEvaluator,
    FakeHoldoutEvaluator as SearchFakeHoldoutEvaluator,
    FakeMaterializer as SearchFakeMaterializer,
    FakePortfolioSelector,
    FakeSelectionEvaluator,
    SequenceWorker,
)


class CrashOnSecondTurnWorker(SequenceWorker):
    """模拟第二轮 Worker 提交前主服务退出。"""

    def submit(self, request):
        if request.turn_index == 2:
            raise KeyboardInterrupt("模拟主服务第二轮前退出")
        return super().submit(request)


class CrashOnSecondFreeTurnWorker(FreeBranchWorker):
    """模拟自由探索第二轮 Worker 提交前主服务退出。"""

    def submit(self, request):
        if request.turn_index == 2:
            raise KeyboardInterrupt("模拟自由探索第二轮前退出")
        return super().submit(request)


class NoWorker:
    """CoSTEER 恢复不应访问搜索 Worker。"""

    def health(self):
        raise AssertionError("CoSTEER 恢复不应检查 Worker")

    def submit(self, request):
        raise AssertionError("CoSTEER 恢复不应提交 Worker")

    def get_job(self, job_id):
        raise AssertionError(job_id)

    def cancel(self, job_id):
        raise AssertionError(job_id)


def open_runtime(
    tmp_path: Path,
) -> tuple[MiningV3RuntimeStore, MiningV3Store]:
    """在既有数据库上重建 V3 契约和运行仓库。"""

    db_path = tmp_path / "factor_mining.duckdb"
    contract = MiningV3Store(db_path, tmp_path / "data")
    return (
        MiningV3RuntimeStore(
            db_path,
            tmp_path / "data",
            contract_store=contract,
        ),
        contract,
    )


def search_orchestrator(
    runtime: MiningV3RuntimeStore,
    worker,
    *,
    holdout_evaluator=None,
) -> MiningV3SearchOrchestrator:
    """构造使用确定性搜索组件的 V3 编排器。"""

    return MiningV3SearchOrchestrator(
        store=runtime,
        worker=worker,
        materializer=SearchFakeMaterializer(),
        development_evaluator=FakeDevelopmentEvaluator(),
        selection_evaluator=FakeSelectionEvaluator(),
        final_evaluator=SearchFakeFinalEvaluator(),
        holdout_evaluator=holdout_evaluator or SearchFakeHoldoutEvaluator(),
        portfolio_selector=FakePortfolioSelector(),
        snapshot_check=lambda *_args, **_kwargs: None,
    )


def research_orchestrator(
    runtime: MiningV3RuntimeStore,
    material_root: Path,
    *,
    interrupting: bool = False,
    holdout_evaluator=None,
) -> MiningV3SearchOrchestrator:
    """构造资料驱动恢复测试使用的 V3 编排器。"""

    return MiningV3SearchOrchestrator(
        store=runtime,
        worker=FakeSearchWorker(),
        research_material_root=material_root,
        research_extractor=FakeResearchExtractor(),
        research_converter=FakeResearchConverter(),
        materializer=FakeMaterializer(),
        development_evaluator=(
            InterruptingEvaluator() if interrupting else FakeEvaluator()
        ),
        selection_evaluator=FakeEvaluator(),
        final_evaluator=FakeFinalEvaluator(),
        holdout_evaluator=holdout_evaluator or FakeHoldoutEvaluator(),
        portfolio_selector=FakePortfolio(),
        snapshot_check=lambda *_args, **_kwargs: None,
    )


def build_hypothesis_fixture(tmp_path: Path):
    """创建可中断的手工假设运行。"""

    runtime, contract = open_runtime(tmp_path)
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=(
            {"name": "close"},
            {"name": "volume_share"},
        ),
        operator_catalog=({"name": "mean"},),
    )
    request = SearchRunRequest(
        "v3-recovery-hypothesis-0001",
        "短期反转可能带来开发期超额收益",
        run_id_prefix="researchrun",
    )
    envelope = hypothesis_input_envelope(
        request,
        context_hash=context.context_hash,
    )
    runtime.configure_v3_run(request, context, envelope)
    return runtime, contract, request, context


def build_free_fixture(tmp_path: Path):
    """创建可中断的自由探索运行。"""

    runtime, contract = open_runtime(tmp_path)
    context = _rich_context()
    base_request = _free_request("v3-recovery-free-0001", seed=53)
    request = replace(base_request, run_id_prefix="researchrun")
    envelope = free_exploration_input_envelope(
        request,
        context_hash=context.context_hash,
    )
    runtime.configure_v3_run(request, context, envelope)
    return runtime, contract, request, context


def build_research_fixture(tmp_path: Path):
    """创建带真实临时 PDF 和冻结资料清单的运行。"""

    root = tmp_path / "reports"
    _write_pdf(root / "report.pdf")
    _template, research_template, budget = _input(
        root,
        "v3-recovery-research-0001",
    )
    request = SearchRunRequest(
        research_template.idempotency_key,
        "资料驱动研究：1 份研报资料",
        budget=budget,
        run_id_prefix="researchrun",
    )
    research_input = replace(
        research_template,
        run_id=request.run_id,
    )
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        budget=budget,
    )
    envelope = research_report_input_envelope(
        research_input,
        context_hash=context.context_hash,
    )
    runtime, contract = open_runtime(tmp_path)
    runtime.configure_v3_run(
        request,
        context,
        envelope,
        research_input=research_input,
    )
    return runtime, contract, request, context, research_input, root


def build_costeer_fixture(tmp_path: Path, suffix: str):
    """创建已持久化人工转写候选的 CoSTEER 运行。"""

    runtime, contract = open_runtime(tmp_path)
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        field_catalog=({"name": "close"},),
        operator_catalog=({"name": "mean"},),
    )
    promotion = PromotionRequest(
        sandbox_run_id=f"sandbox_{suffix:0>24}",
        artifact_manifest_hash=(suffix[-1] or "1") * 64,
        reviewer="reviewer-a",
        reason="人工转写为受控 DSL",
        target_type="dsl",
        target_reference="mean(close, 5)",
    )
    candidate = validate_expression(
        promotion.target_reference,
        direction=1,
    ).to_candidate(
        mechanism="沙箱代码研究人工转写",
        description=f"CoSTEER 恢复候选 {suffix}",
        steps=("读取收盘价", "计算五日均值"),
        discovery_round=1,
        discovery_index=1,
    )
    request = SearchRunRequest(
        f"v3-recovery-costeer-{suffix}",
        "CoSTEER 人工提升候选",
        run_id_prefix="researchrun",
    )
    envelope = costeer_input_envelope(
        promotion,
        context_hash=context.context_hash,
        idempotency_key=request.idempotency_key,
        run_id=request.run_id,
    )
    runtime.configure_v3_run(
        request,
        context,
        envelope,
        seed_candidate=candidate,
    )
    return runtime, contract, request, context, candidate


def approve_first_candidate(
    contract: MiningV3Store,
    run_id: str,
) -> str:
    """批准运行中最先具备资格的候选。"""

    review = next(
        item
        for item in contract.list_candidate_reviews(run_id)
        if item.eligible
    )
    contract.record_approval(
        ApprovalDecision(
            run_id=run_id,
            candidate_id=review.candidate_id,
            decision="approved",
            reviewer="reviewer-a",
            note="恢复审批证据测试",
            candidate_content_hash=review.candidate_content_hash,
            semantic_hash=review.semantic_hash,
            context_hash=review.context_hash,
            evidence_hash=review.approval_evidence_hash,
        ),
        review.source_id,
    )
    return review.candidate_id
