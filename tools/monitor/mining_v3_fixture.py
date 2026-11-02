"""V3 四类来源页面验收夹具。"""

from __future__ import annotations

import atexit
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import backend.mining_v3_service as v3_module
from backend.app import create_app
from backend.mining_service import MiningWebService
from backend.mining_v3_service import MiningV3WebService
from mining.contract_requests import MiningConstraints
from mining.publication_service import MiningPublicationService
from mining.search_context import load_search_config
from mining.search_orchestrator_v3 import MiningV3SearchOrchestrator
from mining.storage import MiningStore
from mining.storage_v3 import MiningV3Store
from mining.storage_v3_runtime import MiningV3RuntimeStore
from mining_sandbox.artifacts import PromotionRequest
from tests.mining_v3_recovery_support import NoWorker
from tests.test_mining_free_exploration import FreeBranchWorker
from tests.test_mining_research_run import (
    FakeEvaluator,
    FakeFinalEvaluator,
    FakeHoldoutEvaluator,
    FakeMaterializer,
    FakePortfolio,
    FakeResearchConverter,
    FakeResearchExtractor,
    FakeSearchWorker,
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


class RecordingNoWorker(NoWorker):
    """记录不应发生的 Worker 请求。"""

    def __init__(self) -> None:
        self.requests = []


class SynchronousV3Orchestrator:
    """把测试用 V3 执行器包装为同步创建。"""

    def __init__(self, delegate: MiningV3SearchOrchestrator) -> None:
        self.delegate = delegate
        self.store = delegate.store

    def create_run(
        self,
        request,
        context,
        *,
        background=True,
        research_input=None,
    ):
        run = self.delegate.create_run(
            request,
            context,
            background=False,
            research_input=research_input,
        )
        return self.delegate.execute(run["run_id"])

    def progress(self, run_id):
        return self.delegate.progress(run_id)

    def cancel(self, run_id):
        return self.delegate.cancel(run_id)

    def resume(self, run_id, *, background=True):
        result = self.delegate.resume(run_id, background=False)
        self.delegate.execute(run_id)
        return result

    def close(self):
        self.delegate.close()


def _staged(candidates, **_kwargs):
    return SimpleNamespace(
        manifest_hash="a" * 64,
        artifact_sha256="b" * 64,
        factor_names=tuple(item["name"] for item in candidates),
    )


def _orchestrator(
    runtime: MiningV3RuntimeStore,
    kind: str,
    material_root: Path,
) -> MiningV3SearchOrchestrator:
    if kind == "research_report":
        return MiningV3SearchOrchestrator(
            store=runtime,
            worker=FakeSearchWorker(),
            research_material_root=material_root,
            research_extractor=FakeResearchExtractor(),
            research_converter=FakeResearchConverter(),
            materializer=FakeMaterializer(),
            development_evaluator=FakeEvaluator(),
            selection_evaluator=FakeEvaluator(),
            final_evaluator=FakeFinalEvaluator(),
            holdout_evaluator=FakeHoldoutEvaluator(),
            portfolio_selector=FakePortfolio(),
            snapshot_check=lambda *_args, **_kwargs: None,
        )
    worker = (
        FreeBranchWorker(("branch", "promote"))
        if kind == "free_exploration"
        else RecordingNoWorker()
        if kind == "costeer"
        else SequenceWorker()
    )
    return MiningV3SearchOrchestrator(
        store=runtime,
        worker=worker,
        materializer=SearchFakeMaterializer(),
        development_evaluator=FakeDevelopmentEvaluator(),
        selection_evaluator=FakeSelectionEvaluator(),
        final_evaluator=SearchFakeFinalEvaluator(),
        holdout_evaluator=SearchFakeHoldoutEvaluator(),
        portfolio_selector=FakePortfolioSelector(),
        research_material_root=material_root,
        snapshot_check=lambda *_args, **_kwargs: None,
    )


_ROOT = Path(tempfile.mkdtemp(prefix="quant-mining-v3-web-"))
_CONTRACT = MiningV3Store(_ROOT / "factor_mining.duckdb", _ROOT / "v3")
_RUNTIME = MiningV3RuntimeStore(
    _ROOT / "factor_mining.duckdb",
    _ROOT / "v3",
    contract_store=_CONTRACT,
)
_V1_STORE = MiningStore(_ROOT / "factor_mining_v1.duckdb", _ROOT / "v1")
_MATERIAL_ROOT = _ROOT / "reports"
_MATERIAL_ROOT.mkdir()
_write_pdf(_MATERIAL_ROOT / "report.pdf")
_PUBLICATION = MiningPublicationService(
    store=_CONTRACT,
    data_root=_ROOT / "publication-data",
    snapshot_check=lambda *_args, **_kwargs: None,
    prepare=_staged,
    commit=lambda **_kwargs: None,
    rollback=lambda **_kwargs: None,
)
_SERVICE = MiningV3WebService(
    contract_store=_CONTRACT,
    runtime_store=_RUNTIME,
    orchestrator=SynchronousV3Orchestrator(
        _orchestrator(_RUNTIME, "hypothesis", _MATERIAL_ROOT)
    ),
    publication_service=_PUBLICATION,
    material_root=_MATERIAL_ROOT,
    snapshot_check=lambda *_args, **_kwargs: None,
)
_SERVICE.initialize()
_PROMOTION = PromotionRequest(
    sandbox_run_id="sandbox_000000000000000000000001",
    artifact_manifest_hash="1" * 64,
    reviewer="fixture-reviewer",
    reason="人工转写为受控 DSL 并重新评估",
    target_type="dsl",
    target_reference="mean(close, 5)",
)
v3_module.list_manual_promotions = lambda **_kwargs: [_PROMOTION]


def _budget(*, research: bool = False) -> dict[str, object]:
    result = dict(load_search_config().budget.to_dict())
    if research:
        result["max_search_turns"] = 1
    return result


def _approve_and_publish(run_id: str) -> None:
    review = next(
        item for item in _CONTRACT.list_candidate_reviews(run_id)
        if item.eligible
    )
    _SERVICE.decide(
        run_id,
        review.candidate_id,
        {
            "decision": "approved",
            "reviewer": "fixture-reviewer",
            "note": "页面验收批准",
            "candidate_content_hash": review.candidate_content_hash,
            "evidence_hash": review.approval_evidence_hash,
        },
    )
    _PUBLICATION.start_batch(
        run_id,
        (review.candidate_id,),
        expected_context_hash=_CONTRACT.load_input(run_id).context_hash,
        background=False,
    )


def _create(kind: str) -> str:
    delegate = _orchestrator(_RUNTIME, kind, _MATERIAL_ROOT)
    _SERVICE.orchestrator = SynchronousV3Orchestrator(delegate)
    if kind == "hypothesis":
        run = _SERVICE.create_run(
            {
                "hypothesis": "手工假设来源候选可能带来开发期超额收益",
                "idempotency_key": "v3-browser-hypothesis-0001",
                "budget": _budget(),
            }
        )
    elif kind == "free_exploration":
        run = _SERVICE.create_run(
            {
                "input_mode": "free_exploration",
                "exploration_seed": "20261102",
                "idempotency_key": "v3-browser-free-0001",
                "budget": _budget(),
            }
        )
    elif kind == "research_report":
        run = _SERVICE.create_research_run(
            {
                "material_path": "report.pdf",
                "idempotency_key": "v3-browser-research-0001",
                "budget": _budget(research=True),
            }
        )
    elif kind == "costeer":
        run = _SERVICE.create_costeer_run(
            {
                "promotion_id": _PROMOTION.promotion_id,
                "idempotency_key": "v3-browser-costeer-0001",
                "budget": _budget(),
                "direction": 1,
                "mechanism": "沙箱代码研究人工转写",
                "description": "页面验收 CoSTEER 候选",
                "steps": ["读取收盘价", "计算五日均值"],
            }
        )
    else:
        raise ValueError(kind)
    run_id = run["run"]["run_id"]
    _approve_and_publish(run_id)
    return run_id


RUN_IDS = {
    "hypothesis": _create("hypothesis"),
    "free_exploration": _create("free_exploration"),
    "research_report": _create("research_report"),
    "costeer": _create("costeer"),
}

app = create_app(
    mining_service=MiningWebService(
        store=_V1_STORE,
        defaults=MiningConstraints(
            max_rounds=1,
            max_candidates=3,
            full_eval_top_k=2,
            max_llm_calls=3,
            min_quality_checks_for_failure_stop=3,
        ),
        reviewer="v3-browser-reviewer",
        snapshot_check=lambda *_args, **_kwargs: None,
        v3_service=_SERVICE,
    )
)


def cleanup() -> None:
    shutil.rmtree(_ROOT, ignore_errors=True)


atexit.register(cleanup)
