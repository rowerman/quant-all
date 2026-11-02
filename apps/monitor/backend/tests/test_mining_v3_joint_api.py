"""V3 四类来源真实 Web/API 和运行器联合验收。"""

from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace
import time

import factors
import pytest
import quant
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.mining_service import MiningWebService
from backend.mining_v3_service import MiningV3WebService
from common.paths import DB_FILE, FACTOR_DB_FILE
from mining.contract_requests import MiningConstraints
from mining.publication_service import MiningPublicationService
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


def _source_orchestrator(
    runtime: MiningV3RuntimeStore,
    kind: str,
    material_root: Path,
) -> tuple[MiningV3SearchOrchestrator, object]:
    if kind == "research_report":
        worker = FakeSearchWorker()
        orchestrator = MiningV3SearchOrchestrator(
            store=runtime,
            worker=worker,
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
        return orchestrator, worker
    worker = (
        FreeBranchWorker(("branch", "promote"))
        if kind == "free_exploration"
        else RecordingNoWorker()
        if kind == "costeer"
        else SequenceWorker()
    )
    orchestrator = MiningV3SearchOrchestrator(
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
    return orchestrator, worker


def _wait_publication(
    client: TestClient,
    run_id: str,
    batch_id: str,
    *,
    timeout: float = 10,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get(
            f"/api/mining/runs/{run_id}/publication-batches/{batch_id}"
        ).json()
        if payload["status"] not in {"staging", "validating"}:
            return payload
        time.sleep(0.02)
    raise AssertionError("共享发布未在超时内结束")


@pytest.mark.parametrize(
    "kind",
    ("hypothesis", "free_exploration", "research_report", "costeer"),
)
def test_v3_four_sources_use_real_web_api_runner_and_publication(
    tmp_path: Path,
    monkeypatch,
    kind: str,
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
    material_root = tmp_path / "reports"
    material_root.mkdir()
    worker = None
    if kind == "research_report":
        _write_pdf(material_root / "report.pdf")
    delegate, worker = _source_orchestrator(runtime, kind, material_root)
    promotion = PromotionRequest(
        sandbox_run_id="sandbox_000000000000000000000001",
        artifact_manifest_hash="1" * 64,
        reviewer="joint-reviewer",
        reason="联合验收人工转写",
        target_type="dsl",
        target_reference="mean(close, 5)",
    )
    monkeypatch.setattr(
        "backend.mining_v3_service.list_manual_promotions",
        lambda **_kwargs: [promotion],
    )
    publication = MiningPublicationService(
        store=contract,
        data_root=tmp_path / "publication-data",
        snapshot_check=lambda *_args, **_kwargs: None,
        prepare=_staged,
        commit=lambda **_kwargs: None,
        rollback=lambda **_kwargs: None,
    )
    service = MiningV3WebService(
        contract_store=contract,
        runtime_store=runtime,
        orchestrator=SynchronousV3Orchestrator(delegate),
        publication_service=publication,
        material_root=material_root,
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    v1_store = MiningStore(
        tmp_path / "factor_mining_v1.duckdb",
        tmp_path / "v1",
    )
    web = MiningWebService(
        store=v1_store,
        defaults=MiningConstraints(
            max_rounds=1,
            max_candidates=3,
            full_eval_top_k=2,
            max_llm_calls=3,
            min_quality_checks_for_failure_stop=3,
        ),
        v3_service=service,
    )
    app = create_app(mining_service=web)

    with TestClient(app) as client:
        options = client.get("/api/mining/options").json()["search_v3"]
        budget = dict(options["defaults"])
        if kind == "research_report":
            budget["max_search_turns"] = 1
        if kind == "hypothesis":
            response = client.post(
                "/api/mining/runs",
                json={
                    "hypothesis": "短期反转可能带来开发期超额收益",
                    "idempotency_key": "v3-joint-hypothesis-0001",
                    "budget": budget,
                },
            )
        elif kind == "free_exploration":
            response = client.post(
                "/api/mining/runs",
                json={
                    "input_mode": "free_exploration",
                    "exploration_seed": "20261102",
                    "idempotency_key": "v3-joint-free-0001",
                    "budget": budget,
                },
            )
        elif kind == "research_report":
            response = client.post(
                "/api/mining/research-runs",
                json={
                    "material_path": "report.pdf",
                    "idempotency_key": "v3-joint-research-0001",
                    "budget": budget,
                },
            )
        else:
            response = client.post(
                "/api/mining/costeer-runs",
                json={
                    "promotion_id": promotion.promotion_id,
                    "idempotency_key": "v3-joint-costeer-0001",
                    "budget": budget,
                    "direction": 1,
                    "mechanism": "沙箱代码研究人工转写",
                    "description": "CoSTEER 联合验收",
                    "steps": ["读取收盘价", "计算五日均值"],
                },
            )
        assert response.status_code == 200, response.text
        detail = response.json()
        run_id = detail["run"]["run_id"]
        assert detail["run"]["status"] == "review_ready", (
            detail["progress"]["stop_reason"]
        )
        assert detail["run"]["input_mode"] == kind

        reviews = contract.list_candidate_reviews(run_id)
        review = next(item for item in reviews if item.eligible)
        candidate_id = review.candidate_id
        candidate = client.get(
            f"/api/mining/runs/{run_id}/candidates/{candidate_id}"
        ).json()
        assert candidate["technical_gate"]["passed"] is True
        assert candidate["v3_source"]["source_kind"] == kind
        if kind == "research_report":
            assert candidate["research_source"]["provenance_hash"]
        if kind == "free_exploration":
            assert detail["exploration"]["initial_direction"]["seed"] == 20261102

        approved = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate_id}/decision",
            json={
                "decision": "approved",
                "reviewer": "joint-reviewer",
                "note": "四来源联合验收",
                "candidate_content_hash": review.candidate_content_hash,
                "evidence_hash": review.approval_evidence_hash,
            },
        )
        assert approved.status_code == 200, approved.text

        started = client.post(
            f"/api/mining/runs/{run_id}/publication-batches",
            json={
                "candidate_ids": [candidate_id],
                "expected_context_hash": contract.load_input(run_id).context_hash,
            },
        )
        assert started.status_code == 202, started.text
        published = _wait_publication(
            client,
            run_id,
            started.json()["batch_id"],
        )
        assert published["status"] == "published"
        assert contract.get_candidate(run_id, candidate_id)["status"] == "published"

    if kind in {"hypothesis", "free_exploration"}:
        assert worker.requests
        encoded = str([item.to_dict() for item in worker.requests])
        context = contract.load_context(run_id)
        assert context.windows.selection_start not in encoded
        assert context.windows.holdout_start not in encoded
    elif kind == "research_report":
        assert worker.requests == []
    else:
        assert worker.requests == []


@pytest.mark.skipif(
    not DB_FILE.exists() or not FACTOR_DB_FILE.exists(),
    reason="需要真实行情库和因子目录库",
)
def test_v3_web_publication_failure_retry_and_downstream_read(tmp_path: Path):
    contract = MiningV3Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    runtime = MiningV3RuntimeStore(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
        contract_store=contract,
    )
    factor_dir = tmp_path / "factors"
    factor_dir.mkdir()
    catalog_file = tmp_path / "factors.duckdb"
    shutil.copy2(FACTOR_DB_FILE, catalog_file)
    delegate, _worker = _source_orchestrator(
        runtime,
        "hypothesis",
        tmp_path / "reports",
    )
    publication = MiningPublicationService(
        store=contract,
        db_file=DB_FILE,
        factor_dir=factor_dir,
        catalog_file=catalog_file,
        data_root=tmp_path / "publication-data",
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    service = MiningV3WebService(
        contract_store=contract,
        runtime_store=runtime,
        orchestrator=SynchronousV3Orchestrator(delegate),
        publication_service=publication,
        db_file=DB_FILE,
        factor_dir=factor_dir,
        catalog_file=catalog_file,
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    v1_store = MiningStore(
        tmp_path / "factor_mining_v1.duckdb",
        tmp_path / "v1",
    )
    web = MiningWebService(
        store=v1_store,
        v3_service=service,
    )
    app = create_app(mining_service=web)

    with TestClient(app) as client:
        options = client.get("/api/mining/options").json()["search_v3"]
        budget = dict(options["defaults"])
        budget["max_search_turns"] = 1
        detail = client.post(
            "/api/mining/runs",
            json={
                "hypothesis": "短期反转可能带来开发期超额收益",
                "idempotency_key": "v3-joint-publication-0001",
                "budget": budget,
            },
        ).json()
        run_id = detail["run"]["run_id"]
        review = next(
            item for item in contract.list_candidate_reviews(run_id)
            if item.eligible
        )
        candidate_id = review.candidate_id
        decision = client.post(
            f"/api/mining/runs/{run_id}/candidates/{candidate_id}/decision",
            json={
                "decision": "approved",
                "reviewer": "joint-reviewer",
                "note": "共享发布联合验收",
                "candidate_content_hash": review.candidate_content_hash,
                "evidence_hash": review.approval_evidence_hash,
            },
        )
        assert decision.status_code == 200, decision.text
        request_body = {
            "candidate_ids": [candidate_id],
            "expected_context_hash": contract.load_input(run_id).context_hash,
        }

        real_prepare = publication._prepare
        publication._prepare = lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("模拟 Web 发布构建失败")
        )
        failed_start = client.post(
            f"/api/mining/runs/{run_id}/publication-batches",
            json=request_body,
        )
        assert failed_start.status_code == 202, failed_start.text
        failed = _wait_publication(
            client,
            run_id,
            failed_start.json()["batch_id"],
        )
        assert failed["status"] == "failed"
        assert contract.get_candidate(run_id, candidate_id)["status"] == "build_failed"
        assert not (factor_dir / "mined").exists()

        publication._prepare = real_prepare
        retry_start = client.post(
            f"/api/mining/runs/{run_id}/publication-batches",
            json=request_body,
        )
        assert retry_start.status_code == 202, retry_start.text
        published = _wait_publication(
            client,
            run_id,
            retry_start.json()["batch_id"],
            timeout=180,
        )
        assert published["status"] == "published"

    candidate = contract.get_candidate(run_id, candidate_id)
    factor_name = candidate["name"]
    assert factors.resolve(factor_name, factor_dir).family == "mined"
    listed = quant.load_factor_list(family="mined", db_path=catalog_file)
    assert factor_name in set(listed["name"])
