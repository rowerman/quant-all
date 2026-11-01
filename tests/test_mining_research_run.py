"""资料驱动运行入口、V2 种子和来源绑定测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import duckdb
import json
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
import pytest

from mining.contract_evidence import EvaluationReport, TechnicalGate
from mining.contract_evidence import ApprovalDecision
from mining.contract_types import sha256_json, utc_now_iso
from mining.evaluation_contract import (
    CandidateEvaluationResult,
    FullEvaluationResult,
    HoldoutEvaluationResult,
)
from mining.errors import MiningStorageError
from mining.errors import MiningContractError
from mining.publication_service import MiningPublicationService
from mining.research_factor_screening_contracts import ResearchConversionLimits
from mining.research_material_io import (
    build_research_material_manifest,
    discover_research_material_paths,
)
from mining.research_material_parser import RESEARCH_PDF_PARSER_VERSION
from mining.research_run_contracts import ResearchRunInput
from mining.research_run_pipeline import prepare_research_run
from mining.search_budget import SearchBudget
from mining.search_context import FrozenSearchContext
from mining.search_cards import ExperimentBatch
from mining.search_orchestrator import MiningSearchOrchestrator
from mining.search_publication_loader import load_search_publication_inputs
from mining.search_portfolio import PortfolioSelectionResult
from mining.search_run import SearchRunRequest
from mining.storage_v2 import MiningV2Store
from backend.mining_search_service import MiningSearchWebService
from mining_worker.protocol import WorkerJobView
from mining_worker.protocol_research import (
    WorkerResearchCitationDraft,
    WorkerResearchExtractionResponse,
    WorkerResearchFactorDraft,
    WorkerResearchFieldDraft,
)
from mining_worker.protocol_research_conversion import (
    WorkerResearchConversionOutcome,
    WorkerResearchConversionResponse,
)


SENTENCE = (
    "Equity research report stock valuation factor: "
    "short reversal rank(ret) is positive."
)


def _write_pdf(path: Path, text: str = SENTENCE) -> None:
    """写入一份可分类的最小金融研报 PDF。"""

    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    page = writer.add_blank_page(width=612, height=792)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font_reference}
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(
        f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    )
    page.replace_contents(stream)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        writer.write(handle)


class FakeResearchExtractor:
    """返回六字段完整的固定研报草稿。"""

    def __init__(self) -> None:
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        citation = WorkerResearchCitationDraft(
            page_number=1,
            quote=SENTENCE,
        )
        values = (
            ("factor_name", ("short reversal",)),
            ("economic_mechanism", ("short-term price repair",)),
            ("formula", ("rank(ret)",)),
            ("variables", ("ret",)),
            ("expected_direction", ("positive",)),
            ("falsification_conditions", ("rank ic below threshold",)),
        )
        return WorkerResearchExtractionResponse(
            request_id=request.request_id,
            factors=(
                WorkerResearchFactorDraft(
                    fields=tuple(
                        WorkerResearchFieldDraft(
                            field_name=name,
                            values=field_values,
                            status="explicit",
                            citations=(citation,),
                        )
                        for name, field_values in values
                    )
                ),
            ),
            usage={"calls": 1},
        )


class FakeResearchConverter:
    """把研报变量 ret 映射到冻结字段 ret。"""

    def __init__(self) -> None:
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return WorkerResearchConversionResponse(
            request_id=request.request_id,
            outcomes=tuple(
                WorkerResearchConversionOutcome(
                    draft_id=item.draft_id,
                    status="converted",
                    expression="rank(ret)",
                    field_mapping={"ret": "ret"},
                )
                for item in request.drafts
            ),
            usage={"calls": 1},
        )


class TwoDraftExtractor(FakeResearchExtractor):
    """返回两个名称不同、可转换为不同表达式的草稿。"""

    def generate(self, request):
        self.requests.append(request)
        citation = WorkerResearchCitationDraft(
            page_number=1,
            quote=SENTENCE,
        )
        drafts = []
        for name in ("short reversal", "trend persistence"):
            values = (
                ("factor_name", (name,)),
                ("economic_mechanism", ("price behavior",)),
                ("formula", ("rank(ret)",)),
                ("variables", ("ret",)),
                ("expected_direction", ("positive",)),
                ("falsification_conditions", ("rank ic below threshold",)),
            )
            drafts.append(
                WorkerResearchFactorDraft(
                    fields=tuple(
                        WorkerResearchFieldDraft(
                            field_name=field_name,
                            values=field_values,
                            status="explicit",
                            citations=(citation,),
                        )
                        for field_name, field_values in values
                    )
                )
            )
        return WorkerResearchExtractionResponse(
            request_id=request.request_id,
            factors=tuple(drafts),
            usage={"calls": 1},
        )


class TwoDraftConverter(FakeResearchConverter):
    """按请求顺序返回两个不同但合法的受控表达式。"""

    def generate(self, request):
        self.requests.append(request)
        expressions = ("rank(ret)", "mean(ret, 5)")
        return WorkerResearchConversionResponse(
            request_id=request.request_id,
            outcomes=tuple(
                WorkerResearchConversionOutcome(
                    draft_id=item.draft_id,
                    status="converted",
                    expression=expressions[index],
                    field_mapping={"ret": "ret"},
                )
                for index, item in enumerate(request.drafts)
            ),
            usage={"calls": 1},
        )


class FakeSearchWorker:
    """资料种子批次不应调用搜索 Worker。"""

    def __init__(self) -> None:
        self.requests = []

    def health(self):
        return {"status": "ready"}

    def submit(self, request):
        self.requests.append(request)
        raise AssertionError("资料种子批次不应提交搜索 Worker")

    def get_job(self, job_id):
        raise AssertionError(job_id)

    def cancel(self, job_id):
        return WorkerJobView(
            job_id=job_id,
            request_id="cancelled",
            status="cancelled",
        )


def _report(evaluation_input, stage, *, status="passed", metrics=None):
    return EvaluationReport(
        run_id=evaluation_input.run_id,
        candidate_id=evaluation_input.candidate_id,
        stage=stage,
        candidate_content_hash=evaluation_input.candidate.content_hash,
        context_hash=evaluation_input.context.context_hash,
        input_hash=sha256_json(
            {
                "candidate": evaluation_input.candidate_id,
                "stage": stage,
            }
        ),
        status=status,
        metrics=metrics or {},
        warnings=(),
        artifact_path=None,
        artifact_sha256=None,
        error_code=None,
        error_message=None,
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
    )


class FakeEvaluator:
    def __call__(self, *, run_id, candidate, context, stage_root):
        evaluation_input = SimpleNamespace(
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            candidate=candidate,
            context=context,
            quick_window=(
                context.windows.development_start,
                context.windows.development_end,
            ),
        )
        return CandidateEvaluationResult(
            gate=TechnicalGate(
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                candidate_content_hash=candidate.content_hash,
                context_hash=context.context_hash,
                rule_version="test-v1",
                passed=True,
                checks={"values": {"coverage": 0.8}},
            ),
            technical=_report(
                evaluation_input,
                "technical",
                metrics={"values": {"coverage": 0.8}},
            ),
            quick=_report(
                evaluation_input,
                "quick",
                metrics={
                    "rank_ic_mean": 0.04,
                    "rank_ic_ir": 0.5,
                    "year_sign_consistency": 0.8,
                    "valid_days": 100,
                },
            ),
            full=None,
            warnings=(),
            factor_glob="/tmp/factor",
            factor_sha256="b" * 64,
        )


class InterruptingEvaluator(FakeEvaluator):
    """模拟资料种子已落库后主服务退出。"""

    def __call__(self, **kwargs):
        raise KeyboardInterrupt("模拟资料种子中断")


class FakeMaterializer:
    def __call__(self, items, context, *, start, end):
        return SimpleNamespace(failures=())


class FakeFinalEvaluator:
    def __call__(self, *, run_id, candidate, context, stage_root):
        evaluation_input = SimpleNamespace(
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            candidate=candidate,
            context=context,
            quick_window=(
                context.windows.final_start,
                context.windows.final_end,
            ),
        )
        return FullEvaluationResult(
            report=_report(evaluation_input, "full", metrics={"complete": True}),
            warnings=(),
            factor_glob="/tmp/full",
            factor_sha256="c" * 64,
        )


class FakeHoldoutEvaluator:
    def __call__(self, *, run_id, candidate, context, stage_root):
        evaluation_input = SimpleNamespace(
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            candidate=candidate,
            context=context,
            quick_window=(
                context.windows.holdout_start,
                context.windows.holdout_end,
            ),
        )
        return HoldoutEvaluationResult(
            report=_report(
                evaluation_input,
                "holdout",
                metrics={"complete": True, "valid_days": 100},
            ),
            passed=True,
            warnings=(),
            factor_glob="/tmp/holdout",
            factor_sha256="d" * 64,
        )


class FakePortfolio:
    def __call__(self, scorecards, policy, *, max_finalists):
        finalists = tuple(
            sorted(item.candidate_id for item in scorecards)[:max_finalists]
        )
        return PortfolioSelectionResult(
            candidates=(),
            fronts=(finalists,),
            finalists=finalists,
        )


def _input(root: Path, key: str, limit: int = 1):
    paths = discover_research_material_paths(root, ".")
    manifest = build_research_material_manifest(
        root,
        paths,
        parser_version=RESEARCH_PDF_PARSER_VERSION,
    )
    budget = SearchBudget(
        max_search_turns=1,
        max_hypothesis_branches=1,
        max_refines_per_branch=1,
        max_candidates_per_turn=1,
        max_candidates_per_run=limit,
        max_full_evaluations=1,
        stop_after_no_improvement_turns=1,
        max_tool_calls_per_turn=0,
    )
    request = SearchRunRequest(key, "资料驱动研究测试假设", budget=budget)
    return (
        request,
        ResearchRunInput(
            run_id=request.run_id,
            idempotency_key=request.idempotency_key,
            material_path=".",
            manifest=manifest,
            limits=ResearchConversionLimits(max_documents=2, max_factors=8),
        ),
        budget,
    )


def test_research_pipeline_builds_source_bound_controlled_candidate(tmp_path: Path):
    root = tmp_path / "reports"
    _write_pdf(root / "report.pdf")
    request, research_input, budget = _input(root, "research-pipeline-0001")
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        llm_model="deepseek/deepseek-chat",
        budget=budget,
    )

    result = prepare_research_run(
        research_input,
        material_root=root,
        context=context,
        extractor=FakeResearchExtractor(),
        converter=FakeResearchConverter(),
        admission_limit=1,
    )

    assert result.summary.status == "ready"
    assert result.summary.admitted_count == 1
    assert len(result.provenances) == 1
    provenance = result.provenances[0]
    assert provenance.document_id == research_input.manifest.materials[0].document_id
    assert provenance.candidate.parameters["source_kind"] == "research_report"
    assert (
        provenance.candidate.parameters["research_provenance_hash"]
        == provenance.provenance_hash
    )
    assert provenance.hypothesis.run_id == request.run_id
    assert provenance.fields[0].citations[0].page_number == 1


def test_research_seed_batch_allows_more_than_worker_turn_limit():
    candidate_ids = tuple(f"cand_seed_{index}" for index in range(5))
    batch = ExperimentBatch(
        run_id="searchrun_batchtest01",
        hypothesis_id="hyp_batchtest01",
        turn_index=1,
        branch_id="branch-0001",
        candidate_ids=candidate_ids,
        parent_candidate_ids=(),
        development_start="2012-01-01",
        development_end="2019-12-31",
        input_hash="a" * 64,
        parameters={"source_kind": "research_report"},
    )
    assert batch.candidate_ids == candidate_ids

    with pytest.raises(MiningContractError, match="candidate_ids"):
        ExperimentBatch(
            run_id="searchrun_batchtest02",
            hypothesis_id="hyp_batchtest02",
            turn_index=1,
            branch_id="branch-0001",
            candidate_ids=candidate_ids,
            parent_candidate_ids=(),
            development_start="2012-01-01",
            development_end="2019-12-31",
            input_hash="a" * 64,
        )


def test_research_pipeline_applies_admission_budget_stably(tmp_path: Path):
    root = tmp_path / "reports"
    _write_pdf(root / "report.pdf")
    _request, research_input, budget = _input(
        root,
        "research-admission-0001",
        limit=1,
    )
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        budget=budget,
    )

    result = prepare_research_run(
        research_input,
        material_root=root,
        context=context,
        extractor=TwoDraftExtractor(),
        converter=TwoDraftConverter(),
        admission_limit=1,
    )

    assert result.summary.accepted_count == 2
    assert result.summary.admitted_count == 1
    assert len(result.provenances) == 1
    assert sorted(item.admitted for item in result.summary.decisions) == [
        False,
        True,
    ]
    rejected = next(
        item for item in result.summary.decisions if not item.admitted
    )
    assert rejected.reason_code == "admission_budget_exhausted"


def test_non_target_material_fails_without_candidate(tmp_path: Path):
    root = tmp_path / "reports"
    _write_pdf(root / "memo.pdf", "Accounting memo without target signals.")
    request, research_input, budget = _input(root, "research-empty-0001")
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        budget=budget,
    )
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    orchestrator = MiningSearchOrchestrator(
        store=store,
        worker=FakeSearchWorker(),
        research_material_root=root,
        research_extractor=FakeResearchExtractor(),
        research_converter=FakeResearchConverter(),
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    try:
        run = orchestrator.create_run(
            request,
            context,
            background=False,
            research_input=research_input,
        )
        result = orchestrator.execute(run["run_id"])
    finally:
        orchestrator.close()

    assert result["status"] == "failed"
    assert store.load_research_summary(run["run_id"]).status == "failed"
    failures = [
        item
        for item in store.list_events(run["run_id"])
        if item["event_type"] == "search_run_failed"
    ]
    assert failures[-1]["payload"]["error_code"] == "MINING-160"


def test_research_candidates_reach_v2_review_ready_without_worker_generation(
    tmp_path: Path,
):
    root = tmp_path / "reports"
    _write_pdf(root / "report.pdf")
    request, research_input, budget = _input(root, "research-loop-0001")
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        budget=budget,
    )
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    search_worker = FakeSearchWorker()
    orchestrator = MiningSearchOrchestrator(
        store=store,
        worker=search_worker,
        research_material_root=root,
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
    try:
        run = orchestrator.create_run(
            request,
            context,
            background=False,
            research_input=research_input,
        )
        result = orchestrator.execute(run["run_id"])
        trace = store.get_search_trace(run["run_id"])
        provenances = store.list_research_candidates(run["run_id"])

        failures = [
            item
            for item in store.list_events(run["run_id"])
            if item["event_type"] == "search_run_failed"
        ]
        assert result["status"] == "review_ready", failures
        assert search_worker.requests == []
        assert len(trace["candidates"]) == 1
        assert trace["candidates"][0]["status"] == "review_ready"
        assert len(provenances) == 1
        candidate = trace["candidates"][0]
        assert candidate["payload"]["parameters"]["source_kind"] == "research_report"
        store.require_research_candidate(
            run["run_id"],
            candidate["candidate_id"],
            candidate_content_hash=candidate["candidate_hash"],
        )
        assert store.validate_search_trace(run["run_id"]) is True
    finally:
        orchestrator.close()


def test_research_seed_resume_skips_reingestion(tmp_path: Path):
    root = tmp_path / "reports"
    _write_pdf(root / "report.pdf")
    request, research_input, budget = _input(root, "research-resume-0001")
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        budget=budget,
    )
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    extractor = FakeResearchExtractor()
    first = MiningSearchOrchestrator(
        store=store,
        worker=FakeSearchWorker(),
        research_material_root=root,
        research_extractor=extractor,
        research_converter=FakeResearchConverter(),
        materializer=FakeMaterializer(),
        development_evaluator=InterruptingEvaluator(),
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    try:
        run = first.create_run(
            request,
            context,
            background=False,
            research_input=research_input,
        )
        try:
            first.execute(run["run_id"])
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("测试应模拟进程中断")
    finally:
        first.close()
    assert len(extractor.requests) == 1
    store.transition_search_run(
        run["run_id"],
        "interrupted",
        event_key="research_seed_interrupted",
        stage="searching",
    )

    second = MiningSearchOrchestrator(
        store=store,
        worker=FakeSearchWorker(),
        research_material_root=root,
        research_preparer=lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(
                AssertionError("恢复不应重新执行资料抽取")
            )
        ),
        materializer=FakeMaterializer(),
        development_evaluator=FakeEvaluator(),
        selection_evaluator=FakeEvaluator(),
        final_evaluator=FakeFinalEvaluator(),
        holdout_evaluator=FakeHoldoutEvaluator(),
        portfolio_selector=FakePortfolio(),
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    try:
        second.resume(run["run_id"], background=False)
        result = second.execute(run["run_id"])
    finally:
        second.close()

    failures = [
        item
        for item in store.list_events(run["run_id"])
        if item["event_type"] == "search_run_failed"
    ]
    assert result["status"] == "review_ready", failures
    assert len(store.list_research_candidates(run["run_id"])) == 1


def _execute_research_run(tmp_path: Path, key: str):
    root = tmp_path / "reports"
    _write_pdf(root / "report.pdf")
    request, research_input, budget = _input(root, key)
    context = FrozenSearchContext(
        data_fingerprint="a" * 64,
        budget=budget,
    )
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    orchestrator = MiningSearchOrchestrator(
        store=store,
        worker=FakeSearchWorker(),
        research_material_root=root,
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
    try:
        run = orchestrator.create_run(
            request,
            context,
            background=False,
            research_input=research_input,
        )
        result = orchestrator.execute(run["run_id"])
    finally:
        orchestrator.close()
    return store, run["run_id"], result


def test_research_candidate_approval_and_publication_bind_source(tmp_path: Path):
    store, run_id, result = _execute_research_run(
        tmp_path,
        "research-publication-0001",
    )
    assert result["status"] == "review_ready"
    trace = store.get_search_trace(run_id)
    candidate_id = trace["candidates"][0]["candidate_id"]
    eligibility = store.search_approval_eligibility(run_id, candidate_id)
    store.record_search_approval(
        ApprovalDecision(
            run_id=run_id,
            candidate_id=candidate_id,
            decision="approved",
            reviewer="reviewer-1",
            note="资料证据完整，允许发布",
            candidate_content_hash=eligibility.candidate_content_hash,
            semantic_hash=eligibility.semantic_hash,
            context_hash=eligibility.context_hash,
            evidence_hash=eligibility.evidence_hash,
        )
    )
    service = MiningPublicationService(
        store=store,
        data_root=tmp_path / "publication",
        snapshot_check=lambda *_args, **_kwargs: None,
        prepare=lambda **kwargs: SimpleNamespace(
            manifest_hash="a" * 64,
            artifact_sha256="b" * 64,
            factor_names=tuple(
                item["name"] for item in kwargs["candidates"]
            ),
        ),
        commit=lambda **_kwargs: None,
        rollback=lambda **_kwargs: None,
    )
    published = service.start_batch(
        run_id,
        (candidate_id,),
        expected_context_hash=store.get_run(run_id)["context_hash"],
        background=False,
    )
    assert published["status"] == "published"
    assert published["source_kind"] == "v2_search"
    assert store.get_candidate(run_id, candidate_id)["status"] == "published"


def test_research_source_tampering_blocks_publication_loading(tmp_path: Path):
    store, run_id, result = _execute_research_run(
        tmp_path,
        "research-tamper-0001",
    )
    assert result["status"] == "review_ready"
    candidate = store.get_search_trace(run_id)["candidates"][0]
    candidate_id = candidate["candidate_id"]
    with duckdb.connect(str(store.db_path)) as conn:
        row = conn.execute(
            "SELECT payload_json FROM mining_v2.research_candidate "
            "WHERE run_id=? AND candidate_id=?",
            [run_id, candidate_id],
        ).fetchone()
        payload = json.loads(row[0])
        payload["fields"][0]["values"][0] = "tampered value"
        conn.execute(
            "UPDATE mining_v2.research_candidate SET payload_json=? "
            "WHERE run_id=? AND candidate_id=?",
            [json.dumps(payload, ensure_ascii=False), run_id, candidate_id],
        )

    try:
        load_search_publication_inputs(store, run_id, (candidate_id,))
    except MiningStorageError:
        pass
    else:
        raise AssertionError("来源篡改后不应加载发布输入")


def test_research_run_and_candidate_views_expose_safe_sources(tmp_path: Path):
    store, run_id, result = _execute_research_run(
        tmp_path,
        "research-view-0001",
    )
    assert result["status"] == "review_ready"
    orchestrator = MiningSearchOrchestrator(
        store=store,
        worker=FakeSearchWorker(),
        snapshot_check=lambda *_args, **_kwargs: None,
    )
    service = MiningSearchWebService(
        store=store,
        orchestrator=orchestrator,
        snapshot_check=lambda *_args, **_kwargs: None,
        material_root=tmp_path / "reports",
    )
    try:
        detail = service.run_detail(run_id)
        candidate_id = detail["candidates"][0]["candidate_id"]
        candidate = service.candidate_detail(run_id, candidate_id)
    finally:
        service.close()

    assert detail["research"]["summary"]["admitted_count"] == 1
    assert detail["research"]["summary"]["materials"][0]["relative_path"] == (
        "report.pdf"
    )
    assert candidate["research_source"]["document_id"].startswith("doc_")
    assert candidate["research_source"]["fields"][0]["citations"][0][
        "page_number"
    ] == 1
    encoded = json.dumps(detail, ensure_ascii=False)
    assert str(tmp_path) not in encoded
