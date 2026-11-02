"""V2 审批与发布页面验收夹具。"""

from __future__ import annotations

import atexit
from dataclasses import replace
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from backend.app import create_app
from backend.mining_search_service import MiningSearchWebService
from backend.mining_service import MiningWebService
from mining.contract_requests import MiningConstraints
from mining.contract_types import sha256_json, stable_id, utc_now_iso
from mining.expression_validation import validate_expression
from mining.catalog import load_catalog
from mining.free_exploration import (
    direction_to_hypothesis,
    generate_initial_direction,
)
from mining.research_extraction_contracts import (
    RESEARCH_EXTRACTION_PROMPT_VERSION,
    RESEARCH_EXTRACTION_RULE_VERSION,
    ResearchExtractionCitation,
    ResearchFieldClaim,
)
from mining.research_factor_screening_contracts import (
    RESEARCH_CONVERSION_PROMPT_VERSION,
    ResearchConversionLimits,
)
from mining.research_material_contracts import (
    ResearchMaterialInput,
    ResearchMaterialManifest,
)
from mining.research_material_parse_contracts import (
    RESEARCH_CLASSIFICATION_RULE_VERSION,
    ResearchMaterialClassification,
)
from mining.research_run_contracts import (
    ResearchCandidateProvenance,
    ResearchMaterialView,
    ResearchRunDecision,
    ResearchRunInput,
    ResearchRunSummary,
)
from mining.orchestration_support import progress_for_run
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_context import FrozenSearchContext
from mining.search_contract_types import FREE_EXPLORATION_INPUT_MODE
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_orchestrator_support import candidate_payload
from mining.search_progress import progress_for_search
from mining.search_review import SEARCH_APPROVAL_REQUIRED_STAGES
from mining.search_run import SearchRunRequest
from mining.storage import MiningStore
from mining.storage_v2 import MiningV2Store


class V1FixtureOrchestrator:
    """让 V1 依赖保持可用，但不创建 V1 运行。"""

    def __init__(self, store: MiningStore) -> None:
        self.store = store

    def progress(self, run_id):
        return progress_for_run(self.store, run_id)

    def cancel(self, run_id):
        return self.store.terminate_run(
            run_id,
            "cancelled",
            event_key="fixture_cancelled",
        )

    def resume(self, run_id, *, background=True):
        return self.store.resume_run(run_id)

    def close(self) -> None:
        return None


class V2FixtureOrchestrator:
    """提供 V2 进度并保留取消和恢复契约。"""

    def __init__(self, store: MiningV2Store) -> None:
        self.store = store

    def progress(self, run_id):
        return progress_for_search(self.store, run_id)

    def cancel(self, run_id):
        return self.store.transition_search_run(
            run_id,
            "cancelled",
            event_key="fixture_cancelled",
            stage="queued",
            stop_reason="用户取消",
        )

    def resume(self, run_id, *, background=True):
        return self.store.search_control_state(run_id)

    def close(self) -> None:
        return None


def _seed_review_ready(
    store: MiningV2Store,
    key: str,
    *,
    candidate_count: int = 3,
) -> tuple[str, tuple[str, ...]]:
    run = store.create_run(
        SearchRunRequest(
            key,
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
    definitions = tuple(
        validate_expression(
            f"mean(close, {5 + index * 5})",
            direction=1,
        ).to_candidate(
            mechanism="短期趋势",
            description=f"{5 + index * 5}日收盘均值",
            steps=("读取收盘价", "计算移动均值"),
            discovery_round=1,
            discovery_index=index + 1,
        )
        for index in range(candidate_count)
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
            store.record_evaluation_evidence(
                SearchEvaluationEvidence(
                    run_id=run["run_id"],
                    candidate_id=candidate.candidate_id,
                    stage=stage,
                    input_hash=(
                        f"{candidate_index:x}{stage_index + 1:x}" * 32
                    ),
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
    return run["run_id"], candidate_ids


def _seed_free_exploration_review_ready(
    store: MiningV2Store,
    key: str,
) -> str:
    """创建带冻结方向的自由探索审核态样例。"""

    request = SearchRunRequest(
        key,
        None,
        input_mode=FREE_EXPLORATION_INPUT_MODE,
        exploration_seed=20261102,
    )
    catalog = load_catalog()
    context = FrozenSearchContext(
        data_fingerprint="e" * 64,
        field_catalog=catalog.frozen_field_mappings(),
        operator_catalog=catalog.frozen_operator_mappings(),
    )
    run = store.create_run(request, context)
    direction = generate_initial_direction(request, context)
    hypothesis = direction_to_hypothesis(direction)
    batch = ExperimentBatch(
        run_id=run["run_id"],
        hypothesis_id=hypothesis.hypothesis_id,
        turn_index=1,
        branch_id=hypothesis.branch_id,
        candidate_ids=(),
        parent_candidate_ids=(),
        development_start=context.windows.development_start,
        development_end=context.windows.development_end,
        input_hash="f" * 64,
    )
    store.append_event(
        run["run_id"],
        event_key="free_exploration_direction_planned",
        stage="searching",
        event_type="free_exploration_direction_planned",
        payload={
            "direction": direction.to_dict(),
            "direction_hash": direction.direction_hash,
            "direction_id": direction.direction_id,
        },
    )
    store.begin_turn(run["run_id"], hypothesis, batch)
    for target in ("searching", "selecting", "evaluating", "holding_out"):
        store.transition_search_run(
            run["run_id"],
            target,
            event_key=f"free_run_{target}",
            stage=target,
        )
    store.transition_search_run(
        run["run_id"],
        "review_ready",
        event_key="free_run_review_ready",
        stage="review_ready",
        stop_reason="搜索轮次预算耗尽，进入人工审核",
    )
    return run["run_id"]


def _seed_research_review_ready(
    store: MiningV2Store,
    key: str,
) -> tuple[str, str]:
    """创建一条带研报来源的审核就绪运行，供页面夹具使用。"""

    quote = "Equity research report stock valuation factor."
    material_sha = "a" * 64
    text_sha = "b" * 64
    material = ResearchMaterialInput(
        relative_path="2026-10/short-reversal.pdf",
        media_type="application/pdf",
        size_bytes=1024,
        sha256=material_sha,
        parser_version="pypdf-6.21.2.page-v1",
    )
    manifest = ResearchMaterialManifest(materials=(material,))
    request = SearchRunRequest(
        key,
        "资料驱动研究：1 份研报资料",
    )
    context = FrozenSearchContext(data_fingerprint="a" * 64)
    research_input = ResearchRunInput(
        run_id=request.run_id,
        idempotency_key=request.idempotency_key,
        material_path="2026-10",
        manifest=manifest,
        limits=ResearchConversionLimits(),
    )
    run = store.create_run(
        request,
        context,
        research_input=research_input,
    )
    hypothesis = HypothesisCard(
        run_id=run["run_id"],
        turn_index=1,
        branch_id="branch-0001",
        statement="短期反转因子在研报中具有正向预期",
        mechanism="短期价格修复",
        expected_direction="positive",
        falsification_criteria=("开发期 RankIC 低于冻结门槛",),
        metadata={
            "source_kind": "research_report",
            "manifest_hash": manifest.manifest_hash,
        },
    )
    base_candidate = validate_expression("rank(close)", direction=1).to_candidate(
        mechanism="短期反转",
        description="收盘价的截面排名",
        steps=("读取收盘价", "做截面排名"),
        discovery_round=1,
        discovery_index=1,
    )
    citation = ResearchExtractionCitation(
        document_id=material.document_id,
        page_number=1,
        start_offset=0,
        end_offset=len(quote),
        quote=quote,
    )
    fields = tuple(
        ResearchFieldClaim(
            field_name=name,
            values=values,
            status="explicit",
            citations=(citation,),
        )
        for name, values in (
            ("factor_name", ("short reversal",)),
            ("economic_mechanism", ("short-term price repair",)),
            ("formula", ("rank(close)",)),
            ("variables", ("close",)),
            ("expected_direction", ("positive",)),
            ("falsification_conditions", ("rank ic below threshold",)),
        )
    )
    factor_hash = sha256_json({"draft": "research-browser-fixture"})
    extraction_hash = sha256_json({"extraction": "research-browser-fixture"})
    provenance_draft = ResearchCandidateProvenance(
        run_id=run["run_id"],
        candidate_id=base_candidate.candidate_id,
        document_id=material.document_id,
        extraction_id=stable_id(
            "extraction",
            {"extraction_hash": extraction_hash},
        ),
        extraction_hash=extraction_hash,
        draft_id=stable_id("factordraft", {"factor_hash": factor_hash}),
        factor_hash=factor_hash,
        material=material,
        classification=ResearchMaterialClassification(
            document_id=material.document_id,
            text_sha256=text_sha,
            kind="financial_research",
            rule_version=RESEARCH_CLASSIFICATION_RULE_VERSION,
            report_signals=("equity research",),
            finance_signals=("factor", "stock", "valuation"),
            created_at=utc_now_iso(),
        ),
        fields=fields,
        checks=(
            "catalog_available",
            "direction_consistent",
            "document_bound",
            "dsl_valid",
            "evidence_complete",
            "mapping_consistent",
            "safe_evidence",
            "supported_direction",
        ),
        field_mapping={"close": "close"},
        catalog_hash=load_catalog().catalog_hash,
        converter_model="deepseek/deepseek-chat",
        conversion_prompt_version=RESEARCH_CONVERSION_PROMPT_VERSION,
        extraction_model="deepseek/deepseek-chat",
        extraction_prompt_version=RESEARCH_EXTRACTION_PROMPT_VERSION,
        hypothesis=hypothesis,
        candidate=base_candidate,
    )
    candidate = replace(
        base_candidate,
        parameters={
            "source_kind": "research_report",
            "research_provenance_hash": provenance_draft.provenance_hash,
        },
    )
    provenance = replace(
        provenance_draft,
        candidate_id=candidate.candidate_id,
        candidate=candidate,
    )
    summary = ResearchRunSummary(
        input=research_input,
        status="ready",
        extraction_ids=(provenance.extraction_id,),
        conversion_id=stable_id(
            "researchconv",
            {"conversion_hash": "c" * 64},
        ),
        conversion_hash="c" * 64,
        model="deepseek/deepseek-chat",
        catalog_hash=load_catalog().catalog_hash,
        materials=(
            ResearchMaterialView(
                document_id=material.document_id,
                relative_path=material.relative_path,
                status="parsed",
                classification="financial_research",
                page_count=1,
            ),
        ),
        decisions=(
            ResearchRunDecision(
                document_id=material.document_id,
                draft_id=provenance.draft_id,
                factor_hash=factor_hash,
                status="accepted",
                reason_code=None,
                checks=provenance.checks,
                candidate_id=candidate.candidate_id,
                admitted=True,
            ),
        ),
        extracted_factor_count=1,
        accepted_count=1,
        admitted_count=1,
    )
    store.save_research_summary(summary, (provenance,))
    batch = ExperimentBatch(
        run_id=run["run_id"],
        hypothesis_id=hypothesis.hypothesis_id,
        turn_index=1,
        branch_id=hypothesis.branch_id,
        candidate_ids=(candidate.candidate_id,),
        parent_candidate_ids=(),
        development_start="2012-01-01",
        development_end="2019-12-31",
        input_hash="d" * 64,
        parameters={"source_kind": "research_report"},
    )
    store.begin_turn(run["run_id"], hypothesis, batch)
    store.record_candidate(
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
    for target in (
        "validating",
        "screening",
        "evaluating",
        "evaluated",
        "holdout_running",
        "review_ready",
    ):
        store.transition_candidate(run["run_id"], candidate.candidate_id, target)
    for stage_index, stage in enumerate(SEARCH_APPROVAL_REQUIRED_STAGES):
        store.record_evaluation_evidence(
            SearchEvaluationEvidence(
                run_id=run["run_id"],
                candidate_id=candidate.candidate_id,
                stage=stage,
                input_hash=f"{stage_index + 1:x}" * 64,
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
            event_key=f"research_run_{target}",
            stage=target,
        )
    return run["run_id"], candidate.candidate_id


_configured_root = os.environ.get("MINING_SEARCH_FIXTURE_ROOT", "").strip()
_TEMP_ROOT = (
    Path(_configured_root)
    if _configured_root
    else Path(tempfile.mkdtemp(prefix="quant-mining-v2-web-"))
)
_V1_STORE = MiningStore(
    _TEMP_ROOT / "factor_mining_v1.duckdb",
    _TEMP_ROOT / "v1",
)
_V2_STORE = MiningV2Store(
    _TEMP_ROOT / "factor_mining_v2.duckdb",
    _TEMP_ROOT / "v2",
)
RUN_ID, CANDIDATE_IDS = _seed_review_ready(
    _V2_STORE,
    "v2-browser-review-0001",
)
RESEARCH_RUN_ID, RESEARCH_CANDIDATE_ID = _seed_research_review_ready(
    _V2_STORE,
    "v2-browser-research-0001",
)
FREE_RUN_ID = _seed_free_exploration_review_ready(
    _V2_STORE,
    "v2-browser-free-0001",
)
_SEARCH_SERVICE = MiningSearchWebService(
    store=_V2_STORE,
    orchestrator=V2FixtureOrchestrator(_V2_STORE),
    snapshot_check=lambda *_args, **_kwargs: None,
)


def _prepare_publication(**kwargs):
    """首次发布失败，重试时返回确定性产物摘要。"""

    marker = _TEMP_ROOT / "publication-attempt"
    if not marker.exists():
        marker.write_text("failed-once", encoding="utf-8")
        raise RuntimeError("injected browser publication failure")
    return SimpleNamespace(
        manifest_hash="a" * 64,
        artifact_sha256="b" * 64,
        factor_names=tuple(item["name"] for item in kwargs["candidates"]),
    )


_SEARCH_SERVICE.publication_service._prepare = _prepare_publication
_SEARCH_SERVICE.publication_service._commit = lambda **_kwargs: None
_SEARCH_SERVICE.publication_service._rollback = lambda **_kwargs: None
_SERVICE = MiningWebService(
    store=_V1_STORE,
    orchestrator=V1FixtureOrchestrator(_V1_STORE),
    defaults=MiningConstraints(
        max_rounds=1,
        max_candidates=3,
        full_eval_top_k=2,
        max_llm_calls=3,
        min_quality_checks_for_failure_stop=3,
    ),
    reviewer="v2-browser-reviewer",
    snapshot_check=lambda *_args, **_kwargs: None,
    search_service=_SEARCH_SERVICE,
)
app = create_app(mining_service=_SERVICE)


def cleanup() -> None:
    """删除浏览器夹具的临时数据。"""

    shutil.rmtree(_TEMP_ROOT, ignore_errors=True)


atexit.register(cleanup)
