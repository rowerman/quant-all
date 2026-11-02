"""V3 四类来源页面验收夹具。"""

from __future__ import annotations

import atexit
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from backend.app import create_app
from backend.mining_service import MiningWebService
from backend.mining_v3_service import MiningV3WebService
from mining.contract_requests import MiningConstraints
from mining.contract_types import stable_id
from mining.expression_validation import validate_expression
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_context import FrozenSearchContext
from mining.search_contract_types import FREE_EXPLORATION_INPUT_MODE
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_orchestrator_support import candidate_payload
from mining.search_progress import progress_for_search
from mining.search_run import SearchRunRequest
from mining.storage import MiningStore
from mining.storage_v2_candidate_trace import V2CandidateTraceMixin
from mining.storage_v2_control import V2SearchControlMixin
from mining.storage_v2_evaluations import V2EvaluationStoreMixin
from mining.storage_v2_trials import V2TrialStoreMixin
from mining.storage_v3 import MiningV3Store
from mining.storage_v3_runtime import MiningV3RuntimeStore
from mining.v3_contracts import (
    CandidateEnvelope,
    EvidenceReference,
    ResearchSourceReference,
)
from mining.v3_sources import (
    costeer_input_envelope,
    free_exploration_input_envelope,
    hypothesis_input_envelope,
    research_report_input_envelope,
)
from mining.research_factor_screening_contracts import ResearchConversionLimits
from mining.research_material_contracts import (
    ResearchMaterialInput,
    ResearchMaterialManifest,
)
from mining.research_run_contracts import ResearchRunInput
from mining_sandbox.artifacts import PromotionRequest


class FixtureV3Orchestrator:
    def __init__(self, store: MiningV3RuntimeStore) -> None:
        self.store = store

    def progress(self, run_id):
        return progress_for_search(self.store, run_id)

    def cancel(self, run_id):
        return self.store.transition_search_run(
            run_id,
            "cancelled",
            event_key="fixture_cancelled",
            stage="review_ready",
            stop_reason="页面夹具取消",
        )

    def resume(self, run_id, *, background=True):
        return self.store.search_control_state(run_id)

    def close(self):
        return None


_ROOT = Path(tempfile.mkdtemp(prefix="quant-mining-v3-web-"))
_CONTRACT = MiningV3Store(_ROOT / "factor_mining.duckdb", _ROOT / "v3")
_RUNTIME = MiningV3RuntimeStore(
    _ROOT / "factor_mining.duckdb",
    _ROOT / "v3",
    contract_store=_CONTRACT,
)
_V1_STORE = MiningStore(_ROOT / "factor_mining_v1.duckdb", _ROOT / "v1")
_SERVICE_V3 = MiningV3WebService(
    contract_store=_CONTRACT,
    runtime_store=_RUNTIME,
    orchestrator=FixtureV3Orchestrator(_RUNTIME),
    snapshot_check=lambda *_args, **_kwargs: None,
)
_SERVICE_V3.publication_service._prepare = lambda **kwargs: SimpleNamespace(
    manifest_hash="a" * 64,
    artifact_sha256="b" * 64,
    factor_names=tuple(item["name"] for item in kwargs["candidates"]),
)
_SERVICE_V3.publication_service._commit = lambda **_kwargs: None
_SERVICE_V3.publication_service._rollback = lambda **_kwargs: None


def _source(kind: str, label: str) -> ResearchSourceReference:
    return ResearchSourceReference(
        source_kind=kind,
        source_id=stable_id(
            "v3src",
            {"source_kind": kind, "label": label},
        ),
        source_hash=__import__("hashlib").sha256(label.encode()).hexdigest(),
        artifact_manifest_hash="1" * 64 if kind == "costeer" else None,
        promotion_hash="2" * 64 if kind == "costeer" else None,
    )


def _seed(kind: str, label: str) -> str:
    free = kind == "free_exploration"
    request = SearchRunRequest(
        f"v3-browser-{kind}-0001",
        None if free else f"{label} 研报或人工研究假设",
        input_mode=FREE_EXPLORATION_INPUT_MODE if free else "hypothesis",
        exploration_seed=20261102 if free else None,
        run_id_prefix="researchrun",
    )
    context = FrozenSearchContext(
        data_fingerprint=__import__("hashlib").sha256(label.encode()).hexdigest(),
    )
    research_input = None
    promotion = None
    if kind == "research_report":
        material = ResearchMaterialInput(
            relative_path="2026-11/v3-browser.pdf",
            media_type="application/pdf",
            size_bytes=1024,
            sha256="3" * 64,
            parser_version="pypdf-6.21.2.page-v1",
        )
        research_input = ResearchRunInput(
            run_id=request.run_id,
            idempotency_key=request.idempotency_key,
            material_path="2026-11",
            manifest=ResearchMaterialManifest(materials=(material,)),
            limits=ResearchConversionLimits(),
        )
        envelope = research_report_input_envelope(
            research_input,
            context_hash=context.context_hash,
        )
    elif kind == "costeer":
        promotion = PromotionRequest(
            sandbox_run_id="sandbox_000000000000000000000001",
            artifact_manifest_hash="1" * 64,
            reviewer="fixture-reviewer",
            reason="人工转写为受控 DSL 并重新评估",
            target_type="dsl",
            target_reference="mean(close, 5)",
        )
        envelope = costeer_input_envelope(
            promotion,
            context_hash=context.context_hash,
            idempotency_key=request.idempotency_key,
            run_id=request.run_id,
        )
    elif free:
        envelope = free_exploration_input_envelope(
            request,
            context_hash=context.context_hash,
        )
    else:
        envelope = hypothesis_input_envelope(
            request,
            context_hash=context.context_hash,
        )
    _RUNTIME.configure_v3_run(request, context, envelope)
    run = _RUNTIME.create_run(
        request,
        context,
        research_input=research_input,
    )
    candidate = validate_expression(
        "mean(close, 5)",
        direction=1,
    ).to_candidate(
        mechanism="短期价格修复",
        description=f"{label} 来源候选",
        steps=("读取收盘价", "计算五日均值"),
        discovery_round=1,
        discovery_index=1,
    )
    hypothesis = HypothesisCard(
        run_id=run["run_id"],
        turn_index=1,
        branch_id="branch-0001",
        statement=f"{label} 来源候选可能带来开发期超额收益",
        mechanism="短期价格修复",
        expected_direction="positive",
        falsification_criteria=("阶段证据未通过",),
        metadata={"source": kind},
    )
    batch = ExperimentBatch(
        run_id=run["run_id"],
        hypothesis_id=hypothesis.hypothesis_id,
        turn_index=1,
        branch_id=hypothesis.branch_id,
        candidate_ids=(candidate.candidate_id,),
        parent_candidate_ids=(),
        development_start=context.windows.development_start,
        development_end=context.windows.development_end,
        input_hash="b" * 64,
    )
    _RUNTIME.begin_turn(run["run_id"], hypothesis, batch)
    V2CandidateTraceMixin.record_candidate(
        _RUNTIME,
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
    evidence = tuple(
        EvidenceReference(
            candidate_id=candidate.candidate_id,
            stage=stage,
            evaluation_id=f"eval_{kind}_{stage}",
            input_hash=f"{index + 4:x}" * 64,
            evidence_hash=f"{index + 7:x}" * 64,
            status="passed",
        )
        for index, stage in enumerate(
            ("technical", "development", "selection", "full", "holdout")
        )
    )
    candidate_source = (
        _source("costeer", label) if promotion is not None else envelope.source
    )
    _CONTRACT.record_candidate_envelope(
        run["run_id"],
        CandidateEnvelope(
            candidate=candidate,
            source=candidate_source,
            evidence=evidence,
        ),
    )
    _CONTRACT.initialize_candidate_state(
        run["run_id"],
        candidate.candidate_id,
        candidate_source.source_id,
        status="proposed",
    )
    for target in (
        "validating",
        "screening",
        "evaluating",
        "evaluated",
        "holdout_running",
        "review_ready",
    ):
        V2TrialStoreMixin.transition_candidate(
            _RUNTIME,
            run["run_id"],
            candidate.candidate_id,
            target,
        )
        _CONTRACT.sync_candidate_state(
            run["run_id"],
            candidate.candidate_id,
            target,
        )
    for target in (
        "searching",
        "selecting",
        "evaluating",
        "holding_out",
        "review_ready",
    ):
        V2SearchControlMixin.transition_search_run(
            _RUNTIME,
            run["run_id"],
            target,
            event_key=f"run_{target}",
            stage=target,
        )
    return run["run_id"]


RUN_IDS = {
    "hypothesis": _seed("hypothesis", "手工假设"),
    "free_exploration": _seed("free_exploration", "自由探索"),
    "research_report": _seed("research_report", "资料驱动"),
    "costeer": _seed("costeer", "CoSTEER"),
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
        v3_service=_SERVICE_V3,
    )
)


def cleanup() -> None:
    shutil.rmtree(_ROOT, ignore_errors=True)


atexit.register(cleanup)
