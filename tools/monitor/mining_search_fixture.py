"""V2 审批与发布页面验收夹具。"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from backend.app import create_app
from backend.mining_search_service import MiningSearchWebService
from backend.mining_service import MiningWebService
from mining.contract_requests import MiningConstraints
from mining.expression_validation import validate_expression
from mining.orchestration_support import progress_for_run
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_context import FrozenSearchContext
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
