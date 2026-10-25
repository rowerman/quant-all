"""编排器共享的上下文、记录转换和进度读取。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from calculate.correlation import available_factors
from common.paths import DB_FILE, FACTOR_DIR
from mining.catalog import load_catalog
from mining.contract_candidates import CandidateDefinition
from mining.contract_requests import (
    FrozenContext,
    MiningConstraints,
    MiningRunRequest,
)
from mining.contract_types import HOLDOUT_SEGMENT_ID
from mining.errors import MiningNotFoundError
from mining.orchestration_contracts import MiningProgress
from mining.snapshot import (
    DEFAULT_DATA_MANIFEST,
    data_snapshot_fingerprint,
)
from mining.storage import MiningStore
from mining_worker.protocol import WorkerGenerateRequest
from mining_worker.worker_config import load_worker_config


def build_frozen_context(
    request: MiningRunRequest,
    *,
    db_file: Path | str = DB_FILE,
    factor_dir: Path | str = FACTOR_DIR,
    manifest_file: Path | str = DEFAULT_DATA_MANIFEST,
) -> FrozenContext:
    """按当前数据和规则冻结一次运行上下文。"""

    catalog = load_catalog()
    existing = tuple(
        sorted(
            name
            for names in available_factors(factor_dir).values()
            for name in names
        )
    )
    worker_config = load_worker_config()
    return FrozenContext(
        data_fingerprint=data_snapshot_fingerprint(db_file, manifest_file),
        llm_profile=request.llm_profile,
        llm_model=worker_config.default_model,
        field_catalog=catalog.frozen_field_mappings(),
        operator_catalog=catalog.frozen_operator_mappings(),
        existing_factors=existing,
        budget=request.constraints,
    )


def request_from_run(run: Mapping[str, Any]) -> MiningRunRequest:
    """从存储记录恢复运行请求。"""

    payload = dict(run["request"])
    payload.pop("schema_version", None)
    constraints = MiningConstraints(**payload.pop("constraints"))
    return MiningRunRequest(
        idempotency_key=str(run["idempotency_key"]),
        constraints=constraints,
        **payload,
    )


def context_from_run(run: Mapping[str, Any]) -> FrozenContext:
    """从存储记录恢复冻结上下文。"""

    payload = dict(run["context"]["payload"])
    payload.pop("schema_version", None)
    payload["budget"] = MiningConstraints(**payload["budget"])
    payload["created_at"] = run["context"]["created_at"]
    return FrozenContext(**payload)


def candidate_from_record(record: Mapping[str, Any]) -> CandidateDefinition:
    """把候选库 JSON 转换为不可变候选定义。"""

    payload = dict(record["definition"])
    return CandidateDefinition(
        name=payload["name"],
        mechanism=payload["mechanism"],
        description=payload["description"],
        canonical_expression=payload["canonical_expression"],
        direction=payload["direction"],
        lookback=payload["lookback"],
        dependencies=tuple(payload["dependencies"]),
        output_type=payload["output_type"],
        steps=tuple(payload["steps"]),
        discovery_round=payload["discovery_round"],
        discovery_index=payload["discovery_index"],
        parameters=payload.get("parameters", {}),
    )


def build_worker_request(
    *,
    request: MiningRunRequest,
    context: FrozenContext,
    round_index: int,
    feedback: tuple[Mapping[str, Any], ...],
    max_candidates: int,
) -> WorkerGenerateRequest:
    """构造一轮固定内容的 Worker 请求。"""

    return WorkerGenerateRequest(
        request_id=f"{request.run_id}:round:{round_index}",
        run_id=request.run_id,
        round_index=round_index,
        hypothesis=request.hypothesis,
        context_hash=context.context_hash,
        field_catalog_hash=context.field_catalog_hash,
        operator_catalog_hash=context.operator_catalog_hash,
        llm_model=context.llm_model,
        max_candidates=max_candidates,
        max_expression_depth=context.budget.max_expression_depth,
        direction=context.budget.direction,
        excluded_fields=context.budget.excluded_fields,
        fields=context.field_catalog,
        operators=context.operator_catalog,
        feedback=feedback,
    )


def events_by_key(store: MiningStore, run_id: str) -> dict[str, dict[str, Any]]:
    """返回运行事件的按键索引。"""

    return {item["event_key"]: item for item in store.list_events(run_id)}


def progress_for_run(store: MiningStore, run_id: str) -> MiningProgress:
    """从数据库、事件和测试段领取记录恢复预算进度。"""

    run = store.get_run(run_id)
    events = store.list_events(run_id)
    candidates = store.list_candidates(run_id)
    budget = MiningConstraints(**run["context"]["payload"]["budget"])
    failed_statuses = {
        "technical_failed",
        "screened_out",
        "holdout_failed",
        "build_failed",
        "failed",
    }
    try:
        store.get_holdout_usage(HOLDOUT_SEGMENT_ID)
        holdout_claimed = True
    except MiningNotFoundError:
        holdout_claimed = False
    current_stage = run.get("interrupted_from") or run["status"]
    return MiningProgress(
        run_id=run_id,
        status=run["status"],
        current_stage=current_stage,
        rounds_requested=sum(
            item["event_type"] == "round_requested" for item in events
        ),
        rounds_completed=sum(
            item["event_type"] == "round_completed" for item in events
        ),
        llm_calls_used=sum(
            item["event_type"] == "round_requested" for item in events
        ),
        candidates_stored=len(candidates),
        quality_checks_used=sum(
            item["event_type"] == "quality_check_started" for item in events
        ),
        full_evaluations_used=sum(
            item["event_type"] == "full_evaluation_started" for item in events
        ),
        failed_candidates=sum(
            item["status"] in failed_statuses for item in candidates
        ),
        holdout_claimed=holdout_claimed,
        budget=budget,
    )
