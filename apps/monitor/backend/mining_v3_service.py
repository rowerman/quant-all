"""监测站 V3 统一研究运行、审核和发布服务。"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from calculate.correlation import available_factors
from common.paths import (
    DB_FILE,
    FACTOR_DB_FILE,
    FACTOR_DIR,
    MINING_DB_FILE,
    MINING_DATA_DIR,
    PROJECT_ROOT,
    RESEARCH_MATERIALS_DIR,
)
from mining.catalog import load_catalog
from mining.contract_evidence import ApprovalDecision
from mining.errors import (
    MiningContractError,
    MiningNotFoundError,
    SnapshotStaleError,
    V3RunStateError,
    V3SourceImportError,
)
from mining.expression_validation import validate_expression
from mining.free_exploration_contracts import EXPLORATION_SEED_MAX
from mining.logging_conf import format_event, get_mining_logger
from mining.publication_service import MiningPublicationService
from mining.research_factor_screening_contracts import ResearchConversionLimits
from mining.research_material_io import (
    build_research_material_manifest,
    discover_research_material_paths,
)
from mining.research_material_parser import RESEARCH_PDF_PARSER_VERSION
from mining.research_run_contracts import ResearchRunInput
from mining.search_budget import SearchBudget
from mining.search_context import FrozenSearchContext, load_search_config
from mining.search_contract_types import (
    FREE_EXPLORATION_INPUT_MODE,
    HYPOTHESIS_INPUT_MODE,
)
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_orchestrator_v3 import MiningV3SearchOrchestrator
from mining.search_progress import progress_for_search
from mining.search_run import SearchRunRequest
from mining.snapshot import (
    DEFAULT_DATA_MANIFEST,
    assert_context_snapshot,
    data_snapshot_fingerprint,
)
from mining.storage_v3 import MiningV3Store
from mining.storage_v3_runtime import MiningV3RuntimeStore
from mining.v3_sources import (
    costeer_input_envelope,
    free_exploration_input_envelope,
    hypothesis_input_envelope,
    research_report_input_envelope,
)
from mining_worker.worker_config import load_worker_config
from mining_sandbox.artifacts import list_manual_promotions, load_manual_promotion
from mining_sandbox.policy import load_sandbox_policy

_LOGGER = get_mining_logger("mining.v3.web")

_CREATE_FIELDS = {
    "hypothesis",
    "idempotency_key",
    "llm_profile",
    "budget",
    "input_mode",
    "exploration_seed",
}
_RESEARCH_FIELDS = {"material_path", "idempotency_key", "llm_profile", "budget"}
_COSTEER_FIELDS = {
    "promotion_id",
    "idempotency_key",
    "llm_profile",
    "budget",
    "direction",
    "mechanism",
    "description",
    "steps",
}
_DECISION_FIELDS = {
    "decision",
    "reviewer",
    "note",
    "candidate_content_hash",
    "evidence_hash",
}
_END_STATUSES = {"review_ready", "failed", "cancelled", "stale"}


class MiningV3WebService:
    """为监测站提供 V3 四类来源的统一运行和审核视图。"""

    def __init__(
        self,
        *,
        contract_store: MiningV3Store | None = None,
        runtime_store: MiningV3RuntimeStore | None = None,
        orchestrator: MiningV3SearchOrchestrator | None = None,
        publication_service: MiningPublicationService | None = None,
        db_file: Path | str = DB_FILE,
        factor_dir: Path | str = FACTOR_DIR,
        catalog_file: Path | str = FACTOR_DB_FILE,
        manifest_file: Path | str = DEFAULT_DATA_MANIFEST,
        snapshot_check: Callable[..., None] | None = None,
        material_root: Path | str | None = None,
        reviewer: str = "v3-reviewer",
    ) -> None:
        self.contract_store = contract_store or MiningV3Store(
            MINING_DB_FILE,
            MINING_DATA_DIR,
        )
        self.runtime_store = runtime_store or MiningV3RuntimeStore(
            MINING_DB_FILE,
            MINING_DATA_DIR,
            contract_store=self.contract_store,
        )
        self.runtime_store.contract_store = self.contract_store
        self.orchestrator = orchestrator or MiningV3SearchOrchestrator(
            store=self.runtime_store,
            db_file=db_file,
            factor_dir=factor_dir,
            manifest_file=manifest_file,
            research_material_root=material_root or RESEARCH_MATERIALS_DIR,
        )
        self.orchestrator.store = self.runtime_store
        self.material_root = Path(material_root or RESEARCH_MATERIALS_DIR)
        self.db_file = Path(db_file)
        self.factor_dir = Path(factor_dir)
        self.catalog_file = Path(catalog_file)
        self.manifest_file = Path(manifest_file)
        self.reviewer = reviewer
        self._snapshot_check = snapshot_check or assert_context_snapshot
        self.publication_service = publication_service or MiningPublicationService(
            store=self.contract_store,
            db_file=self.db_file,
            factor_dir=self.factor_dir,
            catalog_file=self.catalog_file,
            data_root=self.runtime_store.data_root,
            manifest_file=self.manifest_file,
            snapshot_check=self._snapshot_check,
        )
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self.runtime_store.interrupt_orphaned_search_runs()
        self.publication_service.initialize()
        self._initialized = True
        _LOGGER.info(
            format_event(
                {
                    "operation": "initialize_mining_v3",
                    "engine": "v3",
                }
            )
        )

    def close(self) -> None:
        self.publication_service.close()
        self.orchestrator.close()

    def options(self) -> dict[str, Any]:
        config = load_search_config()
        return {
            "defaults": config.budget.to_dict(),
            "control_policy": config.control_policy.to_dict(),
            "windows": config.windows.to_dict(),
            "input_modes": [
                {"value": "hypothesis", "label": "手工研究假设"},
                {"value": "free_exploration", "label": "自由探索"},
                {"value": "research_report", "label": "资料驱动"},
                {"value": "costeer", "label": "CoSTEER 人工提升"},
            ],
            "exploration": {
                "seed_min": 0,
                "seed_max": EXPLORATION_SEED_MAX,
                "seed_optional": True,
                "safety_notice": (
                    "自由探索只替代初始假设，不会扩大冻结预算，也不会把"
                    "选择期、最终研究期或测试段送入 Worker。"
                ),
            },
            "limits": {
                "max_search_turns": 6,
                "max_hypothesis_branches": 3,
                "max_refines_per_branch": 3,
                "max_candidates_per_turn": 4,
                "max_candidates_per_run": 24,
                "max_full_evaluations": 4,
            },
            "research": {
                "material_root": self._public_material_root(),
                "max_documents": ResearchConversionLimits().max_documents,
                "max_factors": ResearchConversionLimits().max_factors,
            },
            "costeer": {
                "enabled": load_sandbox_policy().enabled,
                "selection": "server_promotion",
            },
            "worker_configured": bool(
                os.environ.get("MINING_WORKER_URL", "").strip()
            ),
            "engine": "v3",
        }

    def create_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = self._body(payload, _CREATE_FIELDS)
        request = self._request(body)
        context = self._freeze_context(request)
        if request.input_mode == FREE_EXPLORATION_INPUT_MODE:
            envelope = free_exploration_input_envelope(
                request,
                context_hash=context.context_hash,
            )
        else:
            envelope = hypothesis_input_envelope(
                request,
                context_hash=context.context_hash,
            )
        self.runtime_store.configure_v3_run(request, context, envelope)
        run = self.orchestrator.create_run(request, context, background=True)
        _LOGGER.info(
            format_event(
                {
                    "operation": "create_v3_run",
                    "run_id": run["run_id"],
                    "input_mode": envelope.input_mode,
                    "context_hash": context.context_hash,
                }
            )
        )
        return self.run_detail(run["run_id"])

    def create_research_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = self._body(payload, _RESEARCH_FIELDS)
        key = self._key(body)
        material_path = body.get("material_path")
        if not isinstance(material_path, str):
            raise MiningContractError("material_path 必须是相对路径")
        paths = discover_research_material_paths(
            self.material_root,
            material_path,
            recursive=True,
        )
        manifest = build_research_material_manifest(
            self.material_root,
            paths,
            parser_version=RESEARCH_PDF_PARSER_VERSION,
        )
        request = SearchRunRequest(
            key,
            f"资料驱动研究：{len(paths)} 份研报资料",
            budget=self._budget(body),
            llm_profile=body.get("llm_profile", "default"),
            run_id_prefix="researchrun",
        )
        research_input = ResearchRunInput(
            run_id=request.run_id,
            idempotency_key=request.idempotency_key,
            material_path=material_path,
            manifest=manifest,
            limits=ResearchConversionLimits(),
        )
        context = self._freeze_context(request)
        envelope = research_report_input_envelope(
            research_input,
            context_hash=context.context_hash,
        )
        self.runtime_store.configure_v3_run(
            request,
            context,
            envelope,
            research_input=research_input,
        )
        run = self.orchestrator.create_run(
            request,
            context,
            background=True,
            research_input=research_input,
        )
        _LOGGER.info(
            format_event(
                {
                    "operation": "create_v3_research_run",
                    "run_id": run["run_id"],
                    "document_count": len(paths),
                    "manifest_hash": manifest.manifest_hash,
                }
            )
        )
        return self.run_detail(run["run_id"])

    def create_costeer_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = self._body(payload, _COSTEER_FIELDS)
        promotion = self._find_promotion(str(body.get("promotion_id", "")))
        if promotion.target_type != "dsl":
            raise V3SourceImportError(
                "V3 只导入 target_type=dsl 的 CoSTEER 提升"
            )
        validated = validate_expression(
            promotion.target_reference,
            direction=body.get("direction"),
        )
        steps = body.get("steps")
        if isinstance(steps, str):
            steps = tuple(item.strip() for item in steps.splitlines() if item.strip())
        candidate = validated.to_candidate(
            mechanism=body.get("mechanism"),
            description=body.get("description"),
            steps=steps,
            discovery_round=1,
            discovery_index=1,
        )
        request = SearchRunRequest(
            self._key(body),
            f"CoSTEER 人工提升：{candidate.name}",
            budget=self._budget(body),
            llm_profile=body.get("llm_profile", "default"),
            run_id_prefix="researchrun",
        )
        context = self._freeze_context(request)
        envelope = costeer_input_envelope(
            promotion,
            context_hash=context.context_hash,
            idempotency_key=request.idempotency_key,
            run_id=request.run_id,
        )
        self.runtime_store.configure_v3_run(
            request,
            context,
            envelope,
            seed_candidate=candidate,
        )
        run = self.orchestrator.create_run(request, context, background=True)
        _LOGGER.info(
            format_event(
                {
                    "operation": "create_v3_costeer_run",
                    "run_id": run["run_id"],
                    "promotion_id": promotion.promotion_id,
                    "artifact_manifest_hash": promotion.artifact_manifest_hash,
                }
            )
        )
        return self.run_detail(run["run_id"])

    def costeer_promotions(self) -> list[dict[str, Any]]:
        policy = load_sandbox_policy()
        return [
            {
                "promotion_id": item.promotion_id,
                "sandbox_run_id": item.sandbox_run_id,
                "artifact_manifest_hash": item.artifact_manifest_hash,
                "reviewer": item.reviewer,
                "reason": item.reason,
                "target_type": item.target_type,
                "target_reference": item.target_reference,
                "eligible": item.target_type == "dsl",
            }
            for item in list_manual_promotions(policy=policy)
        ]

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        result = []
        for row in self.runtime_store.list_runs(limit):
            envelope = self.contract_store.load_input(row["run_id"])
            result.append(
                {
                    **dict(row),
                    "engine": "v3",
                    "source_mode": envelope.input_mode,
                    "input_mode": envelope.input_mode,
                }
            )
        result.extend(
            {
                **item,
                "engine": "v3-legacy",
                "source_mode": item["input_mode"],
                "read_only": True,
                "resumable": False,
            }
            for item in self.contract_store.list_legacy_runs(limit)
        )
        return result

    def run_detail(self, run_id: str) -> dict[str, Any]:
        if self.is_legacy_run(run_id):
            return self._legacy_detail(run_id)
        run = self.runtime_store.get_run(run_id)
        envelope = self.contract_store.load_input(run_id)
        progress = progress_for_search(self.runtime_store, run_id)
        candidates = []
        for candidate_id in self._runtime_candidate_ids(run_id):
            runtime = self.runtime_store.get_candidate(run_id, candidate_id)
            review = self._review_for_candidate(run_id, candidate_id)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "name": runtime["name"],
                    "status": review["candidate_status"],
                    "parent_candidate_id": None,
                    "hypothesis_id": None,
                    "batch_id": None,
                    "missing_fields": [],
                    "metadata": {},
                    "attempts": {},
                    "analysis_count": 0,
                }
            )
        return {
            "run": {
                **self._run_view(run, envelope),
            },
            "progress": progress.to_dict(),
            "hypotheses": [],
            "turns": [],
            "candidates": candidates,
            "attempts": progress.trial_counts.to_dict(),
            "branch_count": 0,
            "portfolio": None,
            "multiple_testing": None,
            "analysis_tools": {"calls": {}, "total_calls": 0, "failed_calls": 0},
            "research": None,
            "v3": {
                "input_mode": envelope.input_mode,
                "input_hash": envelope.input_hash,
                "source_hash": envelope.source.source_hash,
                "source_id": envelope.source.source_id,
            },
            "exploration": None,
        }

    def run_status(self, run_id: str) -> dict[str, Any]:
        if self.is_legacy_run(run_id):
            legacy = self.contract_store.get_legacy_run(run_id)
            return {
                "run_id": run_id,
                "status": legacy["status"],
                "current_stage": legacy["status"],
                "updated_at": legacy["updated_at"],
                "error_code": None,
                "error_message": None,
                "finished": True,
            }
        state = self.runtime_store.search_control_state(run_id)
        return {
            "run_id": run_id,
            "status": state["status"],
            "current_stage": state["interrupted_from"] or state["status"],
            "updated_at": state["updated_at"],
            "error_code": None,
            "error_message": state["stop_reason"],
            "finished": state["status"] in _END_STATUSES,
        }

    def events(self, run_id: str, *, after: int = 0, limit: int = 500):
        if self.is_legacy_run(run_id):
            return {
                "events": [],
                "last_sequence": after,
                "has_more": False,
            }
        rows = self.runtime_store.list_events(run_id, after)
        page = rows[: max(1, min(limit, 1000))]
        return {
            "events": page,
            "last_sequence": int(page[-1]["sequence"]) if page else after,
            "has_more": len(rows) > len(page),
        }

    def cancel(self, run_id: str) -> dict[str, Any]:
        if self.is_legacy_run(run_id):
            raise V3RunStateError("迁移的旧运行只读，不能取消")
        self.orchestrator.cancel(run_id)
        return self.runtime_store.search_control_state(run_id)

    def resume(self, run_id: str) -> dict[str, Any]:
        if self.is_legacy_run(run_id):
            raise V3RunStateError("迁移的旧运行只读，不能恢复")
        self.orchestrator.resume(run_id, background=True)
        return self.run_detail(run_id)

    def candidate_detail(self, run_id: str, candidate_id: str) -> dict[str, Any]:
        if self.is_legacy_run(run_id):
            return self._legacy_candidate_detail(run_id, candidate_id)
        run = self.runtime_store.get_run(run_id)
        envelope = self.contract_store.load_input(run_id)
        review = self._review_for_candidate(run_id, candidate_id)
        source_id = review["source_id"]
        candidate_envelope = self.contract_store.get_candidate_envelope(
            run_id,
            candidate_id,
            source_id,
        )
        try:
            approval = self.contract_store.get_approval(run_id, candidate_id)
        except MiningNotFoundError:
            approval = None
        evaluations = [
            row["evidence"]
            for row in self.runtime_store.list_evaluation_evidence(run_id)
            if row["candidate_id"] == candidate_id
        ]
        return {
            "run": self._run_view(run, envelope),
            "candidate": {
                **self.runtime_store.get_candidate(run_id, candidate_id),
                "content_hash": candidate_envelope.candidate.content_hash,
                "definition": {
                    "name": candidate_envelope.candidate.name,
                    "mechanism": candidate_envelope.candidate.mechanism,
                    "description": candidate_envelope.candidate.description,
                    "canonical_expression": candidate_envelope.candidate.canonical_expression,
                    "direction": candidate_envelope.candidate.direction,
                    "lookback": candidate_envelope.candidate.lookback,
                    "dependencies": list(candidate_envelope.candidate.dependencies),
                    "output_type": candidate_envelope.candidate.output_type,
                    "steps": list(candidate_envelope.candidate.steps),
                    "discovery_round": candidate_envelope.candidate.discovery_round,
                    "discovery_index": candidate_envelope.candidate.discovery_index,
                    "parameters": dict(candidate_envelope.candidate.parameters),
                },
            },
            "evaluations": evaluations,
            "technical_gate": next(
                (item for item in evaluations if item["stage"] == "technical"),
                None,
            ),
            "holdout_usage": None,
            "approval": approval,
            "review": review,
            "scorecard": None,
            "publication": self._publication_view(run, review),
            "publication_batches": self.publication_service.list_batches(run_id),
            "v3_source": candidate_envelope.source.to_dict(),
        }

    def decide(self, run_id: str, candidate_id: str, payload: Mapping[str, Any]):
        body = self._body(payload, _DECISION_FIELDS)
        review = self._review_for_candidate(run_id, candidate_id)
        if not review["eligible"]:
            raise MiningContractError("候选当前不能审批：" + review["reason"])
        if (
            body.get("candidate_content_hash") != review["candidate_content_hash"]
            or body.get("evidence_hash") != review["approval_evidence_hash"]
        ):
            raise MiningContractError("候选或审批证据已变化，请刷新后重新审核")
        decision = ApprovalDecision(
            run_id=run_id,
            candidate_id=candidate_id,
            decision=body.get("decision"),
            reviewer=body.get("reviewer"),
            note=body.get("note"),
            candidate_content_hash=review["candidate_content_hash"],
            semantic_hash=review["semantic_hash"],
            context_hash=review["context_hash"],
            evidence_hash=review["approval_evidence_hash"],
        )
        approval = self.contract_store.record_approval(
            decision,
            review["source_id"],
        )
        _LOGGER.info(
            format_event(
                {
                    "operation": "v3_web_decision",
                    "run_id": run_id,
                    "candidate_id": candidate_id,
                    "decision": approval["decision"],
                    "approval_id": approval["approval_id"],
                }
            )
        )
        return {"approval": approval, "candidate": self.runtime_store.get_candidate(run_id, candidate_id)}

    def start_publication_batch(self, run_id: str, payload: Mapping[str, Any]):
        if self.is_legacy_run(run_id):
            raise V3RunStateError("迁移的旧运行只读，不能发布")
        body = self._body(payload, {"candidate_ids", "expected_context_hash"})
        return self.publication_service.start_batch(
            run_id,
            body.get("candidate_ids"),
            expected_context_hash=body.get("expected_context_hash"),
            background=True,
        )

    def publication_batch(self, run_id: str, batch_id: str):
        return self.publication_service.get_batch(run_id, batch_id)

    def publication_batches(self, run_id: str):
        self.runtime_store.get_run(run_id)
        return self.publication_service.list_batches(run_id)

    def monitor_status(self) -> dict[str, Any]:
        runs = self.list_runs(200)
        active = {
            "queued",
            "searching",
            "selecting",
            "evaluating",
            "holding_out",
        }
        reviews = self.contract_store.list_candidate_reviews
        review_count = 0
        for run in runs:
            try:
                review_count += len(reviews(run["run_id"]))
            except MiningNotFoundError:
                continue
        batches = self.contract_store.incomplete_publication_batches()
        legacy = self.contract_store.list_legacy_runs(200)
        source_counts: dict[str, int] = {}
        for run in runs:
            source_counts[run["source_mode"]] = source_counts.get(run["source_mode"], 0) + 1
        return {
            "engine": "v3",
            "schema_version": "mining_v3",
            "runs": len(runs),
            "active_runs": sum(item["status"] in active for item in runs),
            "interrupted_runs": sum(item["status"] == "interrupted" for item in runs),
            "review_candidates": review_count,
            "incomplete_publications": len(batches),
            "legacy_runs": len(legacy),
            "source_counts": source_counts,
        }

    def is_legacy_run(self, run_id: str) -> bool:
        """判断运行是否仅为显式迁移后的只读快照。"""

        try:
            self.contract_store.get_legacy_run(run_id)
        except MiningNotFoundError:
            return False
        return True

    def _review_for_candidate(self, run_id: str, candidate_id: str) -> dict[str, Any]:
        source_id = self.contract_store.candidate_source_id(run_id, candidate_id)
        review = self.contract_store.candidate_review(run_id, candidate_id, source_id)
        run = self.runtime_store.get_run(run_id)
        snapshot_error = None
        try:
            self._assert_snapshot(run["context"])
        except SnapshotStaleError as error:
            snapshot_error = {"code": error.code, "message": error.message}
        blockers = list(review.blocking_reasons)
        if snapshot_error is not None and review.eligible:
            blockers.append("SNAPSHOT_STALE")
        eligible = review.eligible and snapshot_error is None
        return {
            **review.to_dict(),
            "source_id": source_id,
            "eligible": eligible,
            "reason": "；".join(blockers) or None,
            "blocking_reasons": blockers,
            "snapshot_current": snapshot_error is None,
            "snapshot_error": snapshot_error,
        }

    def _assert_snapshot(self, context: Mapping[str, Any]) -> None:
        from mining.search_publication_contracts import publication_context

        self._snapshot_check(
            publication_context(FrozenSearchContext.from_dict(context)),
            db_file=self.db_file,
            factor_dir=self.factor_dir,
            manifest_file=self.manifest_file,
        )

    def _publication_view(self, run: Mapping[str, Any], review: Mapping[str, Any]):
        batches = self.publication_service.list_batches(run["run_id"])
        latest = batches[0] if batches else None
        eligible = bool(review.get("approval_id")) and review.get("decision") == "approved"
        reason = None if eligible else "候选尚未批准"
        if run["status"] != "review_ready":
            eligible = False
            reason = f"运行状态为 {run['status']}，当前不可发布"
        if latest and latest["status"] in {"staging", "validating"}:
            eligible = False
            reason = "已有发布批次正在执行"
        return {
            "eligible": eligible,
            "reason": reason,
            "approved_candidate_ids": [
                item["candidate_id"]
                for item in self.contract_store.list_candidates(run["run_id"])
                if item["status"] in {"approved", "build_failed"}
            ],
            "latest_batch": latest,
        }

    def _runtime_candidate_ids(self, run_id: str) -> tuple[str, ...]:
        return tuple(
            item["candidate_id"]
            for item in self.runtime_store.get_search_trace(run_id)["candidates"]
        )

    def _request(self, body: Mapping[str, Any]) -> SearchRunRequest:
        key = self._key(body)
        mode = body.get("input_mode", HYPOTHESIS_INPUT_MODE)
        if mode not in {HYPOTHESIS_INPUT_MODE, FREE_EXPLORATION_INPUT_MODE}:
            raise MiningContractError("V3 运行输入模式必须是 hypothesis 或 free_exploration")
        seed = body.get("exploration_seed")
        if isinstance(seed, str) and seed.isdecimal():
            seed = int(seed)
        return SearchRunRequest(
            key,
            body.get("hypothesis"),
            budget=self._budget(body),
            llm_profile=body.get("llm_profile", "default"),
            input_mode=mode,
            exploration_seed=seed,
            run_id_prefix="researchrun",
        )

    def _freeze_context(self, request: SearchRunRequest) -> FrozenSearchContext:
        catalog = load_catalog()
        config = replace(load_search_config(), budget=request.budget)
        return FrozenSearchContext.from_config(
            config,
            data_fingerprint=data_snapshot_fingerprint(self.db_file, self.manifest_file),
            llm_profile=request.llm_profile,
            llm_model=load_worker_config().default_model,
            field_catalog=catalog.frozen_field_mappings(),
            operator_catalog=catalog.frozen_operator_mappings(),
            existing_factors=tuple(
                sorted(
                    name
                    for names in available_factors(self.factor_dir).values()
                    for name in names
                )
            ),
        )

    @staticmethod
    def _budget(body: Mapping[str, Any]) -> SearchBudget:
        value = body.get("budget")
        return SearchBudget.from_dict(value) if value is not None else load_search_config().budget

    @staticmethod
    def _key(body: Mapping[str, Any]) -> str:
        value = body.get("idempotency_key")
        if not isinstance(value, str):
            raise MiningContractError("idempotency_key 必须由页面提供")
        return value

    @staticmethod
    def _body(payload: Mapping[str, Any], allowed: set[str]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise MiningContractError("请求体必须是 JSON 对象")
        unknown = set(payload) - allowed
        if unknown:
            raise MiningContractError(f"未知 V3 请求字段：{sorted(unknown)}")
        return dict(payload)

    def _find_promotion(self, promotion_id: str):
        for item in list_manual_promotions(policy=load_sandbox_policy()):
            if item.promotion_id == promotion_id:
                return item
        raise V3SourceImportError(
            f"CoSTEER 人工提升记录不存在：{promotion_id}"
        )

    def _public_material_root(self) -> str:
        try:
            return self.material_root.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            return self.material_root.name

    @staticmethod
    def _run_view(run: Mapping[str, Any], envelope) -> dict[str, Any]:
        context = run["context"]
        windows = context.get("windows", {})
        return {
            "run_id": run["run_id"],
            "idempotency_key": run["idempotency_key"],
            "status": run["status"],
            "hypothesis": run["hypothesis"],
            "engine": "v3",
            "input_mode": envelope.input_mode,
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
            "context_hash": run["context_hash"],
            "context": {
                **context,
                "development_start": windows.get("development_start"),
                "development_end": windows.get("development_end"),
            },
        }

    def _legacy_detail(self, run_id: str) -> dict[str, Any]:
        legacy = self.contract_store.get_legacy_run(run_id)
        return {
            "run": {
                **legacy,
                "engine": "v3-legacy",
                "read_only": True,
                "resumable": False,
            },
            "progress": {"status": legacy["status"], "current_stage": legacy["status"]},
            "hypotheses": [],
            "turns": [],
            "candidates": legacy["candidates"],
            "attempts": {},
            "branch_count": 0,
            "portfolio": None,
            "multiple_testing": None,
            "analysis_tools": {"calls": {}, "total_calls": 0, "failed_calls": 0},
            "research": None,
            "exploration": None,
            "v3": {"legacy": True, "read_only": True},
        }

    def _legacy_candidate_detail(
        self,
        run_id: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        candidate = self.contract_store.get_legacy_candidate(
            run_id,
            candidate_id,
        )
        return {
            "run": {
                **self.contract_store.get_legacy_run(run_id),
                "engine": "v3-legacy",
                "read_only": True,
                "resumable": False,
            },
            "candidate": candidate,
            "evaluations": [],
            "technical_gate": None,
            "holdout_usage": None,
            "approval": None,
            "review": {
                "eligible": False,
                "reason": "迁移旧运行只读",
                "blocking_reasons": ["LEGACY_READ_ONLY"],
                "snapshot_current": False,
                "holdout_candidates": [],
            },
            "scorecard": None,
            "publication": {"eligible": False, "reason": "迁移旧运行只读"},
            "publication_batches": [],
            "v3_source": None,
        }


__all__ = ["MiningV3WebService"]
