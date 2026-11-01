"""监测站 V2 证据驱动搜索用例服务。"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from backend.mining_search_review import MiningSearchReviewAdapter
from calculate.correlation import available_factors
from common.paths import (
    DB_FILE,
    FACTOR_DB_FILE,
    FACTOR_DIR,
    MINING_DB_FILE,
    MINING_DATA_DIR,
)
from mining.analysis_tool_gateway import AnalysisToolGateway
from mining.catalog import load_catalog
from mining.errors import MiningContractError
from mining.logging_conf import format_event, get_mining_logger
from mining.publication_service import MiningPublicationService
from mining.search_budget import SearchBudget
from mining.search_context import FrozenSearchContext, load_search_config
from mining.search_cards import HypothesisCard
from mining.search_orchestrator import MiningSearchOrchestrator
from mining.search_run import SearchRunRequest
from mining.search_research_quality import evaluate_research_quality
from mining.search_scorecard import build_run_scorecards
from mining.search_statistical_diagnostics import MULTIPLE_TESTING_STAGE
from mining.snapshot import (
    DEFAULT_DATA_MANIFEST,
    data_snapshot_fingerprint,
)
from mining.storage_v2 import MiningV2Store
from mining_worker.worker_config import load_worker_config

_LOGGER = get_mining_logger("mining.search.web")
_CREATE_FIELDS = {"hypothesis", "idempotency_key", "llm_profile", "budget"}
_END_STATUSES = {
    "review_ready",
    "failed",
    "cancelled",
    "stale",
}


class MiningSearchWebService:
    """为监测站提供 V2 运行和证据视图。"""

    def __init__(
        self,
        *,
        store: MiningV2Store | None = None,
        orchestrator: MiningSearchOrchestrator | None = None,
        publication_service: MiningPublicationService | None = None,
        db_file: Path | str = DB_FILE,
        factor_dir: Path | str = FACTOR_DIR,
        catalog_file: Path | str = FACTOR_DB_FILE,
        manifest_file: Path | str = DEFAULT_DATA_MANIFEST,
        snapshot_check: Callable[..., None] | None = None,
    ) -> None:
        self.store = store or MiningV2Store(MINING_DB_FILE, MINING_DATA_DIR)
        tools_enabled = os.environ.get(
            "MINING_ANALYSIS_TOOLS_ENABLED",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
        self.orchestrator = orchestrator or MiningSearchOrchestrator(
            store=self.store,
            analysis_tool_gateway=(
                AnalysisToolGateway(store=self.store)
                if tools_enabled
                else None
            ),
            db_file=db_file,
            factor_dir=factor_dir,
            manifest_file=manifest_file,
        )
        self.db_file = Path(db_file)
        self.factor_dir = Path(factor_dir)
        self.catalog_file = Path(catalog_file)
        self.manifest_file = Path(manifest_file)
        self.review = MiningSearchReviewAdapter(
            store=self.store,
            publication_service=publication_service,
            db_file=self.db_file,
            factor_dir=self.factor_dir,
            catalog_file=self.catalog_file,
            manifest_file=self.manifest_file,
            snapshot_check=snapshot_check,
        )
        self.publication_service = self.review.publication_service
        self._initialized = False

    def initialize(self) -> None:
        """初始化 V2 存储并把遗留活动运行标记中断。"""

        if self._initialized:
            return
        interrupted = self.store.interrupt_active_search_runs()
        self.publication_service.initialize()
        self._initialized = True
        _LOGGER.info(
            format_event(
                {
                    "operation": "initialize_search_web",
                    "interrupted_runs": len(interrupted),
                }
            )
        )

    def close(self) -> None:
        """关闭 V2 后台执行器。"""

        self.publication_service.close()
        self.orchestrator.close()

    def options(self) -> dict[str, Any]:
        """返回 V2 预算、质量策略和启用状态。"""

        config = load_search_config()
        return {
            "defaults": config.budget.to_dict(),
            "control_policy": config.control_policy.to_dict(),
            "windows": config.windows.to_dict(),
            "limits": {
                "max_search_turns": 6,
                "max_hypothesis_branches": 3,
                "max_refines_per_branch": 3,
                "max_candidates_per_turn": 4,
                "max_candidates_per_run": 24,
                "max_full_evaluations": 4,
            },
            "worker_configured": bool(
                os.environ.get("MINING_WORKER_URL", "").strip()
            ),
            "analysis_tools_enabled": (
                getattr(self.orchestrator, "analysis_tool_gateway", None)
                is not None
            ),
            "analysis_tool_catalog": (
                list(
                    getattr(
                        self.orchestrator,
                        "analysis_tool_gateway",
                    ).catalog()
                )
                if getattr(self.orchestrator, "analysis_tool_gateway", None)
                is not None
                else []
            ),
        }

    def create_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """创建 V2 实验运行。"""

        if not isinstance(payload, Mapping):
            raise MiningContractError("请求体必须是 JSON 对象")
        unknown = set(payload) - _CREATE_FIELDS
        if unknown:
            raise MiningContractError(f"未知 V2 请求字段：{sorted(unknown)}")
        idempotency_key = payload.get("idempotency_key")
        if not isinstance(idempotency_key, str):
            raise MiningContractError("idempotency_key 必须由页面提供")
        budget_payload = payload.get("budget")
        budget = (
            SearchBudget.from_dict(budget_payload)
            if budget_payload is not None
            else load_search_config().budget
        )
        request = SearchRunRequest(
            idempotency_key=idempotency_key,
            hypothesis=payload.get("hypothesis"),
            budget=budget,
            llm_profile=payload.get("llm_profile", "default"),
        )
        context = self._freeze_context(request)
        run = self.orchestrator.create_run(
            request,
            context,
            background=True,
        )
        return self.run_detail(run["run_id"])

    def _freeze_context(
        self,
        request: SearchRunRequest,
    ) -> FrozenSearchContext:
        catalog = load_catalog()
        worker_config = load_worker_config()
        existing = tuple(
            sorted(
                name
                for names in available_factors(self.factor_dir).values()
                for name in names
            )
        )
        config = replace(load_search_config(), budget=request.budget)
        return FrozenSearchContext.from_config(
            config,
            data_fingerprint=data_snapshot_fingerprint(
                self.db_file,
                self.manifest_file,
            ),
            llm_profile=request.llm_profile,
            llm_model=worker_config.default_model,
            field_catalog=catalog.frozen_field_mappings(),
            operator_catalog=catalog.frozen_operator_mappings(),
            existing_factors=existing,
        )

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """列出 V2 运行摘要。"""

        return [
            {**item, "engine": "v2"}
            for item in self.store.list_runs(limit)
        ]

    def run_detail(self, run_id: str) -> dict[str, Any]:
        """返回 V2 假设树、turn、尝试和候选评分摘要。"""

        run = self.store.get_run(run_id)
        trace = self.store.get_search_trace(run_id)
        progress = self.orchestrator.progress(run_id)
        scorecard_objects = {
            item.candidate_id: item
            for item in build_run_scorecards(self.store, run_id)
        }
        scorecards = {
            candidate_id: card.to_dict()
            for candidate_id, card in scorecard_objects.items()
        }
        decisions = {
            int(item["turn_index"]): item for item in trace["decisions"]
        }
        evidence = {
            int(item["turn_index"]): item for item in trace["evidence"]
        }
        tool_calls = self.store.list_analysis_tool_calls(run_id)
        tool_calls_by_turn: dict[int, list[dict[str, Any]]] = {}
        analysis_count_by_candidate: dict[str, int] = {}
        for call in tool_calls:
            turn_index = int(call["turn_index"])
            tool_calls_by_turn.setdefault(turn_index, []).append(
                {
                    "call_index": call["call_index"],
                    "tool_name": call["tool_name"],
                    "tool_version": call["tool_version"],
                    "candidate_id": call.get("candidate_id"),
                    "status": call["status"],
                    "cache_hit": bool(call["cache_hit"]),
                    "duration_ms": call["duration_ms"],
                    "error_code": call.get("error_code"),
                    "error_message": call.get("error_message"),
                    "input_hash": call.get("input_hash"),
                    "result_hash": call.get("result_hash"),
                    "arguments_summary": call.get("arguments_summary", {}),
                    "evidence_class": (
                        "supplemental"
                        if call["status"] in {"succeeded", "cache_hit"}
                        else "failure"
                    ),
                }
            )
            candidate_id = call.get("candidate_id")
            if candidate_id is not None:
                analysis_count_by_candidate[str(candidate_id)] = (
                    analysis_count_by_candidate.get(str(candidate_id), 0) + 1
                )
        turns = []
        for item in trace["turns"]:
            turn_index = int(item["turn_index"])
            decision = decisions.get(turn_index)
            packet = evidence.get(turn_index)
            turns.append(
                {
                    "turn_index": turn_index,
                    "status": item["status"],
                    "hypothesis_id": item.get("hypothesis_id"),
                    "action": decision["action"] if decision else None,
                    "reason": (
                        decision["reason"][:500] if decision else None
                    ),
                    "evidence_hash": (
                        packet["evidence_hash"] if packet else None
                    ),
                    "feedback": (
                        _feedback_summary(packet)
                        if packet is not None
                        else None
                    ),
                    "tool_calls": tool_calls_by_turn.get(turn_index, []),
                }
            )
        candidates = []
        for item in trace["candidates"]:
            card = scorecards.get(item["candidate_id"], {})
            card_object = scorecard_objects.get(item["candidate_id"])
            selection_quality = (
                evaluate_research_quality(
                    card_object,
                    progress.control_policy,
                    stage="selection",
                ).to_dict()
                if card_object is not None
                else None
            )
            research_stage = (
                "full"
                if card_object is not None
                and "full" in card_object.evidence_status_by_stage
                else "selection"
            )
            research_quality = (
                evaluate_research_quality(
                    card_object,
                    progress.control_policy,
                    stage=research_stage,
                ).to_dict()
                if card_object is not None
                else None
            )
            multiple_testing = card.get("multiple_testing") or {}
            statistical_result = next(
                (
                    candidate
                    for candidate in multiple_testing.get("candidates", [])
                    if candidate.get("candidate_id") == item["candidate_id"]
                ),
                None,
            )
            candidates.append(
                {
                    "candidate_id": item["candidate_id"],
                    "name": item["name"],
                    "status": item["status"],
                    "parent_candidate_id": item["parent_candidate_id"],
                    "hypothesis_id": item["hypothesis_id"],
                    "batch_id": item["batch_id"],
                    "missing_fields": card.get("missing_fields", []),
                    "scorecard_hash": card.get("scorecard_hash"),
                    "decision_stage": card.get("decision_stage"),
                    "metrics": card.get("metrics", {}),
                    "selection_diagnostics": card.get(
                        "selection_diagnostics"
                    ),
                    "stability": card.get("stability"),
                    "selection_quality": selection_quality,
                    "research_quality": research_quality,
                    "multiple_testing": statistical_result,
                    "attempts": card.get("trial_counts", {}),
                    "analysis_count": analysis_count_by_candidate.get(
                        item["candidate_id"],
                        0,
                    ),
                }
            )
        finalist_event = next(
            (
                event
                for event in reversed(self.store.list_events(run_id))
                if event["event_type"] == "finalists_selected"
            ),
            None,
        )
        branch_count = len(
            {str(item["branch_id"]) for item in trace["hypotheses"]}
        )
        statistical_rows = self.store.list_evaluation_evidence(
            run_id,
            stage=MULTIPLE_TESTING_STAGE,
        )
        statistical_summary = (
            statistical_rows[-1]["evidence"]["metrics"]
            if statistical_rows
            else None
        )
        return {
            "run": self._run_view(run),
            "progress": progress.to_dict(),
            "hypotheses": [
                {
                    "hypothesis_id": HypothesisCard.from_dict(
                        item
                    ).hypothesis_id,
                    "parent_hypothesis_id": item["parent_hypothesis_id"],
                    "turn_index": item["turn_index"],
                    "branch_id": item["branch_id"],
                    "statement": item["statement"],
                    "mechanism": item["mechanism"],
                    "expected_direction": item["expected_direction"],
                }
                for item in trace["hypotheses"]
            ],
            "turns": turns,
            "candidates": candidates,
            "attempts": progress.trial_counts.to_dict(),
            "branch_count": branch_count,
            "portfolio": (
                finalist_event["payload"].get("portfolio")
                if finalist_event is not None
                else None
            ),
            "multiple_testing": statistical_summary,
            "analysis_tools": {
                "catalog_version": "mining_analysis_catalog_v1",
                "calls": tool_calls_by_turn,
                "total_calls": len(tool_calls),
                "failed_calls": sum(
                    item["status"] == "failed" for item in tool_calls
                ),
            },
        }

    def run_status(self, run_id: str) -> dict[str, Any]:
        """返回 SSE 终止判断所需状态。"""

        state = self.store.search_control_state(run_id)
        return {
            "run_id": run_id,
            "status": state["status"],
            "current_stage": state["interrupted_from"] or state["status"],
            "updated_at": state["updated_at"],
            "error_code": None,
            "error_message": state["stop_reason"],
            "finished": state["status"] in _END_STATUSES,
        }

    def events(
        self,
        run_id: str,
        *,
        after: int = 0,
        limit: int = 500,
    ) -> dict[str, Any]:
        """读取 V2 脱敏事件。"""

        rows = self.store.list_events(run_id, after)
        page = rows[: max(1, min(limit, 1000))]
        return {
            "events": page,
            "last_sequence": int(page[-1]["sequence"]) if page else after,
            "has_more": len(rows) > len(page),
        }

    def cancel(self, run_id: str) -> dict[str, Any]:
        """请求 V2 运行取消。"""

        self.orchestrator.cancel(run_id)
        return self.store.search_control_state(run_id)

    def resume(self, run_id: str) -> dict[str, Any]:
        """恢复中断 V2 运行。"""

        self.orchestrator.resume(run_id, background=True)
        return self.run_detail(run_id)

    def candidate_detail(
        self,
        run_id: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        """返回 V2 候选、分阶段证据、审批资格和发布状态。"""

        run = self.store.get_run(run_id)
        return {
            "run": self._run_view(run),
            **self.review.candidate_detail(run, candidate_id),
        }

    def decide(
        self,
        run_id: str,
        candidate_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """记录 V2 候选最终审批，不触发正式发布。"""

        return self.review.decide(run_id, candidate_id, payload)

    def start_publication_batch(
        self,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """整批构建并发布 V2 运行内全部可发布候选。"""

        return self.review.start_publication_batch(run_id, payload)

    def publication_batch(
        self,
        run_id: str,
        batch_id: str,
    ) -> dict[str, Any]:
        """读取 V2 发布批次进度。"""

        return self.review.publication_batch(run_id, batch_id)

    def publication_batches(self, run_id: str) -> list[dict[str, Any]]:
        """列出 V2 运行的全部发布批次。"""

        return self.review.publication_batches(run_id)

    @staticmethod
    def _run_view(run: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "run_id": run["run_id"],
            "idempotency_key": run["idempotency_key"],
            "status": run["status"],
            "hypothesis": run["hypothesis"],
            "engine": "v2",
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
            "context": run["context"],
            "context_hash": run["context_hash"],
        }


def _feedback_summary(packet: Mapping[str, Any]) -> dict[str, Any]:
    metrics = packet.get("metrics") or {}
    return {
        "candidate_count": len(metrics),
        "failure_count": len(packet.get("failures") or []),
        "quality_candidates": [
            candidate_id
            for candidate_id, values in metrics.items()
            if values.get("gate_passed") is True
            and values.get("rank_ic") is not None
            and values.get("rank_ic_ir") is not None
            and abs(float(values["rank_ic"])) >= 0.02
            and abs(float(values["rank_ic_ir"])) >= 0.20
            and float(values.get("year_sign_consistency") or 0.0) >= 0.60
            and float(values.get("coverage") or 0.0) >= 0.30
        ],
        "trial_counts": packet.get("trial_counts"),
    }
