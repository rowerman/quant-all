"""挖掘 Web 用例服务：连接候选存储、编排器和审核页面。"""

from __future__ import annotations

from dataclasses import fields
import getpass
import os
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping

from common.paths import (
    DB_FILE,
    FACTOR_DB_FILE,
    FACTOR_DIR,
    MINING_DB_FILE,
    MINING_DATA_DIR,
    MINING_ORCHESTRATOR_FILE,
)
from mining.contracts import (
    ApprovalDecision,
    FrozenContext,
    MiningConstraints,
    MiningRunRequest,
    RESEARCH_END,
    RESEARCH_START,
    FULL_EVAL_END,
    FULL_EVAL_START,
    HOLDOUT_END,
    HOLDOUT_START,
)
from mining.errors import (
    ApprovalConflictError,
    MiningContractError,
    MiningNotFoundError,
    MiningStateError,
    SnapshotStaleError,
)
from mining.logging_conf import format_event, get_mining_logger
from mining.orchestration_contracts import load_orchestration_constraints
from mining.orchestration_support import context_from_run
from mining.orchestrator import MiningOrchestrator
from mining.publication_service import MiningPublicationService
from mining.snapshot import DEFAULT_DATA_MANIFEST, assert_context_snapshot
from mining.storage import MiningStore

_LOGGER = get_mining_logger("mining.web")
_STREAM_END_STATUSES = {
    "review_ready",
    "completed",
    "interrupted",
    "failed",
    "cancelled",
    "stale",
}
_CONSTRAINT_FIELDS = {
    item.name for item in fields(MiningConstraints)
}
_CREATE_FIELDS = {
    "hypothesis",
    "idempotency_key",
    "llm_profile",
    "constraints",
}
_DECISION_FIELDS = {
    "decision",
    "reviewer",
    "note",
    "candidate_content_hash",
    "evidence_hash",
}


class MiningWebService:
    """为监测站提供挖掘运行、事件和人工审批用例。"""

    def __init__(
        self,
        *,
        store: MiningStore | None = None,
        orchestrator: Any | None = None,
        defaults: MiningConstraints | None = None,
        reviewer: str | None = None,
        snapshot_check: Callable[..., None] | None = None,
        db_file: Path | str = DB_FILE,
        factor_dir: Path | str = FACTOR_DIR,
        catalog_file: Path | str = FACTOR_DB_FILE,
        manifest_file: Path | str = DEFAULT_DATA_MANIFEST,
    ) -> None:
        self.store = store or MiningStore(MINING_DB_FILE, MINING_DATA_DIR)
        self.orchestrator = orchestrator or MiningOrchestrator(store=self.store)
        self.defaults = defaults or load_orchestration_constraints(
            MINING_ORCHESTRATOR_FILE
        )
        self.reviewer = reviewer or getpass.getuser()
        self.db_file = Path(db_file)
        self.factor_dir = Path(factor_dir)
        self.catalog_file = Path(catalog_file)
        self.manifest_file = Path(manifest_file)
        self._snapshot_check = snapshot_check or assert_context_snapshot
        self.publication_service = MiningPublicationService(
            store=self.store,
            db_file=self.db_file,
            factor_dir=self.factor_dir,
            catalog_file=self.catalog_file,
            data_root=self.store.data_root,
            manifest_file=self.manifest_file,
            snapshot_check=self._snapshot_check,
        )
        self._init_lock = Lock()
        self._initialized = False
        self._closed = False

    def initialize(self) -> None:
        """初始化候选库；遗留活动运行只标记中断，不自动执行。"""

        with self._init_lock:
            if self._initialized:
                return
            interrupted = self.store.interrupt_active_runs()
            removed = self.store.cleanup_transient_temp()
            self.publication_service.initialize()
            self._initialized = True
            _LOGGER.info(
                format_event(
                    {
                        "operation": "initialize_mining_web",
                        "interrupted_runs": len(interrupted),
                        "removed_temp_paths": len(removed),
                    }
                )
            )

    def close(self) -> None:
        """关闭编排执行器，服务关闭后不再接受运行。"""

        with self._init_lock:
            if self._closed:
                return
            self._closed = True
        self.publication_service.close()
        self.orchestrator.close()

    def options(self) -> dict[str, Any]:
        """返回页面预算默认值、限制和冻结口径。"""

        return {
            "defaults": self.defaults.to_dict(),
            "limits": {
                "max_rounds": 10,
                "max_candidates": 40,
                "max_llm_calls": 60,
                "max_expression_depth": 8,
                "full_eval_top_k": 8,
            },
            "windows": {
                "research_start": RESEARCH_START,
                "research_end": RESEARCH_END,
                "full_eval_start": FULL_EVAL_START,
                "full_eval_end": FULL_EVAL_END,
                "holdout_start": HOLDOUT_START,
                "holdout_end": HOLDOUT_END,
            },
            "fixed": {
                "universe": self.defaults.universe,
                "label": self.defaults.label,
                "price_channel": self.defaults.price_channel,
            },
            "reviewer": self.reviewer,
            "worker_configured": bool(
                os.environ.get("MINING_WORKER_URL", "").strip()
            ),
        }

    def create_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """校验并创建后台挖掘运行。"""

        body = self._body(payload, _CREATE_FIELDS)
        hypothesis = body.get("hypothesis")
        idempotency_key = body.get("idempotency_key")
        if not isinstance(idempotency_key, str):
            raise MiningContractError("idempotency_key 必须由页面提供")
        supplied = body.get("constraints", {})
        if supplied is None:
            supplied = {}
        if not isinstance(supplied, Mapping):
            raise MiningContractError("constraints 必须是 JSON 对象")
        unknown = set(supplied) - _CONSTRAINT_FIELDS
        if unknown:
            raise MiningContractError(f"未知预算字段：{sorted(unknown)}")
        merged = self.defaults.to_dict()
        merged.update(dict(supplied))
        request = MiningRunRequest(
            idempotency_key=idempotency_key,
            hypothesis=hypothesis,
            constraints=MiningConstraints(**merged),
            llm_profile=body.get("llm_profile", "default"),
        )
        run = self.orchestrator.create_run(request, background=True)
        return self.run_detail(run["run_id"])

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """返回近期运行的页面摘要。"""

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise MiningContractError("limit 必须是整数")
        rows = self.store.list_runs(max(1, min(limit, 200)))
        return [self._run_summary(row) for row in rows]

    def run_detail(self, run_id: str) -> dict[str, Any]:
        """返回运行、预算进度和候选摘要。"""

        run = self.store.get_run(run_id)
        return {
            "run": self._run_view(run),
            "progress": self.orchestrator.progress(run_id).to_dict(),
            "candidates": [
                self._candidate_summary(item)
                for item in self.store.list_candidates(run_id)
            ],
        }

    def run_status(self, run_id: str) -> dict[str, Any]:
        """返回 SSE 轮询所需的轻量运行状态。"""

        run = self.store.get_run(run_id)
        return {
            "run_id": run["run_id"],
            "status": run["status"],
            "current_stage": run.get("last_completed_stage"),
            "updated_at": run["updated_at"],
            "error_code": run.get("error_code"),
            "error_message": run.get("error_message"),
            "finished": run["status"] in _STREAM_END_STATUSES,
        }

    def events(
        self,
        run_id: str,
        *,
        after: int = 0,
        limit: int = 500,
    ) -> dict[str, Any]:
        """按序号读取安全业务事件。"""

        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise MiningContractError("after 必须是非负整数")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise MiningContractError("limit 必须是整数")
        limit = max(1, min(limit, 1000))
        self.store.get_run(run_id)
        rows = self.store.list_events(run_id, after)
        page = rows[:limit]
        return {
            "events": page,
            "last_sequence": int(page[-1]["sequence"]) if page else after,
            "has_more": len(rows) > limit,
        }

    def cancel(self, run_id: str) -> dict[str, Any]:
        """请求取消运行并返回最新摘要。"""

        run = self.orchestrator.cancel(run_id)
        return self._run_summary(run)

    def resume(self, run_id: str) -> dict[str, Any]:
        """恢复中断运行并返回最新详情。"""

        self.orchestrator.resume(run_id, background=True)
        return self.run_detail(run_id)

    def candidate_detail(self, run_id: str, candidate_id: str) -> dict[str, Any]:
        """返回候选定义、四阶段证据和当前审批资格。"""

        run = self.store.get_run(run_id)
        candidate = self.store.get_candidate(run_id, candidate_id)
        context = context_from_run(run)
        reports = self.store.list_evaluations(run_id, candidate_id)
        gate = self.store.find_technical_gate(run_id, candidate_id)
        try:
            holdout_usage = self.store.get_holdout_usage()
        except MiningNotFoundError:
            holdout_usage = None
        try:
            approval = self.store.get_approval(run_id, candidate_id)
        except MiningNotFoundError:
            approval = None
        snapshot_error = None
        try:
            self._check_snapshot(context)
        except SnapshotStaleError as exc:
            snapshot_error = {"code": exc.code, "message": exc.message}
        review = self._review_view(
            candidate,
            context,
            gate,
            reports,
            holdout_usage,
            approval,
            snapshot_error,
        )
        return {
            "run": self._run_view(run),
            "candidate": candidate,
            "technical_gate": gate,
            "evaluations": reports,
            "holdout_usage": holdout_usage,
            "approval": approval,
            "review": review,
            "publication": self._publication_view(run_id, candidate),
            "publication_batches": self.publication_service.list_batches(run_id),
        }

    def start_publication_batch(
        self,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """整批构建并发布当前运行的全部已批准候选。"""

        body = self._body(payload, {"candidate_ids", "expected_context_hash"})
        candidate_ids = body.get("candidate_ids")
        expected_context_hash = body.get("expected_context_hash")
        return self.publication_service.start_batch(
            run_id,
            candidate_ids,
            expected_context_hash=expected_context_hash,
            background=True,
        )

    def publication_batch(self, run_id: str, batch_id: str) -> dict[str, Any]:
        """读取发布批次进度。"""

        return self.publication_service.get_batch(run_id, batch_id)

    def publication_batches(self, run_id: str) -> list[dict[str, Any]]:
        """列出运行的全部发布批次。"""

        self.store.get_run(run_id)
        return self.publication_service.list_batches(run_id)

    def _publication_view(
        self,
        run_id: str,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        """构造候选页的发布资格和最近批次摘要。"""

        batches = self.publication_service.list_batches(run_id)
        latest = batches[0] if batches else None
        approved = [
            item
            for item in self.store.list_candidates(run_id)
            if item["status"] in {"approved", "building", "build_failed"}
        ]
        eligible = candidate["status"] in {"approved", "build_failed"}
        reason = None if eligible else f"候选状态为 {candidate['status']}，当前不可发布"
        if latest and latest["status"] in {"staging", "validating"}:
            eligible = False
            reason = "已有发布批次正在执行"
        return {
            "eligible": eligible,
            "reason": reason,
            "approved_candidate_ids": [
                item["candidate_id"] for item in approved
            ],
            "latest_batch": latest,
        }

    def decide(
        self,
        run_id: str,
        candidate_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """记录人工批准或拒绝；不触发正式发布。"""

        body = self._body(payload, _DECISION_FIELDS)
        run = self.store.get_run(run_id)
        context = context_from_run(run)
        try:
            self._check_snapshot(context)
        except SnapshotStaleError:
            self.store.terminate_run(
                run_id,
                "stale",
                event_key=f"web_approval_stale_{candidate_id}",
                error_code="MINING-120",
                error_message="审批前数据、目录或正式因子快照已变化",
            )
            raise
        candidate = self.store.get_candidate(run_id, candidate_id)
        expected_content_hash = body.get("candidate_content_hash")
        expected_evidence_hash = body.get("evidence_hash")
        if expected_content_hash != candidate["content_hash"]:
            raise ApprovalConflictError("候选内容已变化，请刷新后重新审核")
        evidence = self.store.approval_evidence(
            run_id,
            candidate_id,
            candidate_content_hash=candidate["content_hash"],
            context_hash=context.context_hash,
        )
        if expected_evidence_hash != evidence:
            raise ApprovalConflictError("审批证据已变化，请刷新后重新审核")
        decision = ApprovalDecision(
            run_id=run_id,
            candidate_id=candidate_id,
            decision=body.get("decision"),
            reviewer=body.get("reviewer"),
            note=body.get("note"),
            candidate_content_hash=candidate["content_hash"],
            semantic_hash=candidate["semantic_hash"],
            context_hash=context.context_hash,
            evidence_hash=evidence,
        )
        approval = self.store.record_approval(decision)
        _LOGGER.info(
            format_event(
                {
                    "operation": "web_decision",
                    "run_id": run_id,
                    "candidate_id": candidate_id,
                    "decision": approval["decision"],
                }
            )
        )
        return {
            "approval": approval,
            "candidate": self.store.get_candidate(run_id, candidate_id),
        }

    @staticmethod
    def _body(payload: Mapping[str, Any], allowed: set[str]) -> dict[str, Any]:
        """校验 JSON 请求体是对象且没有未知字段。"""

        if not isinstance(payload, Mapping):
            raise MiningContractError("请求体必须是 JSON 对象")
        unknown = set(payload) - allowed
        if unknown:
            raise MiningContractError(f"未知请求字段：{sorted(unknown)}")
        return dict(payload)

    def _check_snapshot(self, context: FrozenContext) -> None:
        """用生产快照口径核对冻结上下文。"""

        self._snapshot_check(
            context,
            db_file=self.db_file,
            factor_dir=self.factor_dir,
            manifest_file=self.manifest_file,
        )

    def _review_view(
        self,
        candidate: Mapping[str, Any],
        context: FrozenContext,
        gate: Mapping[str, Any] | None,
        reports: list[Mapping[str, Any]],
        holdout_usage: Mapping[str, Any] | None,
        approval: Mapping[str, Any] | None,
        snapshot_error: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """构造页面可解释的审批资格和证据摘要。"""

        holdouts = [item for item in reports if item["stage"] == "holdout"]
        reason = None
        if snapshot_error is not None:
            reason = snapshot_error["message"]
        elif candidate["status"] != "review_ready":
            reason = f"候选状态为 {candidate['status']}，当前不可审批"
        elif gate is None or not gate["passed"]:
            reason = "候选尚未通过技术门禁"
        elif not holdouts or any(item["status"] != "passed" for item in holdouts):
            reason = "候选尚无通过的模块测试段报告"
        elif holdout_usage is None:
            reason = "候选尚未领取模块测试段"
        elif candidate["candidate_id"] not in holdout_usage["candidate_ids"]:
            reason = "候选不在本次模块测试段领取批次中"
        elif approval is not None:
            reason = "候选已有最终人工审批"
        evidence = None
        if reason is None:
            try:
                evidence = self.store.approval_evidence(
                    candidate["run_id"],
                    candidate["candidate_id"],
                    candidate_content_hash=candidate["content_hash"],
                    context_hash=context.context_hash,
                )
            except ApprovalConflictError as exc:
                reason = exc.message
        return {
            "eligible": reason is None,
            "reason": reason,
            "candidate_content_hash": candidate["content_hash"],
            "semantic_hash": candidate["semantic_hash"],
            "context_hash": context.context_hash,
            "evidence_hash": evidence,
            "snapshot_current": snapshot_error is None,
            "snapshot_error": snapshot_error,
            "holdout_candidates": (
                list(holdout_usage["candidate_ids"])
                if holdout_usage is not None
                else []
            ),
            "holdout_batch_hash": (
                holdout_usage["batch_hash"] if holdout_usage is not None else None
            ),
        }

    @staticmethod
    def _run_summary(run: Mapping[str, Any]) -> dict[str, Any]:
        """去掉大目录和请求正文后的运行摘要。"""

        return {
            key: run.get(key)
            for key in (
                "run_id",
                "idempotency_key",
                "status",
                "hypothesis",
                "worker_mode",
                "last_completed_stage",
                "interruption_count",
                "error_code",
                "error_message",
                "created_at",
                "updated_at",
                "started_at",
                "finished_at",
            )
        }

    def _run_view(self, run: Mapping[str, Any]) -> dict[str, Any]:
        """返回运行及轻量冻结上下文字段。"""

        result = self._run_summary(run)
        request = run.get("request", {})
        context = run.get("context", {})
        payload = context.get("payload", {})
        result["request"] = {
            "idempotency_key": request.get("idempotency_key"),
            "hypothesis": request.get("hypothesis"),
            "llm_profile": request.get("llm_profile"),
            "constraints": request.get("constraints"),
        }
        result["context"] = {
            key: context.get(key)
            for key in (
                "context_id",
                "context_hash",
                "data_version",
                "data_fingerprint",
                "label",
                "universe",
                "price_channel",
                "research_start",
                "research_end",
                "full_eval_start",
                "full_eval_end",
                "holdout_start",
                "holdout_end",
                "llm_profile",
                "llm_model",
                "field_catalog_hash",
                "operator_catalog_hash",
                "existing_factors_hash",
                "created_at",
            )
        }
        result["context"]["budget"] = payload.get("budget")
        return result

    @staticmethod
    def _candidate_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
        """返回候选表中的轻量字段。"""

        return {
            key: candidate.get(key)
            for key in (
                "run_id",
                "candidate_id",
                "semantic_hash",
                "content_hash",
                "name",
                "status",
                "discovery_round",
                "discovery_index",
                "failure_stage",
                "error_code",
                "error_message",
                "created_at",
                "updated_at",
                "finished_at",
            )
        }
