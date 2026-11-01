"""V2 候选审核、审批和发布批次 Web 适配。"""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Any, Callable, Mapping

from mining.contract_evidence import ApprovalDecision
from mining.errors import (
    ApprovalConflictError,
    MiningContractError,
    MiningNotFoundError,
    SnapshotStaleError,
)
from mining.logging_conf import format_event, get_mining_logger
from mining.publication_service import MiningPublicationService
from mining.search_context import FrozenSearchContext
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_publication_contracts import publication_context
from mining.search_scorecard import build_candidate_scorecard
from mining.snapshot import assert_context_snapshot
from mining.storage_v2 import MiningV2Store

_LOGGER = get_mining_logger("mining.search.web.review")
_DECISION_FIELDS = {
    "decision",
    "reviewer",
    "note",
    "candidate_content_hash",
    "evidence_hash",
}
_PUBLICATION_FIELDS = {"candidate_ids", "expected_context_hash"}
_PUBLISHABLE_STATUSES = {"approved", "build_failed"}


class MiningSearchReviewAdapter:
    """把 V2 存储能力转换为监测站审核和发布接口。"""

    def __init__(
        self,
        *,
        store: MiningV2Store,
        publication_service: MiningPublicationService | None = None,
        db_file: Path | str,
        factor_dir: Path | str,
        catalog_file: Path | str,
        manifest_file: Path | str,
        snapshot_check: Callable[..., None] | None = None,
    ) -> None:
        self.store = store
        self.db_file = Path(db_file)
        self.factor_dir = Path(factor_dir)
        self.catalog_file = Path(catalog_file)
        self.manifest_file = Path(manifest_file)
        self._snapshot_check = snapshot_check or assert_context_snapshot
        self.publication_service = publication_service or MiningPublicationService(
            store=self.store,
            db_file=self.db_file,
            factor_dir=self.factor_dir,
            catalog_file=self.catalog_file,
            data_root=self.store.data_root,
            manifest_file=self.manifest_file,
            snapshot_check=self._snapshot_check,
        )

    def candidate_detail(
        self,
        run: Mapping[str, Any],
        candidate_id: str,
    ) -> dict[str, Any]:
        """返回候选证据、审批资格、评分卡和发布状态。"""

        run_id = run["run_id"]
        record = self.store.get_candidate(run_id, candidate_id)
        definition = dict(record["payload"])
        eligibility = self.store.search_approval_eligibility(
            run_id,
            candidate_id,
        )
        evaluations = [
            self._evaluation_view(row)
            for row in self.store.list_evaluation_evidence(
                run_id,
                candidate_id,
            )
        ]
        try:
            approval = self.store.get_search_approval(run_id, candidate_id)
        except MiningNotFoundError:
            approval = None
        snapshot_error = self._snapshot_error(run)
        blocking_reasons = list(eligibility.blocking_reasons)
        if snapshot_error is not None and eligibility.eligible:
            blocking_reasons.append("SNAPSHOT_STALE")
        reason = (
            snapshot_error["message"]
            if snapshot_error is not None
            else ("；".join(blocking_reasons) or None)
        )
        technical = self._latest_stage(evaluations, "technical")
        holdout = self._latest_stage(evaluations, "holdout")
        candidate = {
            "run_id": record["run_id"],
            "candidate_id": record["candidate_id"],
            "candidate_hash": record["candidate_hash"],
            "semantic_hash": record["semantic_hash"],
            "content_hash": record["candidate_hash"],
            "name": record["name"],
            "status": record["status"],
            "discovery_round": definition.get("discovery_round"),
            "discovery_index": definition.get("discovery_index"),
            "definition": definition,
        }
        return {
            "candidate": candidate,
            "technical_gate": (
                {
                    "passed": technical["status"] == "passed",
                    "rule_version": "mining_search_approval_v1",
                    "checks": technical["metrics"],
                }
                if technical is not None
                else None
            ),
            "evaluations": evaluations,
            "holdout_usage": (
                {
                    "candidate_ids": [candidate_id],
                    "batch_hash": holdout["input_hash"],
                }
                if holdout is not None
                else None
            ),
            "approval": approval,
            "review": {
                **eligibility.to_dict(),
                "eligible": eligibility.eligible and snapshot_error is None,
                "reason": reason,
                "blocking_reasons": blocking_reasons,
                "snapshot_current": snapshot_error is None,
                "snapshot_error": snapshot_error,
                "holdout_candidates": (
                    [candidate_id] if holdout is not None else []
                ),
                "holdout_batch_hash": (
                    holdout["input_hash"] if holdout is not None else None
                ),
            },
            "scorecard": build_candidate_scorecard(
                self.store,
                run_id,
                candidate_id,
            ).to_dict(),
            "publication": self._publication_view(run, candidate),
            "publication_batches": [
                self._publication_batch_view(item)
                for item in self.publication_service.list_batches(run_id)
            ],
        }

    def decide(
        self,
        run_id: str,
        candidate_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """记录 V2 候选最终审批，不触发正式发布。"""

        body = self._body(payload, _DECISION_FIELDS)
        run = self.store.get_run(run_id)
        self._assert_snapshot(run)
        eligibility = self.store.search_approval_eligibility(
            run_id,
            candidate_id,
        )
        expected_content = body.get("candidate_content_hash")
        expected_evidence = body.get("evidence_hash")
        if not (
            isinstance(expected_content, str)
            and hmac.compare_digest(
                expected_content,
                eligibility.candidate_content_hash,
            )
            and isinstance(expected_evidence, str)
            and hmac.compare_digest(
                expected_evidence,
                eligibility.evidence_hash,
            )
        ):
            raise ApprovalConflictError(
                "候选内容或审批证据已变化，请刷新后重新审核"
            )
        decision = ApprovalDecision(
            run_id=run_id,
            candidate_id=candidate_id,
            decision=body.get("decision"),
            reviewer=body.get("reviewer"),
            note=body.get("note"),
            candidate_content_hash=eligibility.candidate_content_hash,
            semantic_hash=eligibility.semantic_hash,
            context_hash=eligibility.context_hash,
            evidence_hash=eligibility.evidence_hash,
        )
        approval = self.store.record_search_approval(decision)
        _LOGGER.info(
            format_event(
                {
                    "operation": "search_web_decision",
                    "run_id": run_id,
                    "candidate_id": candidate_id,
                    "decision": approval["decision"],
                    "approval_id": approval["approval_id"],
                }
            )
        )
        return {
            "approval": approval,
            "candidate": self.store.get_candidate(run_id, candidate_id),
        }

    def start_publication_batch(
        self,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """整批构建并发布 V2 运行内全部可发布候选。"""

        body = self._body(payload, _PUBLICATION_FIELDS)
        batch = self.publication_service.start_batch(
            run_id,
            body.get("candidate_ids"),
            expected_context_hash=body.get("expected_context_hash"),
            background=True,
        )
        return self._publication_batch_view(batch)

    def publication_batch(
        self,
        run_id: str,
        batch_id: str,
    ) -> dict[str, Any]:
        """读取 V2 发布批次进度。"""

        return self._publication_batch_view(
            self.publication_service.get_batch(run_id, batch_id)
        )

    def publication_batches(self, run_id: str) -> list[dict[str, Any]]:
        """列出 V2 运行的全部发布批次。"""

        self.store.get_run(run_id)
        return [
            self._publication_batch_view(item)
            for item in self.publication_service.list_batches(run_id)
        ]

    def _publication_view(
        self,
        run: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        """构造 V2 候选页的发布资格和最近批次摘要。"""

        batches = [
            self._publication_batch_view(item)
            for item in self.publication_service.list_batches(run["run_id"])
        ]
        latest = batches[0] if batches else None
        approved_ids = [
            item["candidate_id"]
            for item in self.store.get_search_trace(run["run_id"])["candidates"]
            if item["status"] in _PUBLISHABLE_STATUSES
        ]
        eligible = candidate["status"] in _PUBLISHABLE_STATUSES
        reason = (
            None
            if eligible
            else f"候选状态为 {candidate['status']}，当前不可发布"
        )
        if run["status"] != "review_ready":
            eligible = False
            reason = f"运行状态为 {run['status']}，当前不可发布"
        if latest and latest["status"] in {"staging", "validating"}:
            eligible = False
            reason = "已有发布批次正在执行"
        return {
            "eligible": eligible,
            "reason": reason,
            "approved_candidate_ids": approved_ids,
            "latest_batch": latest,
        }

    def _snapshot_error(
        self,
        run: Mapping[str, Any],
    ) -> dict[str, str] | None:
        """返回当前冻结快照的公开错误摘要。"""

        try:
            self._assert_snapshot(run)
        except SnapshotStaleError as error:
            return {"code": error.code, "message": error.message}
        return None

    def _assert_snapshot(self, run: Mapping[str, Any]) -> None:
        """按发布构建使用的派生上下文核对数据、目录和正式因子。"""

        context = FrozenSearchContext.from_dict(run["context"])
        self._snapshot_check(
            publication_context(context),
            db_file=self.db_file,
            factor_dir=self.factor_dir,
            manifest_file=self.manifest_file,
        )

    @staticmethod
    def _latest_stage(
        evaluations: list[Mapping[str, Any]],
        stage: str,
    ) -> Mapping[str, Any] | None:
        return next(
            (
                item
                for item in reversed(evaluations)
                if item["stage"] == stage
            ),
            None,
        )

    @staticmethod
    def _evaluation_view(row: Mapping[str, Any]) -> dict[str, Any]:
        """把持久化 V2 证据转换为页面可展示的评估摘要。"""

        evidence = SearchEvaluationEvidence.from_dict(row["evidence"])
        return {
            "evaluation_id": row["evaluation_id"],
            "stage": evidence.stage,
            "status": evidence.status,
            "input_hash": evidence.input_hash,
            "evidence_hash": evidence.evidence_hash,
            "metrics": evidence.metrics,
            "warnings": list(evidence.warnings),
            "error_code": evidence.error_code,
            "error_message": evidence.error_message,
            "created_at": evidence.created_at,
            "started_at": evidence.created_at,
            "finished_at": evidence.created_at,
        }

    @staticmethod
    def _publication_batch_view(
        batch: Mapping[str, Any],
    ) -> dict[str, Any]:
        """移除发布状态的本地暂存路径。"""

        result = dict(batch)
        result.pop("staging_path", None)
        result["publications"] = [
            {
                key: value
                for key, value in item.items()
                if key != "staging_path"
            }
            for item in result.get("publications", [])
        ]
        return result

    @staticmethod
    def _body(
        payload: Mapping[str, Any],
        allowed: set[str],
    ) -> dict[str, Any]:
        """校验 V2 Web 请求体是对象且没有未知字段。"""

        if not isinstance(payload, Mapping):
            raise MiningContractError("请求体必须是 JSON 对象")
        unknown = set(payload) - allowed
        if unknown:
            raise MiningContractError(f"未知 V2 请求字段：{sorted(unknown)}")
        return dict(payload)
