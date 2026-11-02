"""后台批量发布服务、预检和进程重启恢复。"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock, Thread, get_ident
from typing import Any, Callable, Mapping, Sequence

from calculate.correlation import available_factors
from common.paths import (
    DB_FILE,
    FACTOR_DB_FILE,
    FACTOR_DIR,
    MINING_DATA_DIR,
)
from mining.contract_requests import FrozenContext
from mining.contract_types import utc_now_iso
from mining.errors import (
    MiningContractError,
    PublicationConflictError,
    SnapshotStaleError,
)
from mining.logging_conf import format_event, get_mining_logger
from mining.publication_builder import (
    commit_publication,
    prepare_publication,
    rollback_publication,
)
from mining.publication_contracts import PublicationBatch, new_publication_batch
from mining.snapshot import DEFAULT_DATA_MANIFEST, assert_context_snapshot
from mining.storage import MiningStore
from mining.storage_v2 import MiningV2Store
from mining.storage_v3 import MiningV3Store

_LOGGER = get_mining_logger("mining.publication")


class MiningPublicationService:
    """资格检查、后台构建、状态查询和未完成批次恢复。"""

    def __init__(
        self,
        *,
        store: MiningStore | MiningV2Store | MiningV3Store | None = None,
        db_file: Path | str = DB_FILE,
        factor_dir: Path | str = FACTOR_DIR,
        catalog_file: Path | str = FACTOR_DB_FILE,
        data_root: Path | str = MINING_DATA_DIR,
        manifest_file: Path | str = DEFAULT_DATA_MANIFEST,
        snapshot_check: Callable[..., None] | None = None,
        prepare: Callable[..., Any] | None = None,
        commit: Callable[..., None] | None = None,
        rollback: Callable[..., None] | None = None,
    ) -> None:
        self.store = store or MiningStore()
        self.db_file = Path(db_file)
        self.factor_dir = Path(factor_dir)
        self.catalog_file = Path(catalog_file)
        self.data_root = Path(data_root)
        self.manifest_file = Path(manifest_file)
        self._snapshot_check = snapshot_check or assert_context_snapshot
        self._prepare = prepare or prepare_publication
        self._commit = commit or commit_publication
        self._rollback = rollback or rollback_publication
        self._lock = Lock()
        self._threads: list[Thread] = []
        self._closed = False

    def initialize(self) -> None:
        """启动时回滚或收尾上次进程遗留的发布批次。"""

        for batch in self.store.incomplete_publication_batches():
            staging = Path(batch["staging_path"])
            journal_path = staging / "commit.json"
            if not journal_path.is_file():
                self.store.fail_publication_batch(
                    batch["batch_id"],
                    error_code="MINING-123",
                    error_message="发布进程中断，已回滚未提交批次",
                )
                continue
            try:
                journal = json.loads(journal_path.read_text(encoding="utf-8"))
                if journal.get("state") != "committed":
                    self._rollback(
                        staging_root=staging,
                        factor_dir=self.factor_dir,
                        catalog_file=self.catalog_file,
                    )
                    self.store.fail_publication_batch(
                        batch["batch_id"],
                        error_code="MINING-123",
                        error_message="发布进程中断，已恢复旧正式因子",
                    )
                    continue
                if batch["status"] == "staging":
                    batch = self.store.mark_publication_validating(batch["batch_id"])
                names = self._committed_factor_names(batch, self.factor_dir)
                self.store.complete_publication_batch(
                    batch["batch_id"],
                    manifest_hash=batch["manifest_hash"],
                    artifact_sha256=batch["artifact_sha256"],
                    formal_factor_names=names,
                )
            except BaseException as error:
                self.store.fail_publication_batch(
                    batch["batch_id"],
                    error_code="MINING-123",
                    error_message=f"发布恢复失败：{error}"[:2000],
                )

    def start_batch(
        self,
        run_id: str,
        candidate_ids: Sequence[str],
        *,
        expected_context_hash: str,
        background: bool = True,
    ) -> dict[str, Any]:
        """验证整批资格后启动后台正式构建。"""

        if self._closed:
            raise PublicationConflictError("发布服务正在关闭")
        if isinstance(candidate_ids, str) or not isinstance(candidate_ids, Sequence):
            raise MiningContractError("candidate_ids 必须是数组")
        selected = tuple(sorted(dict.fromkeys(str(item) for item in candidate_ids)))
        if not selected:
            raise PublicationConflictError("发布批次至少需要一个候选")
        inputs = self.store.load_publication_inputs(
            run_id,
            selected,
            expected_context_hash=expected_context_hash,
        )
        context = inputs.builder_context
        self._check_snapshot(context)
        candidates = inputs.candidates
        attempt = self.store.next_publication_attempt(run_id, selected)
        staging = (
            self.data_root
            / "publications"
            / run_id
            / f"attempt-{attempt}"
        )
        batch = new_publication_batch(
            run_id=run_id,
            context_hash=inputs.source_context_hash,
            candidate_ids=selected,
            approval_ids=inputs.approval_ids,
            staging_path=str(staging),
            attempt=attempt,
            source_kind=inputs.source_kind,
        )
        saved = self.store.start_publication_batch(batch)
        if background:
            thread = Thread(
                target=self._execute,
                args=(batch, tuple(candidates), context),
                name=f"mining-publication-{batch.batch_id}",
                daemon=True,
            )
            with self._lock:
                self._threads.append(thread)
            thread.start()
        else:
            self._execute(batch, tuple(candidates), context)
        return self.get_batch(run_id, batch.batch_id)

    def _execute(
        self,
        batch: PublicationBatch,
        candidates: tuple[Mapping[str, Any], ...],
        context: FrozenContext,
    ) -> None:
        """执行构建、校验、提交和最终状态写入。"""

        staging = Path(batch.staging_path)
        try:
            staged = self._prepare(
                batch_id=batch.batch_id,
                run_id=batch.run_id,
                candidates=candidates,
                approval_ids=batch.approval_ids,
                context=context,
                staging_root=staging,
                db_file=self.db_file,
                source_catalog=self.catalog_file,
                factor_dir=self.factor_dir,
            )
            self.store.mark_publication_validating(batch.batch_id)
            self.store.validate_publication_manifest(
                batch.batch_id,
                manifest_hash=staged.manifest_hash,
                artifact_sha256=staged.artifact_sha256,
            )
            self.store.assert_publication_batch_current(batch)
            self._commit(
                staged=staged,
                batch_id=batch.batch_id,
                factor_dir=self.factor_dir,
                catalog_file=self.catalog_file,
            )
            self.store.complete_publication_batch(
                batch.batch_id,
                manifest_hash=staged.manifest_hash,
                artifact_sha256=staged.artifact_sha256,
                formal_factor_names=staged.factor_names,
            )
        except BaseException as error:
            try:
                self._rollback(
                    staging_root=staging,
                    factor_dir=self.factor_dir,
                    catalog_file=self.catalog_file,
                )
            except BaseException as rollback_error:
                error = PublicationConflictError(
                    f"{error}；回滚失败：{rollback_error}"
                )
            current = self.store.get_publication_batch(batch.batch_id)
            if current["status"] != "published":
                self.store.fail_publication_batch(
                    batch.batch_id,
                    error_code=getattr(error, "code", "MINING-109"),
                    error_message=str(error)[:2000],
                )
            _LOGGER.error(
                format_event(
                    {
                        "operation": "publish_batch_failed",
                        "batch_id": batch.batch_id,
                        "error_code": getattr(error, "code", "MINING-109"),
                    }
                )
            )
        finally:
            with self._lock:
                self._threads = [
                    item for item in self._threads if item.ident != get_ident()
                ]

    def get_batch(self, run_id: str, batch_id: str) -> dict[str, Any]:
        """读取运行内批次及其候选状态。"""

        batch = self.store.get_publication_batch(batch_id)
        if batch["run_id"] != run_id:
            raise PublicationConflictError("发布批次不属于该运行")
        batch["candidates"] = [
            {
                key: self.store.get_candidate(run_id, candidate_id).get(key)
                for key in (
                    "candidate_id",
                    "name",
                    "status",
                    "error_code",
                    "error_message",
                )
            }
            for candidate_id in batch["candidate_ids"]
        ]
        return batch

    def list_batches(self, run_id: str) -> list[dict[str, Any]]:
        """列出运行发布批次。"""

        return [
            self.get_batch(run_id, item["batch_id"])
            for item in self.store.list_publication_batches(run_id)
        ]

    def close(self) -> None:
        """停止接收新批次，并等待已启动后台任务结束。"""

        self._closed = True
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout=30)

    def _check_snapshot(self, context: FrozenContext) -> None:
        """核对行情、目录与正式因子快照。"""

        self._snapshot_check(
            context,
            db_file=self.db_file,
            factor_dir=self.factor_dir,
            manifest_file=self.manifest_file,
        )

    @staticmethod
    def _committed_factor_names(
        batch: Mapping[str, Any],
        factor_dir: Path,
    ) -> list[str]:
        """从已提交定义清单找出本批正式因子名称。"""

        journal = json.loads(
            (Path(batch["staging_path"]) / "commit.json").read_text(
                encoding="utf-8"
            )
        )
        manifest = factor_dir / journal["new_link"] / "definitions.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        by_candidate = {
            item.get("candidate_id"): item.get("name")
            for item in payload.get("factors", [])
        }
        names = [
            by_candidate.get(candidate_id)
            for candidate_id in batch["candidate_ids"]
        ]
        if any(name is None for name in names):
            raise PublicationConflictError("已提交定义清单与发布批次候选不一致")
        return names
