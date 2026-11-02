"""V3 运行时存储：复用 V2 搜索机制并写入 `mining_v3`。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Mapping

import duckdb

from common.paths import MINING_DATA_DIR, MINING_DB_FILE
from mining.contract_candidates import CandidateDefinition
from mining.contract_types import canonical_json, utc_now_iso
from mining.errors import (
    MiningContractError,
    MiningNotFoundError,
    MiningStorageError,
    V3RunStateError,
)
from mining.free_exploration import direction_from_hypothesis
from mining.research_run_contracts import ResearchRunInput
from mining.search_cards import HypothesisCard
from mining.search_context import FrozenSearchContext
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_orchestrator_support import candidate_from_v2_record
from mining.search_run import SearchRunRequest
from mining.storage_lock import database_state_lock
from mining.storage_schema import ensure_schema
from mining.storage_v2_base import V2StorageBase
from mining.storage_v2_control import V2SearchControlMixin
from mining.storage_v2_evaluations import V2EvaluationStoreMixin
from mining.storage_v2_research import V2ResearchStoreMixin
from mining.storage_v2_runs import V2RunStoreMixin
from mining.storage_v2_schema import V2_DDL
from mining.storage_v2_tools import V2AnalysisToolStoreMixin
from mining.storage_v2_trace import V2TraceStoreMixin
from mining.storage_v2_trial_queries import V2TrialQueryMixin
from mining.storage_v2_trials import V2TrialStoreMixin
from mining.storage_v3 import MiningV3Store
from mining.storage_v3_schema import ensure_v3_schema
from mining.v3_contracts import (
    V3_RESEARCH_CANDIDATE_SCHEMA_VERSION,
    CandidateEnvelope,
    ResearchInputEnvelope,
)
from mining.v3_sources import (
    evidence_reference_from_search,
    free_exploration_candidate_source,
    research_report_candidate_source,
)


def _runtime_sql(sql: str) -> str:
    """把 V2 运行 SQL 映射到 V3 的 runtime 表。"""

    return sql.replace(
        "CREATE SCHEMA IF NOT EXISTS mining_v2",
        "CREATE SCHEMA IF NOT EXISTS mining_v3",
    ).replace("mining_v2.", "mining_v3.runtime_")


_RUNTIME_AUX_DDL = (
    """
    CREATE TABLE IF NOT EXISTS mining_v3.runtime_seed_candidate (
        run_id VARCHAR PRIMARY KEY,
        input_hash VARCHAR NOT NULL,
        source_id VARCHAR NOT NULL,
        candidate_payload_json TEXT NOT NULL,
        content_hash VARCHAR NOT NULL,
        semantic_hash VARCHAR NOT NULL,
        created_at VARCHAR NOT NULL
    )
    """,
)


@dataclass(frozen=True)
class V3RunExecutionState:
    """恢复 V3 运行所需的全部持久化事实。"""

    request: SearchRunRequest
    context: FrozenSearchContext
    input_envelope: ResearchInputEnvelope
    research_input: ResearchRunInput | None
    seed_candidate: CandidateDefinition | None


class _RuntimeConnection:
    """仅转换 SQL 表名的 DuckDB 连接代理。"""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def execute(self, query: str, parameters=None):
        if parameters is None:
            return self._conn.execute(_runtime_sql(query))
        return self._conn.execute(_runtime_sql(query), parameters)

    @property
    def description(self):
        return self._conn.description

    def __getattr__(self, name):
        return getattr(self._conn, name)


class MiningV3RuntimeStore(
    V2RunStoreMixin,
    V2SearchControlMixin,
    V2TraceStoreMixin,
    V2TrialStoreMixin,
    V2EvaluationStoreMixin,
    V2AnalysisToolStoreMixin,
    V2ResearchStoreMixin,
    V2TrialQueryMixin,
    V2StorageBase,
):
    """V3 运行事实仓库；表名统一位于 `mining_v3.runtime_*`。"""

    def __init__(
        self,
        db_path: Path | str = MINING_DB_FILE,
        data_root: Path | str = MINING_DATA_DIR,
        *,
        contract_store: MiningV3Store | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.data_root = Path(data_root)
        self._connection_lock = RLock()
        self.contract_store = contract_store or MiningV3Store(
            self.db_path,
            self.data_root,
        )
        self.logger = self.contract_store.logger
        self._configure_lock = RLock()
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    ensure_v3_schema(conn._conn)
                    for statement in V2_DDL:
                        conn.execute(statement)
                    for statement in _RUNTIME_AUX_DDL:
                        conn.execute(statement)
                    ensure_schema(conn)
        except Exception as error:
            if isinstance(error, MiningContractError):
                raise
            raise MiningStorageError(
                f"无法初始化 V3 运行时存储：{error}"
            ) from error
        self.data_root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connection(self) -> Iterator[_RuntimeConnection]:
        """打开写入 `mining_v3.runtime_*` 的连接。"""

        with self._connection_lock:
            try:
                conn = duckdb.connect(str(self.db_path))
            except duckdb.Error as error:
                raise MiningStorageError(f"V3 候选库连接失败：{error}") from error
            try:
                yield _RuntimeConnection(conn)
            finally:
                conn.close()

    def configure_v3_run(
        self,
        request: SearchRunRequest,
        context: FrozenSearchContext,
        input_envelope: ResearchInputEnvelope,
        *,
        research_input: ResearchRunInput | None = None,
        seed_candidate: CandidateDefinition | None = None,
    ) -> None:
        """在创建运行前冻结 V3 输入、上下文和请求。"""

        if input_envelope.run_id != request.run_id:
            raise MiningContractError("V3 输入与运行 ID 不一致")
        if input_envelope.context_hash != context.context_hash:
            raise MiningContractError("V3 输入与冻结上下文不一致")
        self._validate_execution_facts(
            request,
            context,
            input_envelope,
            research_input=research_input,
            seed_candidate=seed_candidate,
        )
        with self._configure_lock:
            self.contract_store.register_input(input_envelope)
            self.contract_store.register_context(request.run_id, context)
            if seed_candidate is not None:
                self._persist_seed_candidate(
                    request.run_id,
                    input_envelope,
                    seed_candidate,
                )

    def create_run(
        self,
        request: SearchRunRequest,
        context: FrozenSearchContext,
        *,
        research_input: ResearchRunInput | None = None,
    ) -> dict[str, Any]:
        """创建运行时事实并登记统一 V3 输入。"""

        envelope = self.contract_store.load_input(request.run_id)
        self._validate_execution_facts(
            request,
            context,
            envelope,
            research_input=research_input,
            seed_candidate=self.seed_candidate(request.run_id),
        )
        run = super().create_run(
            request,
            context,
            research_input=research_input,
        )
        return run

    def seed_candidate(self, run_id: str) -> CandidateDefinition | None:
        """返回 CoSTEER 运行的人工转写候选。"""

        envelope = self.contract_store.load_input(run_id)
        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining_v3.runtime_seed_candidate WHERE run_id=?",
                [run_id],
            )
        if row is not None:
            if (
                row["input_hash"] != envelope.input_hash
                or row["source_id"] != envelope.source.source_id
            ):
                raise V3RunStateError("V3 预置候选与运行输入绑定不一致")
            candidate = self._candidate_from_payload(
                json.loads(row["candidate_payload_json"]),
                envelope,
            )
            if (
                row["content_hash"] != candidate.content_hash
                or row["semantic_hash"] != candidate.semantic_hash
            ):
                raise V3RunStateError("V3 预置候选哈希与持久化记录不一致")
            return candidate
        if envelope.input_mode != "costeer":
            return None
        for candidate_envelope in self.contract_store.list_candidate_envelopes(
            run_id
        ):
            if candidate_envelope.source.source_id == envelope.source.source_id:
                return candidate_envelope.candidate
        raise V3RunStateError("CoSTEER 运行缺少持久化预置候选")

    def load_run_request(self, run_id: str) -> SearchRunRequest:
        """从运行表重建并核对 V3 搜索请求。"""

        run = self.get_run(run_id)
        try:
            request = SearchRunRequest.from_dict(
                {
                    "idempotency_key": run["idempotency_key"],
                    **run["request"],
                }
            )
        except (MiningContractError, TypeError, ValueError) as error:
            raise V3RunStateError(f"V3 运行请求无法恢复：{error}") from error
        if request.run_id != run_id:
            raise V3RunStateError("V3 运行请求 ID 与持久化运行不一致")
        return request

    def load_input_envelope(self, run_id: str) -> ResearchInputEnvelope:
        """读取并返回运行绑定的 V3 输入包络。"""

        return self.contract_store.load_input(run_id)

    def load_run_context(self, run_id: str) -> FrozenSearchContext:
        """读取并返回运行绑定的冻结上下文。"""

        return self.contract_store.load_context(run_id)

    def recover_v3_run(self, run_id: str) -> V3RunExecutionState:
        """从持久化事实重建并完整校验一次 V3 运行。"""

        try:
            run = self.get_run(run_id)
            request = self.load_run_request(run_id)
            input_envelope = self.load_input_envelope(run_id)
            context = self.load_run_context(run_id)
            runtime_context = FrozenSearchContext.from_dict(run["context"])
            research_input = self.load_research_input(run_id)
            seed_candidate = self.seed_candidate(run_id)
            if runtime_context != context:
                raise V3RunStateError("V3 运行时上下文与契约上下文不一致")
            if run["context_hash"] != context.context_hash:
                raise V3RunStateError("V3 运行时上下文哈希不一致")
            self._validate_execution_facts(
                request,
                context,
                input_envelope,
                research_input=research_input,
                seed_candidate=seed_candidate,
            )
            self.contract_store.list_candidate_envelopes(run_id)
            for review in self.contract_store.list_candidate_reviews(run_id):
                if (
                    review.approval_id is not None
                    and not self.contract_store.approval_is_current(
                        run_id,
                        review.candidate_id,
                    )
                ):
                    raise V3RunStateError(
                        f"V3 候选审批证据已失效：{review.candidate_id}"
                    )
        except V3RunStateError:
            raise
        except (
            MiningContractError,
            MiningNotFoundError,
            MiningStorageError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise V3RunStateError(f"V3 运行恢复事实校验失败：{error}") from error
        return V3RunExecutionState(
            request=request,
            context=context,
            input_envelope=input_envelope,
            research_input=research_input,
            seed_candidate=seed_candidate,
        )

    def interrupt_orphaned_search_runs(self) -> list[dict[str, Any]]:
        """把上次进程遗留的队列态和活动 V3 运行标记为中断。"""

        statuses = (
            "queued",
            "searching",
            "selecting",
            "evaluating",
            "holding_out",
        )
        placeholders = ", ".join("?" for _ in statuses)
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT r.run_id, coalesce(ctl.status, r.status) AS status "
                "FROM mining_v3.runtime_run r "
                "LEFT JOIN mining_v3.runtime_run_control ctl "
                "ON ctl.run_id=r.run_id "
                f"WHERE coalesce(ctl.status, r.status) IN ({placeholders}) "
                "ORDER BY r.run_id",
                list(statuses),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for run_id, status in rows:
            result.append(
                self.transition_search_run(
                    str(run_id),
                    "interrupted",
                    event_key=f"run_interrupted_{status}",
                    stage=str(status),
                    payload={"reason": "主服务重启"},
                )
            )
        return result

    def _persist_seed_candidate(
        self,
        run_id: str,
        envelope: ResearchInputEnvelope,
        candidate: CandidateDefinition,
    ) -> None:
        """幂等保存 CoSTEER 运行创建前的人工转写候选。"""

        if envelope.input_mode != "costeer":
            raise MiningContractError("只有 CoSTEER 输入可以配置预置候选")
        payload = CandidateEnvelope(
            candidate=candidate,
            source=envelope.source,
        ).to_dict()["candidate"]
        encoded = canonical_json(payload)
        with database_state_lock(self.db_path):
            with self._connection() as conn:
                with self._transaction(conn):
                    existing = self._row(
                        conn,
                        "SELECT * FROM mining_v3.runtime_seed_candidate "
                        "WHERE run_id=?",
                        [run_id],
                    )
                    if existing is not None:
                        if (
                            existing["input_hash"] != envelope.input_hash
                            or existing["source_id"] != envelope.source.source_id
                            or existing["candidate_payload_json"] != encoded
                            or existing["content_hash"] != candidate.content_hash
                            or existing["semantic_hash"] != candidate.semantic_hash
                        ):
                            raise MiningContractError(
                                "V3 运行已绑定不同 CoSTEER 预置候选"
                            )
                        return
                    conn.execute(
                        "INSERT INTO mining_v3.runtime_seed_candidate VALUES "
                        "(?, ?, ?, ?, ?, ?, ?)",
                        [
                            run_id,
                            envelope.input_hash,
                            envelope.source.source_id,
                            encoded,
                            candidate.content_hash,
                            candidate.semantic_hash,
                            utc_now_iso(),
                        ],
                    )

    @staticmethod
    def _candidate_from_payload(
        payload: Any,
        envelope: ResearchInputEnvelope,
    ) -> CandidateDefinition:
        """从持久化候选 JSON 恢复不可变候选定义。"""

        restored = CandidateEnvelope.from_dict(
            {
                "schema_version": V3_RESEARCH_CANDIDATE_SCHEMA_VERSION,
                "candidate": payload,
                "source": envelope.source.to_dict(),
                "evidence": [],
            }
        )
        return restored.candidate

    @staticmethod
    def _validate_execution_facts(
        request: SearchRunRequest,
        context: FrozenSearchContext,
        envelope: ResearchInputEnvelope,
        *,
        research_input: ResearchRunInput | None,
        seed_candidate: CandidateDefinition | None,
    ) -> None:
        """核对恢复运行所需的全部身份和来源事实。"""

        if (
            envelope.run_id != request.run_id
            or envelope.idempotency_key != request.idempotency_key
        ):
            raise V3RunStateError("V3 请求与输入包络身份不一致")
        if envelope.context_hash != context.context_hash:
            raise V3RunStateError("V3 输入包络与冻结上下文不一致")
        if request.budget != context.budget:
            raise V3RunStateError("V3 请求预算与冻结上下文不一致")
        if request.llm_profile != context.llm_profile:
            raise V3RunStateError("V3 请求模型配置与冻结上下文不一致")
        if envelope.input_mode in {"hypothesis", "free_exploration"}:
            if (
                request.input_mode != envelope.input_mode
                or request.request_hash != envelope.request_hash
            ):
                raise V3RunStateError("V3 搜索请求与输入包络不一致")
            if research_input is not None or seed_candidate is not None:
                raise V3RunStateError("普通 V3 运行包含未知预置输入")
            return
        if envelope.input_mode == "research_report":
            if not isinstance(research_input, ResearchRunInput):
                raise V3RunStateError("资料驱动 V3 运行缺少持久化资料输入")
            if (
                research_input.run_id != request.run_id
                or research_input.idempotency_key != request.idempotency_key
                or research_input.input_hash != envelope.request_hash
            ):
                raise V3RunStateError("资料输入与 V3 运行包络不一致")
            if seed_candidate is not None:
                raise V3RunStateError("资料驱动 V3 运行包含未知预置候选")
            return
        if not isinstance(seed_candidate, CandidateDefinition):
            raise V3RunStateError("CoSTEER 运行缺少持久化预置候选")
        if research_input is not None:
            raise V3RunStateError("CoSTEER 运行包含未知资料输入")

    def record_candidate(self, *args, **kwargs):
        """登记运行候选并同步创建 V3 候选来源包络。"""

        result = super().record_candidate(*args, **kwargs)
        self._sync_candidate(result)
        return result

    def record_evaluation_evidence(self, evidence: SearchEvaluationEvidence):
        """登记运行证据并追加到 V3 候选包络。"""

        result = super().record_evaluation_evidence(evidence)
        if evidence.stage in {
            "technical",
            "development",
            "selection",
            "full",
            "holdout",
        }:
            self._sync_evidence(evidence)
        return result

    def transition_candidate(self, run_id: str, candidate_id: str, target: str, **kwargs):
        """迁移运行候选状态并同步 V3 状态。"""

        result = super().transition_candidate(run_id, candidate_id, target, **kwargs)
        self.contract_store.sync_candidate_state(run_id, candidate_id, target)
        return result

    def _sync_candidate(self, record: Mapping[str, Any]) -> None:
        run_id = str(record["run_id"])
        candidate_id = str(record["candidate_id"])
        request = self.load_run_request(run_id)
        input_envelope = self.load_input_envelope(run_id)
        candidate = candidate_from_v2_record(record)
        source = input_envelope.source
        if input_envelope.input_mode == "free_exploration":
            hypothesis_row = self._candidate_hypothesis(run_id, candidate_id)
            direction = direction_from_hypothesis(
                HypothesisCard.from_dict(hypothesis_row)
            )
            if direction is not None:
                source = free_exploration_candidate_source(request, direction)
        elif input_envelope.input_mode == "research_report":
            provenance = self.get_research_candidate(run_id, candidate_id)
            if provenance is not None:
                source = research_report_candidate_source(provenance)
        envelope = CandidateEnvelope(
            candidate=candidate,
            source=source,
        )
        self.contract_store.record_candidate_envelope(run_id, envelope)
        self.contract_store.initialize_candidate_state(
            run_id,
            candidate_id,
            source.source_id,
        )

    def _sync_evidence(self, evidence: SearchEvaluationEvidence) -> None:
        source_id = self.contract_store.candidate_source_id(
            evidence.run_id,
            evidence.candidate_id,
        )
        current = self.contract_store.get_candidate_envelope(
            evidence.run_id,
            evidence.candidate_id,
            source_id,
        )
        reference = evidence_reference_from_search(evidence)
        updated = CandidateEnvelope(
            candidate=current.candidate,
            source=current.source,
            evidence=(*current.evidence, reference),
        )
        self.contract_store.record_candidate_envelope(
            evidence.run_id,
            updated,
        )

    def _candidate_hypothesis(
        self,
        run_id: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT hypothesis_id FROM mining_v2.candidate_lineage "
                "WHERE run_id=? AND candidate_id=?",
                [run_id, candidate_id],
            )
            if row is None:
                raise MiningStorageError("运行候选缺少假设来源")
            hypothesis = self._row(
                conn,
                "SELECT payload_json FROM mining_v2.hypothesis "
                "WHERE run_id=? AND hypothesis_id=?",
                [run_id, row["hypothesis_id"]],
            )
        if hypothesis is None:
            raise MiningStorageError("运行候选绑定的假设不存在")
        import json

        return json.loads(hypothesis["payload_json"])


__all__ = ["MiningV3RuntimeStore", "V3RunExecutionState"]
