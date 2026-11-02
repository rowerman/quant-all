"""V3 运行时存储：复用 V2 搜索机制并写入 `mining_v3`。"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Mapping

import duckdb

from common.paths import MINING_DATA_DIR, MINING_DB_FILE
from mining.contract_candidates import CandidateDefinition
from mining.contract_types import canonical_json, utc_now_iso
from mining.errors import MiningContractError, MiningStorageError
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
from mining.v3_contracts import CandidateEnvelope, ResearchInputEnvelope
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
        self._run_inputs: dict[str, ResearchInputEnvelope] = {}
        self._run_contexts: dict[str, FrozenSearchContext] = {}
        self._run_requests: dict[str, SearchRunRequest] = {}
        self._run_seed_candidates: dict[str, CandidateDefinition] = {}
        self._configure_lock = RLock()
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    ensure_v3_schema(conn._conn)
                    for statement in V2_DDL:
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
        seed_candidate: CandidateDefinition | None = None,
    ) -> None:
        """在创建运行前冻结 V3 输入、上下文和请求。"""

        if input_envelope.run_id != request.run_id:
            raise MiningContractError("V3 输入与运行 ID 不一致")
        if input_envelope.context_hash != context.context_hash:
            raise MiningContractError("V3 输入与冻结上下文不一致")
        with self._configure_lock:
            self._run_inputs[request.run_id] = input_envelope
            self._run_contexts[request.run_id] = context
            self._run_requests[request.run_id] = request
            if seed_candidate is not None:
                if input_envelope.input_mode != "costeer":
                    raise MiningContractError(
                        "只有 CoSTEER 输入可以配置预置候选"
                    )
                self._run_seed_candidates[request.run_id] = seed_candidate

    def create_run(
        self,
        request: SearchRunRequest,
        context: FrozenSearchContext,
        *,
        research_input: ResearchRunInput | None = None,
    ) -> dict[str, Any]:
        """创建运行时事实并登记统一 V3 输入。"""

        envelope = self._run_inputs.get(request.run_id)
        if envelope is None:
            raise MiningContractError("V3 运行创建前必须先配置输入")
        self.contract_store.register_input(envelope)
        self.contract_store.register_context(request.run_id, context)
        run = super().create_run(
            request,
            context,
            research_input=research_input,
        )
        return run

    def seed_candidate(self, run_id: str) -> CandidateDefinition | None:
        """返回 CoSTEER 运行的人工转写候选。"""

        return self._run_seed_candidates.get(run_id)

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
        request = self._run_requests[run_id]
        input_envelope = self._run_inputs[run_id]
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


__all__ = ["MiningV3RuntimeStore"]
