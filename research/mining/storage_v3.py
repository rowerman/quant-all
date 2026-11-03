"""挖掘 V3 统一输入、候选来源和证据引用存储。"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Sequence

import duckdb

from common.paths import MINING_DATA_DIR, MINING_DB_FILE
from mining.contract_types import canonical_json, sha256_json, utc_now_iso
from mining.errors import (
    MiningContractError,
    MiningDuplicateError,
    MiningMigrationError,
    MiningNotFoundError,
    MiningStorageError,
)
from mining.logging_conf import get_mining_logger
from mining.storage_lock import database_state_lock
from mining.storage_schema import ensure_schema
from mining.storage_v2_schema import schema_exists
from mining.storage_v3_publication import V3PublicationStoreMixin
from mining.storage_v3_review import V3ReviewStoreMixin
from mining.storage_v3_legacy import V3LegacyStoreMixin
from mining.storage_v3_schema import (
    ensure_v3_schema,
    initialize_v3_schema,
)
from mining.v3_contracts import (
    CandidateEnvelope,
    EvidenceReference,
    ResearchInputEnvelope,
    ResearchSourceReference,
)


class MiningV3Store(
    V3ReviewStoreMixin,
    V3PublicationStoreMixin,
    V3LegacyStoreMixin,
):
    """持久化 V3 来源无关候选及独立来源绑定。"""

    def __init__(
        self,
        db_path: Path | str = MINING_DB_FILE,
        data_root: Path | str = MINING_DATA_DIR,
    ) -> None:
        self.db_path = Path(db_path)
        self.data_root = Path(data_root)
        self._connection_lock = RLock()
        self.logger = get_mining_logger("mining.storage_v3")
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    if schema_exists(conn, "mining_v3"):
                        ensure_v3_schema(conn)
                        ensure_schema(conn)
                    elif schema_exists(conn, "mining_v2") or schema_exists(
                        conn,
                        "mining",
                    ):
                        raise MiningMigrationError(
                            "检测到 V1/V2 候选库，请先执行显式 V3 迁移"
                        )
                    else:
                        with self._transaction(conn):
                            initialize_v3_schema(conn)
                            ensure_schema(conn)
            self.data_root.mkdir(parents=True, exist_ok=True)
        except (MiningContractError, MiningMigrationError, MiningStorageError):
            raise
        except (OSError, duckdb.Error) as error:
            raise MiningStorageError(
                f"无法初始化 V3 候选存储：{error}"
            ) from error

    @contextmanager
    def _connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """打开并可靠关闭一个写连接。"""

        with self._connection_lock:
            try:
                conn = duckdb.connect(str(self.db_path))
            except duckdb.Error as error:
                raise MiningStorageError(
                    f"V3 候选库连接失败：{error}"
                ) from error
            try:
                yield conn
            finally:
                conn.close()

    @contextmanager
    def _transaction(self, conn: duckdb.DuckDBPyConnection) -> Iterator[None]:
        """执行显式事务，异常时整体回滚。"""

        conn.begin()
        try:
            yield
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()

    @staticmethod
    def _row(
        conn: duckdb.DuckDBPyConnection,
        sql: str,
        params: Sequence[Any] = (),
    ) -> dict[str, Any] | None:
        """读取一行并保留列名。"""

        row = conn.execute(sql, list(params)).fetchone()
        if row is None:
            return None
        columns = [item[0] for item in conn.description]
        return dict(zip(columns, row, strict=True))

    def register_input(self, envelope: ResearchInputEnvelope) -> dict[str, Any]:
        """幂等登记 V3 输入及其主来源。"""

        if not isinstance(envelope, ResearchInputEnvelope):
            raise MiningContractError("envelope 必须是 ResearchInputEnvelope")
        encoded = canonical_json(envelope.to_dict())
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        by_run = self._row(
                            conn,
                            "SELECT * FROM mining_v3.research_input WHERE run_id=?",
                            [envelope.run_id],
                        )
                        by_key = self._row(
                            conn,
                            "SELECT * FROM mining_v3.research_input "
                            "WHERE idempotency_key=?",
                            [envelope.idempotency_key],
                        )
                        existing = by_run or by_key
                        if existing is not None:
                            if (
                                existing["run_id"] != envelope.run_id
                                or existing["idempotency_key"]
                                != envelope.idempotency_key
                                or existing["input_mode"] != envelope.input_mode
                                or existing["context_hash"]
                                != envelope.context_hash
                                or existing["request_hash"]
                                != envelope.request_hash
                                or existing["source_id"]
                                != envelope.source.source_id
                                or existing["input_hash"] != envelope.input_hash
                                or existing["payload_json"] != encoded
                            ):
                                raise MiningDuplicateError(
                                    "V3 运行或幂等键已绑定不同输入"
                                )
                            self._require_source_conn(
                                conn,
                                envelope.run_id,
                                envelope.source,
                            )
                            reused = True
                        else:
                            self._insert_source_conn(
                                conn,
                                envelope.run_id,
                                envelope.source,
                            )
                            conn.execute(
                                "INSERT INTO mining_v3.research_input VALUES "
                                "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                [
                                    envelope.run_id,
                                    envelope.idempotency_key,
                                    envelope.input_mode,
                                    envelope.context_hash,
                                    envelope.request_hash,
                                    envelope.source.source_id,
                                    envelope.input_hash,
                                    encoded,
                                    utc_now_iso(),
                                ],
                            )
                            reused = False
        except (
            MiningContractError,
            MiningDuplicateError,
            MiningStorageError,
        ):
            raise
        except duckdb.Error as error:
            raise MiningStorageError(f"登记 V3 输入失败：{error}") from error
        self.logger.info(
            "operation=register_v3_input run_id=%s input_mode=%s reused=%s",
            envelope.run_id,
            envelope.input_mode,
            reused,
        )
        return {
            "run_id": envelope.run_id,
            "input_hash": envelope.input_hash,
            "source_reference_hash": envelope.source.reference_hash,
            "reused": reused,
        }

    def load_input(self, run_id: str) -> ResearchInputEnvelope:
        """读取并校验 V3 输入包络。"""

        try:
            with self._connection() as conn:
                envelope = self._input_envelope_conn(conn, run_id)
        except (MiningContractError, MiningNotFoundError, MiningStorageError):
            raise
        except (duckdb.Error, json.JSONDecodeError) as error:
            raise MiningStorageError(f"读取 V3 输入失败：{error}") from error
        return envelope

    def get_source_reference(
        self,
        run_id: str,
        source_id: str,
    ) -> ResearchSourceReference:
        """读取并校验一个 V3 来源引用。"""

        with self._connection() as conn:
            return self._source_conn(conn, run_id, source_id)

    def record_candidate_envelope(
        self,
        run_id: str,
        envelope: CandidateEnvelope,
    ) -> dict[str, Any]:
        """追加登记候选来源和证据；重复包络保持幂等。"""

        if not isinstance(envelope, CandidateEnvelope):
            raise MiningContractError("envelope 必须是 CandidateEnvelope")
        encoded_candidate = canonical_json(
            CandidateEnvelope(
                candidate=envelope.candidate,
                source=envelope.source,
                evidence=(),
            ).to_dict()
        )
        candidate_source_hash = sha256_json(
            {
                "candidate_content_hash": envelope.candidate.content_hash,
                "semantic_hash": envelope.candidate.semantic_hash,
                "source_reference_hash": envelope.source.reference_hash,
            }
        )
        reused = True
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        stored_input = self._input_envelope_conn(conn, run_id)
                        if stored_input.input_mode != envelope.source.source_kind:
                            raise MiningContractError(
                                "候选来源类型与 V3 运行输入模式不一致"
                            )
                        self._insert_source_conn(
                            conn,
                            run_id,
                            envelope.source,
                        )
                        existing = self._row(
                            conn,
                            "SELECT * FROM mining_v3.candidate_source "
                            "WHERE run_id=? AND candidate_id=? AND source_id=?",
                            [
                                run_id,
                                envelope.candidate.candidate_id,
                                envelope.source.source_id,
                            ],
                        )
                        if existing is None:
                            conn.execute(
                                "INSERT INTO mining_v3.candidate_source VALUES "
                                "(?, ?, ?, ?, ?, ?, ?, ?)",
                                [
                                    run_id,
                                    envelope.candidate.candidate_id,
                                    envelope.source.source_id,
                                    envelope.candidate.content_hash,
                                    envelope.candidate.semantic_hash,
                                    candidate_source_hash,
                                    encoded_candidate,
                                    utc_now_iso(),
                                ],
                            )
                            reused = False
                        elif (
                            existing["content_hash"]
                            != envelope.candidate.content_hash
                            or existing["semantic_hash"]
                            != envelope.candidate.semantic_hash
                            or existing["candidate_source_hash"]
                            != candidate_source_hash
                            or existing["candidate_payload_json"]
                            != encoded_candidate
                        ):
                            raise MiningDuplicateError(
                                "V3 候选和来源已绑定不同内容"
                            )
                        for evidence in envelope.evidence:
                            if self._insert_evidence_conn(
                                conn,
                                run_id,
                                envelope,
                                evidence,
                            ):
                                reused = False
            stored = self.get_candidate_envelope(
                run_id,
                envelope.candidate.candidate_id,
                envelope.source.source_id,
            )
        except (
            MiningContractError,
            MiningDuplicateError,
            MiningNotFoundError,
            MiningStorageError,
        ):
            raise
        except (duckdb.Error, json.JSONDecodeError) as error:
            raise MiningStorageError(
                f"登记 V3 候选来源失败：{error}"
            ) from error
        self.logger.info(
            "operation=record_v3_candidate_source run_id=%s candidate_id=%s "
            "source_id=%s reused=%s",
            run_id,
            envelope.candidate.candidate_id,
            envelope.source.source_id,
            reused,
        )
        return {
            "run_id": run_id,
            "candidate_id": stored.candidate.candidate_id,
            "source_id": stored.source.source_id,
            "envelope_hash": stored.envelope_hash,
            "reused": reused,
        }

    def get_candidate_envelope(
        self,
        run_id: str,
        candidate_id: str,
        source_id: str,
    ) -> CandidateEnvelope:
        """读取一个候选与来源绑定的完整包络。"""

        with self._connection() as conn:
            return self._candidate_envelope_conn(
                conn,
                run_id,
                candidate_id,
                source_id,
            )

    def list_candidate_envelopes(
        self,
        run_id: str,
    ) -> tuple[CandidateEnvelope, ...]:
        """按候选和来源稳定顺序列出运行内全部包络。"""

        with self._connection() as conn:
            self._input_row_conn(conn, run_id)
            rows = conn.execute(
                "SELECT candidate_id, source_id "
                "FROM mining_v3.candidate_source WHERE run_id=? "
                "ORDER BY candidate_id, source_id",
                [run_id],
            ).fetchall()
            return tuple(
                self._candidate_envelope_conn(
                    conn,
                    run_id,
                    str(candidate_id),
                    str(source_id),
                )
                for candidate_id, source_id in rows
            )

    def _input_row_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
    ) -> dict[str, Any]:
        row = self._row(
            conn,
            "SELECT * FROM mining_v3.research_input WHERE run_id=?",
            [run_id],
        )
        if row is None:
            raise MiningNotFoundError(f"V3 研究输入不存在：{run_id}")
        return row

    def _input_envelope_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
    ) -> ResearchInputEnvelope:
        """读取输入并核对所有公开列与正文及来源表。"""

        row = self._input_row_conn(conn, run_id)
        envelope = ResearchInputEnvelope.from_dict(
            json.loads(row["payload_json"])
        )
        source = self._source_conn(conn, run_id, envelope.source.source_id)
        if (
            row["run_id"] != envelope.run_id
            or row["idempotency_key"] != envelope.idempotency_key
            or row["input_mode"] != envelope.input_mode
            or row["context_hash"] != envelope.context_hash
            or row["request_hash"] != envelope.request_hash
            or row["source_id"] != envelope.source.source_id
            or row["input_hash"] != envelope.input_hash
            or source != envelope.source
        ):
            raise MiningStorageError("V3 输入列与包络或来源内容不一致")
        return envelope

    def _insert_source_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        source: ResearchSourceReference,
    ) -> None:
        existing = self._row(
            conn,
            "SELECT * FROM mining_v3.source_reference "
            "WHERE run_id=? AND source_id=?",
            [run_id, source.source_id],
        )
        encoded = canonical_json(source.to_dict())
        if existing is not None:
            if (
                existing["reference_hash"] != source.reference_hash
                or existing["payload_json"] != encoded
            ):
                raise MiningDuplicateError("V3 来源 ID 已绑定不同来源内容")
            return
        conn.execute(
            "INSERT INTO mining_v3.source_reference VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                source.source_id,
                source.source_kind,
                source.source_hash,
                source.artifact_manifest_hash,
                source.promotion_hash,
                source.reference_hash,
                encoded,
                utc_now_iso(),
            ],
        )

    def _require_source_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        source: ResearchSourceReference,
    ) -> None:
        stored = self._source_conn(conn, run_id, source.source_id)
        if stored != source:
            raise MiningDuplicateError("V3 来源引用内容冲突")

    def _source_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        source_id: str,
    ) -> ResearchSourceReference:
        row = self._row(
            conn,
            "SELECT * FROM mining_v3.source_reference "
            "WHERE run_id=? AND source_id=?",
            [run_id, source_id],
        )
        if row is None:
            raise MiningNotFoundError(f"V3 来源引用不存在：{source_id}")
        source = ResearchSourceReference.from_dict(json.loads(row["payload_json"]))
        if (
            row["reference_hash"] != source.reference_hash
            or row["source_hash"] != source.source_hash
            or row["source_kind"] != source.source_kind
        ):
            raise MiningStorageError(
                f"V3 来源引用列与正文不一致：{source_id}"
            )
        return source

    def _insert_evidence_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        envelope: CandidateEnvelope,
        evidence: EvidenceReference,
    ) -> bool:
        """插入一条证据；返回是否新增。"""

        existing = self._row(
            conn,
            "SELECT * FROM mining_v3.evaluation_evidence "
            "WHERE run_id=? AND candidate_id=? AND source_id=? "
            "AND evaluation_id=?",
            [
                run_id,
                envelope.candidate.candidate_id,
                envelope.source.source_id,
                evidence.evaluation_id,
            ],
        )
        encoded = canonical_json(evidence.to_dict())
        if existing is not None:
            if (
                existing["evidence_hash"] != evidence.evidence_hash
                or existing["payload_json"] != encoded
            ):
                raise MiningDuplicateError("V3 评估 ID 已绑定不同证据")
            return False
        conn.execute(
            "INSERT INTO mining_v3.evaluation_evidence VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                evidence.candidate_id,
                envelope.source.source_id,
                evidence.evaluation_id,
                evidence.stage,
                evidence.input_hash,
                evidence.evidence_hash,
                evidence.status,
                encoded,
                utc_now_iso(),
            ],
        )
        return True

    def _candidate_envelope_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        candidate_id: str,
        source_id: str,
    ) -> CandidateEnvelope:
        row = self._row(
            conn,
            "SELECT * FROM mining_v3.candidate_source "
            "WHERE run_id=? AND candidate_id=? AND source_id=?",
            [run_id, candidate_id, source_id],
        )
        if row is None:
            raise MiningNotFoundError(
                f"V3 候选来源不存在：{run_id}/{candidate_id}/{source_id}"
            )
        input_envelope = self._input_envelope_conn(conn, run_id)
        stored_source = self._source_conn(conn, run_id, source_id)
        evidence_rows = conn.execute(
            "SELECT payload_json FROM mining_v3.evaluation_evidence "
            "WHERE run_id=? AND candidate_id=? AND source_id=? "
            "ORDER BY stage, evaluation_id",
            [run_id, candidate_id, source_id],
        ).fetchall()
        payload = json.loads(row["candidate_payload_json"])
        payload["evidence"] = [
            json.loads(values[0]) for values in evidence_rows
        ]
        envelope = CandidateEnvelope.from_dict(payload)
        if (
            envelope.source != stored_source
            or envelope.source.source_kind != input_envelope.input_mode
        ):
            raise MiningStorageError(
                "V3 候选来源与运行输入或来源表不一致"
            )
        expected_source_hash = sha256_json(
            {
                "candidate_content_hash": envelope.candidate.content_hash,
                "semantic_hash": envelope.candidate.semantic_hash,
                "source_reference_hash": envelope.source.reference_hash,
            }
        )
        if (
            row["content_hash"] != envelope.candidate.content_hash
            or row["semantic_hash"] != envelope.candidate.semantic_hash
            or row["candidate_source_hash"] != expected_source_hash
        ):
            raise MiningStorageError("V3 候选来源列与正文不一致")
        for evidence in envelope.evidence:
            matched = self._row(
                conn,
                "SELECT * FROM mining_v3.evaluation_evidence "
                "WHERE run_id=? AND candidate_id=? AND source_id=? "
                "AND evaluation_id=?",
                [
                    run_id,
                    candidate_id,
                    source_id,
                    evidence.evaluation_id,
                ],
            )
            if (
                matched is None
                or matched["evidence_hash"] != evidence.evidence_hash
                or matched["candidate_id"] != evidence.candidate_id
                or matched["source_id"] != source_id
                or matched["stage"] != evidence.stage
                or matched["input_hash"] != evidence.input_hash
                or matched["status"] != evidence.status
                or json.loads(matched["payload_json"]) != evidence.to_dict()
            ):
                raise MiningStorageError(
                    f"V3 证据列与正文不一致：{evidence.evaluation_id}"
                )
        return envelope


__all__ = ["MiningV3Store"]
