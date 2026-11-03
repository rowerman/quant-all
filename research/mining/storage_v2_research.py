"""V2 资料运行和候选来源的持久化。"""

from __future__ import annotations

import json
from typing import Any

import duckdb

from mining.contract_types import canonical_json, identifier, sha256_json, utc_now_iso
from mining.errors import (
    MiningContractError,
    MiningDuplicateError,
    MiningNotFoundError,
    MiningStorageError,
)
from mining.research_run_contracts import (
    ResearchCandidateProvenance,
    ResearchRunInput,
    ResearchRunSummary,
)
from mining.storage_lock import database_state_lock


class V2ResearchStoreMixin:
    """资料驱动运行输入、摘要和候选来源仓库。"""

    def load_research_input(self, run_id: str) -> ResearchRunInput | None:
        """读取资料运行输入；普通搜索运行返回 None。"""

        run_id = identifier(run_id, "run_id")
        with self._connection() as conn:
            self._require_run_conn(conn, run_id)
            row = self._row(
                conn,
                "SELECT input_json FROM mining_v2.research_run WHERE run_id=?",
                [run_id],
            )
        if row is None:
            return None
        return ResearchRunInput.from_dict(json.loads(row["input_json"]))

    def load_research_summary(self, run_id: str) -> ResearchRunSummary | None:
        """读取已完成的资料运行摘要。"""

        run_id = identifier(run_id, "run_id")
        with self._connection() as conn:
            self._require_run_conn(conn, run_id)
            row = self._row(
                conn,
                "SELECT summary_json FROM mining_v2.research_run WHERE run_id=?",
                [run_id],
            )
        if row is None or row["summary_json"] is None:
            return None
        return ResearchRunSummary.from_dict(json.loads(row["summary_json"]))

    def research_run_state(self, run_id: str) -> dict[str, Any] | None:
        """返回资料运行状态和摘要哈希，不暴露正文。"""

        run_id = identifier(run_id, "run_id")
        with self._connection() as conn:
            self._require_run_conn(conn, run_id)
            row = self._row(
                conn,
                "SELECT input_hash, manifest_hash, status, summary_json, "
                "created_at, updated_at FROM mining_v2.research_run WHERE run_id=?",
                [run_id],
            )
        if row is None:
            return None
        summary = (
            ResearchRunSummary.from_dict(json.loads(row.pop("summary_json")))
            if row.get("summary_json") is not None
            else None
        )
        row.pop("summary_json", None)
        row["summary_hash"] = None if summary is None else summary.summary_hash
        return row

    def save_research_summary(
        self,
        summary: ResearchRunSummary,
        provenances: tuple[ResearchCandidateProvenance, ...],
    ) -> dict[str, Any]:
        """原子保存资料运行摘要和全部准入候选来源。"""

        if not isinstance(summary, ResearchRunSummary):
            raise MiningContractError("summary 必须是 ResearchRunSummary")
        if summary.status not in {"ready", "failed"}:
            raise MiningContractError("总结状态必须是 ready 或 failed")
        provenances = tuple(provenances)
        if any(
            not isinstance(item, ResearchCandidateProvenance)
            for item in provenances
        ):
            raise MiningContractError("provenances 包含未知对象")
        if any(item.run_id != summary.input.run_id for item in provenances):
            raise MiningContractError("候选来源必须属于当前资料运行")
        if len({item.candidate_id for item in provenances}) != len(provenances):
            raise MiningContractError("候选来源不能包含重复候选")
        if len(provenances) != summary.admitted_count:
            raise MiningContractError("准入候选数量与来源记录数量不一致")
        run_id = summary.input.run_id
        encoded_summary = canonical_json(summary.to_dict())
        try:
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    with self._transaction(conn):
                        state = self._row(
                            conn,
                            "SELECT input_hash FROM mining_v2.research_run "
                            "WHERE run_id=?",
                            [run_id],
                        )
                        if state is None:
                            raise MiningNotFoundError("资料运行输入不存在")
                        if state["input_hash"] != summary.input.input_hash:
                            raise MiningDuplicateError("资料运行摘要与输入哈希不一致")
                        now = utc_now_iso()
                        conn.execute(
                            "UPDATE mining_v2.research_run SET status=?, "
                            "summary_json=?, updated_at=? WHERE run_id=?",
                            [summary.status, encoded_summary, now, run_id],
                        )
                        for provenance in provenances:
                            self._insert_research_candidate_conn(
                                conn,
                                provenance,
                            )
                        self._append_trace_conn(
                            conn,
                            run_id,
                            "research_run",
                            run_id,
                            summary.summary_hash,
                            now,
                        )
            result = self.research_run_state(run_id)
        except (
            MiningContractError,
            MiningDuplicateError,
            MiningNotFoundError,
        ):
            raise
        except duckdb.Error as error:
            raise MiningStorageError(f"保存资料运行摘要失败：{error}") from error
        if result is None:
            raise MiningStorageError("资料运行摘要写入后无法读取")
        return result

    def list_research_candidates(
        self,
        run_id: str,
    ) -> tuple[ResearchCandidateProvenance, ...]:
        """读取运行内全部准入候选来源并校验哈希。"""

        run_id = identifier(run_id, "run_id")
        with self._connection() as conn:
            self._require_run_conn(conn, run_id)
            rows = conn.execute(
                "SELECT candidate_id, provenance_hash, payload_json "
                "FROM mining_v2.research_candidate WHERE run_id=? "
                "ORDER BY document_id, draft_id, candidate_id",
                [run_id],
            ).fetchall()
        result: list[ResearchCandidateProvenance] = []
        for candidate_id, provenance_hash, payload in rows:
            provenance = ResearchCandidateProvenance.from_dict(json.loads(payload))
            if (
                provenance.candidate_id != candidate_id
                or provenance.provenance_hash != provenance_hash
            ):
                raise MiningStorageError(
                    f"资料候选来源哈希不一致：{candidate_id}"
                )
            result.append(provenance)
        return tuple(result)

    def get_research_candidate(
        self,
        run_id: str,
        candidate_id: str,
    ) -> ResearchCandidateProvenance:
        """读取并校验一个资料候选来源。"""

        run_id = identifier(run_id, "run_id")
        candidate_id = identifier(candidate_id, "candidate_id")
        with self._connection() as conn:
            self._require_run_conn(conn, run_id)
            row = self._row(
                conn,
                "SELECT provenance_hash, payload_json "
                "FROM mining_v2.research_candidate "
                "WHERE run_id=? AND candidate_id=?",
                [run_id, candidate_id],
            )
        if row is None:
            raise MiningNotFoundError(f"资料候选来源不存在：{candidate_id}")
        provenance = ResearchCandidateProvenance.from_dict(
            json.loads(row["payload_json"])
        )
        if provenance.provenance_hash != row["provenance_hash"]:
            raise MiningStorageError(
                f"资料候选来源哈希不一致：{candidate_id}"
            )
        return provenance

    def require_research_candidate(
        self,
        run_id: str,
        candidate_id: str,
        *,
        candidate_content_hash: str,
    ) -> ResearchCandidateProvenance:
        """确认公开候选内容哈希绑定的来源记录未被篡改。"""

        provenance = self.get_research_candidate(run_id, candidate_id)
        candidate = self.get_candidate(run_id, candidate_id)
        expected = candidate["payload"].get("parameters", {}).get(
            "research_provenance_hash"
        )
        if expected != provenance.provenance_hash:
            raise MiningStorageError("候选未绑定当前资料来源哈希")
        if provenance.candidate.content_hash != candidate_content_hash:
            raise MiningStorageError("资料候选内容哈希与来源记录不一致")
        return provenance

    def _insert_research_candidate_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        provenance: ResearchCandidateProvenance,
    ) -> None:
        existing = self._row(
            conn,
            "SELECT provenance_hash, payload_json "
            "FROM mining_v2.research_candidate "
            "WHERE run_id=? AND candidate_id=?",
            [provenance.run_id, provenance.candidate_id],
        )
        encoded = canonical_json(provenance.to_dict())
        if existing is not None:
            if (
                existing["provenance_hash"] != provenance.provenance_hash
                or existing["payload_json"] != encoded
            ):
                raise MiningDuplicateError("资料候选来源已绑定不同内容")
            return
        now = utc_now_iso()
        conn.execute(
            "INSERT INTO mining_v2.research_candidate VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?)",
            [
                provenance.run_id,
                provenance.candidate_id,
                provenance.document_id,
                provenance.extraction_id,
                provenance.draft_id,
                provenance.provenance_hash,
                encoded,
                now,
            ],
        )
        self._append_trace_conn(
            conn,
            provenance.run_id,
            "research_candidate",
            provenance.candidate_id,
            sha256_json(
                {
                    "provenance_hash": provenance.provenance_hash,
                    "candidate_content_hash": provenance.candidate.content_hash,
                }
            ),
            now,
        )
