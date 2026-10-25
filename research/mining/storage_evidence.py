"""候选评估报告与技术门禁存储。"""

from __future__ import annotations

import json
from typing import Any

from mining.contracts import EvaluationReport, TechnicalGate, canonical_json
from mining.errors import (
    ApprovalConflictError,
    MiningEvaluationError,
    MiningNotFoundError,
)
from mining.logging_conf import format_event


class EvidenceStoreMixin:
    """评估报告和技术门禁的持久化方法。"""

    def record_evaluation(self, report: EvaluationReport) -> dict[str, Any]:
        """幂等保存评估报告。"""

        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    candidate = self._require_current_candidate(
                        conn,
                        run_id=report.run_id,
                        candidate_id=report.candidate_id,
                        content_hash=report.candidate_content_hash,
                    )
                    context = self._context_for_run_conn(conn, report.run_id)
                    if context is None or context["context_hash"] != report.context_hash:
                        raise ApprovalConflictError("评估报告绑定了不同或未知上下文")
                    expected_status = {
                        "technical": "validating",
                        "quick": "screening",
                        "full": "evaluating",
                        "holdout": "holdout_running",
                    }[report.stage]
                    if candidate["status"] != expected_status:
                        raise MiningEvaluationError(
                            f"{report.stage} 报告要求候选处于 {expected_status}，"
                            f"当前为 {candidate['status']}"
                        )
                    existing = self._row(
                        conn,
                        "SELECT * FROM mining.evaluation WHERE report_id=?",
                        [report.report_id],
                    )
                    if existing is not None:
                        expected = self._evaluation_row(report)
                        self._assert_same_row(
                            existing,
                            expected,
                            tuple(expected),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO mining.evaluation VALUES "
                            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            self._evaluation_values(report),
                        )
                        self._append_event_conn(
                            conn,
                            report.run_id,
                            f"evaluation_{report.report_id}",
                            report.stage,
                            "evaluation_recorded",
                            {
                                "candidate_id": report.candidate_id,
                                "report_id": report.report_id,
                                "status": report.status,
                                "input_hash": report.input_hash,
                            },
                        )
            result = self._get_evaluation(report.report_id)
        self.logger.info(
            format_event(
                {
                    "operation": "record_evaluation",
                    "run_id": report.run_id,
                    "candidate_id": report.candidate_id,
                    "stage": report.stage,
                    "status": report.status,
                }
            )
        )
        return result

    @staticmethod
    def _evaluation_values(report: EvaluationReport) -> list[Any]:
        """把评估对象转为表字段顺序。"""

        return [
            report.report_id,
            report.run_id,
            report.candidate_id,
            report.stage,
            report.candidate_content_hash,
            report.context_hash,
            report.input_hash,
            report.status,
            canonical_json(dict(report.metrics)),
            canonical_json(list(report.warnings)),
            report.artifact_path,
            report.artifact_sha256,
            report.error_code,
            report.error_message,
            report.attempt,
            report.started_at,
            report.finished_at,
        ]

    @classmethod
    def _evaluation_row(cls, report: EvaluationReport) -> dict[str, Any]:
        """返回用于重复写入比较的评估字段。"""

        columns = (
            "report_id",
            "run_id",
            "candidate_id",
            "stage",
            "candidate_content_hash",
            "context_hash",
            "input_hash",
            "status",
            "metrics_json",
            "warnings_json",
            "artifact_path",
            "artifact_sha256",
            "error_code",
            "error_message",
            "attempt",
            "started_at",
            "finished_at",
        )
        return dict(
            zip(columns, cls._evaluation_values(report), strict=True)
        )

    @staticmethod
    def _decode_evaluation(row: dict[str, Any]) -> dict[str, Any]:
        """解码评估报告 JSON。"""

        result = dict(row)
        result["metrics"] = json.loads(result.pop("metrics_json"))
        result["warnings"] = json.loads(result.pop("warnings_json"))
        return result

    def _get_evaluation(self, report_id: str) -> dict[str, Any]:
        """按 ID 读取评估报告。"""

        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining.evaluation WHERE report_id=?",
                [report_id],
            )
        if row is None:
            raise MiningNotFoundError(f"评估报告不存在：{report_id}")
        return self._decode_evaluation(row)

    def list_evaluations(
        self,
        run_id: str,
        candidate_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """列出运行或候选的评估报告。"""

        sql = "SELECT * FROM mining.evaluation WHERE run_id=?"
        params: list[Any] = [run_id]
        if candidate_id is not None:
            sql += " AND candidate_id=?"
            params.append(candidate_id)
        sql += " ORDER BY started_at, stage, attempt"
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            columns = [item[0] for item in conn.description]
        return [
            self._decode_evaluation(dict(zip(columns, row, strict=True)))
            for row in rows
        ]

    def find_evaluation(
        self,
        run_id: str,
        candidate_id: str,
        stage: str,
    ) -> dict[str, Any] | None:
        """查找候选指定阶段的最近报告。"""

        rows = self.list_evaluations(run_id, candidate_id)
        matching = [row for row in rows if row["stage"] == stage]
        return matching[-1] if matching else None

    def record_technical_gate(self, gate: TechnicalGate) -> dict[str, Any]:
        """幂等保存技术门禁结论。"""

        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    self._require_current_candidate(
                        conn,
                        run_id=gate.run_id,
                        candidate_id=gate.candidate_id,
                        content_hash=gate.candidate_content_hash,
                    )
                    context = self._context_for_run_conn(conn, gate.run_id)
                    if context is None or context["context_hash"] != gate.context_hash:
                        raise ApprovalConflictError("技术门禁绑定了不同或未知上下文")
                    existing = self._row(
                        conn,
                        "SELECT * FROM mining.technical_gate WHERE gate_id=?",
                        [gate.gate_id],
                    )
                    if existing is not None:
                        self._assert_same_row(
                            existing,
                            {
                                "passed": gate.passed,
                                "checks_json": canonical_json(dict(gate.checks)),
                                "check_hash": gate.check_hash,
                            },
                            ("passed", "checks_json", "check_hash"),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO mining.technical_gate VALUES "
                            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            [
                                gate.gate_id,
                                gate.run_id,
                                gate.candidate_id,
                                gate.candidate_content_hash,
                                gate.context_hash,
                                gate.rule_version,
                                gate.passed,
                                canonical_json(dict(gate.checks)),
                                gate.check_hash,
                                gate.created_at,
                            ],
                        )
                        self._append_event_conn(
                            conn,
                            gate.run_id,
                            f"gate_{gate.gate_id}",
                            "technical",
                            "technical_gate_recorded",
                            {
                                "candidate_id": gate.candidate_id,
                                "gate_id": gate.gate_id,
                                "passed": gate.passed,
                                "check_hash": gate.check_hash,
                            },
                        )
            return self.get_technical_gate(gate.gate_id)

    @staticmethod
    def _decode_gate(row: dict[str, Any]) -> dict[str, Any]:
        """解码技术门禁 JSON。"""

        result = dict(row)
        result["checks"] = json.loads(result.pop("checks_json"))
        return result

    def get_technical_gate(self, gate_id: str) -> dict[str, Any]:
        """读取技术门禁。"""

        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining.technical_gate WHERE gate_id=?",
                [gate_id],
            )
        if row is None:
            raise MiningNotFoundError(f"技术门禁不存在：{gate_id}")
        return self._decode_gate(row)

    def find_technical_gate(
        self,
        run_id: str,
        candidate_id: str,
    ) -> dict[str, Any] | None:
        """查找候选最近一次技术门禁。"""

        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining.technical_gate "
                "WHERE run_id=? AND candidate_id=? "
                "ORDER BY created_at DESC LIMIT 1",
                [run_id, candidate_id],
            )
        return self._decode_gate(row) if row is not None else None
