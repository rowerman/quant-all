"""Worker 本地作业状态存储；不接触候选、行情或正式因子数据。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

from mining_worker.protocol import WorkerJobView


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class JobConflictError(RuntimeError):
    """同一 request_id 对应不同请求内容。"""


class JobBusyError(RuntimeError):
    """已有作业占用唯一执行槽位。"""


class JobStore:
    """单进程 Worker 使用的小型 SQLite 作业状态库。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS worker_job (
                    job_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    payload_hash TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_worker_job_status "
                "ON worker_job(status)"
            )
            self.interrupt_unfinished()

    def close(self) -> None:
        """关闭数据库连接。"""

        with self._lock:
            self._connection.close()

    def interrupt_unfinished(self) -> int:
        """启动时把上次进程遗留的作业标记为中断。"""

        with self._lock:
            cursor = self._connection.execute(
                "UPDATE worker_job SET status='interrupted', "
                "error_code='MINING-117', error_message=?, finished_at=? "
                "WHERE status IN ('queued', 'running')",
                ("Worker 重启，未完成作业已中断", _now()),
            )
            return int(cursor.rowcount)

    def create(
        self,
        *,
        job_id: str,
        request_id: str,
        payload_hash: str,
        request: dict[str, Any],
    ) -> tuple[WorkerJobView, bool]:
        """幂等创建作业，返回视图及是否新建。"""

        encoded = json.dumps(request, ensure_ascii=False, sort_keys=True)
        with self._lock:
            existing = self._connection.execute(
                "SELECT * FROM worker_job WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_hash"] != payload_hash:
                    raise JobConflictError("request_id 已被不同请求内容占用")
                return self._decode(existing), False
            active = self._connection.execute(
                "SELECT 1 FROM worker_job WHERE status IN ('queued', 'running') LIMIT 1"
            ).fetchone()
            if active is not None:
                raise JobBusyError("Worker 已有作业正在运行")
            created_at = _now()
            self._connection.execute(
                "INSERT INTO worker_job "
                "(job_id, request_id, payload_hash, request_json, status, created_at) "
                "VALUES (?, ?, ?, ?, 'queued', ?)",
                (job_id, request_id, payload_hash, encoded, created_at),
            )
            row = self._connection.execute(
                "SELECT * FROM worker_job WHERE job_id=?",
                (job_id,),
            ).fetchone()
            assert row is not None
            return self._decode(row), True

    def get(self, job_id: str) -> WorkerJobView | None:
        """按作业 ID 读取状态。"""

        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM worker_job WHERE job_id=?",
                (job_id,),
            ).fetchone()
            return self._decode(row) if row is not None else None

    def request(self, job_id: str) -> dict[str, Any] | None:
        """读取已持久化的原请求。"""

        with self._lock:
            row = self._connection.execute(
                "SELECT request_json FROM worker_job WHERE job_id=?",
                (job_id,),
            ).fetchone()
            return json.loads(row["request_json"]) if row is not None else None

    def mark_running(self, job_id: str) -> bool:
        """仅在作业仍排队时切换到运行。"""

        with self._lock:
            cursor = self._connection.execute(
                "UPDATE worker_job SET status='running', started_at=? "
                "WHERE job_id=? AND status='queued'",
                (_now(), job_id),
            )
            return cursor.rowcount == 1

    def finish_success(self, job_id: str, result: dict[str, Any]) -> bool:
        """写入成功结果，不覆盖已经取消的作业。"""

        return self._finish(job_id, "succeeded", result=result)

    def finish_failure(self, job_id: str, code: str, message: str) -> bool:
        """写入失败状态，不覆盖已经取消的作业。"""

        return self._finish(job_id, "failed", code=code, message=message)

    def cancel(self, job_id: str, message: str) -> WorkerJobView | None:
        """把排队或运行作业标记为取消。"""

        with self._lock:
            self._connection.execute(
                "UPDATE worker_job SET status='cancelled', error_code='MINING-117', "
                "error_message=?, finished_at=? "
                "WHERE job_id=? AND status IN ('queued', 'running')",
                (message[:500], _now(), job_id),
            )
            return self.get(job_id)

    def _finish(
        self,
        job_id: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        code: str | None = None,
        message: str | None = None,
    ) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE worker_job SET status=?, result_json=?, error_code=?, "
                "error_message=?, finished_at=? "
                "WHERE job_id=? AND status='running'",
                (
                    status,
                    json.dumps(result, ensure_ascii=False, sort_keys=True)
                    if result is not None
                    else None,
                    code,
                    message[:500] if message else None,
                    _now(),
                    job_id,
                ),
            )
            return cursor.rowcount == 1

    @staticmethod
    def _decode(row: sqlite3.Row) -> WorkerJobView:
        return WorkerJobView(
            job_id=row["job_id"],
            request_id=row["request_id"],
            status=row["status"],
            result=json.loads(row["result_json"])
            if row["result_json"] is not None
            else None,
            error_code=row["error_code"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )
