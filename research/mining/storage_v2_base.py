"""挖掘 V2 存储的连接、事务和事件基础能力。"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Mapping, Sequence

import duckdb

from common.paths import MINING_DATA_DIR, MINING_DB_FILE
from mining.contract_types import canonical_json, utc_now_iso
from mining.errors import (
    MiningContractError,
    MiningDuplicateError,
    MiningMigrationError,
    MiningStorageError,
)
from mining.logging_conf import ensure_safe_event_payload, get_mining_logger
from mining.storage_lock import database_state_lock
from mining.storage_v2_schema import (
    ensure_v2_schema,
    initialize_v2_schema,
    schema_exists,
)


class V2StorageBase:
    """为 V2 运行、轨迹和台账仓库提供持久化基础设施。"""

    def __init__(
        self,
        db_path: Path | str = MINING_DB_FILE,
        data_root: Path | str = MINING_DATA_DIR,
    ) -> None:
        self.db_path = Path(db_path)
        self.data_root = Path(data_root)
        self._connection_lock = RLock()
        self.logger = get_mining_logger("mining.storage_v2")
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.data_root.mkdir(parents=True, exist_ok=True)
            with database_state_lock(self.db_path):
                with self._connection() as conn:
                    if schema_exists(conn, "mining_v2"):
                        ensure_v2_schema(conn)
                    elif schema_exists(conn, "mining"):
                        raise MiningMigrationError(
                            "检测到 mining_v1，请先执行显式 V2 迁移"
                        )
                    else:
                        with self._transaction(conn):
                            initialize_v2_schema(conn)
        except (MiningContractError, MiningMigrationError, MiningStorageError):
            raise
        except (OSError, duckdb.Error) as error:
            raise MiningStorageError(f"无法初始化 V2 候选存储：{error}") from error

    @contextmanager
    def _connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """打开并可靠关闭一个写连接。"""

        with self._connection_lock:
            try:
                conn = duckdb.connect(str(self.db_path))
            except duckdb.Error as error:
                raise MiningStorageError(f"V2 候选库连接失败：{error}") from error
            try:
                yield conn
            finally:
                conn.close()

    def run_dir(self, run_id: str) -> Path:
        """返回 V2 运行目录并按需创建。"""

        path = self.data_root / "runs" / run_id
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise MiningStorageError(f"无法创建 V2 运行目录：{error}") from error
        return path

    def candidate_dir(self, run_id: str, candidate_id: str) -> Path:
        """返回 V2 候选目录并按需创建。"""

        path = self.run_dir(run_id) / "candidates" / candidate_id
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise MiningStorageError(f"无法创建 V2 候选目录：{error}") from error
        return path

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

    def _append_event_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        event_key: str,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        created_at: str | None = None,
    ) -> int:
        """在已有运行事务中追加幂等事件。"""

        ensure_safe_event_payload(dict(payload))
        encoded = canonical_json(dict(payload))
        existing = self._row(
            conn,
            "SELECT sequence, event_type, payload_json FROM mining_v2.event "
            "WHERE run_id=? AND event_key=?",
            [run_id, event_key],
        )
        if existing is not None:
            if existing["event_type"] != event_type or existing["payload_json"] != encoded:
                raise MiningDuplicateError(f"事件键 {event_key} 已用于不同事件")
            return int(existing["sequence"])
        sequence = int(
            conn.execute(
                "SELECT coalesce(max(sequence), 0) + 1 "
                "FROM mining_v2.event WHERE run_id=?",
                [run_id],
            ).fetchone()[0]
        )
        conn.execute(
            "INSERT INTO mining_v2.event VALUES (?, ?, ?, ?, ?, ?)",
            [
                run_id,
                sequence,
                event_key,
                event_type,
                encoded,
                created_at or utc_now_iso(),
            ],
        )
        return sequence

    def list_events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        """读取 V2 运行的不可变事件。"""

        import json

        with self._connection() as conn:
            rows = conn.execute(
                "SELECT sequence, event_key, event_type, payload_json, created_at "
                "FROM mining_v2.event WHERE run_id=? AND sequence>? "
                "ORDER BY sequence",
                [run_id, max(0, int(after))],
            ).fetchall()
        return [
            {
                "sequence": int(sequence),
                "event_key": event_key,
                "event_type": event_type,
                "payload": json.loads(payload),
                "created_at": created_at,
            }
            for sequence, event_key, event_type, payload, created_at in rows
        ]

    def append_event(
        self,
        run_id: str,
        *,
        event_key: str,
        stage: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> int:
        """在独立事务中追加 V2 运行事件。"""

        with database_state_lock(self.db_path):
            with self._connection() as conn:
                with self._transaction(conn):
                    return self._append_event_conn(
                        conn,
                        run_id,
                        event_key,
                        event_type,
                        {"stage": stage, **dict(payload)},
                    )
