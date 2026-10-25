"""候选存储的连接、事务、文件锁和事件基础能力。"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
import re
from threading import RLock
from typing import Any, Iterator, Mapping, Sequence

import duckdb

from common.paths import MINING_DATA_DIR, MINING_DB_FILE
from mining.contracts import (
    FrozenContext,
    MiningRunRequest,
    canonical_json,
    utc_now_iso,
)
from mining.errors import (
    MiningContractError,
    MiningDuplicateError,
    MiningStorageError,
)
from mining.logging_conf import ensure_safe_event_payload, get_mining_logger
from mining.storage_schema import ensure_schema

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")


class StorageBase:
    """为各领域存储仓库提供共享的持久化基础设施。"""

    def __init__(
        self,
        db_path: Path | str = MINING_DB_FILE,
        data_root: Path | str = MINING_DATA_DIR,
    ) -> None:
        self.db_path = Path(db_path)
        self.data_root = Path(data_root)
        self._connection_lock = RLock()
        self.logger = get_mining_logger("mining.storage")
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.data_root.mkdir(parents=True, exist_ok=True)
            with self._state_lock():
                with self._connection() as conn:
                    ensure_schema(conn)
        except MiningStorageError:
            raise
        except (OSError, duckdb.Error) as error:
            raise MiningStorageError(f"无法初始化候选存储：{error}") from error

    @contextmanager
    def _connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """打开并可靠关闭一个写连接。"""

        with self._connection_lock:
            try:
                conn = duckdb.connect(str(self.db_path))
            except duckdb.Error as error:
                raise MiningStorageError(f"候选库连接失败：{error}") from error
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

    @contextmanager
    def _state_lock(self) -> Iterator[None]:
        """用本机文件锁串行化写入与状态迁移。"""

        lock_path = self.db_path.with_name(f"{self.db_path.name}.state.lock")
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = lock_path.open("a+b")
        except OSError as error:
            raise MiningStorageError(f"候选库状态锁不可用：{error}") from error
        with handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except OSError as error:
                raise MiningStorageError(f"候选库状态锁不可用：{error}") from error
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _validate_id(value: str, name: str) -> str:
        """校验数据库和目录共用的标识。"""

        if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
            raise MiningContractError(f"{name} 不合法")
        return value

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

    def _context_for_run_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
    ) -> dict[str, Any] | None:
        """通过运行读取共享的冻结上下文快照。"""

        return self._row(
            conn,
            "SELECT c.* FROM mining.run_context c "
            "JOIN mining.run r ON r.context_id=c.context_id "
            "WHERE r.run_id=?",
            [run_id],
        )

    def run_dir(self, run_id: str) -> Path:
        """返回运行目录，仅在需要时创建。"""

        path = self.data_root / "runs" / self._validate_id(run_id, "run_id")
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise MiningStorageError(f"无法创建运行目录：{error}") from error
        return path

    def candidate_dir(self, run_id: str, candidate_id: str) -> Path:
        """返回候选目录，仅在需要时创建。"""

        path = (
            self.run_dir(run_id)
            / "candidates"
            / self._validate_id(candidate_id, "candidate_id")
        )
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise MiningStorageError(f"无法创建候选目录：{error}") from error
        return path

    @property
    def cache_dir(self) -> Path:
        """返回与正式因子目录隔离的评估缓存目录。"""

        path = self.data_root / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _append_event_conn(
        self,
        conn: duckdb.DuckDBPyConnection,
        run_id: str,
        event_key: str,
        stage: str,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        created_at: str | None = None,
    ) -> int:
        """在已有事务中追加幂等事件。"""

        self._validate_id(event_key, "event_key")
        ensure_safe_event_payload(dict(payload))
        encoded = canonical_json(dict(payload))
        existing = self._row(
            conn,
            "SELECT sequence, payload_json, stage, event_type FROM mining.event "
            "WHERE run_id=? AND event_key=?",
            [run_id, event_key],
        )
        if existing is not None:
            if (
                existing["payload_json"] != encoded
                or existing["stage"] != stage
                or existing["event_type"] != event_type
            ):
                raise MiningDuplicateError(f"事件键 {event_key} 已用于不同事件")
            return int(existing["sequence"])
        sequence = int(
            conn.execute(
                "SELECT coalesce(max(sequence), 0) + 1 FROM mining.event WHERE run_id=?",
                [run_id],
            ).fetchone()[0]
        )
        conn.execute(
            "INSERT INTO mining.event VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                sequence,
                event_key,
                stage,
                event_type,
                encoded,
                created_at or utc_now_iso(),
            ],
        )
        return sequence

    @staticmethod
    def _check_context(request: MiningRunRequest, context: FrozenContext) -> None:
        """确认请求和冻结上下文没有互相矛盾。"""

        if request.constraints != context.budget:
            raise MiningContractError("请求预算与冻结上下文预算不一致")
        if request.llm_profile != context.llm_profile:
            raise MiningContractError("请求模型配置与冻结上下文不一致")

    @staticmethod
    def _assert_same_row(
        existing: Mapping[str, Any],
        expected: Mapping[str, Any],
        fields: Sequence[str],
    ) -> None:
        """幂等写入时拒绝同 ID 不同内容。"""

        changed = [name for name in fields if existing.get(name) != expected.get(name)]
        if changed:
            raise MiningDuplicateError(f"相同记录 ID 已绑定不同内容：{changed}")
