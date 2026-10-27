"""运行事件读取和中断临时目录清理。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from mining.errors import MiningContractError, MiningNotFoundError, MiningStorageError

TRANSIENT_AGE = timedelta(hours=24)


class EventStoreMixin:
    """追加事件、读取事件和清理临时产物。"""

    def append_event(
        self,
        run_id: str,
        *,
        event_key: str,
        stage: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> int:
        """公开追加幂等事件。"""

        run_id = self._validate_id(run_id, "run_id")
        with self._state_lock():
            with self._connection() as conn:
                with self._transaction(conn):
                    exists = self._row(
                        conn, "SELECT run_id FROM mining.run WHERE run_id=?", [run_id]
                    )
                    if exists is None:
                        raise MiningNotFoundError(f"挖掘运行不存在：{run_id}")
                    sequence = self._append_event_conn(
                        conn,
                        run_id,
                        event_key,
                        stage,
                        event_type,
                        payload,
                    )
            return sequence

    def list_events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        """按序号读取不可变事件。"""

        run_id = self._validate_id(run_id, "run_id")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT sequence, event_key, stage, event_type, payload_json, created_at "
                "FROM mining.event WHERE run_id=? AND sequence>? ORDER BY sequence",
                [run_id, max(0, int(after))],
            ).fetchall()
        return [
            {
                "sequence": int(sequence),
                "event_key": event_key,
                "stage": stage,
                "event_type": event_type,
                "payload": json.loads(payload),
                "created_at": created_at,
            }
            for sequence, event_key, stage, event_type, payload, created_at in rows
        ]

    def cleanup_transient_temp(
        self,
        *,
        now: datetime | None = None,
        max_age: timedelta = TRANSIENT_AGE,
    ) -> list[Path]:
        """清理中断运行下超过期限的临时文件和 staging 目录。"""

        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise MiningContractError("清理时间必须带时区")
        cutoff = current.timestamp() - max_age.total_seconds()
        removed: list[Path] = []
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT run_id FROM mining.run WHERE status='interrupted'"
            ).fetchall()
        root = self.data_root.resolve()
        for (run_id,) in rows:
            run_root = (self.data_root / "runs" / run_id).resolve()
            if root not in run_root.parents or not run_root.is_dir():
                continue
            targets = sorted(
                (
                    path
                    for path in run_root.rglob("*")
                    if (
                        path.name.endswith(".tmp")
                        or path.name == "staging"
                        or ".staging-" in path.name
                    )
                ),
                key=lambda path: len(path.parts),
                reverse=True,
            )
            for path in targets:
                resolved = path.resolve()
                if root not in resolved.parents or resolved.is_symlink():
                    continue
                try:
                    if resolved.stat().st_mtime > cutoff:
                        continue
                    if resolved.is_dir():
                        shutil.rmtree(resolved)
                    else:
                        resolved.unlink()
                    removed.append(resolved)
                except FileNotFoundError:
                    continue
                except OSError as error:
                    raise MiningStorageError(
                        f"清理临时产物失败：{resolved}"
                    ) from error
        return removed
