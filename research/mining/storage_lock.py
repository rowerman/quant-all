"""候选数据库跨连接文件锁。"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Iterator

from mining.errors import MiningStorageError


@contextmanager
def database_state_lock(db_path: Path | str) -> Iterator[None]:
    """独占串行化同一候选数据库的写入和迁移。"""

    target = Path(db_path)
    lock_path = target.with_name(f"{target.name}.state.lock")
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
