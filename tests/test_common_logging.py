"""共享日志处理器的命名、惰性创建与轮转。"""

from datetime import datetime
import logging

from common.logging_conf import TimestampedFileHandler


def _record(created: float, message: str) -> logging.LogRecord:
    """构造带指定时间的日志记录。"""
    record = logging.LogRecord("common.test", logging.INFO, "", 0, message, (), None)
    record.created = created
    return record


def _stamp(created: float) -> str:
    """返回与处理器一致的时间戳。"""
    return datetime.fromtimestamp(created).strftime("%Y%m%d%H%M%S%f")


def test_shared_handler_tracks_last_message_and_rotates(tmp_path):
    """文件名结束时间随最后一条记录推进，超限前切换新段。"""
    handler = TimestampedFileHandler(tmp_path, max_bytes=8)
    handler.setFormatter(logging.Formatter("%(message)s"))
    assert not list(tmp_path.glob("*.log"))
    times = [1_700_000_000.1, 1_700_000_001.2, 1_700_000_002.3]
    for created, message in zip(times, ("a", "bb", "ccc")):
        handler.handle(_record(created, message))

    old_path = tmp_path / f"{_stamp(times[0])}-{_stamp(times[1])}.log"
    new_path = tmp_path / f"{_stamp(times[2])}-{_stamp(times[2])}.log"
    assert old_path.read_text(encoding="utf-8").splitlines() == ["a", "bb"]
    assert new_path.read_text(encoding="utf-8").splitlines() == ["ccc"]
    handler.close()
