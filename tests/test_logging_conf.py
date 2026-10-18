"""回测日志文件名与轮转测试。"""

from datetime import datetime
import logging

import backtest.logging_conf as logging_conf


def _record(created: float, message: str) -> logging.LogRecord:
    """构造带指定时间的日志记录。"""
    record = logging.LogRecord("backtest.test", logging.INFO, "", 0, message, (), None)
    record.created = created
    return record


def _stamp(created: float) -> str:
    """返回与处理器一致的时间戳。"""
    return datetime.fromtimestamp(created).strftime("%Y%m%d%H%M%S%f")


def test_segment_name_tracks_last_message(tmp_path):
    """文件名里的结束时间必须随最后一条记录推进。"""
    handler = logging_conf.TimestampedFileHandler(tmp_path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    first = _stamp(1_700_000_000.123456)
    second = _stamp(1_700_000_001.654321)

    handler.handle(_record(1_700_000_000.123456, "first"))
    assert handler.path.name == f"{first}-{first}.log"
    handler.handle(_record(1_700_000_001.654321, "second"))

    assert handler.path.name == f"{first}-{second}.log"
    assert not (tmp_path / f"{first}-{first}.log").exists()
    assert handler.path.read_text(encoding="utf-8").splitlines() == ["first", "second"]
    handler.close()


def test_handler_is_lazy_and_rotates_before_limit(tmp_path, monkeypatch):
    """没有日志时不建空文件，达到上限前切换到新段。"""
    handler = logging_conf.TimestampedFileHandler(tmp_path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    assert not list(tmp_path.glob("*.log"))

    monkeypatch.setattr(logging_conf, "MAX_BYTES", 8)
    times = [1_700_000_000.1, 1_700_000_001.2, 1_700_000_002.3]
    handler.handle(_record(times[0], "a"))
    handler.handle(_record(times[1], "bb"))
    handler.handle(_record(times[2], "ccc"))

    old_path = tmp_path / f"{_stamp(times[0])}-{_stamp(times[1])}.log"
    new_path = tmp_path / f"{_stamp(times[2])}-{_stamp(times[2])}.log"
    assert old_path.read_text(encoding="utf-8").splitlines() == ["a", "bb"]
    assert new_path.read_text(encoding="utf-8").splitlines() == ["ccc"]
    handler.close()
