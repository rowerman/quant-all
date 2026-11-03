"""按模块分目录记录日志的共享实现。

调用方只传入日志根目录；文件名使用该日志段首条与最后一条记录的微秒时间，
单文件达到上限后切换新文件。
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Callable

MAX_LOG_BYTES = 128 * 1024 * 1024
_HANDLER_MARKER = "_quant_timestamped_log_dir"


class TimestampedFileHandler(logging.Handler):
    """按「起始时间-结束时间」命名并支持轮转的文件日志处理器。"""

    def __init__(
        self,
        directory: Path,
        *,
        max_bytes: int = MAX_LOG_BYTES,
    ) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.segment_start: str | None = None
        self._path: Path | None = None
        self._stream = None
        setattr(self, _HANDLER_MARKER, str(self.directory))

    @staticmethod
    def _stamp(created: float) -> str:
        """把日志记录时间转成微秒级文件名时间戳。"""
        return datetime.fromtimestamp(created).strftime("%Y%m%d%H%M%S%f")

    def _open_segment(self, stamp: str) -> None:
        """创建并打开一个新日志段。"""
        self.segment_start = stamp
        self._path = self.directory / f"{stamp}-{stamp}.log"
        self._stream = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path | None:
        """当前日志段路径；尚未写入时为 None。"""
        return self._path

    @property
    def started(self) -> str | None:
        """当前日志段起始时间戳。"""
        return self.segment_start

    def emit(self, record: logging.LogRecord) -> None:
        """写入一条日志，并让文件名中的结束时间跟随最后一条记录。"""
        try:
            message = self.format(record) + "\n"
            stamp = self._stamp(record.created)
            encoded = len(message.encode("utf-8"))
            if self._stream is None:
                self._open_segment(stamp)
            elif self._stream.tell() + encoded > self.max_bytes:
                self._stream.close()
                self._open_segment(stamp)
            target = self.directory / f"{self.segment_start}-{stamp}.log"
            if target != self._path:
                self._path.rename(target)
                self._path = target
            self._stream.write(message)
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """关闭底层文件。"""
        try:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        finally:
            super().close()


def get_logger(
    module_name: str,
    log_root: Path,
    *,
    handler_factory: Callable[..., logging.Handler] | None = None,
) -> logging.Logger:
    """取得按模块分目录记录的 logger。

    参数:
        module_name: 模块名，同时作为日志根目录下的子目录名。
        log_root: 该项目的日志根目录。
        handler_factory: 可选处理器工厂，供各模块覆盖单文件上限。
    """
    directory = Path(log_root) / module_name.replace(".", "_")
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    marker = str(directory)
    if any(getattr(handler, _HANDLER_MARKER, None) == marker for handler in logger.handlers):
        return logger
    factory = handler_factory or TimestampedFileHandler
    handler = factory(directory)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    return logger
