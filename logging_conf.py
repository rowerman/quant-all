"""计算包的日志配置。

按模块分目录写到 ``calculate/logs/<模块名>/``，单文件超过 128MB 自动轮转，
文件名用「起始时间-结束时间.log」的格式，便于按时间定位；
与数据项目的 ``logs/`` 分开，避免两个项目的运行记录混在一起。
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

from calculate.config import LOG_DIR

MAX_BYTES = 128 * 1024 * 1024
_LOGGERS: dict[str, logging.Logger] = {}


class TimestampedFileHandler(logging.Handler):
    """按「起始时间-结束时间」命名并支持轮转的文件日志处理器。"""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.segment_start: str | None = None
        self._path: Path | None = None
        self._stream = None

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
        """写入一条日志，超出单文件上限时切到新段。"""
        try:
            message = self.format(record) + "\n"
            stamp = self._stamp(record.created)
            encoded = len(message.encode("utf-8"))
            if self._stream is None:
                self._open_segment(stamp)
            elif self._stream.tell() + encoded > MAX_BYTES:
                self._stream.close()
                self._open_segment(stamp)
            target = self.directory / f"{self.segment_start}-{stamp}.log"
            if target != self._path:
                self._path.rename(target)
                self._path = target
            self._stream.write(message)
            self._stream.flush()
        except Exception:  # 日志写入失败不能反过来炸掉计算主流程
            self.handleError(record)

    def close(self) -> None:
        """关闭底层文件。"""
        try:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        finally:
            super().close()


def get_logger(module_name: str) -> logging.Logger:
    """取得按模块分目录记录的 logger。

    参数:
        module_name: 模块名，同时作为 ``calculate/logs/`` 下的子目录名，
            例如 ``calculate.panel``。

    返回:
        配置好的 logger，重复调用返回同一实例。
    """
    if module_name in _LOGGERS:
        return _LOGGERS[module_name]
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    directory = LOG_DIR / module_name.replace(".", "_")
    handler = TimestampedFileHandler(directory)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    _LOGGERS[module_name] = logger
    return logger
