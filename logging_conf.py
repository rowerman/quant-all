"""回测模块的日志配置。

按模块分目录写到 ``backtest/logs/<模块名>/``，单文件超过 128MB 自动轮转，
文件名用「起始时间-结束时间.log」的格式，便于按时间定位。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from backtest.config import LOG_DIR

MAX_BYTES = 128 * 1024 * 1024
_SEGMENT_START = time.strftime("%Y%m%d%H%M%S")
_LOGGERS: dict[str, logging.Logger] = {}


class TimestampedFileHandler(logging.Handler):
    """按「起始时间-结束时间」命名并支持轮转的文件日志处理器。"""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.segment_start = _SEGMENT_START
        self._path = self._build_path()
        self._stream = self._path.open("a", encoding="utf-8")

    def _build_path(self) -> Path:
        """生成当前日志段文件名。"""
        stamp = time.strftime("%Y%m%d%H%M%S")
        return self.directory / f"{self.segment_start}-{stamp}.log"

    def emit(self, record: logging.LogRecord) -> None:
        """写入一条日志，超出单文件上限时切到新段。"""
        try:
            message = self.format(record)
            if self._stream.tell() >= MAX_BYTES:
                self._rotate()
            self._stream.write(message + "\n")
            self._stream.flush()
        except Exception:  # 日志写入失败不能反过来炸掉回测主流程
            self.handleError(record)

    def _rotate(self) -> None:
        """关闭当前文件并开启新的日志段。"""
        self._stream.close()
        self.segment_start = time.strftime("%Y%m%d%H%M%S")
        self._path = self._build_path()
        self._stream = self._path.open("a", encoding="utf-8")

    def close(self) -> None:
        """关闭底层文件。"""
        try:
            self._stream.close()
        finally:
            super().close()


def get_logger(module_name: str) -> logging.Logger:
    """取得按模块分目录记录的 logger。

    参数:
        module_name: 模块名，同时作为 ``backtest/logs/`` 下的子目录名，
            例如 ``backtest.engine``。

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

