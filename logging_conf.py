"""计算包的日志配置。

共享实现位于 :mod:`common.logging_conf`；本模块只绑定 ``calculate/logs/``。
"""

from __future__ import annotations

from pathlib import Path
import logging

from common.logging_conf import MAX_LOG_BYTES
from common.logging_conf import TimestampedFileHandler as _SharedTimestampedFileHandler
from common.logging_conf import get_logger as _get_logger
from calculate.config import LOG_DIR

MAX_BYTES = MAX_LOG_BYTES


class TimestampedFileHandler(_SharedTimestampedFileHandler):
    """使用计算包当前单文件上限的共享处理器。"""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory, max_bytes=MAX_BYTES)


def get_logger(module_name: str) -> logging.Logger:
    """取得写入 ``calculate/logs/<模块名>/`` 的 logger。"""
    return _get_logger(
        module_name,
        LOG_DIR,
        handler_factory=TimestampedFileHandler,
    )
