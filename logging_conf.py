"""Alpha 模型层的按模块日志配置。"""

from __future__ import annotations

from pathlib import Path
import logging

from alpha.config import LOG_DIR
from common.logging_conf import MAX_LOG_BYTES
from common.logging_conf import TimestampedFileHandler as _SharedTimestampedFileHandler
from common.logging_conf import get_logger as _get_logger

MAX_BYTES = MAX_LOG_BYTES


class TimestampedFileHandler(_SharedTimestampedFileHandler):
    """使用共享单文件上限的日志处理器。"""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory, max_bytes=MAX_BYTES)


def get_logger(module_name: str) -> logging.Logger:
    """取得写入 ``logs/alpha/<模块名>/`` 的 logger。"""

    return _get_logger(
        module_name,
        LOG_DIR,
        handler_factory=TimestampedFileHandler,
    )
