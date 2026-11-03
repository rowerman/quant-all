"""Worker 服务的事件日志，只记录白名单标量。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from common.logging_conf import get_logger


def get_worker_logger(module_name: str) -> logging.Logger:
    """返回写入项目挖掘日志目录的 Worker logger。"""

    root = Path(os.environ.get("MINING_WORKER_LOG_DIR", "logs/research/mining"))
    return get_logger(module_name, root)


def log_worker_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """记录脱敏后的单行作业事件。"""

    safe: dict[str, str | int | float | bool] = {"event": event}
    for key, value in fields.items():
        if key in {"hypothesis", "expression", "formula", "response", "prompt"}:
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            safe[key] = value if not isinstance(value, str) else value[:200]
    body = " ".join(f"{key}={safe[key]!r}" for key in sorted(safe))
    logger.info(body)
