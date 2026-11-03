"""沙箱研究事件日志，只记录脱敏后的白名单字段。"""

from __future__ import annotations

import logging
from typing import Any

from common.logging_conf import get_logger
from common.paths import log_root_for
from mining_sandbox.feedback import SandboxCodeFeedback, sanitize_feedback_text


def get_sandbox_logger(module_name: str) -> logging.Logger:
    """返回写入沙箱业务域目录的模块 logger。"""

    if module_name != "mining_sandbox" and not module_name.startswith(
        "mining_sandbox."
    ):
        raise ValueError("沙箱 logger 名必须以 mining_sandbox 开头")
    return get_logger(module_name, log_root_for(module_name))


def sandbox_feedback_log_fields(
    *,
    run_id: str,
    iteration: int,
    code_hash: str,
    feedback: SandboxCodeFeedback,
) -> dict[str, Any]:
    """构造不包含原始代码、异常文本或路径的事件字段。"""

    return {
        "run_id": sanitize_feedback_text(run_id, maximum=160),
        "iteration": int(iteration),
        "code_hash": sanitize_feedback_text(code_hash, maximum=64),
        "error_code": feedback.error_code,
        "category": feedback.category.value,
        "stage": feedback.stage.value,
        "retryable": feedback.retryable,
        "feedback_hash": feedback.feedback_hash,
    }


def log_sandbox_feedback(
    logger: logging.Logger,
    *,
    run_id: str,
    iteration: int,
    code_hash: str,
    feedback: SandboxCodeFeedback,
) -> None:
    """记录一次失败尝试的安全元数据。"""

    safe = sandbox_feedback_log_fields(
        run_id=run_id,
        iteration=iteration,
        code_hash=code_hash,
        feedback=feedback,
    )
    body = " ".join(f"{key}={safe[key]!r}" for key in sorted(safe))
    logger.info(body)


__all__ = [
    "get_sandbox_logger",
    "log_sandbox_feedback",
    "sandbox_feedback_log_fields",
]
