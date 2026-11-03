"""挖掘模块日志入口与敏感字段约束。"""

from __future__ import annotations

import logging
import re
from typing import Any

from common.logging_conf import get_logger
from common.paths import log_root_for
from mining.errors import MiningContractError

_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|authorization|credential|password|secret|token)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE = re.compile(
    r"(?i)(sk-[A-Za-z0-9_-]{8,}|bearer\s+[A-Za-z0-9._~+/=-]{8,})"
)
_RAW_TEXT_KEYS = {"hypothesis", "expression", "formula", "raw_request"}


def get_mining_logger(module_name: str) -> logging.Logger:
    """返回写入挖掘业务域目录的模块 logger。"""

    if module_name != "mining" and not module_name.startswith("mining."):
        raise ValueError("挖掘 logger 名必须以 mining 开头")
    return get_logger(module_name, log_root_for(module_name))


def safe_event_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """只保留可写日志的标量字段并脱敏。

    参数:
        payload: 待写入状态日志的字段。
    """

    result: dict[str, Any] = {}
    for key, value in payload.items():
        lowered = str(key).lower()
        if _SENSITIVE_KEY.search(lowered) or lowered in _RAW_TEXT_KEYS:
            continue
        if value is None or isinstance(value, (bool, int, float)):
            result[str(key)] = value
        elif isinstance(value, str):
            result[str(key)] = _SENSITIVE_VALUE.sub("[REDACTED]", value[:200])
    return result


def format_event(fields: dict[str, Any]) -> str:
    """把白名单字段格式化为稳定的单行日志正文。"""

    safe = safe_event_fields(fields)
    return " ".join(f"{key}={safe[key]!r}" for key in sorted(safe))


def ensure_safe_event_payload(value: Any, *, path: str = "payload") -> None:
    """拒绝可能进入数据库事件表的敏感键和超长文本。"""

    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if _SENSITIVE_KEY.search(lowered) or lowered in _RAW_TEXT_KEYS:
                raise MiningContractError(f"{path} 不允许包含字段 {key!r}")
            ensure_safe_event_payload(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ensure_safe_event_payload(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and len(value) > 10000:
        raise MiningContractError(f"{path} 文本超过 10000 个字符")
