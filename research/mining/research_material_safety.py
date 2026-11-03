"""资料路径、密钥和业务日志的安全规则。"""

from __future__ import annotations

from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from mining.contract_types import text
from mining.errors import MiningContractError, ResearchMaterialSecurityError

_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|authorization|credential|password|secret|token)"
    r"\b\s*[:=]\s*['\"]?([^\s,;'\"]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_SK_TOKEN_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")
_POSIX_HOST_PATH_PATTERN = re.compile(
    r"(?<![\w.])/(?:home|Users|root|tmp|var|opt|mnt|srv|workspace)/[^\s\"'<>]+"
)
_WINDOWS_HOST_PATH_PATTERN = re.compile(
    r"\b[A-Za-z]:\\(?:[^\\\s\"'<>]+\\)*[^\\\s\"'<>]+"
)
_LOG_FIELDS = {
    "document_id",
    "manifest_id",
    "manifest_hash",
    "record_id",
    "record_hash",
    "source_reference_id",
    "error_code",
    "category",
    "status",
    "page_count",
    "size_bytes",
    "retryable",
    "classification",
    "elapsed_ms",
    "financial_research_count",
    "filtered_count",
    "failed_count",
    "document_count",
    "failure_type",
    "factor_count",
    "missing_field_count",
    "extraction_id",
    "window_index",
    "window_count",
    "prompt_version",
    "model",
    "operation",
    "parser_version",
}


def redact_research_material_text(value: str) -> str:
    """遮蔽资料文本中的密钥和主机绝对路径。"""

    if not isinstance(value, str):
        raise MiningContractError("资料文本必须是字符串")
    result = _SECRET_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}=[REDACTED]",
        value,
    )
    result = _BEARER_PATTERN.sub("Bearer [REDACTED]", result)
    result = _SK_TOKEN_PATTERN.sub("[REDACTED]", result)
    result = _POSIX_HOST_PATH_PATTERN.sub("[PATH]", result)
    result = _WINDOWS_HOST_PATH_PATTERN.sub("[PATH]", result)
    return result


def safe_research_material_log_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """只保留可写入资料业务日志的标识、状态和统计字段。"""

    if not isinstance(payload, Mapping):
        raise MiningContractError("资料日志字段必须是 JSON 对象")
    result: dict[str, Any] = {}
    for key in sorted(_LOG_FIELDS & set(payload)):
        value = payload[key]
        if value is None or isinstance(value, (bool, int, float)):
            result[key] = value
        elif isinstance(value, str):
            result[key] = redact_research_material_text(value[:200])
    return result


def validate_relative_material_path(value: Any) -> str:
    """校验不暴露主机根路径的 POSIX 相对路径。"""

    try:
        result = text(value, "relative_path", maximum=1000)
    except MiningContractError as error:
        raise ResearchMaterialSecurityError(str(error)) from error
    if (
        "\\" in result
        or result.startswith(("/", "~"))
        or _WINDOWS_DRIVE_PATTERN.match(result)
    ):
        raise ResearchMaterialSecurityError("relative_path 必须是安全的 POSIX 相对路径")
    parts = result.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ResearchMaterialSecurityError("relative_path 不能包含空段、. 或 ..")
    normalized = PurePosixPath(*parts).as_posix()
    if normalized != result:
        raise ResearchMaterialSecurityError("relative_path 不是规范化 POSIX 路径")
    return result


__all__ = [
    "redact_research_material_text",
    "safe_research_material_log_fields",
    "validate_relative_material_path",
]
