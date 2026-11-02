"""挖掘 V2 搜索契约共用的严格解析和字段限制工具。"""

from __future__ import annotations

from datetime import date
import re
from typing import Any, Mapping

from mining.contract_types import canonical_json, identifier, text
from mining.errors import MiningContractError

SEARCH_SCHEMA_VERSION = "mining_v2"
SEARCH_CONTEXT_SCHEMA_VERSION = "mining_search_v2"
SEARCH_DEVELOPMENT_START = "2012-01-01"
SEARCH_DEVELOPMENT_END = "2019-12-31"
SEARCH_SELECTION_START = "2020-01-01"
SEARCH_SELECTION_END = "2022-12-31"
SEARCH_FINAL_START = "2023-01-01"
SEARCH_FINAL_END = "2025-10-15"
SEARCH_HOLDOUT_START = "2025-10-16"
SEARCH_HOLDOUT_END = "2026-10-15"
SEARCH_ACTION_VALUES = ("refine", "branch", "stop", "promote")
SEARCH_DIRECTION_VALUES = ("positive", "negative", "non_monotonic")
SEARCH_INPUT_MODE_VALUES = ("hypothesis", "free_exploration")
HYPOTHESIS_INPUT_MODE = SEARCH_INPUT_MODE_VALUES[0]
FREE_EXPLORATION_INPUT_MODE = SEARCH_INPUT_MODE_VALUES[1]
TRIAL_KIND_VALUES = (
    "proposed",
    "rejected",
    "duplicate",
    "invalid",
    "gate_failed",
    "evaluated",
    "full_evaluated",
    "holdout_claimed",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def strict_fields(
    value: Any,
    name: str,
    expected: set[str],
) -> dict[str, Any]:
    """校验 JSON 对象字段集合恰好等于契约字段集合。"""

    if not isinstance(value, Mapping):
        raise MiningContractError(f"{name} 必须是 JSON 对象")
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise MiningContractError(
            f"{name} 字段不匹配：缺少 {sorted(missing)}，未知 {sorted(unknown)}"
        )
    return dict(value)


def bounded_integer(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """校验整数范围，并明确拒绝布尔值。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise MiningContractError(f"{name} 必须是整数")
    if not minimum <= value <= maximum:
        raise MiningContractError(f"{name} 必须在 {minimum}～{maximum}")
    return value


def nullable_identifier(value: Any, name: str) -> str | None:
    """校验可空稳定标识。"""

    if value is None:
        return None
    return identifier(value, name)


def hash_text(value: Any, name: str) -> str:
    """校验小写十六进制 SHA-256。"""

    result = text(value, name, minimum=64, maximum=64)
    if not _SHA256_PATTERN.fullmatch(result):
        raise MiningContractError(f"{name} 必须是 64 位小写十六进制 SHA-256")
    return result


def choice(value: Any, name: str, allowed: tuple[str, ...]) -> str:
    """校验字符串枚举。"""

    if not isinstance(value, str) or value not in allowed:
        raise MiningContractError(f"{name} 必须是 {', '.join(allowed)} 之一")
    return value


def validate_date_range(
    start: Any,
    end: Any,
    name: str,
) -> tuple[str, str]:
    """校验 ISO 日期范围，开始日期不得晚于结束日期。"""

    left = text(start, f"{name}_start", minimum=10, maximum=10)
    right = text(end, f"{name}_end", minimum=10, maximum=10)
    try:
        parsed_left = date.fromisoformat(left)
        parsed_right = date.fromisoformat(right)
    except ValueError as error:
        raise MiningContractError(f"{name} 包含非法 ISO 日期") from error
    if parsed_left.isoformat() != left or parsed_right.isoformat() != right:
        raise MiningContractError(f"{name} 必须使用 YYYY-MM-DD 格式")
    if parsed_left > parsed_right:
        raise MiningContractError(f"{name}_start 不得晚于 {name}_end")
    return left, right


def mapping_with_size(
    value: Any,
    name: str,
    *,
    maximum_bytes: int,
) -> dict[str, Any]:
    """校验 JSON 对象及其规范化 UTF-8 大小。"""

    if not isinstance(value, Mapping):
        raise MiningContractError(f"{name} 必须是 JSON 对象")
    copied = dict(value)
    size = len(canonical_json(copied).encode("utf-8"))
    if size > maximum_bytes:
        raise MiningContractError(f"{name} 超过 {maximum_bytes} 字节")
    return copied


def reject_dates_outside(
    start: str,
    end: str,
    *,
    allowed_start: str,
    allowed_end: str,
    name: str,
) -> None:
    """确认日期范围完全位于允许窗口内。"""

    if start < allowed_start or end > allowed_end:
        raise MiningContractError(
            f"{name} 必须位于 {allowed_start}～{allowed_end}"
        )
