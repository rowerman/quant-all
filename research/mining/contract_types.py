"""挖掘契约共用的常量、校验和稳定哈希工具。"""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from mining.errors import MiningContractError

SCHEMA_VERSION = "mining_v1"
FULL_FACTOR_START = "2000-01-04"
RESEARCH_START = "2012-01-01"
RESEARCH_END = "2025-10-15"
FULL_EVAL_START = "2020-01-01"
FULL_EVAL_END = RESEARCH_END
HOLDOUT_START = "2025-10-16"
HOLDOUT_END = "2026-10-15"
DATA_VERSION = "tushare_fullA_20000104_20261015"
LABEL_NAME = "fwd_ret_open_5"
UNIVERSE = "all_a"
PRICE_CHANNEL = "hfq"
HOLDOUT_SEGMENT_ID = f"{HOLDOUT_START}_{HOLDOUT_END}"

IDEMPOTENCY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
PROFILE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")


def utc_now_iso() -> str:
    """返回微秒精度的 UTC ISO 时间。"""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def canonical_json(value: Any) -> str:
    """生成与字典键顺序无关的稳定 JSON。"""

    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    """计算规范化 JSON 的 SHA-256。"""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    """把领域对象递归转成可稳定序列化的 JSON 值。"""

    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise MiningContractError("时间必须携带时区")
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MiningContractError("JSON 中不能包含 NaN 或无穷值")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise MiningContractError(f"不支持序列化类型：{type(value).__name__}")


def text(value: Any, name: str, *, minimum: int = 1, maximum: int = 4000) -> str:
    """校验并返回去首尾空白的字符串。"""

    if not isinstance(value, str):
        raise MiningContractError(f"{name} 必须是字符串")
    result = value.strip()
    if not minimum <= len(result) <= maximum:
        raise MiningContractError(f"{name} 长度必须在 {minimum}～{maximum} 个字符")
    return result


def identifier(value: Any, name: str) -> str:
    """校验稳定标识的字符和长度。"""

    result = text(value, name, maximum=160)
    if not IDENTIFIER_PATTERN.fullmatch(result):
        raise MiningContractError(f"{name} 只能包含字母、数字、点、下划线、冒号和连字符")
    return result


def sha256_text(value: Any, name: str) -> str:
    """校验小写十六进制 SHA-256。"""

    result = text(value, name, minimum=64, maximum=64)
    if not HASH_PATTERN.fullmatch(result):
        raise MiningContractError(f"{name} 必须是 64 位小写十六进制 SHA-256")
    return result


def validate_date(value: Any, name: str) -> str:
    """校验 ISO 日期并返回原值。"""

    result = text(value, name, minimum=10, maximum=10)
    try:
        parsed = date.fromisoformat(result)
    except ValueError as error:
        raise MiningContractError(f"{name} 不是合法 ISO 日期：{result}") from error
    if parsed.isoformat() != result:
        raise MiningContractError(f"{name} 必须使用 YYYY-MM-DD 格式")
    return result


def validate_utc_timestamp(value: Any, name: str) -> str:
    """校验带时区时间并规范为 UTC ISO 字符串。"""

    result = text(value, name, maximum=64)
    try:
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as error:
        raise MiningContractError(f"{name} 不是合法 ISO 时间：{result}") from error
    if parsed.tzinfo is None:
        raise MiningContractError(f"{name} 必须携带时区")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def stable_id(prefix: str, value: Any) -> str:
    """由内容哈希构造带前缀的稳定标识。"""

    return f"{prefix}_{sha256_json(value)[:24]}"


def ensure_string_tuple(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = 500,
) -> tuple[str, ...]:
    """把字符串序列规范为去空白、去重、保序的元组。"""

    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise MiningContractError(f"{name} 必须是字符串序列")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        current = text(item, name, maximum=500)
        if current not in seen:
            result.append(current)
            seen.add(current)
    if not minimum <= len(result) <= maximum:
        raise MiningContractError(f"{name} 数量必须在 {minimum}～{maximum}")
    return tuple(result)


def mapping(value: Any, name: str) -> dict[str, Any]:
    """校验 JSON 对象并返回独立副本。"""

    if not isinstance(value, Mapping):
        raise MiningContractError(f"{name} 必须是 JSON 对象")
    return json.loads(canonical_json(dict(value)))


def mapping_sequence(
    value: Any,
    name: str,
) -> tuple[dict[str, Any], ...]:
    """把对象序列规范为按内容排序且无重复的元组。"""

    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise MiningContractError(f"{name} 必须是 JSON 对象序列")
    unique = {
        canonical_json(mapping(item, f"{name} item")): mapping(
            item, f"{name} item"
        )
        for item in value
    }
    return tuple(unique[key] for key in sorted(unique))
