"""提供者页面参数结构的校验与规范化。"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from backtest.errors import SpecError
from backtest.strategies._param_schema import PARAM_SCHEMAS

_PARAM_TYPES = frozenset(
    {"integer", "number", "percent", "boolean", "string", "enum"}
)
_PARAM_FIELDS = frozenset(
    {
        "name",
        "label",
        "type",
        "default",
        "min",
        "max",
        "step",
        "options",
        "description",
    }
)


def _finite_number(value: Any) -> bool:
    """返回是否为有限数值（布尔值不算数值）。"""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def provider_param_schema(provider: Any) -> list[dict[str, Any]]:
    """校验并返回提供者的扁平参数字段说明。"""
    name = str(provider.name).strip()
    version = str(provider.version).strip()
    declared = getattr(provider, "param_schema", None)
    if declared is None:
        declared = PARAM_SCHEMAS.get((name, version), ())
    if not isinstance(declared, (tuple, list)):
        raise SpecError(f"提供者 {name} 的 param_schema 必须是字段序列")

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in declared:
        if not isinstance(raw, Mapping):
            raise SpecError(f"提供者 {name} 的参数字段必须是对象")
        unknown = set(raw) - _PARAM_FIELDS
        if unknown:
            raise SpecError(
                f"提供者 {name} 的参数字段包含未知属性：{sorted(unknown)}"
            )
        key = raw.get("name")
        label = raw.get("label")
        kind = raw.get("type")
        if not isinstance(key, str) or not key.strip():
            raise SpecError(f"提供者 {name} 的参数字段 name 必须是非空字符串")
        key = key.strip()
        if key in seen:
            raise SpecError(f"提供者 {name} 的参数字段重复：{key}")
        seen.add(key)
        if not isinstance(label, str) or not label.strip():
            raise SpecError(f"提供者 {name} 的参数字段 {key} 缺少有效 label")
        if kind not in _PARAM_TYPES:
            raise SpecError(f"提供者 {name} 的参数字段 {key} 类型非法：{kind!r}")
        if "default" not in raw:
            raise SpecError(f"提供者 {name} 的参数字段 {key} 缺少 default")
        default = raw["default"]
        minimum = raw.get("min")
        maximum = raw.get("max")
        step = raw.get("step")
        description = raw.get("description", "")
        if not isinstance(description, str):
            raise SpecError(f"提供者 {name} 的参数字段 {key} description 必须是字符串")

        if kind == "integer":
            if type(default) is not int:
                raise SpecError(f"提供者 {name} 的参数字段 {key} 默认值必须是整数")
            for bound_name, bound in (("min", minimum), ("max", maximum), ("step", step)):
                if bound is not None and (
                    type(bound) is not int or (bound_name == "step" and bound <= 0)
                ):
                    raise SpecError(
                        f"提供者 {name} 的参数字段 {key} {bound_name} 必须是"
                        + ("正整数" if bound_name == "step" else "整数")
                    )
        elif kind in {"number", "percent"}:
            if not _finite_number(default):
                raise SpecError(f"提供者 {name} 的参数字段 {key} 默认值必须是有限数字")
            for bound_name, bound in (("min", minimum), ("max", maximum), ("step", step)):
                if bound is not None and not _finite_number(bound):
                    raise SpecError(
                        f"提供者 {name} 的参数字段 {key} {bound_name} 必须是有限数字"
                    )
            if step is not None and float(step) <= 0:
                raise SpecError(f"提供者 {name} 的参数字段 {key} step 必须大于 0")
        elif kind == "boolean":
            if not isinstance(default, bool):
                raise SpecError(f"提供者 {name} 的参数字段 {key} 默认值必须是布尔值")
            if minimum is not None or maximum is not None or step is not None:
                raise SpecError(f"提供者 {name} 的布尔参数字段 {key} 不能设置范围")
        elif kind == "string":
            if not isinstance(default, str):
                raise SpecError(f"提供者 {name} 的参数字段 {key} 默认值必须是字符串")
            if minimum is not None or maximum is not None or step is not None:
                raise SpecError(f"提供者 {name} 的字符串参数字段 {key} 不能设置范围")
        else:
            options = raw.get("options")
            if not isinstance(options, (tuple, list)) or not options:
                raise SpecError(f"提供者 {name} 的枚举参数字段 {key} 缺少 options")
            if any(not isinstance(item, (str, int, float, bool)) for item in options):
                raise SpecError(f"提供者 {name} 的枚举参数字段 {key} options 必须是标量")
            if default not in options:
                raise SpecError(f"提供者 {name} 的枚举参数字段 {key} 默认值不在 options 中")
            if minimum is not None or maximum is not None or step is not None:
                raise SpecError(f"提供者 {name} 的枚举参数字段 {key} 不能设置范围")

        if minimum is not None and maximum is not None and minimum > maximum:
            raise SpecError(f"提供者 {name} 的参数字段 {key} min 不能大于 max")
        if _finite_number(default) and minimum is not None and default < minimum:
            raise SpecError(f"提供者 {name} 的参数字段 {key} 默认值低于 min")
        if _finite_number(default) and maximum is not None and default > maximum:
            raise SpecError(f"提供者 {name} 的参数字段 {key} 默认值高于 max")
        try:
            json.dumps(default, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise SpecError(
                f"提供者 {name} 的参数字段 {key} 默认值不能序列化：{error}"
            ) from error

        result.append(
            {
                "name": key,
                "label": label.strip(),
                "type": kind,
                "default": default,
                "min": minimum,
                "max": maximum,
                "step": step,
                "options": list(raw.get("options", [])),
                "description": description,
            }
        )

    validate = getattr(provider, "validate_params", None)
    if callable(validate):
        defaults = {item["name"]: item["default"] for item in result}
        normalized = validate(defaults)
        if not isinstance(normalized, dict):
            raise SpecError(f"提供者 {name} 的 validate_params 必须返回对象")
        missing = sorted(seen - set(normalized))
        if missing:
            raise SpecError(f"提供者 {name} 的参数规范化结果缺少字段：{missing}")
    return result
