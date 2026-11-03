"""自动因子挖掘的字段与算子目录。

目录文件是版本化的规则来源。运行时不从数据库猜测可用字段，只用只读连接核对
目录和实际 schema 是否一致。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

import duckdb

from common.paths import DB_FILE, MINING_CATALOG_FILE
from factors.sources import SOURCES
from mining.contract_types import sha256_json
from mining.errors import CatalogDriftError

_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_OPERATOR_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_ALLOWED_KINDS = {
    "arithmetic",
    "scalar",
    "time_series",
    "cross_section",
    "parameterized",
}


@dataclass(frozen=True)
class FieldSpec:
    """一个可审查的因子输入字段。"""

    name: str
    label: str
    dtype: str
    dimension: str
    unit: str
    source: str
    table: str
    join_kind: str
    available_at: str
    allowed: bool
    nullable: bool
    reason: str = ""

    def as_mapping(self) -> dict[str, Any]:
        """转为可写入冻结上下文和目录哈希的稳定结构。"""

        return {
            "name": self.name,
            "label": self.label,
            "dtype": self.dtype,
            "dimension": self.dimension,
            "unit": self.unit,
            "source": self.source,
            "table": self.table,
            "join_kind": self.join_kind,
            "available_at": self.available_at,
            "allowed": self.allowed,
            "nullable": self.nullable,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OperatorSpec:
    """一个允许进入受控 DSL 的算子。"""

    name: str
    kind: str
    min_args: int
    max_args: int
    description: str
    window_arg: int | None = None
    min_window: int | None = None
    max_window: int | None = None
    parameter_arg: int | None = None
    min_parameter: float | None = None
    max_parameter: float | None = None

    def as_mapping(self) -> dict[str, Any]:
        """转为可写入冻结上下文和目录哈希的稳定结构。"""

        return {
            "name": self.name,
            "kind": self.kind,
            "min_args": self.min_args,
            "max_args": self.max_args,
            "description": self.description,
            "window_arg": self.window_arg,
            "min_window": self.min_window,
            "max_window": self.max_window,
            "parameter_arg": self.parameter_arg,
            "min_parameter": self.min_parameter,
            "max_parameter": self.max_parameter,
        }


@dataclass(frozen=True)
class CatalogValidation:
    """目录与数据库 schema 的核对结果。"""

    catalog_hash: str
    table_count: int
    field_count: int
    operator_count: int


@dataclass(frozen=True)
class MiningCatalog:
    """冻结的字段、算子目录。"""

    schema_version: str
    data_version: str
    fields: tuple[FieldSpec, ...]
    operators: tuple[OperatorSpec, ...]

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise CatalogDriftError("字段目录存在重名")
        operator_names = [operator.name for operator in self.operators]
        if len(operator_names) != len(set(operator_names)):
            raise CatalogDriftError("算子目录存在重名")

    @property
    def catalog_hash(self) -> str:
        """返回目录内容的稳定哈希。"""

        return sha256_json(
            {
                "schema_version": self.schema_version,
                "data_version": self.data_version,
                "fields": [field.as_mapping() for field in self.fields],
                "operators": [operator.as_mapping() for operator in self.operators],
            }
        )

    @property
    def allowed_fields(self) -> tuple[FieldSpec, ...]:
        """返回允许 Agent 使用的字段。"""

        return tuple(field for field in self.fields if field.allowed)

    def field(self, name: str, *, require_allowed: bool = True) -> FieldSpec:
        """按名字取字段，默认拒绝不可用字段。"""

        for field in self.fields:
            if field.name == name:
                if require_allowed and not field.allowed:
                    raise CatalogDriftError(f"字段 {name!r} 未获准进入因子表达式")
                return field
        raise CatalogDriftError(f"字段 {name!r} 不在冻结目录中")

    def operator(self, name: str) -> OperatorSpec:
        """按名字取算子。"""

        for operator in self.operators:
            if operator.name == name:
                return operator
        raise CatalogDriftError(f"算子 {name!r} 不在冻结目录中")

    def frozen_field_mappings(self) -> tuple[Mapping[str, Any], ...]:
        """返回允许进入冻结上下文的字段映射。"""

        return tuple(field.as_mapping() for field in self.allowed_fields)

    def frozen_operator_mappings(self) -> tuple[Mapping[str, Any], ...]:
        """返回进入冻结上下文的算子映射。"""

        return tuple(operator.as_mapping() for operator in self.operators)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogDriftError(f"字段目录文件不存在：{path}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogDriftError(f"字段目录 JSON 无法解析：{exc}") from exc
    if not isinstance(payload, dict):
        raise CatalogDriftError("字段目录根节点必须是对象")
    return payload


def _parse_fields(payload: Mapping[str, Any]) -> tuple[FieldSpec, ...]:
    groups = payload.get("field_groups")
    if not isinstance(groups, list) or not groups:
        raise CatalogDriftError("字段目录缺少 field_groups")

    fields: list[FieldSpec] = []
    seen_sources: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise CatalogDriftError("field_groups 的元素必须是对象")
        source = str(group.get("source", "")).strip()
        table = str(group.get("table", "")).strip()
        join_kind = str(group.get("join_kind", "")).strip()
        available_at = str(group.get("available_at", "")).strip()
        if not source or source in seen_sources:
            raise CatalogDriftError(f"字段来源缺失或重复：{source!r}")
        if not table or "." not in table:
            raise CatalogDriftError(f"字段来源 {source} 的表名非法")
        if join_kind not in {"base", "date", "asof"}:
            raise CatalogDriftError(f"字段来源 {source} 的连接方式非法")
        if available_at not in {"close", "next_trading_day"}:
            raise CatalogDriftError(f"字段来源 {source} 的可用时点非法")
        seen_sources.add(source)

        raw_fields = group.get("fields")
        if not isinstance(raw_fields, dict) or not raw_fields:
            raise CatalogDriftError(f"字段来源 {source} 没有字段")
        for name, raw in raw_fields.items():
            if not isinstance(name, str) or not _FIELD_NAME.fullmatch(name):
                raise CatalogDriftError(f"字段名不合法：{name!r}")
            if not isinstance(raw, dict):
                raise CatalogDriftError(f"字段 {name} 的定义必须是对象")
            allowed = raw.get("allowed", True)
            if not isinstance(allowed, bool):
                raise CatalogDriftError(f"字段 {name} 的 allowed 必须是布尔值")
            dtype = str(raw.get("dtype", "")).strip()
            dimension = str(raw.get("dimension", "")).strip()
            unit = str(raw.get("unit", "")).strip()
            label = str(raw.get("label", "")).strip()
            reason = str(raw.get("reason", "")).strip()
            if dtype not in {"number", "boolean", "string", "date"}:
                raise CatalogDriftError(f"字段 {name} 的类型非法")
            if allowed and dtype not in {"number", "boolean"}:
                raise CatalogDriftError(f"字段 {name} 获准使用但不是数值或布尔类型")
            if not all((dimension, unit, label)):
                raise CatalogDriftError(f"字段 {name} 缺少维度、单位或中文名")
            if not allowed and not reason:
                raise CatalogDriftError(f"禁用字段 {name} 必须说明原因")
            nullable = raw.get("nullable", True)
            if not isinstance(nullable, bool):
                raise CatalogDriftError(f"字段 {name} 的 nullable 必须是布尔值")
            fields.append(
                FieldSpec(
                    name=name,
                    label=label,
                    dtype=dtype,
                    dimension=dimension,
                    unit=unit,
                    source=source,
                    table=table,
                    join_kind=join_kind,
                    available_at=available_at,
                    allowed=allowed,
                    nullable=nullable,
                    reason=reason,
                )
            )
    return tuple(sorted(fields, key=lambda item: item.name))


def _parse_operators(payload: Mapping[str, Any]) -> tuple[OperatorSpec, ...]:
    raw_operators = payload.get("operators")
    if not isinstance(raw_operators, dict) or not raw_operators:
        raise CatalogDriftError("字段目录缺少 operators")

    operators: list[OperatorSpec] = []
    for name, raw in raw_operators.items():
        if not isinstance(name, str) or not _OPERATOR_NAME.fullmatch(name):
            raise CatalogDriftError(f"算子名不合法：{name!r}")
        if not isinstance(raw, dict):
            raise CatalogDriftError(f"算子 {name} 的定义必须是对象")
        kind = str(raw.get("kind", "")).strip()
        if kind not in _ALLOWED_KINDS:
            raise CatalogDriftError(f"算子 {name} 的类型非法")
        try:
            min_args = int(raw["min_args"])
            max_args = int(raw["max_args"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CatalogDriftError(f"算子 {name} 的参数个数非法") from exc
        if not 1 <= min_args <= max_args <= 8:
            raise CatalogDriftError(f"算子 {name} 的参数个数范围非法")
        description = str(raw.get("description", "")).strip()
        if not description:
            raise CatalogDriftError(f"算子 {name} 缺少说明")
        window_arg = raw.get("window_arg")
        min_window = raw.get("min_window")
        max_window = raw.get("max_window")
        if window_arg is not None:
            window_arg = int(window_arg)
            min_window = int(min_window)
            max_window = int(max_window)
            if not 0 <= window_arg < max_args:
                raise CatalogDriftError(f"算子 {name} 的窗口参数位置非法")
            if not 1 <= min_window <= max_window <= 252:
                raise CatalogDriftError(f"算子 {name} 的窗口范围非法")
        parameter_arg = raw.get("parameter_arg")
        min_parameter = raw.get("min_parameter")
        max_parameter = raw.get("max_parameter")
        if parameter_arg is not None:
            parameter_arg = int(parameter_arg)
            min_parameter = float(min_parameter)
            max_parameter = float(max_parameter)
            if not 0 <= parameter_arg < max_args:
                raise CatalogDriftError(f"算子 {name} 的参数位置非法")
            if min_parameter > max_parameter:
                raise CatalogDriftError(f"算子 {name} 的参数范围非法")
        operators.append(
            OperatorSpec(
                name=name,
                kind=kind,
                min_args=min_args,
                max_args=max_args,
                description=description,
                window_arg=window_arg,
                min_window=min_window,
                max_window=max_window,
                parameter_arg=parameter_arg,
                min_parameter=min_parameter,
                max_parameter=max_parameter,
            )
        )
    return tuple(sorted(operators, key=lambda item: item.name))


def load_catalog(path: Path | str = MINING_CATALOG_FILE) -> MiningCatalog:
    """读取并构造冻结目录。"""

    payload = _read_manifest(Path(path))
    schema_version = str(payload.get("schema_version", "")).strip()
    data_version = str(payload.get("data_version", "")).strip()
    if schema_version != "mining_catalog_v1":
        raise CatalogDriftError(f"不支持的字段目录版本：{schema_version!r}")
    if not data_version:
        raise CatalogDriftError("字段目录缺少数据版本")
    return MiningCatalog(
        schema_version=schema_version,
        data_version=data_version,
        fields=_parse_fields(payload),
        operators=_parse_operators(payload),
    )


def _expected_type(dtype: str) -> tuple[str, ...]:
    if dtype == "number":
        return ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "FLOAT", "DOUBLE", "DECIMAL")
    if dtype == "boolean":
        return ("BOOLEAN",)
    if dtype == "date":
        return ("DATE",)
    return ("VARCHAR",)


def validate_catalog(
    catalog: MiningCatalog | None = None,
    *,
    db_path: Path | str = DB_FILE,
) -> CatalogValidation:
    """只读核对目录字段、来源与行情库 schema。"""

    catalog = catalog or load_catalog()
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {
            (str(schema), str(table))
            for schema, table in connection.execute(
                "SELECT table_schema, table_name "
                "FROM information_schema.tables "
                "WHERE table_schema NOT IN ('information_schema', 'pg_catalog')"
            ).fetchall()
        }
        columns: dict[tuple[str, str], dict[str, str]] = {}
        for schema, table in tables:
            rows = connection.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = ? AND table_name = ?",
                [schema, table],
            ).fetchall()
            columns[(schema, table)] = {
                str(column): str(data_type).upper() for column, data_type in rows
            }
    finally:
        connection.close()

    catalog_sources = {field.source for field in catalog.fields}
    if catalog_sources - {"daily_hfq"} != set(SOURCES):
        raise CatalogDriftError("字段目录的附加来源与 factors.sources.SOURCES 不一致")
    for qualified_table in sorted({field.table for field in catalog.fields}):
        schema, table = qualified_table.split(".", 1)
        if (schema, table) not in tables:
            raise CatalogDriftError(f"行情库缺少字段目录声明的表 {qualified_table}")

    for source in catalog_sources:
        source_fields = {field.name for field in catalog.fields if field.source == source}
        if source == "daily_hfq":
            table_name = ("md", "daily_hfq")
            if table_name not in tables:
                raise CatalogDriftError("行情库缺少 md.daily_hfq")
            actual = set(columns[table_name])
            if actual != source_fields:
                missing = sorted(source_fields - actual)
                extra = sorted(actual - source_fields)
                raise CatalogDriftError(
                    f"md.daily_hfq 与字段目录不一致；缺少 {missing}，新增 {extra}"
                )
            for field in catalog.fields:
                if field.source != source:
                    continue
                expected = _expected_type(field.dtype)
                actual_type = columns[table_name][field.name]
                if not any(actual_type.startswith(item) for item in expected):
                    raise CatalogDriftError(
                        f"字段 {field.name} 类型漂移：目录 {field.dtype}，"
                        f"库内 {actual_type}"
                    )
            continue

        spec = SOURCES.get(source)
        if spec is None:
            raise CatalogDriftError(f"未知附加来源：{source}")
        if source_fields != set(spec.columns):
            raise CatalogDriftError(f"附加来源 {source} 的列与 factors.sources 不一致")
        group = next(field for field in catalog.fields if field.source == source)
        if group.join_kind != spec.join_kind:
            raise CatalogDriftError(f"附加来源 {source} 的连接方式不一致")

    return CatalogValidation(
        catalog_hash=catalog.catalog_hash,
        table_count=len({field.table for field in catalog.fields}),
        field_count=len(catalog.fields),
        operator_count=len(catalog.operators),
    )


def render_catalog_markdown(catalog: MiningCatalog | None = None) -> str:
    """生成供人工审查的字段和算子文档。"""

    catalog = catalog or load_catalog()
    lines = [
        "# 自动因子挖掘字段与算子目录",
        "",
        f"- schema：`{catalog.schema_version}`",
        f"- 数据版本：`{catalog.data_version}`",
        f"- 目录哈希：`{catalog.catalog_hash}`",
        f"- 字段总数：{len(catalog.fields)}，允许使用：{len(catalog.allowed_fields)}",
        f"- 算子总数：{len(catalog.operators)}",
        "",
        "## 字段",
        "",
        "| 来源 | 字段 | 中文名 | 类型 | 维度 | 单位 | 可用时点 | 允许 | "
        "说明 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for field in catalog.fields:
        allowed = "是" if field.allowed else "否"
        reason = field.reason or ("可空" if field.nullable else "非空")
        lines.append(
            f"| `{field.source}` | `{field.name}` | {field.label} | {field.dtype} | "
            f"{field.dimension} | {field.unit} | {field.available_at} | {allowed} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## 算子",
            "",
            "| 算子 | 类型 | 参数数 | 窗口范围 | 说明 |",
            "|---|---|---|---|---|",
        ]
    )
    for operator in catalog.operators:
        window = (
            f"{operator.min_window}～{operator.max_window}"
            if operator.min_window is not None
            else "无"
        )
        lines.append(
            f"| `{operator.name}` | {operator.kind} | "
            f"{operator.min_args}～{operator.max_args} | {window} | {operator.description} |"
        )
    lines.extend(
        [
            "",
            "## 安全边界",
            "",
            "- 财务字段只在公告日次日及以后可用，"
            "连接条件固定为 `trade_date > available_date`。",
            "- 表达式只接受本目录中的字段和算子，"
            "不接受字符串、原始 SQL、注释或多语句。",
            "- 股票代码、日期、状态、标签、审计列和未复权价格字段"
            "不能进入表达式。",
        ]
    )
    return "\n".join(lines) + "\n"
