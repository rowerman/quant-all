"""把受控表达式 AST 编译为分层 DuckDB SQL。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from factors import ops
from mining.catalog import FieldSpec, MiningCatalog, OperatorSpec
from mining.errors import ExpressionValidationError
from mining.expression_ast import (
    BinaryNode,
    CallNode,
    ExpressionNode,
    FieldNode,
    NumberNode,
    UnaryNode,
    expression_fields,
)


@dataclass(frozen=True)
class CompiledExpression:
    """通过安全校验、可直接交给因子查询层执行的分层表达式。"""

    canonical_expression: str
    layers: tuple[str, ...]
    dependencies: tuple[str, ...]
    lookback: int
    extra_sources: tuple[str, ...]
    output_type: str
    dimension: str


@dataclass(frozen=True)
class CompiledValue:
    """一个 AST 节点的 SQL、量纲和是否必须单独成层的标记。"""

    sql: str
    dimension: str
    lookback: int
    needs_layer: bool = False


_Value = CompiledValue


def _calculation_sql(name: str, arguments: tuple[str, ...], window: int | None) -> str:
    if name == "add":
        return f"({arguments[0]} + {arguments[1]})"
    if name == "sub":
        return f"({arguments[0]} - {arguments[1]})"
    if name == "mul":
        return f"({arguments[0]} * {arguments[1]})"
    if name == "div":
        return ops.safe_div(arguments[0], arguments[1])
    if name == "neg":
        return ops.neg(arguments[0])
    if name in {"abs", "ln", "sqrt", "sign"}:
        return f"{name}({arguments[0]})"
    if name in {"least", "greatest"}:
        return f"{name}({', '.join(arguments)})"
    if name == "rank":
        return ops.cs_rank(arguments[0])
    if name == "zscore":
        return ops.cs_zscore(arguments[0])
    if name == "demean":
        return ops.cs_demean(arguments[0])
    if name == "scale":
        return ops.cs_scale(arguments[0])
    if name == "signed_power":
        return ops.signed_power(arguments[0], arguments[1])
    if window is None:
        raise ExpressionValidationError(
            f"算子 {name} 缺少编译窗口",
            reason="invalid_operator",
        )
    if name == "lag":
        return ops.ts_delay(arguments[0], window)
    if name == "delta":
        return ops.ts_delta(arguments[0], window)
    if name == "mean":
        return ops.ts_mean(arguments[0], window)
    if name == "sum":
        return ops.ts_sum(arguments[0], window)
    if name == "min":
        return ops.ts_min(arguments[0], window)
    if name == "max":
        return ops.ts_max(arguments[0], window)
    if name == "median":
        return ops.ts_median(arguments[0], window)
    if name == "std":
        return ops.ts_std(arguments[0], window)
    if name == "skew":
        return ops.ts_skew(arguments[0], window)
    if name == "kurt":
        return ops.ts_kurt(arguments[0], window)
    if name == "product":
        return ops.ts_product(arguments[0], window)
    if name == "count":
        return ops.ts_count(arguments[0], window)
    if name == "corr":
        return ops.ts_corr(arguments[0], arguments[1], window)
    if name == "cov":
        return ops.ts_cov(arguments[0], arguments[1], window)
    if name == "slope":
        return ops.ts_slope(arguments[0], arguments[1], window)
    if name == "r2":
        return ops.ts_r2(arguments[0], arguments[1], window)
    if name == "downside_std":
        return ops.ts_std_downside(arguments[0], window)
    if name == "ts_rank":
        return ops.ts_rank(arguments[0], window)
    raise ExpressionValidationError(
        f"算子 {name} 没有 SQL 编译规则",
        reason="unknown_operator",
    )


def _add_dimension(left: str, right: str, *, operator: str) -> str:
    if left == "scalar":
        return right
    if right == "scalar":
        return left
    if left != right:
        raise ExpressionValidationError(
            f"{operator} 的维度不兼容：{left} 与 {right}",
            reason="dimension_mismatch",
        )
    return left


def _multiply_dimension(left: str, right: str) -> str:
    if left == "scalar":
        return right
    if right == "scalar":
        return left
    return "mixed"


def _divide_dimension(left: str, right: str) -> str:
    if right == "scalar":
        return left
    if left == right:
        return "scalar"
    if left == "scalar":
        return "inverse"
    return "mixed"


def _combine_dimensions(values: tuple[_Value, ...], operator: str) -> str:
    dimensions = [value.dimension for value in values]
    non_scalar = {dimension for dimension in dimensions if dimension != "scalar"}
    if len(non_scalar) > 1:
        raise ExpressionValidationError(
            f"{operator} 的维度不兼容：{sorted(non_scalar)}",
            reason="dimension_mismatch",
        )
    return next(iter(non_scalar), "scalar")


class ExpressionCompiler:
    """校验白名单并编译表达式；不修改全局因子注册表。"""

    def __init__(self, catalog: MiningCatalog, *, excluded_fields=()) -> None:
        self._catalog = catalog
        self._excluded_fields = frozenset(str(item) for item in excluded_fields)
        self._layers: list[str] = []

    def compile(self, root: ExpressionNode) -> CompiledExpression:
        """编译完整表达式。"""

        self._layers = []
        value = self._compile_node(root)
        current_layer_alias = f"v{len(self._layers)}" if self._layers else ""
        if value.sql != current_layer_alias:
            self._layers.append(value.sql)
        if not self._layers:
            raise ExpressionValidationError("表达式没有生成可执行层", reason="empty")
        dependencies = expression_fields(root)
        sources = sorted(
            {
                field.source
                for field in (self._find_field(name) for name in dependencies)
                if field.source != "daily_hfq"
            }
        )
        return CompiledExpression(
            canonical_expression=root.canonical(),
            layers=tuple(self._layers),
            dependencies=dependencies,
            lookback=self._lookback(root),
            extra_sources=tuple(sources),
            output_type="number",
            dimension=value.dimension,
        )

    def _find_field(self, name: str) -> FieldSpec:
        if name in self._excluded_fields:
            raise ExpressionValidationError(
                f"字段 {name!r} 已被本次运行排除",
                reason="excluded_field",
            )
        for field in self._catalog.fields:
            if field.name == name:
                if not field.allowed:
                    raise ExpressionValidationError(
                        f"字段 {name!r} 不允许进入因子表达式：{field.reason}",
                        reason="forbidden_field",
                    )
                return field
        raise ExpressionValidationError(
            f"字段 {name!r} 不在冻结目录中",
            reason="unknown_field",
        )

    def _find_operator(self, name: str) -> OperatorSpec:
        for operator in self._catalog.operators:
            if operator.name == name:
                return operator
        raise ExpressionValidationError(
            f"算子 {name!r} 不在冻结目录中",
            reason="unknown_operator",
        )

    def _emit(self, sql: str) -> str:
        alias = f"v{len(self._layers) + 1}"
        self._layers.append(sql)
        return alias

    def _compile_node(self, node: ExpressionNode) -> _Value:
        value = self.compile_value(node, self._compile_node)
        if value.needs_layer:
            return CompiledValue(
                self._emit(value.sql),
                value.dimension,
                value.lookback,
            )
        return value

    def compile_value(
        self,
        node: ExpressionNode,
        resolve_child: Callable[[ExpressionNode], CompiledValue],
    ) -> CompiledValue:
        """按调用方提供的子节点结果编译当前节点，不直接写入单候选层列表。"""

        if isinstance(node, NumberNode):
            return _Value(node.canonical(), "scalar", 0)
        if isinstance(node, FieldNode):
            field = self._find_field(node.name)
            if field.dtype == "boolean":
                sql = f"(CASE WHEN {field.name} THEN 1.0 ELSE 0.0 END)"
                return _Value(sql, "scalar", 0)
            return _Value(field.name, field.dimension, 0)
        if isinstance(node, UnaryNode):
            operand = resolve_child(node.operand)
            return _Value(ops.neg(operand.sql), operand.dimension, operand.lookback)
        if isinstance(node, BinaryNode):
            left = resolve_child(node.left)
            right = resolve_child(node.right)
            name = {"+": "add", "-": "sub", "*": "mul", "/": "div"}[node.operator]
            if name in {"add", "sub"}:
                dimension = _add_dimension(left.dimension, right.dimension, operator=node.operator)
            elif name == "mul":
                dimension = _multiply_dimension(left.dimension, right.dimension)
            else:
                dimension = _divide_dimension(left.dimension, right.dimension)
            sql = _calculation_sql(name, (left.sql, right.sql), None)
            return _Value(sql, dimension, max(left.lookback, right.lookback))
        return self._compile_call(node, resolve_child)

    def _require_literal(self, node: ExpressionNode, *, label: str) -> Decimal:
        if not isinstance(node, NumberNode):
            raise ExpressionValidationError(
                f"{label} 必须是数字字面量",
                reason="invalid_parameter",
            )
        return node.value

    def _window(self, operator: OperatorSpec, node: CallNode) -> int | None:
        if operator.window_arg is None:
            return None
        value = self._require_literal(node.arguments[operator.window_arg], label="窗口参数")
        if value != value.to_integral_value():
            raise ExpressionValidationError("窗口参数必须是整数", reason="invalid_window")
        window = int(value)
        if not operator.min_window <= window <= operator.max_window:
            raise ExpressionValidationError(
                f"算子 {operator.name} 的窗口必须在 "
                f"{operator.min_window}～{operator.max_window}",
                reason="invalid_window",
            )
        return window

    def _compile_call(
        self,
        node: CallNode,
        resolve_child: Callable[[ExpressionNode], CompiledValue],
    ) -> _Value:
        operator = self._find_operator(node.name)
        if not operator.min_args <= len(node.arguments) <= operator.max_args:
            raise ExpressionValidationError(
                f"算子 {node.name} 需要 {operator.min_args}～{operator.max_args} 个参数，"
                f"收到 {len(node.arguments)} 个",
                reason="invalid_arity",
            )
        window = self._window(operator, node)
        argument_nodes = tuple(
            argument
            for index, argument in enumerate(node.arguments)
            if index != operator.window_arg and index != operator.parameter_arg
        )
        values = tuple(resolve_child(argument) for argument in argument_nodes)
        if not values:
            raise ExpressionValidationError(
                f"算子 {node.name} 缺少计算参数",
                reason="invalid_arity",
            )

        dimension = values[0].dimension
        if node.name in {"add", "sub", "least", "greatest"}:
            dimension = _combine_dimensions(values, node.name)
        elif node.name == "mul":
            dimension = _multiply_dimension(values[0].dimension, values[1].dimension)
        elif node.name == "div":
            dimension = _divide_dimension(values[0].dimension, values[1].dimension)
        elif node.name in {"abs", "sign", "lag", "delta", "mean", "sum", "min", "max",
                           "median", "std", "downside_std", "demean", "scale"}:
            dimension = values[0].dimension
        elif node.name in {
            "ln",
            "sqrt",
            "corr",
            "r2",
            "count",
            "ts_rank",
            "rank",
            "zscore",
        }:
            dimension = "scalar"
        elif node.name in {"skew", "kurt", "cov", "slope", "product", "signed_power"}:
            dimension = "mixed" if values[0].dimension != "scalar" else "scalar"
        elif node.name == "neg":
            dimension = values[0].dimension

        parameters: tuple[Any, ...] = ()
        if operator.parameter_arg is not None:
            parameter = self._require_literal(
                node.arguments[operator.parameter_arg],
                label="算子参数",
            )
            if not operator.min_parameter <= float(parameter) <= operator.max_parameter:
                raise ExpressionValidationError(
                    f"算子 {node.name} 的参数必须在 "
                    f"{operator.min_parameter:g}～{operator.max_parameter:g}",
                    reason="invalid_parameter",
                )
            parameters = (parameter,)

        sql_arguments = tuple(value.sql for value in values) + tuple(
            str(parameter) for parameter in parameters
        )
        sql = _calculation_sql(node.name, sql_arguments, window)
        inner_lookback = max(value.lookback for value in values)
        if window is not None:
            inner_lookback += window - 1
        return _Value(
            sql,
            dimension,
            inner_lookback,
            needs_layer=operator.kind in {"time_series", "cross_section"},
        )

    def _lookback(self, node: ExpressionNode) -> int:
        if isinstance(node, (NumberNode, FieldNode)):
            return 0
        if isinstance(node, UnaryNode):
            return self._lookback(node.operand)
        if isinstance(node, BinaryNode):
            return max(self._lookback(node.left), self._lookback(node.right))
        operator = self._find_operator(node.name)
        window = self._window(operator, node)
        arguments = [
            self._lookback(argument)
            for index, argument in enumerate(node.arguments)
            if index != operator.window_arg and index != operator.parameter_arg
        ]
        result = max(arguments, default=0)
        if window is not None:
            result += window - 1
        return result
