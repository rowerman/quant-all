"""不依赖执行环境的受控因子表达式解析器。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re

from mining.errors import ExpressionValidationError
from mining.expression_ast import (
    BinaryNode,
    CallNode,
    ExpressionNode,
    FieldNode,
    NumberNode,
    UnaryNode,
    expression_depth,
    expression_node_count,
)

_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MAX_NUMBER = Decimal("1e12")
_MIN_NUMBER = Decimal("1e-12")
_FUNCTION_ALIASES = {
    "average": "mean",
    "delay": "lag",
    "stddev": "std",
}


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    position: int


def _tokenize(expression: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    position = 0
    while position < len(expression):
        character = expression[position]
        if character.isspace():
            position += 1
            continue
        number = _NUMBER.match(expression, position)
        if number is not None:
            tokens.append(_Token("number", number.group(0), position))
            position = number.end()
            continue
        identifier = _IDENTIFIER.match(expression, position)
        if identifier is not None:
            tokens.append(_Token("identifier", identifier.group(0), position))
            position = identifier.end()
            continue
        if character in "+-*/(),":
            tokens.append(_Token("symbol", character, position))
            position += 1
            continue
        raise ExpressionValidationError(
            f"表达式在位置 {position} 含有不允许的字符 {character!r}",
            reason="illegal_character",
        )
    tokens.append(_Token("eof", "", len(expression)))
    return tuple(tokens)


class _Parser:
    def __init__(self, expression: str) -> None:
        self._tokens = _tokenize(expression)
        self._index = 0

    @property
    def _current(self) -> _Token:
        return self._tokens[self._index]

    def _take(self, value: str | None = None) -> _Token:
        token = self._current
        if value is not None and token.value != value:
            raise ExpressionValidationError(
                f"位置 {token.position} 需要 {value!r}，收到 {token.value!r}",
                reason="syntax",
            )
        self._index += 1
        return token

    def parse(self) -> ExpressionNode:
        node = self._additive()
        if self._current.kind != "eof":
            raise ExpressionValidationError(
                f"位置 {self._current.position} 存在未消费内容 {self._current.value!r}",
                reason="syntax",
            )
        return node

    def _additive(self) -> ExpressionNode:
        node = self._multiplicative()
        while self._current.value in {"+", "-"}:
            operator = self._take().value
            node = BinaryNode(operator, node, self._multiplicative())
        return node

    def _multiplicative(self) -> ExpressionNode:
        node = self._unary()
        while self._current.value in {"*", "/"}:
            operator = self._take().value
            node = BinaryNode(operator, node, self._unary())
        return node

    def _unary(self) -> ExpressionNode:
        if self._current.value == "+":
            self._take("+")
            return self._unary()
        if self._current.value == "-":
            self._take("-")
            operand = self._unary()
            if isinstance(operand, NumberNode):
                return NumberNode(-operand.value)
            return UnaryNode("-", operand)
        return self._primary()

    def _primary(self) -> ExpressionNode:
        token = self._current
        if token.kind == "number":
            self._take()
            try:
                value = Decimal(token.value)
            except InvalidOperation as exc:
                raise ExpressionValidationError(
                    f"位置 {token.position} 的数字无法解析",
                    reason="invalid_number",
                ) from exc
            magnitude = abs(value)
            if (
                not value.is_finite()
                or magnitude > _MAX_NUMBER
                or (magnitude != 0 and magnitude < _MIN_NUMBER)
            ):
                raise ExpressionValidationError(
                    f"位置 {token.position} 的数字超出允许范围",
                    reason="invalid_number",
                )
            return NumberNode(value)
        if token.value == "(":
            self._take("(")
            node = self._additive()
            self._take(")")
            return node
        if token.kind == "identifier":
            self._take()
            if self._current.value != "(":
                return FieldNode(token.value)
            self._take("(")
            arguments: list[ExpressionNode] = []
            if self._current.value == ")":
                raise ExpressionValidationError(
                    f"函数 {token.value} 至少需要一个参数",
                    reason="invalid_arity",
                )
            while True:
                arguments.append(self._additive())
                if self._current.value == ")":
                    self._take(")")
                    break
                self._take(",")
            name = _FUNCTION_ALIASES.get(token.value.lower(), token.value.lower())
            if name == "neg" and len(arguments) == 1:
                return UnaryNode("-", arguments[0])
            return CallNode(name, tuple(arguments))
        raise ExpressionValidationError(
            f"位置 {token.position} 缺少表达式",
            reason="syntax",
        )


def parse_expression(
    expression: str,
    *,
    max_depth: int = 6,
    max_nodes: int = 256,
) -> ExpressionNode:
    """解析受控表达式并检查结构与规模。"""

    if not isinstance(expression, str):
        raise ExpressionValidationError("expression 必须是字符串", reason="invalid_type")
    if not expression.strip():
        raise ExpressionValidationError("expression 不能为空", reason="empty")
    try:
        node = _Parser(expression).parse()
    except RecursionError as exc:
        raise ExpressionValidationError(
            "表达式嵌套过深",
            reason="too_deep",
        ) from exc
    try:
        depth = expression_depth(node)
        count = expression_node_count(node)
    except RecursionError as exc:
        raise ExpressionValidationError(
            "表达式嵌套过深",
            reason="too_deep",
        ) from exc
    if depth > max_depth:
        raise ExpressionValidationError(
            f"表达式深度 {depth} 超过上限 {max_depth}",
            reason="too_deep",
        )
    if count > max_nodes:
        raise ExpressionValidationError(
            f"表达式节点数 {count} 超过上限 {max_nodes}",
            reason="too_many_nodes",
        )
    return node
