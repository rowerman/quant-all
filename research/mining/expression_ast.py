"""受控因子表达式的最小 AST。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TypeAlias


def format_number(value: Decimal) -> str:
    """把有限 Decimal 规范化为不带指数的十进制文本。"""

    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


@dataclass(frozen=True)
class NumberNode:
    """数字字面量。"""

    value: Decimal

    def canonical(self) -> str:
        """返回规范化文本。"""

        return format_number(self.value)


@dataclass(frozen=True)
class FieldNode:
    """字段引用。"""

    name: str

    def canonical(self) -> str:
        """返回规范化文本。"""

        return self.name


@dataclass(frozen=True)
class UnaryNode:
    """一元正负号。"""

    operator: str
    operand: ExpressionNode

    def canonical(self) -> str:
        """返回规范化文本。"""

        return f"(-{self.operand.canonical()})"


@dataclass(frozen=True)
class BinaryNode:
    """二元四则运算。"""

    operator: str
    left: ExpressionNode
    right: ExpressionNode

    def canonical(self) -> str:
        """返回规范化文本；加法和乘法忽略交换顺序。"""

        if self.operator in {"+", "*"}:
            left, right = sorted(
                (self.left.canonical(), self.right.canonical()),
            )
        else:
            left, right = self.left.canonical(), self.right.canonical()
        return f"({left} {self.operator} {right})"


@dataclass(frozen=True)
class CallNode:
    """受控函数调用。"""

    name: str
    arguments: tuple[ExpressionNode, ...]

    def canonical(self) -> str:
        """返回规范化文本。"""

        rendered = [argument.canonical() for argument in self.arguments]
        if self.name in {"least", "greatest"}:
            rendered.sort()
        arguments = ", ".join(rendered)
        return f"{self.name}({arguments})"


ExpressionNode: TypeAlias = NumberNode | FieldNode | UnaryNode | BinaryNode | CallNode


def expression_structural_key(node: ExpressionNode) -> tuple[object, ...]:
    """返回与规范化表达式语义一致、可稳定哈希的结构键。"""

    if isinstance(node, NumberNode):
        return ("number", node.canonical())
    if isinstance(node, FieldNode):
        return ("field", node.name)
    if isinstance(node, UnaryNode):
        return ("unary", node.operator, expression_structural_key(node.operand))
    if isinstance(node, BinaryNode):
        left = expression_structural_key(node.left)
        right = expression_structural_key(node.right)
        if node.operator in {"+", "*"} and right < left:
            left, right = right, left
        return ("binary", node.operator, left, right)
    arguments = tuple(expression_structural_key(argument) for argument in node.arguments)
    if node.name in {"least", "greatest"}:
        arguments = tuple(sorted(arguments))
    return ("call", node.name, arguments)


def expression_depth(node: ExpressionNode) -> int:
    """返回以叶节点为 1 的 AST 深度。"""

    if isinstance(node, (NumberNode, FieldNode)):
        return 1
    if isinstance(node, UnaryNode):
        return 1 + expression_depth(node.operand)
    if isinstance(node, BinaryNode):
        return 1 + max(expression_depth(node.left), expression_depth(node.right))
    return 1 + max(expression_depth(argument) for argument in node.arguments)


def expression_node_count(node: ExpressionNode) -> int:
    """返回 AST 节点数。"""

    if isinstance(node, (NumberNode, FieldNode)):
        return 1
    if isinstance(node, UnaryNode):
        return 1 + expression_node_count(node.operand)
    if isinstance(node, BinaryNode):
        return 1 + expression_node_count(node.left) + expression_node_count(node.right)
    return 1 + sum(expression_node_count(argument) for argument in node.arguments)


def expression_fields(node: ExpressionNode) -> tuple[str, ...]:
    """返回表达式引用的字段，按名字排序去重。"""

    found: set[str] = set()

    def visit(item: ExpressionNode) -> None:
        if isinstance(item, FieldNode):
            found.add(item.name)
        elif isinstance(item, UnaryNode):
            visit(item.operand)
        elif isinstance(item, BinaryNode):
            visit(item.left)
            visit(item.right)
        elif isinstance(item, CallNode):
            for argument in item.arguments:
                visit(argument)

    visit(node)
    return tuple(sorted(found))
