"""确定性邻域变体生成。"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from mining.catalog import load_catalog
from mining.contract_candidates import CandidateDefinition
from mining.expression_ast import (
    CallNode,
    ExpressionNode,
    NumberNode,
)
from mining.expression_parser import parse_expression
from mining.expression_validation import validate_expression
from mining.search_stability_contracts import (
    STABILITY_VARIANT_LIMIT,
    VariantPlan,
    WindowVariant,
)


def _window_sites(
    node: ExpressionNode,
    path: tuple[int, ...] = (),
):
    """按确定性深度优先顺序列出整数窗口参数。"""

    if isinstance(node, CallNode):
        operator = load_catalog().operator(node.name)
        if operator.window_arg is not None:
            index = operator.window_arg
            argument = node.arguments[index]
            if (
                isinstance(argument, NumberNode)
                and argument.value == argument.value.to_integral_value()
            ):
                yield path, node, index, operator
        for child_index, child in enumerate(node.arguments):
            yield from _window_sites(child, (*path, child_index))


def _replace_argument(
    node: ExpressionNode,
    path: tuple[int, ...],
    argument_index: int,
    value: int,
) -> ExpressionNode:
    """替换指定路径调用节点的一个参数。"""

    if not path:
        if not isinstance(node, CallNode):
            raise ValueError("窗口路径没有指向函数调用")
        arguments = list(node.arguments)
        arguments[argument_index] = NumberNode(Decimal(value))
        return replace(node, arguments=tuple(arguments))
    child_index = path[0]
    if not isinstance(node, CallNode):
        raise ValueError("窗口路径越过了函数调用")
    arguments = list(node.arguments)
    arguments[child_index] = _replace_argument(
        arguments[child_index],
        path[1:],
        argument_index,
        value,
    )
    return replace(node, arguments=tuple(arguments))


def generate_primary_window_variants(
    candidate: CandidateDefinition,
) -> VariantPlan:
    """为一个确定的整数窗口生成最多两个相邻变体。"""

    ast = parse_expression(candidate.canonical_expression, max_depth=8)
    selected = None
    for path, call, argument_index, operator in _window_sites(ast):
        window = int(call.arguments[argument_index].value)
        lower = window - 1 if operator.min_window is None else max(
            operator.min_window,
            window - 1,
        )
        upper = window + 1 if operator.max_window is None else min(
            operator.max_window,
            window + 1,
        )
        options = tuple(
            value
            for value in (lower, upper)
            if value != window
        )[:STABILITY_VARIANT_LIMIT]
        if options:
            selected = (path, argument_index, window, options)
            break
    if selected is None:
        return VariantPlan(
            base_candidate_id=candidate.candidate_id,
            base_window=None,
            variants=(),
            status="not_applicable",
            reason="候选没有可用整数窗口",
        )
    path, argument_index, window, options = selected
    variants: list[WindowVariant] = []
    for value in options:
        expression = _replace_argument(ast, path, argument_index, value).canonical()
        validated = validate_expression(
            expression,
            direction=candidate.direction,
            max_depth=8,
        )
        variant = validated.to_candidate(
            mechanism=candidate.mechanism,
            description=f"{candidate.description}（稳定性变体）",
            steps=candidate.steps,
            discovery_round=candidate.discovery_round,
            discovery_index=candidate.discovery_index,
        )
        variants.append(
            WindowVariant(
                candidate=variant,
                base_window=window,
                variant_window=value,
                delta=value - window,
            )
        )
    return VariantPlan(
        base_candidate_id=candidate.candidate_id,
        base_window=window,
        variants=tuple(variants),
        status="ready",
    )


__all__ = ["generate_primary_window_variants"]
