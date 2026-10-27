"""受控表达式的公共校验、稳定命名和 S2 候选适配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mining.catalog import MiningCatalog, load_catalog
from mining.contract_candidates import CandidateDefinition
from mining.contract_requests import FrozenContext
from mining.contract_types import SCHEMA_VERSION, sha256_json
from mining.errors import CandidateError, ExpressionValidationError
from mining.expression_compiler import CompiledExpression, ExpressionCompiler
from mining.expression_parser import parse_expression
from mining.logging_conf import format_event, get_mining_logger

_LOGGER = get_mining_logger("mining.expression")
MAX_EXPRESSION_BYTES = 12000
MAX_AST_NODES = 256


@dataclass(frozen=True)
class ValidatedExpression:
    """通过受控 DSL 校验并绑定稳定语义的表达式。"""

    canonical_expression: str
    factor_name: str
    semantic_hash: str
    dependencies: tuple[str, ...]
    lookback: int
    output_type: str
    layers: tuple[str, ...]
    extra_sources: tuple[str, ...]
    parameters: dict[str, Any]
    direction: int
    catalog_hash: str

    def to_candidate(
        self,
        *,
        mechanism: str,
        description: str,
        steps,
        discovery_round: int,
        discovery_index: int,
    ) -> CandidateDefinition:
        """把已验证表达式包装为 S2 候选，不写入候选库。"""

        candidate = CandidateDefinition(
            name=self.factor_name,
            mechanism=mechanism,
            description=description,
            canonical_expression=self.canonical_expression,
            direction=self.direction,
            lookback=self.lookback,
            dependencies=self.dependencies,
            output_type=self.output_type,
            steps=steps,
            discovery_round=discovery_round,
            discovery_index=discovery_index,
            parameters=self.parameters,
        )
        if candidate.semantic_hash != self.semantic_hash:
            raise ExpressionValidationError(
                "候选语义哈希与已验证表达式不一致",
                reason="hash_mismatch",
            )
        return candidate


def _semantic_hash(
    compiled: CompiledExpression,
    *,
    direction: int,
    parameters: dict[str, Any],
) -> str:
    return sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "canonical_expression": compiled.canonical_expression,
            "direction": direction,
            "lookback": compiled.lookback,
            "dependencies": list(compiled.dependencies),
            "output_type": compiled.output_type,
            "parameters": parameters,
        }
    )


def validate_expression(
    expression: str,
    *,
    direction: int,
    max_depth: int = 6,
    max_length: int = MAX_EXPRESSION_BYTES,
    excluded_fields=(),
    catalog: MiningCatalog | None = None,
) -> ValidatedExpression:
    """解析、校验并编译一个受控表达式。"""

    catalog = catalog or load_catalog()
    try:
        if direction not in (-1, 1):
            raise ExpressionValidationError(
                "direction 必须是 -1 或 1",
                reason="invalid_direction",
            )
        if (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, int)
            or not 2 <= max_depth <= 8
        ):
            raise ExpressionValidationError(
                "max_depth 必须在 2～8",
                reason="invalid_depth_limit",
            )
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or not 1 <= max_length <= MAX_EXPRESSION_BYTES
        ):
            raise ExpressionValidationError(
                f"max_length 必须在 1～{MAX_EXPRESSION_BYTES}",
                reason="invalid_length_limit",
            )
        if not isinstance(expression, str):
            raise ExpressionValidationError("expression 必须是字符串", reason="invalid_type")
        size = len(expression.encode("utf-8"))
        if size == 0:
            raise ExpressionValidationError("expression 不能为空", reason="empty")
        if size > max_length:
            raise ExpressionValidationError(
                f"表达式为 {size} 字节，超过上限 {max_length}",
                reason="too_long",
            )
        ast = parse_expression(
            expression,
            max_depth=max_depth,
            max_nodes=MAX_AST_NODES,
        )
        compiled = ExpressionCompiler(catalog, excluded_fields=excluded_fields).compile(ast)
        canonical_size = len(compiled.canonical_expression.encode("utf-8"))
        if canonical_size > max_length:
            raise ExpressionValidationError(
                f"规范化表达式为 {canonical_size} 字节，超过上限 {max_length}",
                reason="too_long",
            )
        parameters: dict[str, Any] = {}
        semantic_hash = _semantic_hash(compiled, direction=direction, parameters=parameters)
        return ValidatedExpression(
            canonical_expression=compiled.canonical_expression,
            factor_name=f"MINE_{semantic_hash[:24].upper()}",
            semantic_hash=semantic_hash,
            dependencies=compiled.dependencies,
            lookback=compiled.lookback,
            output_type=compiled.output_type,
            layers=compiled.layers,
            extra_sources=compiled.extra_sources,
            parameters=parameters,
            direction=direction,
            catalog_hash=catalog.catalog_hash,
        )
    except ExpressionValidationError as exc:
        _LOGGER.warning(
            format_event(
                {
                    "event": "expression_rejected",
                    "error_code": exc.code,
                    "reason": exc.reason,
                    "catalog_hash": catalog.catalog_hash,
                }
            )
        )
        raise


def validate_candidate_expression(
    candidate: CandidateDefinition,
    context: FrozenContext,
    *,
    catalog: MiningCatalog | None = None,
) -> ValidatedExpression:
    """重新校验候选表达式及其冻结语义身份。"""

    if not isinstance(candidate, CandidateDefinition):
        raise CandidateError("candidate 必须是 CandidateDefinition")
    if not isinstance(context, FrozenContext):
        raise CandidateError("context 必须是 FrozenContext")
    validated = validate_expression(
        candidate.canonical_expression,
        direction=candidate.direction,
        max_depth=context.budget.max_expression_depth,
        excluded_fields=context.budget.excluded_fields,
        catalog=catalog,
    )
    if (
        validated.semantic_hash != candidate.semantic_hash
        or validated.factor_name != candidate.name
        or validated.lookback != candidate.lookback
        or validated.dependencies != candidate.dependencies
        or validated.output_type != candidate.output_type
        or validated.parameters != dict(candidate.parameters)
    ):
        raise CandidateError("候选表达式、稳定名或语义哈希与冻结定义不一致")
    return validated
