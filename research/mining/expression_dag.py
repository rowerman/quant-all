"""候选表达式的确定性 DAG 规划与共享子树诊断。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from mining.catalog import MiningCatalog, load_catalog
from mining.contract_candidates import CandidateDefinition
from mining.contract_requests import FrozenContext
from mining.contract_types import canonical_json, sha256_json
from mining.errors import CandidateError, CatalogDriftError
from mining.expression_ast import (
    BinaryNode,
    CallNode,
    ExpressionNode,
    FieldNode,
    NumberNode,
    UnaryNode,
    expression_fields,
    expression_node_count,
    expression_structural_key,
)
from mining.expression_parser import parse_expression
from mining.expression_validation import validate_candidate_expression
from mining.logging_conf import format_event, get_mining_logger

DAG_SCHEMA_VERSION = "mining_expression_dag_v1"
_LOGGER = get_mining_logger("mining.expression.dag")


@dataclass(frozen=True)
class ExpressionDagNode:
    """一个按规范化语义去重后的表达式节点。"""

    node_id: str
    structural_hash: str
    canonical_expression: str
    node_kind: str
    child_ids: tuple[str, ...]
    dependencies: tuple[str, ...]
    reference_count: int

    def to_dict(self) -> dict[str, object]:
        """返回稳定可序列化的节点内容。"""

        return {
            "node_id": self.node_id,
            "structural_hash": self.structural_hash,
            "canonical_expression": self.canonical_expression,
            "node_kind": self.node_kind,
            "child_ids": list(self.child_ids),
            "dependencies": list(self.dependencies),
            "reference_count": self.reference_count,
        }


@dataclass(frozen=True)
class CandidateExpressionRoot:
    """候选身份与表达式根节点的映射。"""

    candidate_id: str
    content_hash: str
    semantic_hash: str
    canonical_expression: str
    root_node_id: str

    def to_dict(self) -> dict[str, str]:
        """返回稳定可序列化的候选根映射。"""

        return {
            "candidate_id": self.candidate_id,
            "content_hash": self.content_hash,
            "semantic_hash": self.semantic_hash,
            "canonical_expression": self.canonical_expression,
            "root_node_id": self.root_node_id,
        }


@dataclass(frozen=True)
class ExpressionDagDiagnostics:
    """候选组的只读节点复用诊断。"""

    candidate_count: int
    total_node_occurrences: int
    unique_node_count: int
    deduplicated_node_count: int
    shared_node_count: int
    shared_node_ratio: float
    potential_evaluation_reduction_ratio: float

    def to_dict(self) -> dict[str, int | float]:
        """返回稳定可序列化的诊断。"""

        return {
            "candidate_count": self.candidate_count,
            "total_node_occurrences": self.total_node_occurrences,
            "unique_node_count": self.unique_node_count,
            "deduplicated_node_count": self.deduplicated_node_count,
            "shared_node_count": self.shared_node_count,
            "shared_node_ratio": self.shared_node_ratio,
            "potential_evaluation_reduction_ratio": (
                self.potential_evaluation_reduction_ratio
            ),
        }


@dataclass(frozen=True)
class CandidateExpressionPlan:
    """同一冻结上下文内一个候选组的表达式 DAG。"""

    schema_version: str
    context_hash: str
    catalog_hash: str
    candidates: tuple[CandidateExpressionRoot, ...]
    nodes: tuple[ExpressionDagNode, ...]
    diagnostics: ExpressionDagDiagnostics

    def to_dict(self) -> dict[str, object]:
        """返回可持久化或比较的完整计划。"""

        return {
            "schema_version": self.schema_version,
            "context_hash": self.context_hash,
            "catalog_hash": self.catalog_hash,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "nodes": [node.to_dict() for node in self.nodes],
            "diagnostics": self.diagnostics.to_dict(),
        }

    @property
    def plan_hash(self) -> str:
        """返回完整计划的稳定哈希。"""

        return sha256_json(self.to_dict())


def _node_kind(node: ExpressionNode) -> str:
    if isinstance(node, NumberNode):
        return "number"
    if isinstance(node, FieldNode):
        return "field"
    if isinstance(node, UnaryNode):
        return "unary"
    if isinstance(node, BinaryNode):
        return "binary"
    return "call"


def _node_children(node: ExpressionNode) -> tuple[ExpressionNode, ...]:
    if isinstance(node, (NumberNode, FieldNode)):
        return ()
    if isinstance(node, UnaryNode):
        return (node.operand,)
    if isinstance(node, BinaryNode):
        return (node.left, node.right)
    return node.arguments


class _DagBuilder:
    def __init__(self) -> None:
        self._nodes: dict[str, ExpressionDagNode] = {}
        self._references: dict[str, int] = {}

    def intern(self, node: ExpressionNode) -> str:
        """登记节点并返回稳定 ID，相同结构只保留一次。"""

        structural_hash = sha256_json(
            {
                "schema_version": DAG_SCHEMA_VERSION,
                "node": expression_structural_key(node),
            }
        )
        node_id = f"expr_{structural_hash[:24]}"
        if structural_hash in self._nodes:
            self._references[structural_hash] += 1
            return node_id
        child_ids = tuple(self.intern(child) for child in _node_children(node))
        self._references[structural_hash] = 1
        self._nodes[structural_hash] = ExpressionDagNode(
            node_id=node_id,
            structural_hash=structural_hash,
            canonical_expression=node.canonical(),
            node_kind=_node_kind(node),
            child_ids=child_ids,
            dependencies=expression_fields(node),
            reference_count=1,
        )
        return node_id

    def finish(self) -> tuple[ExpressionDagNode, ...]:
        """按后序构建顺序返回带最终引用次数的节点。"""

        return tuple(
            ExpressionDagNode(
                node_id=node.node_id,
                structural_hash=node.structural_hash,
                canonical_expression=node.canonical_expression,
                node_kind=node.node_kind,
                child_ids=node.child_ids,
                dependencies=node.dependencies,
                reference_count=self._references[node.structural_hash],
            )
            for node in self._nodes.values()
        )


def _ensure_frozen_catalog(context: FrozenContext, catalog: MiningCatalog) -> None:
    if (
        catalog.data_version != context.data_version
        or context.field_catalog_hash
        != sha256_json(
            sorted(catalog.frozen_field_mappings(), key=canonical_json)
        )
        or context.operator_catalog_hash
        != sha256_json(
            sorted(catalog.frozen_operator_mappings(), key=canonical_json)
        )
    ):
        raise CatalogDriftError("表达式计划目录与冻结上下文不一致")


def _candidate_order(candidate: CandidateDefinition) -> tuple[str, str, int, int]:
    return (
        candidate.candidate_id,
        candidate.semantic_hash,
        candidate.discovery_round,
        candidate.discovery_index,
    )


def build_candidate_expression_plan(
    candidates: Iterable[CandidateDefinition],
    context: FrozenContext,
    *,
    catalog: MiningCatalog | None = None,
) -> CandidateExpressionPlan:
    """为同一冻结上下文中的候选组构建确定性的表达式 DAG。"""

    if not isinstance(context, FrozenContext):
        raise CandidateError("context 必须是 FrozenContext")
    candidate_list = tuple(candidates)
    if not candidate_list:
        raise CandidateError("候选组不能为空")
    if not all(isinstance(candidate, CandidateDefinition) for candidate in candidate_list):
        raise CandidateError("候选组只能包含 CandidateDefinition")
    candidate_ids = [candidate.candidate_id for candidate in candidate_list]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise CandidateError("候选组包含重复 candidate_id")
    ordered = tuple(sorted(candidate_list, key=_candidate_order))
    catalog = catalog or load_catalog()
    _ensure_frozen_catalog(context, catalog)

    builder = _DagBuilder()
    roots: list[CandidateExpressionRoot] = []
    total_node_occurrences = 0
    for candidate in ordered:
        validated = validate_candidate_expression(candidate, context, catalog=catalog)
        ast = parse_expression(
            validated.canonical_expression,
            max_depth=context.budget.max_expression_depth,
        )
        total_node_occurrences += expression_node_count(ast)
        roots.append(
            CandidateExpressionRoot(
                candidate_id=candidate.candidate_id,
                content_hash=candidate.content_hash,
                semantic_hash=candidate.semantic_hash,
                canonical_expression=validated.canonical_expression,
                root_node_id=builder.intern(ast),
            )
        )

    nodes = builder.finish()
    deduplicated = total_node_occurrences - len(nodes)
    shared = sum(node.reference_count > 1 for node in nodes)
    diagnostics = ExpressionDagDiagnostics(
        candidate_count=len(roots),
        total_node_occurrences=total_node_occurrences,
        unique_node_count=len(nodes),
        deduplicated_node_count=deduplicated,
        shared_node_count=shared,
        shared_node_ratio=shared / len(nodes),
        potential_evaluation_reduction_ratio=deduplicated / total_node_occurrences,
    )
    plan = CandidateExpressionPlan(
        schema_version=DAG_SCHEMA_VERSION,
        context_hash=context.context_hash,
        catalog_hash=catalog.catalog_hash,
        candidates=tuple(roots),
        nodes=nodes,
        diagnostics=diagnostics,
    )
    _LOGGER.info(
        format_event(
            {
                "event": "expression_dag_built",
                "context_hash": context.context_hash,
                "catalog_hash": catalog.catalog_hash,
                "plan_hash": plan.plan_hash,
                "candidate_count": diagnostics.candidate_count,
                "unique_node_count": diagnostics.unique_node_count,
                "shared_node_count": diagnostics.shared_node_count,
            }
        )
    )
    return plan
