"""候选表达式 DAG 的确定性、共享身份和错误边界测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mining.catalog import load_catalog
from mining.contract_candidates import CandidateDefinition
from mining.contract_requests import FrozenContext, MiningConstraints
from mining.errors import CandidateError, CatalogDriftError, ExpressionValidationError
from mining.expression_ast import expression_structural_key
from mining.expression_dag import build_candidate_expression_plan
from mining.expression_parser import parse_expression
from mining.expression_validation import validate_expression


def _context(**constraints) -> FrozenContext:
    """构造与当前目录一致的最小冻结上下文。"""

    catalog = load_catalog()
    return FrozenContext(
        data_fingerprint="a" * 64,
        field_catalog=catalog.frozen_field_mappings(),
        operator_catalog=catalog.frozen_operator_mappings(),
        budget=MiningConstraints(**constraints),
    )


def _candidate(
    expression: str,
    *,
    index: int = 1,
    description: str | None = None,
    direction: int = 1,
) -> CandidateDefinition:
    """由合法表达式构造稳定候选。"""

    validated = validate_expression(expression, direction=direction)
    return validated.to_candidate(
        mechanism="测试候选",
        description=description or f"候选 {index}",
        steps=("执行受控表达式",),
        discovery_round=1,
        discovery_index=index,
    )


def _node(plan, canonical_expression: str):
    """按规范化表达式查找唯一 DAG 节点。"""

    matches = [
        node
        for node in plan.nodes
        if node.canonical_expression == canonical_expression
    ]
    assert len(matches) == 1
    return matches[0]


def test_structural_key_follows_expression_canonicalization():
    """别名、数字格式和交换顺序不改变结构键。"""

    first = parse_expression("close * 1.00 + open")
    second = parse_expression("open + 1 * close")
    alias = parse_expression("delay(close, 5)")
    canonical = parse_expression("lag(close, 5)")
    reordered_least = parse_expression("least(open, close)")
    canonical_least = parse_expression("least(close, open)")

    assert expression_structural_key(first) == expression_structural_key(second)
    assert expression_structural_key(alias) == expression_structural_key(canonical)
    assert expression_structural_key(reordered_least) == expression_structural_key(
        canonical_least
    )
    assert expression_structural_key(parse_expression("mean(close, 5)")) != (
        expression_structural_key(parse_expression("mean(close, 20)"))
    )


def test_plan_reuses_shared_subtrees_across_candidates():
    """相同窗口子树跨候选共享，候选根仍保持独立。"""

    candidates = (
        _candidate("mean(close, 5) + rank(open)", index=1),
        _candidate("mean(close, 5) - open", index=2),
    )
    plan = build_candidate_expression_plan(candidates, _context())

    mean_node = _node(plan, "mean(close, 5)")
    assert mean_node.reference_count == 2
    assert len(plan.candidates) == 2
    assert len({root.root_node_id for root in plan.candidates}) == 2
    assert plan.diagnostics.candidate_count == 2
    assert plan.diagnostics.shared_node_count >= 1
    assert plan.diagnostics.deduplicated_node_count >= 1
    assert 0 < plan.diagnostics.potential_evaluation_reduction_ratio < 1


def test_same_expression_shares_root_without_merging_candidate_identity():
    """完整表达式相同的候选共享根节点，但内容身份保持独立。"""

    first = _candidate("mean(close, 5)", index=1, description="候选一")
    second = _candidate("mean(close, 5)", index=2, description="候选二")
    plan = build_candidate_expression_plan((first, second), _context())

    assert first.candidate_id != second.candidate_id
    assert first.semantic_hash == second.semantic_hash
    assert len({root.root_node_id for root in plan.candidates}) == 1
    assert {root.content_hash for root in plan.candidates} == {
        first.content_hash,
        second.content_hash,
    }


def test_different_windows_fields_and_parameters_are_not_shared():
    """窗口、字段和算子参数属于结构身份。"""

    candidates = (
        _candidate("mean(close, 5)", index=1),
        _candidate("mean(close, 20)", index=2),
        _candidate("signed_power(ret, 2)", index=3),
        _candidate("signed_power(ret, 3)", index=4),
    )
    plan = build_candidate_expression_plan(candidates, _context())

    assert plan.diagnostics.total_node_occurrences == 12
    assert plan.diagnostics.unique_node_count == 10
    assert plan.diagnostics.deduplicated_node_count == 2
    assert _node(plan, "ret").reference_count == 2
    assert len(
        [
            node
            for node in plan.nodes
            if node.node_kind == "call" and node.canonical_expression.startswith("mean")
        ]
    ) == 2


def test_plan_is_independent_of_candidate_input_order():
    """重复规划和候选换序得到相同节点顺序与计划哈希。"""

    candidates = (
        _candidate("rank(mean(close, 5))", index=1),
        _candidate("mean(close, 5) + open", index=2),
        _candidate("open / close", index=3),
    )
    first = build_candidate_expression_plan(candidates, _context())
    second = build_candidate_expression_plan(tuple(reversed(candidates)), _context())
    repeated = build_candidate_expression_plan(candidates, _context())

    assert first.to_dict() == second.to_dict()
    assert first.plan_hash == second.plan_hash == repeated.plan_hash
    assert first.diagnostics == repeated.diagnostics


def test_repeated_nodes_in_one_expression_are_counted():
    """单表达式内重复叶子也计入共享和潜在求值减少。"""

    plan = build_candidate_expression_plan(
        (_candidate("close + close"),),
        _context(),
    )

    assert plan.diagnostics.total_node_occurrences == 3
    assert plan.diagnostics.unique_node_count == 2
    assert plan.diagnostics.deduplicated_node_count == 1
    assert plan.diagnostics.shared_node_count == 1
    assert plan.diagnostics.shared_node_ratio == 0.5
    assert plan.diagnostics.potential_evaluation_reduction_ratio == pytest.approx(
        1 / 3
    )


def test_plan_rejects_empty_duplicate_and_non_candidate_groups():
    """候选组基础契约失败时明确报错。"""

    with pytest.raises(CandidateError, match="不能为空"):
        build_candidate_expression_plan((), _context())

    candidate = _candidate("mean(close, 5)")
    with pytest.raises(CandidateError, match="重复 candidate_id"):
        build_candidate_expression_plan((candidate, candidate), _context())

    with pytest.raises(CandidateError, match="CandidateDefinition"):
        build_candidate_expression_plan(("not-a-candidate",), _context())


def test_plan_rejects_catalog_drift_and_candidate_semantic_drift():
    """冻结目录或候选身份漂移时 fail-closed。"""

    candidate = _candidate("mean(close, 5)")
    wrong_context = FrozenContext(data_fingerprint="a" * 64)
    with pytest.raises(CatalogDriftError, match="冻结上下文"):
        build_candidate_expression_plan((candidate,), wrong_context)

    wrong_version = replace(load_catalog(), data_version="other_version")
    with pytest.raises(CatalogDriftError, match="冻结上下文"):
        build_candidate_expression_plan(
            (candidate,),
            _context(),
            catalog=wrong_version,
        )

    drifted = replace(candidate, lookback=candidate.lookback + 1)
    with pytest.raises(CandidateError, match="语义哈希"):
        build_candidate_expression_plan((drifted,), _context())


def test_plan_honors_excluded_fields():
    """计划重新执行冻结上下文的字段排除规则。"""

    candidate = _candidate("mean(close, 5)")
    context = _context(excluded_fields=("close",))
    with pytest.raises(ExpressionValidationError, match="排除"):
        build_candidate_expression_plan((candidate,), context)


def test_plan_rejects_uncontrolled_expression():
    """DAG 入口继续执行受控 DSL 安全校验。"""

    candidate = CandidateDefinition(
        name="MINE_UNCONTROLLED_00000001",
        mechanism="非法表达式",
        description="不应进入计划",
        canonical_expression="close; DROP TABLE md.daily_hfq",
        direction=1,
        lookback=0,
        dependencies=("close",),
        output_type="number",
        steps=("触发校验",),
        discovery_round=1,
        discovery_index=1,
    )
    with pytest.raises(ExpressionValidationError):
        build_candidate_expression_plan((candidate,), _context())
