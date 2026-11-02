"""自由探索首轮方向的确定性生成和假设转换。"""

from __future__ import annotations

import random
from typing import Any, Mapping

from mining.errors import FreeExplorationError, MiningContractError
from mining.free_exploration_contracts import (
    EXPRESSION_COMPLEXITY_VALUES,
    EXPRESSION_DEPTH_BY_COMPLEXITY,
    ResearchDirection,
)
from mining.free_exploration_profiles import (
    FREE_EXPLORATION_PROFILE_HASH,
    MECHANISM_PROFILES,
    MechanismProfile,
)
from mining.search_cards import HypothesisCard
from mining.search_context import FrozenSearchContext
from mining.search_contract_types import FREE_EXPLORATION_INPUT_MODE
from mining.search_run import SearchRunRequest

DIRECTION_HYPOTHESIS_ROLES = ("initial", "branch")


def _supported_fields(
    profile: MechanismProfile,
    context: FrozenSearchContext,
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    by_name = {
        str(item.get("name")): item
        for item in context.field_catalog
        if isinstance(item, Mapping) and item.get("name")
    }
    return tuple(
        (name, by_name[name])
        for name in profile.fields
        if name in by_name
    )


def _direction_from_profile(
    *,
    profile: MechanismProfile,
    dependency_name: str,
    dependency: Mapping[str, Any],
    expression_complexity: str,
    request: SearchRunRequest,
    context: FrozenSearchContext,
) -> ResearchDirection:
    """把冻结机制和字段组合转换为可审计方向。"""

    label = str(dependency.get("label") or dependency_name)
    return ResearchDirection(
        run_id=request.run_id,
        mechanism_category=profile.category,
        statement=profile.statement_template.format(label=label),
        mechanism=profile.mechanism,
        expected_direction=profile.expected_direction,
        falsification_criteria=profile.falsification_criteria,
        data_dependencies=(dependency_name,),
        expression_complexity=expression_complexity,
        max_expression_depth=EXPRESSION_DEPTH_BY_COMPLEXITY[
            expression_complexity
        ],
        seed=request.effective_exploration_seed,
        seed_source=(
            "request" if request.exploration_seed is not None else "request_hash"
        ),
        request_hash=request.request_hash,
        context_hash=context.context_hash,
        profile_hash=FREE_EXPLORATION_PROFILE_HASH,
    )


def generate_initial_direction(
    request: SearchRunRequest,
    context: FrozenSearchContext,
) -> ResearchDirection:
    """从冻结字段目录确定一个可复现的首轮研究方向。"""

    if request.input_mode != FREE_EXPLORATION_INPUT_MODE:
        raise MiningContractError("只有自由探索请求可以生成初始方向")
    if request.budget != context.budget:
        raise MiningContractError("请求预算与冻结上下文预算不一致")
    if not context.operator_catalog:
        raise FreeExplorationError("自由探索上下文没有可用算子")

    supported: list[
        tuple[MechanismProfile, tuple[tuple[str, Mapping[str, Any]], ...]]
    ] = []
    for profile in MECHANISM_PROFILES:
        fields = _supported_fields(profile, context)
        if fields:
            supported.append((profile, fields))
    if not supported:
        raise FreeExplorationError("冻结字段目录没有可支持的研究机制")

    seed = request.effective_exploration_seed
    rng = random.Random(seed)
    profile, available_fields = rng.choice(tuple(supported))
    dependency_name, dependency = rng.choice(available_fields)
    expression_complexity = rng.choice(EXPRESSION_COMPLEXITY_VALUES)
    return _direction_from_profile(
        profile=profile,
        dependency_name=str(dependency_name),
        dependency=dependency,
        expression_complexity=expression_complexity,
        request=request,
        context=context,
    )


def direction_to_hypothesis(
    direction: ResearchDirection,
    *,
    turn_index: int = 1,
    branch_id: str = "branch-0001",
    parent_hypothesis_id: str | None = None,
    direction_role: str = "initial",
) -> HypothesisCard:
    """把研究方向转换为绑定指定 turn 和分支的假设卡。"""

    if not isinstance(direction, ResearchDirection):
        raise MiningContractError("direction 必须是 ResearchDirection")
    if direction_role not in DIRECTION_HYPOTHESIS_ROLES:
        raise MiningContractError(
            "direction_role 必须是 initial 或 branch"
        )
    return HypothesisCard(
        run_id=direction.run_id,
        turn_index=turn_index,
        branch_id=branch_id,
        statement=direction.statement,
        mechanism=direction.mechanism,
        expected_direction=direction.expected_direction,
        falsification_criteria=direction.falsification_criteria,
        parent_hypothesis_id=parent_hypothesis_id,
        metadata={
            "source": "free_exploration",
            "direction_role": direction_role,
            "research_direction": direction.to_dict(),
        },
    )


def direction_from_hypothesis(
    hypothesis: HypothesisCard,
) -> ResearchDirection | None:
    """从自由探索假设恢复方向，手工假设返回空。"""

    metadata = hypothesis.metadata
    source = metadata.get("source")
    if source not in {"free_exploration", "free_exploration_refine"}:
        return None
    raw = metadata.get("research_direction")
    if not isinstance(raw, Mapping):
        raise MiningContractError("自由探索假设缺少 research_direction")
    direction = ResearchDirection.from_dict(raw)
    if direction.run_id != hypothesis.run_id:
        raise MiningContractError("自由探索方向与假设卡不属于同一运行")
    role = metadata.get("direction_role")
    if (
        source == "free_exploration"
        and role is not None
        and role not in DIRECTION_HYPOTHESIS_ROLES
    ):
        raise MiningContractError("自由探索假设包含未知 direction_role")
    if source == "free_exploration" and (
        direction.statement != hypothesis.statement
        or direction.mechanism != hypothesis.mechanism
        or direction.expected_direction != hypothesis.expected_direction
        or direction.falsification_criteria
        != hypothesis.falsification_criteria
    ):
        raise MiningContractError("自由探索方向与假设卡内容不一致")
    if source == "free_exploration_refine" and role != "refine":
        raise MiningContractError("自由探索修订假设缺少 refine 方向角色")
    return direction


__all__ = [
    "DIRECTION_HYPOTHESIS_ROLES",
    "direction_from_hypothesis",
    "direction_to_hypothesis",
    "generate_initial_direction",
]
