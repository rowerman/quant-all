"""自由探索方向池、语义判重和可恢复的分支方向规划。"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable, Mapping

from mining.contract_types import identifier, sha256_json
from mining.errors import FreeExplorationError, MiningContractError
from mining.free_exploration import (
    _direction_from_profile,
    _supported_fields,
    direction_from_hypothesis,
)
from mining.free_exploration_contracts import (
    EXPRESSION_COMPLEXITY_VALUES,
    ResearchDirection,
)
from mining.free_exploration_profiles import (
    FREE_EXPLORATION_PROFILE_HASH,
    MECHANISM_PROFILES,
    MechanismProfile,
)
from mining.search_cards import HypothesisCard
from mining.search_context import FrozenSearchContext
from mining.search_contract_types import FREE_EXPLORATION_INPUT_MODE, hash_text
from mining.search_run import SearchRunRequest

FREE_EXPLORATION_POOL_SCHEMA_VERSION = "mining_free_exploration_pool_v1"


@dataclass(frozen=True)
class DuplicateDirectionAttempt:
    """方向池中因语义重复而被跳过的方向。"""

    direction: ResearchDirection
    duplicate_of_direction_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.direction, ResearchDirection):
            raise MiningContractError(
                "duplicate direction attempt 必须包含 ResearchDirection"
            )
        object.__setattr__(
            self,
            "duplicate_of_direction_id",
            identifier(
                self.duplicate_of_direction_id,
                "duplicate_of_direction_id",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """返回尝试台账可安全持久化的摘要。"""

        return {
            "direction": self.direction.to_dict(),
            "semantic_hash": self.direction.semantic_hash,
            "duplicate_of_direction_id": self.duplicate_of_direction_id,
        }


@dataclass(frozen=True)
class BranchDirectionPlan:
    """一次确定性的下一分支方向规划。"""

    direction: ResearchDirection
    duplicate_attempts: tuple[DuplicateDirectionAttempt, ...]
    pool_hash: str
    schema_version: str = FREE_EXPLORATION_POOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FREE_EXPLORATION_POOL_SCHEMA_VERSION:
            raise MiningContractError(
                "direction plan schema_version 必须为 "
                f"{FREE_EXPLORATION_POOL_SCHEMA_VERSION}"
            )
        if not isinstance(self.direction, ResearchDirection):
            raise MiningContractError("direction plan 缺少 ResearchDirection")
        object.__setattr__(
            self,
            "duplicate_attempts",
            tuple(self.duplicate_attempts),
        )
        if any(
            not isinstance(item, DuplicateDirectionAttempt)
            for item in self.duplicate_attempts
        ):
            raise MiningContractError(
                "duplicate_attempts 每项必须是 DuplicateDirectionAttempt"
            )
        object.__setattr__(
            self,
            "pool_hash",
            hash_text(self.pool_hash, "pool_hash"),
        )

    def material(self) -> dict[str, Any]:
        """返回排除规划时间的稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.direction.run_id,
            "direction_semantic_hash": self.direction.semantic_hash,
            "duplicate_attempts": [
                {
                    "semantic_hash": item.direction.semantic_hash,
                    "duplicate_of_direction_id": item.duplicate_of_direction_id,
                }
                for item in self.duplicate_attempts
            ],
            "pool_hash": self.pool_hash,
        }

    @property
    def plan_hash(self) -> str:
        """返回方向规划内容哈希。"""

        return sha256_json(self.material())

    def to_dict(self) -> dict[str, Any]:
        """返回事件和测试使用的稳定字典。"""

        return {
            **self.material(),
            "direction": self.direction.to_dict(),
            "duplicate_attempts": [
                item.to_dict() for item in self.duplicate_attempts
            ],
            "plan_hash": self.plan_hash,
        }


def _validate_inputs(
    request: SearchRunRequest,
    context: FrozenSearchContext,
) -> None:
    if not isinstance(request, SearchRunRequest):
        raise MiningContractError("request 必须是 SearchRunRequest")
    if not isinstance(context, FrozenSearchContext):
        raise MiningContractError("context 必须是 FrozenSearchContext")
    if request.input_mode != FREE_EXPLORATION_INPUT_MODE:
        raise MiningContractError("只有自由探索请求可以规划分支方向")
    if request.budget != context.budget:
        raise MiningContractError("请求预算与冻结上下文预算不一致")
    if not context.operator_catalog:
        raise FreeExplorationError("自由探索上下文没有可用算子")


def _supported_profiles(
    context: FrozenSearchContext,
) -> tuple[
    tuple[MechanismProfile, tuple[tuple[str, Mapping[str, Any]], ...]],
    ...,
]:
    supported = []
    for profile in MECHANISM_PROFILES:
        fields = _supported_fields(profile, context)
        if fields:
            supported.append((profile, fields))
    if not supported:
        raise FreeExplorationError("冻结字段目录没有可支持的研究机制")
    return tuple(supported)


def build_direction_pool(
    request: SearchRunRequest,
    context: FrozenSearchContext,
) -> tuple[ResearchDirection, ...]:
    """按机制轮转顺序构造冻结且可复现的方向池。"""

    _validate_inputs(request, context)
    supported = list(_supported_profiles(context))
    seed = request.effective_exploration_seed
    random.Random(seed).shuffle(supported)
    queues: list[
        tuple[
            MechanismProfile,
            list[tuple[str, Mapping[str, Any], str]],
        ]
    ] = []
    for profile, fields in supported:
        variants = [
            (field_name, dependency, complexity)
            for field_name, dependency in fields
            for complexity in EXPRESSION_COMPLEXITY_VALUES
        ]
        variants.sort(
            key=lambda item: sha256_json(
                {
                    "schema_version": FREE_EXPLORATION_POOL_SCHEMA_VERSION,
                    "seed": seed,
                    "mechanism_category": profile.category,
                    "field": item[0],
                    "complexity": item[2],
                }
            )
        )
        queues.append((profile, variants))

    pool: list[ResearchDirection] = []
    while any(variants for _, variants in queues):
        for profile, variants in queues:
            if not variants:
                continue
            field_name, dependency, complexity = variants.pop(0)
            pool.append(
                _direction_from_profile(
                    profile=profile,
                    dependency_name=field_name,
                    dependency=dependency,
                    expression_complexity=complexity,
                    request=request,
                    context=context,
                )
            )
    return tuple(pool)


def direction_pool_hash(directions: Iterable[ResearchDirection]) -> str:
    """返回完整方向池的稳定哈希。"""

    items = tuple(directions)
    return sha256_json(
        {
            "schema_version": FREE_EXPLORATION_POOL_SCHEMA_VERSION,
            "profile_hash": FREE_EXPLORATION_PROFILE_HASH,
            "semantic_hashes": [item.semantic_hash for item in items],
        }
    )


def _direction_usage(
    hypotheses: Iterable[HypothesisCard | Mapping[str, Any]],
) -> dict[str, str]:
    usage: dict[str, str] = {}
    for value in hypotheses:
        hypothesis = (
            value
            if isinstance(value, HypothesisCard)
            else HypothesisCard.from_dict(dict(value))
        )
        direction = direction_from_hypothesis(hypothesis)
        if direction is not None:
            usage.setdefault(direction.semantic_hash, direction.direction_id)
    return usage


def plan_next_branch_direction(
    request: SearchRunRequest,
    context: FrozenSearchContext,
    prior_hypotheses: Iterable[HypothesisCard | Mapping[str, Any]],
) -> BranchDirectionPlan:
    """选择下一个语义唯一的分支方向，并保留跳过的重复尝试。"""

    pool = build_direction_pool(request, context)
    used = _direction_usage(prior_hypotheses)
    duplicates: list[DuplicateDirectionAttempt] = []
    selected: ResearchDirection | None = None
    for direction in pool:
        previous_id = used.get(direction.semantic_hash)
        if previous_id is not None:
            duplicates.append(
                DuplicateDirectionAttempt(
                    direction=direction,
                    duplicate_of_direction_id=previous_id,
                )
            )
            continue
        selected = direction
        break
    if selected is None:
        raise FreeExplorationError("冻结方向池没有未使用的研究方向")
    return BranchDirectionPlan(
        direction=selected,
        duplicate_attempts=tuple(duplicates),
        pool_hash=direction_pool_hash(pool),
    )


__all__ = [
    "BranchDirectionPlan",
    "DuplicateDirectionAttempt",
    "FREE_EXPLORATION_POOL_SCHEMA_VERSION",
    "build_direction_pool",
    "direction_pool_hash",
    "plan_next_branch_direction",
]
