"""挖掘 V2 搜索运行请求。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from mining.contract_types import (
    IDEMPOTENCY_PATTERN,
    PROFILE_PATTERN,
    sha256_json,
    stable_id,
    text,
)
from mining.errors import MiningContractError
from mining.search_budget import SearchBudget
from mining.free_exploration_contracts import EXPLORATION_SEED_MAX
from mining.search_contract_types import (
    FREE_EXPLORATION_INPUT_MODE,
    HYPOTHESIS_INPUT_MODE,
    SEARCH_INPUT_MODE_VALUES,
    SEARCH_SCHEMA_VERSION,
    choice,
)

FREE_EXPLORATION_DISPLAY_LABEL = "自由探索"


@dataclass(frozen=True)
class SearchRunRequest:
    """提交一次 V2 搜索运行的幂等请求。"""

    idempotency_key: str
    hypothesis: str | None
    budget: SearchBudget = field(default_factory=SearchBudget)
    llm_profile: str = "default"
    schema_version: str = SEARCH_SCHEMA_VERSION
    input_mode: str = HYPOTHESIS_INPUT_MODE
    exploration_seed: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_SCHEMA_VERSION:
            raise MiningContractError(
                f"SearchRunRequest schema_version 必须为 {SEARCH_SCHEMA_VERSION}"
            )
        if not isinstance(self.idempotency_key, str) or not IDEMPOTENCY_PATTERN.fullmatch(
            self.idempotency_key
        ):
            raise MiningContractError(
                "idempotency_key 长度为 8～128，只能包含字母、数字、点、下划线、冒号和连字符"
            )
        object.__setattr__(
            self,
            "input_mode",
            choice(self.input_mode, "input_mode", SEARCH_INPUT_MODE_VALUES),
        )
        if self.input_mode == HYPOTHESIS_INPUT_MODE:
            object.__setattr__(
                self,
                "hypothesis",
                text(self.hypothesis, "hypothesis", minimum=8, maximum=4000),
            )
        elif self.hypothesis is not None:
            raise MiningContractError("free_exploration 模式不得提供 hypothesis")
        if not isinstance(self.budget, SearchBudget):
            raise MiningContractError("budget 必须是 SearchBudget")
        if not isinstance(self.llm_profile, str) or not PROFILE_PATTERN.fullmatch(
            self.llm_profile
        ):
            raise MiningContractError(
                "llm_profile 只能包含字母、数字、下划线、点和连字符"
            )
        if self.exploration_seed is not None:
            if self.input_mode != FREE_EXPLORATION_INPUT_MODE:
                raise MiningContractError(
                    "hypothesis 模式不得提供 exploration_seed"
                )
            if (
                isinstance(self.exploration_seed, bool)
                or not isinstance(self.exploration_seed, int)
                or not 0 <= self.exploration_seed <= EXPLORATION_SEED_MAX
            ):
                raise MiningContractError(
                    "exploration_seed 必须是 0～2^64-1 的整数"
                )

    def material(self) -> dict[str, Any]:
        """返回参与幂等请求哈希的字段。"""

        payload = {
            "schema_version": self.schema_version,
            "hypothesis": self.hypothesis,
            "budget": self.budget.to_dict(),
            "llm_profile": self.llm_profile,
        }
        if self.input_mode == FREE_EXPLORATION_INPUT_MODE:
            payload["input_mode"] = self.input_mode
            if self.exploration_seed is not None:
                payload["exploration_seed"] = self.exploration_seed
        return payload

    @property
    def request_hash(self) -> str:
        """返回请求内容哈希。"""

        return sha256_json(self.material())

    @property
    def run_id(self) -> str:
        """返回与 V1 不冲突的稳定运行 ID。"""

        return stable_id(
            "searchrun",
            {
                "schema_version": self.schema_version,
                "idempotency_key": self.idempotency_key,
            },
        )

    @property
    def display_hypothesis(self) -> str:
        """返回页面和列表可安全展示的输入标签。"""

        return self.hypothesis or FREE_EXPLORATION_DISPLAY_LABEL

    @property
    def effective_exploration_seed(self) -> int:
        """返回自由探索使用的有效种子。"""

        if self.input_mode != FREE_EXPLORATION_INPUT_MODE:
            raise MiningContractError("只有自由探索模式具有有效探索种子")
        if self.exploration_seed is not None:
            return self.exploration_seed
        return int(self.request_hash[:16], 16)

    @classmethod
    def from_dict(cls, value: Any) -> SearchRunRequest:
        """严格解析请求，并兼容缺少 input_mode 的旧字典。"""

        if not isinstance(value, Mapping):
            raise MiningContractError("search_run_request 必须是 JSON 对象")
        expected = set(cls.__dataclass_fields__)
        actual = set(value)
        missing = expected - actual - {"exploration_seed", "input_mode"}
        unknown = actual - expected
        if missing or unknown:
            raise MiningContractError(
                f"search_run_request 字段不匹配：缺少 {sorted(missing)}，"
                f"未知 {sorted(unknown)}"
            )
        payload = dict(value)
        payload.setdefault("input_mode", HYPOTHESIS_INPUT_MODE)
        payload.setdefault("exploration_seed", None)
        payload["budget"] = SearchBudget.from_dict(payload["budget"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定持久化字典。"""

        return {
            "input_mode": self.input_mode,
            "hypothesis": self.hypothesis,
            "budget": self.budget.to_dict(),
            "llm_profile": self.llm_profile,
            "schema_version": self.schema_version,
            "exploration_seed": self.exploration_seed,
        }
