"""挖掘 V2 搜索运行请求。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mining.contract_types import (
    IDEMPOTENCY_PATTERN,
    PROFILE_PATTERN,
    sha256_json,
    stable_id,
    text,
)
from mining.errors import MiningContractError
from mining.search_budget import SearchBudget
from mining.search_contract_types import SEARCH_SCHEMA_VERSION, strict_fields


@dataclass(frozen=True)
class SearchRunRequest:
    """提交一次 V2 搜索运行的幂等请求。"""

    idempotency_key: str
    hypothesis: str
    budget: SearchBudget = field(default_factory=SearchBudget)
    llm_profile: str = "default"
    schema_version: str = SEARCH_SCHEMA_VERSION

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
            "hypothesis",
            text(self.hypothesis, "hypothesis", minimum=8, maximum=4000),
        )
        if not isinstance(self.budget, SearchBudget):
            raise MiningContractError("budget 必须是 SearchBudget")
        if not isinstance(self.llm_profile, str) or not PROFILE_PATTERN.fullmatch(
            self.llm_profile
        ):
            raise MiningContractError(
                "llm_profile 只能包含字母、数字、下划线、点和连字符"
            )

    def material(self) -> dict[str, Any]:
        """返回参与幂等请求哈希的字段。"""

        return {
            "schema_version": self.schema_version,
            "hypothesis": self.hypothesis,
            "budget": self.budget.to_dict(),
            "llm_profile": self.llm_profile,
        }

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

    @classmethod
    def from_dict(cls, value: Any) -> SearchRunRequest:
        """严格解析持久化请求。"""

        payload = strict_fields(
            value,
            "search_run_request",
            set(cls.__dataclass_fields__),
        )
        payload["budget"] = SearchBudget.from_dict(payload["budget"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定持久化字典。"""

        return self.material()
