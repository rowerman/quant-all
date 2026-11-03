"""候选因子定义及语义、内容哈希。"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping

from mining.contract_types import (
    SCHEMA_VERSION,
    ensure_string_tuple,
    identifier,
    mapping,
    sha256_json,
    stable_id,
    text,
)
from mining.errors import CandidateError


@dataclass(frozen=True)
class CandidateDefinition:
    """Worker 或测试提交的不可变候选定义。"""

    name: str
    mechanism: str
    description: str
    canonical_expression: str
    direction: int
    lookback: int
    dependencies: tuple[str, ...]
    output_type: str
    steps: tuple[str, ...]
    discovery_round: int
    discovery_index: int
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", text(self.name, "name", maximum=120))
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", self.name):
            raise CandidateError("候选名必须以大写字母开头且只含大写字母、数字和下划线")
        object.__setattr__(
            self,
            "mechanism",
            text(self.mechanism, "mechanism", maximum=200),
        )
        object.__setattr__(
            self,
            "description",
            text(self.description, "description", maximum=2000),
        )
        object.__setattr__(
            self,
            "canonical_expression",
            text(
                self.canonical_expression,
                "canonical_expression",
                maximum=12000,
            ),
        )
        if self.direction not in (-1, 1):
            raise CandidateError("direction 必须是 -1 或 1")
        if isinstance(self.lookback, bool) or not isinstance(self.lookback, int):
            raise CandidateError("lookback 必须是整数")
        if not 0 <= self.lookback <= 10000:
            raise CandidateError("lookback 必须在 0～10000")
        object.__setattr__(
            self,
            "dependencies",
            tuple(sorted(ensure_string_tuple(self.dependencies, "dependencies"))),
        )
        if not self.dependencies:
            raise CandidateError("候选至少依赖一个字段")
        object.__setattr__(
            self,
            "output_type",
            identifier(self.output_type, "output_type"),
        )
        object.__setattr__(
            self,
            "steps",
            ensure_string_tuple(self.steps, "steps", minimum=1, maximum=100),
        )
        for name, value in (
            ("discovery_round", self.discovery_round),
            ("discovery_index", self.discovery_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CandidateError(f"{name} 必须是正整数")
        object.__setattr__(self, "parameters", mapping(self.parameters, "parameters"))

    def semantic_material(self) -> dict[str, Any]:
        """返回只决定计算语义的哈希输入。"""

        return {
            "schema_version": SCHEMA_VERSION,
            "canonical_expression": self.canonical_expression,
            "direction": self.direction,
            "lookback": self.lookback,
            "dependencies": list(self.dependencies),
            "output_type": self.output_type,
            "parameters": self.parameters,
        }

    def content_material(self) -> dict[str, Any]:
        """返回审核时绑定、含展示说明的完整哈希输入。"""

        return {
            "schema_version": SCHEMA_VERSION,
            "semantic_hash": self.semantic_hash,
            "name": self.name,
            "mechanism": self.mechanism,
            "description": self.description,
            "steps": list(self.steps),
        }

    @property
    def semantic_hash(self) -> str:
        """返回候选计算语义哈希。"""

        return sha256_json(self.semantic_material())

    @property
    def content_hash(self) -> str:
        """返回候选完整内容哈希。"""

        return sha256_json(self.content_material())

    @property
    def candidate_id(self) -> str:
        """返回由完整内容确定的稳定候选 ID。"""

        return stable_id("cand", {"content_hash": self.content_hash})
