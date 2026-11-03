"""研究请求、预算约束与冻结上下文。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from mining.contract_types import (
    DATA_VERSION,
    FULL_EVAL_END,
    FULL_EVAL_START,
    HOLDOUT_END,
    HOLDOUT_START,
    IDEMPOTENCY_PATTERN,
    LABEL_NAME,
    PRICE_CHANNEL,
    PROFILE_PATTERN,
    RESEARCH_END,
    RESEARCH_START,
    SCHEMA_VERSION,
    UNIVERSE,
    ensure_string_tuple,
    mapping_sequence,
    sha256_json,
    sha256_text,
    stable_id,
    text,
    utc_now_iso,
    validate_date,
    validate_utc_timestamp,
)
from mining.errors import MiningContractError


@dataclass(frozen=True)
class MiningConstraints:
    """一次运行冻结的资源与范围约束。"""

    universe: str = UNIVERSE
    label: str = LABEL_NAME
    price_channel: str = PRICE_CHANNEL
    direction: str = "either"
    max_rounds: int = 3
    max_candidates: int = 12
    full_eval_top_k: int = 4
    max_llm_calls: int = 30
    max_expression_depth: int = 6
    min_coverage: float = 0.30
    max_failure_rate: float = 0.75
    min_quality_checks_for_failure_stop: int = 4
    seed: int = 42
    excluded_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.universe != UNIVERSE:
            raise MiningContractError(f"首版股票池固定为 {UNIVERSE}")
        if self.label != LABEL_NAME:
            raise MiningContractError(f"首版标签固定为 {LABEL_NAME}")
        if self.price_channel != PRICE_CHANNEL:
            raise MiningContractError(f"首版价格通道固定为 {PRICE_CHANNEL}")
        if self.direction not in {"either", "positive", "negative"}:
            raise MiningContractError("direction 必须是 either、positive 或 negative")
        for name, value, upper in (
            ("max_rounds", self.max_rounds, 10),
            ("max_candidates", self.max_candidates, 40),
            ("max_llm_calls", self.max_llm_calls, 60),
            ("max_expression_depth", self.max_expression_depth, 8),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise MiningContractError(f"{name} 必须在 1～{upper}")
        if self.max_expression_depth < 2:
            raise MiningContractError("max_expression_depth 必须在 2～8")
        if not 1 <= self.full_eval_top_k <= min(8, self.max_candidates):
            raise MiningContractError(
                "full_eval_top_k 必须在 1～min(8, max_candidates)"
            )
        if (
            isinstance(self.min_coverage, bool)
            or not isinstance(self.min_coverage, (int, float))
            or not 0 <= self.min_coverage <= 1
        ):
            raise MiningContractError("min_coverage 必须在 0～1")
        if (
            isinstance(self.max_failure_rate, bool)
            or not isinstance(self.max_failure_rate, (int, float))
            or not 0 < self.max_failure_rate <= 1
        ):
            raise MiningContractError("max_failure_rate 必须在 (0, 1]")
        if (
            isinstance(self.min_quality_checks_for_failure_stop, bool)
            or not isinstance(self.min_quality_checks_for_failure_stop, int)
            or not 1 <= self.min_quality_checks_for_failure_stop <= self.max_candidates
        ):
            raise MiningContractError(
                "min_quality_checks_for_failure_stop 必须在 1～max_candidates"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise MiningContractError("seed 必须是整数")
        object.__setattr__(
            self,
            "excluded_fields",
            tuple(sorted(ensure_string_tuple(self.excluded_fields, "excluded_fields"))),
        )

    def to_dict(self) -> dict[str, Any]:
        """转为稳定持久化字典。"""

        payload = asdict(self)
        payload["excluded_fields"] = list(self.excluded_fields)
        return payload


@dataclass(frozen=True)
class MiningRunRequest:
    """提交一次挖掘运行的幂等请求。"""

    idempotency_key: str
    hypothesis: str
    constraints: MiningConstraints = field(default_factory=MiningConstraints)
    llm_profile: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.idempotency_key, str):
            raise MiningContractError("idempotency_key 必须是字符串")
        if not IDEMPOTENCY_PATTERN.fullmatch(self.idempotency_key):
            raise MiningContractError(
                "idempotency_key 长度为 8～128，只能包含字母、数字、点、下划线、冒号和连字符"
            )
        object.__setattr__(
            self,
            "hypothesis",
            text(self.hypothesis, "hypothesis", minimum=8, maximum=4000),
        )
        if not isinstance(self.constraints, MiningConstraints):
            raise MiningContractError("constraints 必须是 MiningConstraints")
        if not isinstance(self.llm_profile, str) or not PROFILE_PATTERN.fullmatch(
            self.llm_profile
        ):
            raise MiningContractError(
                "llm_profile 只能包含字母、数字、下划线、点和连字符"
            )

    def material(self) -> dict[str, Any]:
        """返回参与幂等请求哈希的字段。"""

        return {
            "schema_version": SCHEMA_VERSION,
            "hypothesis": self.hypothesis,
            "constraints": self.constraints.to_dict(),
            "llm_profile": self.llm_profile,
        }

    @property
    def request_hash(self) -> str:
        """请求内容哈希。"""

        return sha256_json(self.material())

    @property
    def run_id(self) -> str:
        """由幂等键生成稳定运行 ID。"""

        return stable_id(
            "run",
            {"schema_version": SCHEMA_VERSION, "idempotency_key": self.idempotency_key},
        )


@dataclass(frozen=True)
class FrozenContext:
    """运行开始后不可改变的数据、规则和预算快照。"""

    data_version: str = DATA_VERSION
    data_fingerprint: str = ""
    label: str = LABEL_NAME
    universe: str = UNIVERSE
    price_channel: str = PRICE_CHANNEL
    research_start: str = RESEARCH_START
    research_end: str = RESEARCH_END
    full_eval_start: str = FULL_EVAL_START
    full_eval_end: str = FULL_EVAL_END
    holdout_start: str = HOLDOUT_START
    holdout_end: str = HOLDOUT_END
    llm_profile: str = "default"
    llm_model: str = "deepseek/deepseek-chat"
    field_catalog: tuple[Mapping[str, Any], ...] = ()
    operator_catalog: tuple[Mapping[str, Any], ...] = ()
    existing_factors: tuple[str, ...] = ()
    budget: MiningConstraints = field(default_factory=MiningConstraints)
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if self.data_version != DATA_VERSION:
            raise MiningContractError(f"首版数据版本固定为 {DATA_VERSION}")
        object.__setattr__(
            self,
            "data_fingerprint",
            sha256_text(self.data_fingerprint, "data_fingerprint"),
        )
        if (self.label, self.universe, self.price_channel) != (
            LABEL_NAME,
            UNIVERSE,
            PRICE_CHANNEL,
        ):
            raise MiningContractError("上下文的数据口径与首版冻结口径不一致")
        for name in (
            "research_start",
            "research_end",
            "full_eval_start",
            "full_eval_end",
            "holdout_start",
            "holdout_end",
        ):
            object.__setattr__(self, name, validate_date(getattr(self, name), name))
        if not (
            self.research_start <= self.full_eval_start <= self.full_eval_end
            <= self.research_end < self.holdout_start <= self.holdout_end
        ):
            raise MiningContractError("研究、完整评估和模块测试段时间窗口不连续或重叠非法")
        if not isinstance(self.budget, MiningConstraints):
            raise MiningContractError("budget 必须是 MiningConstraints")
        if not isinstance(self.llm_profile, str) or not PROFILE_PATTERN.fullmatch(
            self.llm_profile
        ):
            raise MiningContractError(
                "llm_profile 只能包含字母、数字、下划线、点和连字符"
            )
        object.__setattr__(
            self,
            "llm_model",
            text(self.llm_model, "llm_model", maximum=200),
        )
        object.__setattr__(
            self,
            "field_catalog",
            mapping_sequence(self.field_catalog, "field_catalog"),
        )
        object.__setattr__(
            self,
            "operator_catalog",
            mapping_sequence(self.operator_catalog, "operator_catalog"),
        )
        object.__setattr__(
            self,
            "existing_factors",
            tuple(sorted(ensure_string_tuple(self.existing_factors, "existing_factors"))),
        )
        object.__setattr__(
            self,
            "created_at",
            validate_utc_timestamp(self.created_at, "created_at"),
        )

    def material(self) -> dict[str, Any]:
        """返回参与上下文哈希的完整内容。"""

        return {
            "schema_version": SCHEMA_VERSION,
            "data_version": self.data_version,
            "data_fingerprint": self.data_fingerprint,
            "label": self.label,
            "universe": self.universe,
            "price_channel": self.price_channel,
            "research_start": self.research_start,
            "research_end": self.research_end,
            "full_eval_start": self.full_eval_start,
            "full_eval_end": self.full_eval_end,
            "holdout_start": self.holdout_start,
            "holdout_end": self.holdout_end,
            "llm_profile": self.llm_profile,
            "llm_model": self.llm_model,
            "field_catalog": list(self.field_catalog),
            "operator_catalog": list(self.operator_catalog),
            "existing_factors": list(self.existing_factors),
            "budget": self.budget.to_dict(),
        }

    @property
    def context_hash(self) -> str:
        """返回冻结上下文哈希。"""

        return sha256_json(self.material())

    @property
    def context_id(self) -> str:
        """返回稳定上下文 ID。"""

        return stable_id("ctx", {"context_hash": self.context_hash})

    @property
    def field_catalog_hash(self) -> str:
        """返回字段目录哈希。"""

        return sha256_json(list(self.field_catalog))

    @property
    def operator_catalog_hash(self) -> str:
        """返回算子目录哈希。"""

        return sha256_json(list(self.operator_catalog))
