"""挖掘 V2 的分层时序窗口、冻结上下文和角色可见性。"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Mapping

from common.paths import MINING_SEARCH_CONFIG_FILE
from mining.contract_types import (
    DATA_VERSION,
    LABEL_NAME,
    PRICE_CHANNEL,
    UNIVERSE,
    canonical_json,
    ensure_string_tuple,
    mapping_sequence,
    sha256_json,
    stable_id,
    text,
    utc_now_iso,
    validate_date,
    validate_utc_timestamp,
)
from mining.errors import MiningContractError, MiningVisibilityError
from mining.search_budget import SearchBudget
from mining.search_control import SearchControlPolicy
from mining.search_contract_types import (
    SEARCH_CONTEXT_SCHEMA_VERSION,
    SEARCH_DEVELOPMENT_END,
    SEARCH_DEVELOPMENT_START,
    SEARCH_FINAL_END,
    SEARCH_FINAL_START,
    SEARCH_HOLDOUT_END,
    SEARCH_HOLDOUT_START,
    SEARCH_SELECTION_END,
    SEARCH_SELECTION_START,
    hash_text,
    strict_fields,
    validate_date_range,
)

AGENT_ROLE = "agent"
CONTROLLER_ROLE = "controller"
EVALUATOR_ROLE = "evaluator"
HOLDOUT_ROLE = "holdout_executor"
SEARCH_WINDOW_ROLES = (
    AGENT_ROLE,
    CONTROLLER_ROLE,
    EVALUATOR_ROLE,
    HOLDOUT_ROLE,
)
_FORBIDDEN_AGENT_KEY = re.compile(
    r"(?<![a-z0-9])"
    r"(selection|final|holdout|test_segment|module_test)"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)
_DATE_RANGE_KEY = re.compile(r"^(.+)_(start|end)$")


@dataclass(frozen=True)
class SearchWindows:
    """四段互不重叠、按时间递增的研究窗口。"""

    development_start: str = SEARCH_DEVELOPMENT_START
    development_end: str = SEARCH_DEVELOPMENT_END
    selection_start: str = SEARCH_SELECTION_START
    selection_end: str = SEARCH_SELECTION_END
    final_start: str = SEARCH_FINAL_START
    final_end: str = SEARCH_FINAL_END
    holdout_start: str = SEARCH_HOLDOUT_START
    holdout_end: str = SEARCH_HOLDOUT_END

    def __post_init__(self) -> None:
        ordered: dict[str, str] = {}
        for name in self.__dataclass_fields__:
            ordered[name] = validate_date(getattr(self, name), name)
            object.__setattr__(self, name, ordered[name])
        pairs = (
            ("development", self.development_start, self.development_end),
            ("selection", self.selection_start, self.selection_end),
            ("final", self.final_start, self.final_end),
            ("holdout", self.holdout_start, self.holdout_end),
        )
        for name, start, end in pairs:
            validate_date_range(start, end, name)
        if not (
            self.development_end < self.selection_start
            and self.selection_end < self.final_start
            and self.final_end < self.holdout_start
        ):
            raise MiningContractError("研发、选择、最终研究和测试段必须按顺序且不重叠")

    @classmethod
    def from_dict(cls, value: Any) -> SearchWindows:
        """严格解析分层窗口。"""

        return cls(
            **strict_fields(value, "search_windows", set(cls.__dataclass_fields__))
        )

    def to_dict(self) -> dict[str, str]:
        """返回稳定持久化字典。"""

        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    def role_ranges(self, role: str) -> tuple[tuple[str, str], ...]:
        """返回角色可访问的日期范围。"""

        if role == AGENT_ROLE:
            return ((self.development_start, self.development_end),)
        if role == CONTROLLER_ROLE:
            return (
                (self.development_start, self.development_end),
                (self.selection_start, self.selection_end),
            )
        if role == EVALUATOR_ROLE:
            return ((self.final_start, self.final_end),)
        if role == HOLDOUT_ROLE:
            return ((self.holdout_start, self.holdout_end),)
        raise MiningVisibilityError(f"未知搜索访问角色：{role!r}")

    def assert_access(self, role: str, start: str, end: str) -> None:
        """确认请求日期范围完全位于角色可访问窗口内。"""

        left, right = validate_date_range(start, end, "requested")
        for allowed_start, allowed_end in self.role_ranges(role):
            if allowed_start <= left and right <= allowed_end:
                return
        raise MiningVisibilityError(
            f"{role} 无权访问日期范围 {left}～{right}"
        )


DEFAULT_SEARCH_WINDOWS = SearchWindows()


@dataclass(frozen=True)
class SearchConfig:
    """版本化 V2 搜索窗口和预算配置。"""

    windows: SearchWindows = field(default_factory=SearchWindows)
    budget: SearchBudget = field(default_factory=SearchBudget)
    control_policy: SearchControlPolicy = field(default_factory=SearchControlPolicy)
    schema_version: str = SEARCH_CONTEXT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_CONTEXT_SCHEMA_VERSION:
            raise MiningContractError(
                f"搜索配置 schema_version 必须为 {SEARCH_CONTEXT_SCHEMA_VERSION}"
            )
        if not isinstance(self.windows, SearchWindows):
            raise MiningContractError("搜索配置 windows 必须是 SearchWindows")
        if not isinstance(self.budget, SearchBudget):
            raise MiningContractError("搜索配置 budget 必须是 SearchBudget")
        if not isinstance(self.control_policy, SearchControlPolicy):
            raise MiningContractError(
                "搜索配置 control_policy 必须是 SearchControlPolicy"
            )

    @classmethod
    def from_dict(cls, value: Any) -> SearchConfig:
        """严格解析搜索配置。"""

        payload = strict_fields(
            value,
            "search_config",
            set(cls.__dataclass_fields__),
        )
        payload["windows"] = SearchWindows.from_dict(payload["windows"])
        payload["budget"] = SearchBudget.from_dict(payload["budget"])
        payload["control_policy"] = SearchControlPolicy.from_dict(
            payload["control_policy"]
        )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定持久化字典。"""

        return {
            "schema_version": self.schema_version,
            "windows": self.windows.to_dict(),
            "budget": self.budget.to_dict(),
            "control_policy": self.control_policy.to_dict(),
        }

    @property
    def policy_hash(self) -> str:
        """返回窗口、预算和配置版本的综合哈希。"""

        return sha256_json(self.to_dict())


def load_search_config(
    path: Path | str = MINING_SEARCH_CONFIG_FILE,
) -> SearchConfig:
    """读取 V2 搜索配置并拒绝非冻结窗口。"""

    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise MiningContractError(f"V2 搜索配置不存在：{target}") from error
    except json.JSONDecodeError as error:
        raise MiningContractError(f"V2 搜索配置 JSON 无法解析：{error}") from error
    config = SearchConfig.from_dict(payload)
    if config.windows != DEFAULT_SEARCH_WINDOWS:
        raise MiningContractError("V2 搜索窗口与冻结定义不一致")
    return config


def ensure_agent_visible(value: Any, *, path: str = "payload") -> None:
    """拒绝 Agent 载荷中的选择期、最终研究期或测试段字段。"""

    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if _FORBIDDEN_AGENT_KEY.search(lowered):
                raise MiningVisibilityError(f"{path} 不允许字段 {key!r}")
            ensure_agent_visible(item, path=f"{path}.{key}")
        for key, start in value.items():
            match = _DATE_RANGE_KEY.fullmatch(str(key))
            if match is None or match.group(2) != "start":
                continue
            end_key = f"{match.group(1)}_end"
            if end_key not in value:
                raise MiningVisibilityError(f"{path} 日期范围缺少 {end_key}")
            left, right = validate_date_range(start, value[end_key], str(match.group(1)))
            if left < SEARCH_DEVELOPMENT_START or right > SEARCH_DEVELOPMENT_END:
                raise MiningVisibilityError(
                    f"{path}.{match.group(1)} 超出开发期"
                )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ensure_agent_visible(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class FrozenSearchContext:
    """V2 运行开始后不可改变的窗口、规则和预算快照。"""

    data_version: str = DATA_VERSION
    data_fingerprint: str = ""
    label: str = LABEL_NAME
    universe: str = UNIVERSE
    price_channel: str = PRICE_CHANNEL
    windows: SearchWindows = field(default_factory=SearchWindows)
    llm_profile: str = "default"
    llm_model: str = "deepseek/deepseek-chat"
    field_catalog: tuple[Mapping[str, Any], ...] = ()
    operator_catalog: tuple[Mapping[str, Any], ...] = ()
    existing_factors: tuple[str, ...] = ()
    budget: SearchBudget = field(default_factory=SearchBudget)
    control_policy: SearchControlPolicy = field(default_factory=SearchControlPolicy)
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SEARCH_CONTEXT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_CONTEXT_SCHEMA_VERSION:
            raise MiningContractError(
                f"FrozenSearchContext schema_version 必须为 "
                f"{SEARCH_CONTEXT_SCHEMA_VERSION}"
            )
        if self.data_version != DATA_VERSION:
            raise MiningContractError(f"首版数据版本固定为 {DATA_VERSION}")
        if (self.label, self.universe, self.price_channel) != (
            LABEL_NAME,
            UNIVERSE,
            PRICE_CHANNEL,
        ):
            raise MiningContractError("V2 上下文的数据口径与冻结口径不一致")
        object.__setattr__(
            self,
            "data_fingerprint",
            hash_text(self.data_fingerprint, "data_fingerprint"),
        )
        if not isinstance(self.windows, SearchWindows):
            raise MiningContractError("windows 必须是 SearchWindows")
        if not isinstance(self.budget, SearchBudget):
            raise MiningContractError("budget 必须是 SearchBudget")
        if not isinstance(self.control_policy, SearchControlPolicy):
            raise MiningContractError(
                "control_policy 必须是 SearchControlPolicy"
            )
        if not isinstance(self.field_catalog, (list, tuple)):
            raise MiningContractError("field_catalog 必须是对象序列")
        if not isinstance(self.operator_catalog, (list, tuple)):
            raise MiningContractError("operator_catalog 必须是对象序列")
        object.__setattr__(
            self,
            "llm_profile",
            text(self.llm_profile, "llm_profile", maximum=40),
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

    @classmethod
    def from_config(
        cls,
        config: SearchConfig,
        *,
        data_fingerprint: str,
        llm_profile: str = "default",
        llm_model: str = "deepseek/deepseek-chat",
        field_catalog: tuple[Mapping[str, Any], ...] = (),
        operator_catalog: tuple[Mapping[str, Any], ...] = (),
        existing_factors: tuple[str, ...] = (),
        control_policy: SearchControlPolicy | None = None,
        created_at: str | None = None,
    ) -> FrozenSearchContext:
        """从配置冻结一次 V2 运行上下文。"""

        if config.windows != DEFAULT_SEARCH_WINDOWS:
            raise MiningContractError("V2 搜索窗口与冻结定义不一致")
        return cls(
            data_fingerprint=data_fingerprint,
            windows=config.windows,
            llm_profile=llm_profile,
            llm_model=llm_model,
            field_catalog=field_catalog,
            operator_catalog=operator_catalog,
            existing_factors=existing_factors,
            budget=config.budget,
            control_policy=control_policy or config.control_policy,
            created_at=created_at or utc_now_iso(),
        )

    @classmethod
    def from_dict(cls, value: Any) -> FrozenSearchContext:
        """严格解析已持久化的 V2 上下文。"""

        payload = strict_fields(
            value,
            "frozen_search_context",
            set(cls.__dataclass_fields__),
        )
        payload["windows"] = SearchWindows.from_dict(payload["windows"])
        payload["budget"] = SearchBudget.from_dict(payload["budget"])
        payload["control_policy"] = SearchControlPolicy.from_dict(
            payload["control_policy"]
        )
        payload["field_catalog"] = tuple(payload["field_catalog"])
        payload["operator_catalog"] = tuple(payload["operator_catalog"])
        payload["existing_factors"] = tuple(payload["existing_factors"])
        return cls(**payload)

    def material(self) -> dict[str, Any]:
        """返回参与上下文哈希的完整冻结内容。"""

        return {
            "schema_version": self.schema_version,
            "data_version": self.data_version,
            "data_fingerprint": self.data_fingerprint,
            "label": self.label,
            "universe": self.universe,
            "price_channel": self.price_channel,
            "windows": self.windows.to_dict(),
            "llm_profile": self.llm_profile,
            "llm_model": self.llm_model,
            "field_catalog": list(self.field_catalog),
            "operator_catalog": list(self.operator_catalog),
            "existing_factors": list(self.existing_factors),
            "budget": self.budget.to_dict(),
            "control_policy": self.control_policy.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        """返回含创建时间的持久化字典。"""

        return {**self.material(), "created_at": self.created_at}

    @property
    def context_hash(self) -> str:
        """返回冻结搜索上下文哈希。"""

        return sha256_json(self.material())

    @property
    def context_id(self) -> str:
        """返回稳定 V2 上下文 ID。"""

        return stable_id("searchctx", {"context_hash": self.context_hash})

    @property
    def field_catalog_hash(self) -> str:
        """返回字段目录哈希。"""

        return sha256_json(list(self.field_catalog))

    @property
    def operator_catalog_hash(self) -> str:
        """返回算子目录哈希。"""

        return sha256_json(list(self.operator_catalog))

    @property
    def windows_hash(self) -> str:
        """返回分层窗口哈希。"""

        return sha256_json(self.windows.to_dict())

    def worker_view(self) -> dict[str, Any]:
        """返回 Agent 唯一可见的脱敏上下文。"""

        payload = {
            "schema_version": self.schema_version,
            "context_hash": self.context_hash,
            "data_version": self.data_version,
            "data_fingerprint": self.data_fingerprint,
            "label": self.label,
            "universe": self.universe,
            "price_channel": self.price_channel,
            "development_start": self.windows.development_start,
            "development_end": self.windows.development_end,
            "llm_profile": self.llm_profile,
            "llm_model": self.llm_model,
            "field_catalog": list(self.field_catalog),
            "operator_catalog": list(self.operator_catalog),
            "existing_factors": list(self.existing_factors),
            "budget": self.budget.to_dict(),
            "control_policy": self.control_policy.to_dict(),
        }
        ensure_agent_visible(payload)
        return payload

    def assert_worker_payload(self, payload: Any) -> None:
        """在发送给 Agent 前校验载荷不含受控窗口或证据。"""

        ensure_agent_visible(payload)
        if not isinstance(payload, Mapping):
            raise MiningContractError("Worker 载荷根节点必须是 JSON 对象")
        encoded = canonical_json(payload)
        for secret in (
            self.windows.selection_start,
            self.windows.selection_end,
            self.windows.final_start,
            self.windows.final_end,
            self.windows.holdout_start,
            self.windows.holdout_end,
        ):
            if secret in encoded:
                raise MiningVisibilityError("Worker 载荷包含非开发期日期")
