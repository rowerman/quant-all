"""目标权重提供者的注册、发现与校验。

提供者是用户写在 ``backtest/strategies/`` 下的普通 Python 对象，只负责把行情和因子
转换成信号日目标权重。执行、风险约束、交易规则和账本仍由回测引擎统一负责。
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd

from backtest.errors import SpecError
from backtest.execution_state import ProviderState
from backtest.logging_conf import get_logger
from backtest.specs import ProviderSpec, SignalSpec, StrategySpec

if TYPE_CHECKING:
    from backtest.data import MarketData


@runtime_checkable
class TargetWeightProvider(Protocol):
    """目标权重提供者协议。

    ``eligible`` 是引擎按统一股票池口径算好的布尔矩阵；提供者只能给 True 的格子
    正权重。``build`` 必须返回与行情面板同形的权重矩阵。

    需要真实持仓的提供者额外声明 ``uses_position_state = True``，此时 ``build``
    接收第四个关键字参数 ``state``；状态由执行层账本生成，提供者不得自行模拟持仓。
    """

    name: str
    version: str
    description: str
    required_factors: tuple[str, ...]

    def build(
        self,
        data: "MarketData",
        eligible: pd.DataFrame,
        params: dict[str, Any],
        state: ProviderState | None = None,
    ) -> pd.DataFrame:
        """返回信号日目标权重。"""


_REGISTRY: dict[str, TargetWeightProvider] = {}
_ORIGINS: dict[str, str] = {}
_DISCOVERED = False
logger = get_logger("backtest.providers")


def _validate_provider(provider: TargetWeightProvider) -> None:
    """检查提供者对象的基本属性。"""
    for name in ("name", "version", "description"):
        value = getattr(provider, name, None)
        if not isinstance(value, str) or not value.strip():
            raise SpecError(f"目标权重提供者缺少有效的 {name}")
    factors = getattr(provider, "required_factors", None)
    if not isinstance(factors, (tuple, list)) or any(
        not isinstance(item, str) or not item for item in factors
    ):
        raise SpecError("目标权重提供者 required_factors 必须是字符串序列")
    if not callable(getattr(provider, "build", None)):
        raise SpecError(
            "目标权重提供者必须实现 build(data, eligible, params[, state])"
        )
    uses_state = getattr(provider, "uses_position_state", False)
    if not isinstance(uses_state, bool):
        raise SpecError("目标权重提供者 uses_position_state 必须是布尔值")
    if uses_state:
        try:
            signature = inspect.signature(provider.build)
        except (TypeError, ValueError) as error:
            raise SpecError(
                f"目标权重提供者 {provider.name} 的 build 签名无法检查"
            ) from error
        accepts_state = "state" in signature.parameters or any(
            item.kind is inspect.Parameter.VAR_KEYWORD
            for item in signature.parameters.values()
        )
        if not accepts_state:
            raise SpecError(
                f"目标权重提供者 {provider.name} 声明 uses_position_state，"
                "但 build 没有 state 参数"
            )


def register_provider(provider: TargetWeightProvider | type[TargetWeightProvider]) -> TargetWeightProvider:
    """注册一个提供者；重复名称直接报错。"""
    if isinstance(provider, type):
        provider = provider()
    _validate_provider(provider)
    name = str(provider.name).strip()
    origin = type(provider).__module__
    existing = _REGISTRY.get(name)
    if existing is not None:
        if existing is provider:
            return existing
        existing_origin = _ORIGINS.get(name, type(existing).__module__)
        raise SpecError(
            f"目标权重提供者名称重复：{name}；已由 {existing_origin} 注册，"
            f"当前来自 {origin}"
        )
    _REGISTRY[name] = provider
    _ORIGINS[name] = origin
    logger.info("注册提供者 %s v%s（%s）", name, provider.version, origin)
    return provider


def discover_providers() -> None:
    """导入 ``backtest.strategies`` 下的模块，触发模块内的注册。"""
    global _DISCOVERED
    if _DISCOVERED:
        return
    import backtest.strategies as package

    for item in sorted(pkgutil.iter_modules(package.__path__), key=lambda row: row.name):
        if item.name.startswith("_"):
            continue
        module_name = f"{package.__name__}.{item.name}"
        try:
            importlib.import_module(module_name)
        except SpecError as error:
            logger.error("提供者模块 %s 注册失败：%s", module_name, error)
            raise SpecError(f"提供者模块 {module_name} 注册失败：{error}") from error
        except Exception as error:
            logger.exception("提供者模块 %s 导入失败", module_name)
            raise SpecError(
                f"提供者模块 {module_name} 导入失败：{type(error).__name__}: {error}"
            ) from error
    _DISCOVERED = True
    logger.info("发现 %d 个目标权重提供者：%s", len(_REGISTRY), sorted(_REGISTRY))


def get_provider(spec: ProviderSpec | str) -> TargetWeightProvider:
    """按配置或名称取得提供者，并检查版本。"""
    discover_providers()
    if isinstance(spec, ProviderSpec):
        name, version = spec.name, spec.version
    else:
        name, version = str(spec), None
    provider = _REGISTRY.get(name)
    if provider is None:
        available = ", ".join(sorted(_REGISTRY)) or "无"
        logger.warning("未注册的目标权重提供者：%s；可选：%s", name, available)
        raise SpecError(f"未注册的目标权重提供者：{name}；可选：{available}")
    if version is not None and str(provider.version) != version:
        origin = _ORIGINS.get(name, type(provider).__module__)
        logger.warning(
            "提供者 %s 版本不匹配：配置=%s 当前=%s 模块=%s",
            name, version, provider.version, origin,
        )
        raise SpecError(
            f"目标权重提供者 {name} 的版本不匹配：配置为 {version}，"
            f"当前注册为 {provider.version}（模块 {origin}）；请升级策略版本"
        )
    return provider


def provider_uses_position_state(provider: TargetWeightProvider) -> bool:
    """返回提供者是否读取执行层持仓状态。"""
    return bool(getattr(provider, "uses_position_state", False))


def _code_source(provider: TargetWeightProvider) -> tuple[str, str]:
    """读取提供者模块源码并返回源码文本和 SHA-256。"""
    source = inspect.getsourcefile(type(provider))
    if source and Path(source).is_file():
        payload = Path(source).read_text(encoding="utf-8")
        return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()
    try:
        text = inspect.getsource(type(provider))
    except (OSError, TypeError) as error:
        raise SpecError(f"无法读取目标权重提供者源码：{type(provider).__name__}") from error
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _code_sha256(provider: TargetWeightProvider) -> str:
    """读取提供者模块源码并计算哈希。"""
    return _code_source(provider)[1]


def _metadata(provider: TargetWeightProvider) -> dict[str, Any]:
    """返回提供者的可审计元数据。"""
    return {
        "name": provider.name,
        "version": str(provider.version),
        "description": provider.description,
        "required_factors": list(provider.required_factors),
        "uses_position_state": provider_uses_position_state(provider),
        "module": type(provider).__module__,
        "code_sha256": _code_sha256(provider),
    }


def provider_metadata(spec: ProviderSpec) -> dict[str, Any]:
    """返回当前注册提供者的元数据。"""
    return _metadata(get_provider(spec))


def provider_snapshot_metadata(spec: ProviderSpec) -> dict[str, Any]:
    """返回提供者的完整快照信息，包含模块源码。

    该接口只供回测输入快照使用；常规运行元数据的 :func:`signal_metadata`
    不包含源码正文，避免接口和日志膨胀。

    参数:
        spec: 策略中的提供者配置。
    """
    provider = get_provider(spec)
    payload = _metadata(provider)
    source, digest = _code_source(provider)
    payload["params"] = dict(spec.params)
    payload["source"] = source
    payload["source_sha256"] = digest
    return payload


def list_providers() -> list[dict[str, Any]]:
    """列出当前可用的提供者。"""
    discover_providers()
    return [_metadata(provider) for _, provider in sorted(_REGISTRY.items())]


def validate_strategy_provider(spec: StrategySpec) -> None:
    """校验策略引用的提供者、版本、必需因子和参数。"""
    if spec.signal.provider is None:
        return
    provider = get_provider(spec.signal.provider)
    missing = sorted(set(provider.required_factors) - set(spec.signal.factors))
    if missing:
        raise SpecError(
            f"提供者 {provider.name} 需要因子 {missing}，"
            f"但 signal.factors 只声明了 {list(spec.signal.factors)}"
        )
    validate = getattr(provider, "validate_params", None)
    if callable(validate):
        validate(dict(spec.signal.provider.params))


def signal_metadata(signal: SignalSpec) -> dict[str, Any]:
    """返回信号层的运行元数据。"""
    if signal.provider is None:
        return {
            "kind": "factor",
            "factors": list(signal.factors),
            "weights": list(signal.weights),
        }
    payload = {"kind": "provider", **provider_metadata(signal.provider)}
    payload["params"] = dict(signal.provider.params)
    return payload
