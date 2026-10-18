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
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from backtest.data import MarketData
from backtest.errors import SpecError
from backtest.specs import ProviderSpec, SignalSpec, StrategySpec


@runtime_checkable
class TargetWeightProvider(Protocol):
    """目标权重提供者协议。

    ``eligible`` 是引擎按统一股票池口径算好的布尔矩阵；提供者只能给 True 的格子
    正权重。``build`` 必须返回与行情面板同形的权重矩阵。
    """

    name: str
    version: str
    description: str
    required_factors: tuple[str, ...]

    def build(
        self,
        data: MarketData,
        eligible: pd.DataFrame,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """返回信号日目标权重。"""


_REGISTRY: dict[str, TargetWeightProvider] = {}
_DISCOVERED = False


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
        raise SpecError("目标权重提供者必须实现 build(data, eligible, params)")


def register_provider(provider: TargetWeightProvider | type[TargetWeightProvider]) -> TargetWeightProvider:
    """注册一个提供者；重复名称直接报错。"""
    if isinstance(provider, type):
        provider = provider()
    _validate_provider(provider)
    name = str(provider.name).strip()
    existing = _REGISTRY.get(name)
    if existing is not None:
        if existing is provider:
            return existing
        raise SpecError(f"目标权重提供者名称重复：{name}")
    _REGISTRY[name] = provider
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
        importlib.import_module(f"{package.__name__}.{item.name}")
    _DISCOVERED = True


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
        raise SpecError(f"未注册的目标权重提供者：{name}；可选：{available}")
    if version is not None and str(provider.version) != version:
        raise SpecError(
            f"目标权重提供者 {name} 的版本不匹配：配置为 {version}，"
            f"当前注册为 {provider.version}；请升级策略版本"
        )
    return provider


def _code_sha256(provider: TargetWeightProvider) -> str:
    """读取提供者类所在源码文件并计算哈希。"""
    source = inspect.getsourcefile(type(provider))
    if source and Path(source).is_file():
        return hashlib.sha256(Path(source).read_bytes()).hexdigest()
    try:
        text = inspect.getsource(type(provider))
    except (OSError, TypeError) as error:
        raise SpecError(f"无法读取目标权重提供者源码：{type(provider).__name__}") from error
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _metadata(provider: TargetWeightProvider) -> dict[str, Any]:
    """返回提供者的可审计元数据。"""
    return {
        "name": provider.name,
        "version": str(provider.version),
        "description": provider.description,
        "required_factors": list(provider.required_factors),
        "module": type(provider).__module__,
        "code_sha256": _code_sha256(provider),
    }


def provider_metadata(spec: ProviderSpec) -> dict[str, Any]:
    """返回当前注册提供者的元数据。"""
    return _metadata(get_provider(spec))


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
