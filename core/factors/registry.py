"""因子注册表：因子定义的数据结构与注册、查询入口。

因子的唯一来源就是这个包：构建脚本、目录库、文档生成都从这里读定义。
新增一个因子只要在对应族模块里调用一次 :func:`define`，构建脚本无需改动。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scripts.errors import FactorRegistryError

from factors.sources import SOURCES

# 因子族，顺序即文档与落盘目录的顺序
FAMILIES: tuple[str, ...] = (
    "momentum",
    "volatility",
    "liquidity",
    "volume_price",
    "technical",
    "valuation",
    "fundamental",
    "cn_special",
    "market",
    "alpha101",
    "gtja191",
)
OPTIONAL_FAMILIES: tuple[str, ...] = ("mined",)
ALL_FAMILIES: tuple[str, ...] = FAMILIES + OPTIONAL_FAMILIES

FAMILY_LABELS: dict[str, str] = {
    "momentum": "动量与反转",
    "volatility": "波动与风险",
    "liquidity": "流动性",
    "volume_price": "量价关系",
    "technical": "技术形态",
    "valuation": "估值与规模",
    "fundamental": "财务与质量",
    "cn_special": "A 股特色",
    "market": "市场与相对强度",
    "alpha101": "WorldQuant Alpha101 复现",
    "gtja191": "国泰君安 Alpha191 复现",
}

_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_LAG_PATTERN = re.compile(r"lag\(.*?,\s*(\d+)\)\s*over\s+w\b")
_FRAME_PATTERN = re.compile(r"rows between\s+(\d+)\s+preceding")


def required_lookback(layers: tuple[str, ...]) -> int:
    """从表达式里推出最少需要回看的交易日数。

    参数:
        layers: 因子的计算表达式序列。

    返回:
        表达式里最长的滞后或窗口长度。
    """
    need = 0
    for layer in layers:
        for size in _LAG_PATTERN.findall(layer):
            need = max(need, int(size))
        for size in _FRAME_PATTERN.findall(layer):
            need = max(need, int(size) + 1)
    return need


@dataclass(frozen=True)
class FactorSpec:
    """一个因子的完整定义。

    参数:
        name: 因子名，大写字母开头，只能含大写字母、数字与下划线。
        family: 所属族，取值见 ``FAMILIES``，决定落盘目录与文档分组。
        formula: 公式文本，用于生成文档。
        description: 一句话说明因子的金融含义。
        steps: 白话计算步骤，一条一步；已实现因子必须有，供初学者对照实现。
        layers: 计算表达式序列，按顺序逐层用 DuckDB SQL 求值，最后一行即因子值。
            单层够用就写一个；需要嵌套窗口（先算行号再算 arg_max 之类）才写多层。
        lookback: 计算时最多需要回看的交易日数，跨年补窗口用；留空按表达式自动推导。
        direction: +1 表示因子值越大越看多，-1 表示越大越看空。
        fields: 依赖的日线字段，仅用于文档与依赖检查。
        extra_sources: 依赖的附加数据源标识，见 ``factors.sources``。
    """

    name: str
    family: str
    formula: str
    description: str
    steps: tuple[str, ...]
    layers: tuple[str, ...]
    lookback: int
    direction: int = 1
    fields: tuple[str, ...] = ()
    extra_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _NAME_PATTERN.match(self.name):
            raise FactorRegistryError(
                f"因子名 {self.name!r} 不合规范：要求大写字母开头，只含大写字母、数字与下划线"
            )
        if self.family not in ALL_FAMILIES:
            raise FactorRegistryError(
                f"未知因子族 {self.family!r}，可选：{list(ALL_FAMILIES)}"
            )
        if not self.layers or any(not layer.strip() for layer in self.layers):
            raise FactorRegistryError(f"因子 {self.name} 必须有非空的计算表达式")
        if not self.steps or any(not step.strip() for step in self.steps):
            raise FactorRegistryError(f"因子 {self.name} 必须有白话计算步骤 steps，供文档展示")
        if self.direction not in (-1, 1):
            raise FactorRegistryError(f"因子 {self.name} 的 direction 只能是 1 或 -1")
        need = required_lookback(self.layers)
        if self.lookback < need:
            raise FactorRegistryError(
                f"因子 {self.name} 声明 lookback={self.lookback}，"
                f"但表达式至少需要回看 {need} 个交易日"
            )
        for source in self.extra_sources:
            if source not in SOURCES:
                raise FactorRegistryError(
                    f"因子 {self.name} 引用了未知数据源 {source!r}，可选：{sorted(SOURCES)}"
                )

    @property
    def source_columns(self) -> tuple[str, ...]:
        """该因子用到的附加数据源列。"""
        columns: list[str] = []
        for key in self.extra_sources:
            columns.extend(SOURCES[key].columns)
        return tuple(columns)


_REGISTRY: dict[str, FactorSpec] = {}


def register(spec: FactorSpec, overwrite: bool = False) -> FactorSpec:
    """登记一个因子定义。

    参数:
        spec: 因子定义。
        overwrite: 同名因子已存在时是否覆盖，默认报错。

    返回:
        登记后的因子定义。
    """
    if spec.name in _REGISTRY and not overwrite:
        raise FactorRegistryError(
            f"因子 {spec.name} 重复注册：已存在同名定义，"
            "要么改名字，要么显式传 overwrite=True"
        )
    _REGISTRY[spec.name] = spec
    return spec


def define(
    *,
    name: str,
    family: str,
    formula: str,
    description: str,
    steps,
    layers,
    lookback: int | None = None,
    direction: int = 1,
    fields=(),
    extra_sources=(),
    overwrite: bool = False,
) -> FactorSpec:
    """定义并登记一个因子。

    参数:
        name: 因子名。
        family: 因子族。
        formula: 公式文本。
        description: 金融含义说明。
        steps: 白话计算步骤。
        layers: 计算表达式序列。
        lookback: 回看交易日数；留空按表达式自动推导。
        direction: +1 / -1。
        fields: 依赖的日线字段。
        extra_sources: 依赖的附加数据源。
        overwrite: 同名时是否覆盖。
    """
    layer_tuple = (layers,) if isinstance(layers, str) else tuple(layers)
    step_tuple = (steps,) if isinstance(steps, str) else tuple(steps)
    spec = FactorSpec(
        name=name,
        family=family,
        formula=formula,
        description=description,
        steps=step_tuple,
        layers=layer_tuple,
        lookback=required_lookback(layer_tuple) if lookback is None else lookback,
        direction=direction,
        fields=tuple(fields),
        extra_sources=tuple(extra_sources),
    )
    return register(spec, overwrite=overwrite)


def unregister(name: str) -> None:
    """注销一个因子，找不到就静默返回。

    参数:
        name: 因子名。
    """
    _REGISTRY.pop(name, None)


def get(name: str) -> FactorSpec:
    """按名字取因子定义。

    参数:
        name: 因子名。
    """
    spec = _REGISTRY.get(name)
    if spec is None:
        raise FactorRegistryError(f"未知因子 {name!r}；已注册 {len(_REGISTRY)} 个：{sorted(_REGISTRY)}")
    return spec


def resolve(name: str, factor_dir=None) -> FactorSpec:
    """解析源码内置或已正式发布的因子定义。"""

    spec = _REGISTRY.get(name)
    if spec is not None:
        return spec
    from common.paths import FACTOR_DIR
    from factors.published import resolve_spec

    target = factor_dir if factor_dir is not None else FACTOR_DIR
    published = resolve_spec(name, target)
    if published is not None:
        return published
    raise FactorRegistryError(
        f"未知因子 {name!r}；已注册 {len(_REGISTRY)} 个内置因子"
    )


def available_specs(factor_dir=None) -> tuple[FactorSpec, ...]:
    """返回源码内置与当前已发布定义，去重后按族和名称排序。"""

    from common.paths import FACTOR_DIR
    from factors.published import published_specs

    target = factor_dir if factor_dir is not None else FACTOR_DIR
    merged = {item.name: item for item in all_factors()}
    for item in published_specs(target):
        if item.name in merged:
            raise FactorRegistryError(f"已发布因子与内置因子重名：{item.name}")
        merged[item.name] = item
    order = {family: index for index, family in enumerate(ALL_FAMILIES)}
    return tuple(
        sorted(merged.values(), key=lambda item: (order[item.family], item.name))
    )


def all_factors() -> tuple[FactorSpec, ...]:
    """返回全部已注册因子，按族与名字排序。"""
    order = {family: index for index, family in enumerate(FAMILIES)}
    return tuple(sorted(_REGISTRY.values(), key=lambda item: (order.get(item.family, 99), item.name)))


def family_factors(family: str) -> tuple[FactorSpec, ...]:
    """返回某个族下的因子。

    参数:
        family: 因子族。
    """
    if family not in FAMILIES:
        raise FactorRegistryError(f"未知因子族 {family!r}，可选：{list(FAMILIES)}")
    return tuple(spec for spec in all_factors() if spec.family == family)


def used_families() -> tuple[str, ...]:
    """返回已注册因子覆盖到的族，顺序与 ``FAMILIES`` 一致。"""
    present = {spec.family for spec in _REGISTRY.values()}
    return tuple(family for family in FAMILIES if family in present)


def clear() -> None:
    """清空注册表，仅供测试使用。"""
    _REGISTRY.clear()
