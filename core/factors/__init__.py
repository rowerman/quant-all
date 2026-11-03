"""因子库：因子定义、算子与注册表。

导入本包即完成全部已实现因子的注册，构建脚本与读取层都依赖这一行为。
"""

from factors.registry import (
    ALL_FAMILIES,
    FAMILIES,
    FAMILY_LABELS,
    OPTIONAL_FAMILIES,
    FactorSpec,
    all_factors,
    available_specs,
    clear,
    define,
    family_factors,
    get,
    register,
    resolve,
    unregister,
    used_families,
)
from factors.sources import SOURCES, SourceSpec
from factors import published  # noqa: F401

# 导入族模块触发因子注册，顺序与 FAMILIES 一致
from factors import (  # noqa: E402,F401
    alpha101,
    cn_special,
    fundamental,
    gtja191,
    liquidity,
    market,
    momentum,
    technical,
    valuation,
    volatility,
    volume_price,
)

__all__ = [
    "FAMILIES",
    "OPTIONAL_FAMILIES",
    "ALL_FAMILIES",
    "FAMILY_LABELS",
    "SOURCES",
    "FactorSpec",
    "SourceSpec",
    "all_factors",
    "available_specs",
    "clear",
    "define",
    "family_factors",
    "get",
    "register",
    "resolve",
    "unregister",
    "used_families",
]
