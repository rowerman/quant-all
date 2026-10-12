"""指标包：导入本包即完成全部已实现指标的注册。"""

from calculate.metrics.base import (
    Metric,
    MetricParam,
    clear_metrics,
    default_options,
    resolve_options,
    get_metric,
    list_metrics,
    metrics_by_stage,
    register_metric,
    unregister_metric,
)

# 按模块名稳定导入，新增文件不需要修改这份清单。
from importlib import import_module
from pkgutil import iter_modules


def discover_metrics() -> None:
    """发现 metrics 包内的指标模块；每个模块通过导入完成注册。"""
    modules = sorted(iter_modules(__path__), key=lambda item: item.name)
    for module in modules:
        if module.name != "base" and not module.name.startswith("_"):
            import_module(f"{__name__}.{module.name}")


discover_metrics()

__all__ = [
    "Metric",
    "MetricParam",
    "clear_metrics",
    "default_options",
    "resolve_options",
    "discover_metrics",
    "get_metric",
    "list_metrics",
    "metrics_by_stage",
    "register_metric",
    "unregister_metric",
]
