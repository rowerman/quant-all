"""指标包：导入本包即完成全部已实现指标的注册。"""

from calculate.metrics.base import (
    Metric,
    MetricParam,
    clear_metrics,
    default_options,
    get_metric,
    list_metrics,
    metrics_by_stage,
    register_metric,
    unregister_metric,
)

# 导入即注册，注册顺序就是前端卡片顺序
from calculate.metrics import bins, coverage, decay, ic, turnover  # noqa: E402,F401

__all__ = [
    "Metric",
    "MetricParam",
    "clear_metrics",
    "default_options",
    "get_metric",
    "list_metrics",
    "metrics_by_stage",
    "register_metric",
    "unregister_metric",
]
