"""因子评估计算包：只算，不展示、不落库。

- :mod:`calculate.panel`：把因子值、标签、池子标记与后续阶段原料对齐成评估面板；
- :mod:`calculate.metrics`：指标注册表与已实现指标（覆盖率、IC 家族）；
- :mod:`calculate.evaluate`：按 stage 逐段产出结果，接口层直接转成 SSE。

数据读取一律走 DuckDB 只读连接（主库的 ``md`` / ``ref`` 表 + ``factor_data`` 的 parquet）。
"""

from calculate.evaluate import STAGE_ORDER, evaluate, meta_payload, planned_stages
from calculate.metrics import (
    Metric,
    MetricParam,
    get_metric,
    list_metrics,
    metrics_by_stage,
    register_metric,
    unregister_metric,
)
from calculate.panel import (
    MAX_PANEL_ROWS,
    PANEL_COLUMNS,
    EvalContext,
    build_panel,
    default_params,
    open_source,
    panel_sql,
    resolve_factor,
    summarize_panel,
)
from calculate.params import (
    HORIZONS,
    MAX_TRADE_DAYS,
    PRICE_CHANNELS,
    QUANTILE_CHOICES,
    EvalParams,
)
from calculate.universe import (
    ALL_A,
    MIN_LISTED_DAYS,
    TRADABLE_FILTER,
    UniverseSpec,
    list_universes,
    resolve_universe,
)

__all__ = [
    "ALL_A",
    "HORIZONS",
    "MAX_PANEL_ROWS",
    "MAX_TRADE_DAYS",
    "MIN_LISTED_DAYS",
    "PANEL_COLUMNS",
    "PRICE_CHANNELS",
    "QUANTILE_CHOICES",
    "STAGE_ORDER",
    "TRADABLE_FILTER",
    "EvalContext",
    "EvalParams",
    "Metric",
    "MetricParam",
    "UniverseSpec",
    "build_panel",
    "default_params",
    "evaluate",
    "get_metric",
    "list_metrics",
    "list_universes",
    "meta_payload",
    "metrics_by_stage",
    "open_source",
    "panel_sql",
    "planned_stages",
    "register_metric",
    "resolve_factor",
    "resolve_universe",
    "summarize_panel",
    "unregister_metric",
]
