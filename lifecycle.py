"""模型生命周期的稳定公开接口。"""

from alpha.lifecycle_artifacts import load_model_lifecycle, run_model_lifecycle
from alpha.lifecycle_engine import (
    build_retrain_schedule,
    evaluate_lifecycle,
)
from alpha.lifecycle_policy import (
    DEFAULT_FALLBACK_MODEL_ID,
    DEFAULT_MONITOR_ID,
    DEFAULT_PRIMARY_MODEL_ID,
    LIFECYCLE_COMPLETE_SCHEMA_VERSION,
    LIFECYCLE_SCHEMA_VERSION,
    LifecyclePolicy,
    LifecycleResult,
)

__all__ = [
    "DEFAULT_FALLBACK_MODEL_ID",
    "DEFAULT_MONITOR_ID",
    "DEFAULT_PRIMARY_MODEL_ID",
    "LIFECYCLE_COMPLETE_SCHEMA_VERSION",
    "LIFECYCLE_SCHEMA_VERSION",
    "LifecyclePolicy",
    "LifecycleResult",
    "build_retrain_schedule",
    "evaluate_lifecycle",
    "load_model_lifecycle",
    "run_model_lifecycle",
]
