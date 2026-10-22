"""模型生命周期固定策略、状态和结果类型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from alpha.errors import AlphaLifecycleError
from alpha.monitor_policy import MonitorPolicy

LIFECYCLE_SCHEMA_VERSION = "alpha_model_lifecycle_v1"
LIFECYCLE_COMPLETE_SCHEMA_VERSION = "alpha_model_lifecycle_complete_v1"
LIFECYCLE_RESULT_FILES = (
    "events.csv",
    "manifest.json",
    "report.json",
    "report.md",
    "timeseries.csv",
)
DEFAULT_PRIMARY_MODEL_ID = "lightgbm_ensemble_label_rank_6aea5c9299d5"
DEFAULT_FALLBACK_MODEL_ID = "ridge_label_rank_aaff2cb2fbcf"
DEFAULT_MONITOR_ID = "p5_2_0a9e6523e9e7"
STATES = ("normal", "reduced", "rollback", "paused")


@dataclass(frozen=True)
class LifecyclePolicy:
    """模型生命周期的固定模型、去抖窗口和执行动作。"""

    primary_model_id: str = DEFAULT_PRIMARY_MODEL_ID
    fallback_model_id: str = DEFAULT_FALLBACK_MODEL_ID
    monitor: MonitorPolicy = field(default_factory=MonitorPolicy)
    reduce_after: int = 4
    rollback_after: int = 2
    pause_after: int = 4
    recover_after: int = 4
    reduced_multiplier: float = 0.5
    retrain_start: str = "2012-01-04"

    def __post_init__(self) -> None:
        for name in ("primary_model_id", "fallback_model_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise AlphaLifecycleError(f"{name} 必须是非空字符串")
        if self.primary_model_id == self.fallback_model_id:
            raise AlphaLifecycleError("主模型与回滚模型不能相同")
        if not isinstance(self.monitor, MonitorPolicy):
            raise AlphaLifecycleError("monitor 必须是 MonitorPolicy")
        for name in ("reduce_after", "rollback_after", "pause_after", "recover_after"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise AlphaLifecycleError(f"{name} 必须是正整数")
        if self.rollback_after >= self.pause_after:
            raise AlphaLifecycleError("rollback_after 必须小于 pause_after")
        if not 0.0 < self.reduced_multiplier < 1.0:
            raise AlphaLifecycleError("reduced_multiplier 必须在 0 和 1 之间")
        try:
            retrain_start = pd.Timestamp(self.retrain_start)
        except (TypeError, ValueError) as error:
            raise AlphaLifecycleError("retrain_start 必须是有效日期") from error
        if retrain_start.date().isoformat() != self.retrain_start:
            raise AlphaLifecycleError("retrain_start 必须是 YYYY-MM-DD")

    def to_dict(self) -> dict:
        """返回稳定策略字典。"""

        payload = asdict(self)
        payload["monitor"] = self.monitor.to_dict()
        return payload


@dataclass(frozen=True)
class LifecycleResult:
    """一次生命周期计算的状态时序、事件和摘要。"""

    timeseries: pd.DataFrame
    events: pd.DataFrame
    retrain_schedule: pd.DataFrame
    summary: dict


def _number(value: object) -> float | None:
    """返回有限浮点数。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) and abs(number) != float("inf") else None


__all__ = [
    "DEFAULT_FALLBACK_MODEL_ID",
    "DEFAULT_MONITOR_ID",
    "DEFAULT_PRIMARY_MODEL_ID",
    "LIFECYCLE_COMPLETE_SCHEMA_VERSION",
    "LIFECYCLE_RESULT_FILES",
    "LIFECYCLE_SCHEMA_VERSION",
    "LifecyclePolicy",
    "LifecycleResult",
    "STATES",
]
