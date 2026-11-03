"""模型监控的公共类型、固定口径和阈值策略。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from alpha.errors import AlphaMonitorError
from alpha.evaluation_plan import current_model_ids

MONITOR_SCHEMA_VERSION = "alpha_model_monitor_v1"
DEFAULT_MODEL_ID, _ = current_model_ids()
DEFAULT_ACCEPTANCE_ID = "p4_4a_23c7e101aee9"
DEFAULT_EXPOSURE_ANALYSIS_ID = "p3_4_8152f500bb86"
SEVERITY_ORDER = {"insufficient_data": 0, "ok": 1, "warning": 2, "alert": 3}


@dataclass(frozen=True)
class MonitorPolicy:
    """模型监控的固定窗口和告警阈值。"""

    rolling_weeks: int = 13
    min_periods: int = 8
    rank_ic_warning: float = 0.0
    rank_ic_alert_consecutive: int = 2
    excess_warning: float = 0.0
    excess_alert: float = -0.10
    drawdown_warning: float = -0.28
    drawdown_alert: float = -0.35
    turnover_warning_multiple: float = 1.5
    turnover_alert_multiple: float = 2.0
    missing_warning: float = 0.01
    missing_alert: float = 0.05
    style_warning: float = 1.5
    style_alert: float = 2.0
    industry_warning: float = 0.25
    industry_alert: float = 0.35

    def __post_init__(self) -> None:
        if type(self.rolling_weeks) is not int or self.rolling_weeks <= 0:
            raise AlphaMonitorError("rolling_weeks 必须是正整数")
        if (
            type(self.min_periods) is not int
            or not 1 <= self.min_periods <= self.rolling_weeks
        ):
            raise AlphaMonitorError("min_periods 必须在 1 和滚动窗口之间")
        if type(self.rank_ic_alert_consecutive) is not int:
            raise AlphaMonitorError("rank_ic_alert_consecutive 必须是整数")
        if self.rank_ic_alert_consecutive < 2:
            raise AlphaMonitorError("rank_ic_alert_consecutive 不能小于 2")
        if self.turnover_warning_multiple <= 0:
            raise AlphaMonitorError("turnover_warning_multiple 必须为正")
        if self.turnover_alert_multiple <= self.turnover_warning_multiple:
            raise AlphaMonitorError("换手 alert 倍数必须大于 warning 倍数")
        if not 0.0 < self.missing_warning < self.missing_alert:
            raise AlphaMonitorError("缺失率阈值必须满足 0 < warning < alert")
        if not 0.0 < self.style_warning < self.style_alert:
            raise AlphaMonitorError("风格阈值必须满足 0 < warning < alert")
        if not 0.0 < self.industry_warning < self.industry_alert:
            raise AlphaMonitorError("行业阈值必须满足 0 < warning < alert")

    def to_dict(self) -> dict:
        """返回稳定字典。"""

        return asdict(self)


@dataclass(frozen=True)
class MonitorResult:
    """一次监控计算的时间序列、告警和摘要。"""

    timeseries: pd.DataFrame
    alerts: pd.DataFrame
    style_summary: pd.DataFrame = field(repr=False)
    industry_summary: pd.DataFrame = field(repr=False)
    summary: dict = field(default_factory=dict)


def number(value: object) -> float | None:
    """把数值规范为有限浮点数。"""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def first_alert_date(alerts: pd.DataFrame) -> str | None:
    """返回第一条 alert 日期。"""

    if alerts.empty:
        return None
    selected = alerts.loc[alerts["severity"].eq("alert"), "trade_date"]
    if selected.empty:
        return None
    return str(pd.Timestamp(selected.min()).date())


__all__ = [
    "DEFAULT_ACCEPTANCE_ID",
    "DEFAULT_EXPOSURE_ANALYSIS_ID",
    "DEFAULT_MODEL_ID",
    "MONITOR_SCHEMA_VERSION",
    "MonitorPolicy",
    "MonitorResult",
    "SEVERITY_ORDER",
    "first_alert_date",
    "number",
]
