"""沙箱代码研究汇总指标的严格读取和校验。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from mining_sandbox.feedback import (
    SandboxCodeFeedback,
    SandboxFailureCategory,
    SandboxFeedbackError,
)

_RESEARCH_METRICS_FIELDS = {"rank_ic", "icir", "coverage", "valid_days"}


def _metrics_feedback(
    category: SandboxFailureCategory,
    summary: str,
    repair_hint: str,
) -> SandboxFeedbackError:
    return SandboxFeedbackError(
        SandboxCodeFeedback.create(
            category,
            summary=summary,
            repair_hint=repair_hint,
        )
    )


def parse_research_metrics(payload: Any) -> dict[str, Any]:
    """严格校验脚本汇总指标并返回规范化结果。"""

    if not isinstance(payload, Mapping) or set(payload) != _RESEARCH_METRICS_FIELDS:
        raise _metrics_feedback(
            SandboxFailureCategory.OUTPUT_FORMAT,
            "沙箱指标字段不符合契约",
            "metrics.json 必须且只能包含 rank_ic、icir、coverage、valid_days",
        )
    valid_days = payload["valid_days"]
    if isinstance(valid_days, bool) or not isinstance(valid_days, int):
        raise _metrics_feedback(
            SandboxFailureCategory.OUTPUT_FORMAT,
            "valid_days 类型不符合契约",
            "valid_days 必须写成大于零的整数",
        )
    if valid_days <= 0:
        raise _metrics_feedback(
            SandboxFailureCategory.INDEX,
            "valid_days 没有形成有效交易日索引",
            "对齐冻结交易日和股票索引，并确认至少有一个有效观察日",
        )
    normalized: dict[str, Any] = {"valid_days": valid_days}
    for name in ("rank_ic", "icir", "coverage"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _metrics_feedback(
                SandboxFailureCategory.OUTPUT_FORMAT,
                f"{name} 类型不符合契约",
                f"{name} 必须写成 JSON 数值",
            )
        number = float(value)
        if not math.isfinite(number):
            raise _metrics_feedback(
                SandboxFailureCategory.NUMERIC,
                f"{name} 不是有限数值",
                f"修正 {name} 的计算，过滤空值和不可计算的样本",
            )
        normalized[name] = number
    if not -1 <= normalized["rank_ic"] <= 1:
        raise _metrics_feedback(
            SandboxFailureCategory.NUMERIC,
            "rank_ic 超出合法范围",
            "检查收益和因子方向，确保 rank_ic 落在 -1～1",
        )
    if not 0 <= normalized["coverage"] <= 1:
        raise _metrics_feedback(
            SandboxFailureCategory.COVERAGE,
            "coverage 超出合法范围",
            "按冻结股票和交易日重算覆盖率，确保结果落在 0～1",
        )
    return normalized


def read_research_metrics(path: Path | str) -> dict[str, Any]:
    """读取并严格校验沙箱 metrics.json。"""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _metrics_feedback(
            SandboxFailureCategory.OUTPUT_FORMAT,
            f"沙箱指标不可读：{type(exc).__name__}",
            "重新写出合法的 metrics.json，字段固定为四个研究指标",
        ) from exc
    return parse_research_metrics(payload)


__all__ = ["parse_research_metrics", "read_research_metrics"]
