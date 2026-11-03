"""模型生命周期 JSON 与 Markdown 报告。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha.artifacts import file_sha256, write_json
from alpha.lifecycle_policy import (
    LIFECYCLE_SCHEMA_VERSION,
    LifecycleResult,
)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """渲染紧凑 Markdown 表。"""

    if frame.empty:
        return "无数据。\n"
    selected = frame.loc[:, columns].copy()
    header = "| " + " | ".join(selected.columns) + " |"
    divider = "| " + " | ".join("---" for _ in selected.columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows]) + "\n"


def write_lifecycle_package(
    folder: Path,
    *,
    lifecycle_id: str,
    identity: dict,
    result: LifecycleResult,
    sources: dict,
) -> dict:
    """写入生命周期报告并返回 manifest 负载。"""

    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "lifecycle_id": lifecycle_id,
        "generated_at": generated_at,
        "policy": identity["policy"],
        "summary": result.summary,
        "sources": sources,
        "retrain_schedule": result.retrain_schedule.to_dict(orient="records"),
        "events": result.events.to_dict(orient="records"),
    }
    write_json(folder / "report.json", report)

    transitions = result.events.loc[
        result.events["event_type"].isin(
            ("reduce", "rollback", "pause", "recover")
        )
    ]
    schedule = result.events.loc[
        result.events["event_type"].isin(("initial_fit", "retrain_due"))
    ]
    summary_rows = pd.DataFrame(
        [
            {"指标": "生命周期 ID", "值": lifecycle_id},
            {"指标": "区间", "值": f"{result.summary['start']} ~ {result.summary['end']}"},
            {"指标": "信号日", "值": result.summary["signal_days"]},
            {"指标": "最新状态", "值": result.summary["latest_state"]},
            {
                "指标": "最新活动模型",
                "值": result.summary["latest_active_model_id"] or "空仓",
            },
            {
                "指标": "最新仓位乘数",
                "值": result.summary["latest_exposure_multiplier"],
            },
        ]
    )
    lines = [
        "# P5-3 模型生命周期",
        "",
        _markdown_table(summary_rows, ["指标", "值"]),
        "## 状态分布",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"状态": key, "信号日": value}
                    for key, value in result.summary["state_counts"].items()
                ]
            ),
            ["状态", "信号日"],
        ),
        "## 状态切换",
        "",
        _markdown_table(
            transitions,
            [
                "trade_date",
                "event_type",
                "from_state",
                "to_state",
                "active_model_id",
                "exposure_multiplier",
                "reason",
            ],
        ),
        "## 重训日程",
        "",
        _markdown_table(
            schedule,
            [
                "trade_date",
                "event_type",
                "fold_id",
                "train_start",
                "train_end",
                "valid_start",
                "valid_end",
            ],
        ),
        "## 口径与限制",
        "",
        "- 信号日收盘生成状态，次一交易日开盘执行。",
        "- RankIC 使用上一信号日值；其余指标使用信号日收盘已知值。",
        "- 重训日程只记录到期和窗口，挑战者晋升仍需独立完整样本外验证。",
        "- 正式规则不按回测收益调参，不读取 2024-2026 最终测试集。",
        "",
    ]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")

    files = {
        name: file_sha256(folder / name)
        for name in (
            "events.csv",
            "report.json",
            "report.md",
            "timeseries.csv",
        )
    }
    return {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "lifecycle_id": lifecycle_id,
        "created_at": generated_at,
        "policy": identity["policy"],
        "summary": result.summary,
        "sources": sources,
        "code_sha256": identity["code_sha256"],
        "files": files,
    }


__all__ = ["write_lifecycle_package"]
