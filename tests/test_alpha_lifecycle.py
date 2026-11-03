"""P5-3 生命周期状态、因果性、产物和公开接口测试。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from alpha.artifacts import file_sha256
from alpha.errors import AlphaLifecycleError
from alpha.lifecycle import (
    LifecyclePolicy,
    evaluate_lifecycle,
    load_model_lifecycle,
    run_model_lifecycle,
)
from scripts.build_alpha.lifecycle import main as lifecycle_main


def _timeseries(levels: list[str]) -> pd.DataFrame:
    """构造只通过超额收益触发状态的周频监控序列。"""

    dates = pd.date_range("2023-01-06", periods=len(levels), freq="W-FRI")
    rows = []
    for day, level in zip(dates, levels, strict=True):
        rows.append(
            {
                "trade_date": day,
                "rolling_deployed_rank_ic": 0.10,
                "rolling_excess": (
                    -0.20
                    if level == "alert"
                    else -0.05
                    if level == "warning"
                    else 0.02
                ),
                "drawdown": -0.01,
                "rolling_turnover_two_way_annual": 20.0,
                "missing_rate": (
                    0.06 if level == "hard_pause" else 0.0
                ),
                "max_style_abs_exposure": 0.5,
                "max_abs_industry_active_weight": 0.1,
            }
        )
    return pd.DataFrame(rows)


def _folds() -> list[dict]:
    """构造与固定契约一致的最小训练折。"""

    return [
        {
            "fold_id": "wf_2023",
            "train_start": "2012-01-04",
            "train_end": "2021-12-23",
            "valid_start": "2022-01-04",
            "valid_end": "2022-12-30",
            "test_start": "2023-01-06",
            "test_end": "2023-12-29",
            "purge_days": 6,
            "embargo_days": 6,
        }
    ]


def test_lifecycle_state_machine_executes_all_actions():
    """降权、恢复、回滚、暂停和恢复按固定去抖顺序执行。"""

    levels = (
        ["ok"]
        + ["warning"] * 4
        + ["ok"] * 4
        + ["alert"] * 4
        + ["ok"] * 4
    )
    result = evaluate_lifecycle(
        _timeseries(levels),
        reference_turnover=40.0,
        folds=_folds(),
    )
    events = result.events.loc[
        result.events["event_type"].isin(
            ("reduce", "recover", "rollback", "pause")
        )
    ]
    assert events["event_type"].tolist() == [
        "reduce",
        "recover",
        "rollback",
        "pause",
        "recover",
    ]
    assert events.loc[events["event_type"].eq("rollback"), "active_model_id"].eq(
        "ridge_label_rank_aaff2cb2fbcf"
    ).all()
    assert events.loc[events["event_type"].eq("pause"), "exposure_multiplier"].eq(
        0.0
    ).all()
    assert events.loc[events["event_type"].eq("reduce"), "exposure_multiplier"].eq(
        0.5
    ).all()


def test_missing_alert_pauses_immediately_and_recovers_after_four_ok():
    """缺失率超过硬上限时立即暂停，连续四期正常后恢复。"""

    result = evaluate_lifecycle(
        _timeseries(["ok", "hard_pause"] + ["ok"] * 4),
        reference_turnover=40.0,
        folds=_folds(),
    )
    assert result.timeseries.iloc[1]["state"] == "paused"
    assert result.timeseries.iloc[4]["state"] == "paused"
    assert result.timeseries.iloc[5]["state"] == "normal"


def test_lifecycle_is_causal_for_future_monitor_changes():
    """修改未来监控数据不改写历史生命周期状态。"""

    frame = _timeseries(["ok"] * 12)
    first = evaluate_lifecycle(
        frame,
        reference_turnover=40.0,
        folds=_folds(),
    )
    changed = frame.copy()
    changed.loc[8:, "rolling_excess"] = -0.20
    second = evaluate_lifecycle(
        changed,
        reference_turnover=40.0,
        folds=_folds(),
    )
    pd.testing.assert_frame_equal(
        first.timeseries.iloc[:8],
        second.timeseries.iloc[:8],
    )


def test_retrain_schedule_requires_fixed_expanding_start():
    """年度折必须使用固定的 2012-01-04 扩展训练起点。"""

    folds = _folds()
    folds[0]["train_start"] = "2012-01-05"
    with pytest.raises(AlphaLifecycleError, match="训练起点"):
        evaluate_lifecycle(
            _timeseries(["ok"] * 8),
            reference_turnover=40.0,
            folds=folds,
        )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"reduce_after": 0}, "reduce_after"),
        (
            {"rollback_after": 4, "pause_after": 4},
            "rollback_after",
        ),
        ({"reduced_multiplier": 1.0}, "reduced_multiplier"),
        ({"fallback_model_id": "same", "primary_model_id": "same"}, "不能相同"),
    ],
)
def test_lifecycle_policy_rejects_invalid_rules(changes, match):
    """非法去抖窗口、仓位乘数和模型组合明确失败。"""

    with pytest.raises(AlphaLifecycleError, match=match):
        LifecyclePolicy(**changes)


def test_formal_lifecycle_package_is_complete_and_reusable():
    """正式监控生成可复用、可校验并包含完整动作链的生命周期包。"""

    first = run_model_lifecycle()
    second = run_model_lifecycle()
    assert first["lifecycle_id"] == second["lifecycle_id"]
    assert first["summary"]["signal_days"] == 300
    assert first["summary"]["state_counts"]["reduced"] > 0
    assert first["summary"]["event_counts"]["rollback"] >= 1
    assert first["summary"]["event_counts"]["pause"] >= 1
    assert first["summary"]["event_counts"]["recover"] >= 1
    assert first["summary"]["retrain_due_count"] == 5
    loaded = load_model_lifecycle(first["lifecycle_id"])
    assert len(loaded["timeseries"]) == 300
    folder = loaded["folder"]
    complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    assert complete["complete"] is True
    assert set(complete["files"]) == {
        "events.csv",
        "manifest.json",
        "report.json",
        "report.md",
        "timeseries.csv",
    }
    for name, expected in complete["files"].items():
        assert file_sha256(folder / name) == expected


def test_lifecycle_cli_prints_summary(capsys):
    """命令行复用完整生命周期包并输出动作摘要。"""

    assert lifecycle_main([]) == 0
    output = capsys.readouterr().out
    assert "p5_3_" in output
    assert "rollback=" in output


def test_lifecycle_loader_rejects_changed_result_file(tmp_path: Path):
    """已有生命周期包任一结果文件变化后拒绝读取。"""

    manifest = run_model_lifecycle()
    lifecycle_id = manifest["lifecycle_id"]
    from alpha.artifacts import model_dir

    source = (
        model_dir("lightgbm_ensemble_label_rank_6aea5c9299d5")
        / "lifecycle"
        / lifecycle_id
    )
    target = (
        tmp_path
        / "models"
        / "lightgbm_ensemble_label_rank_6aea5c9299d5"
        / "lifecycle"
        / lifecycle_id
    )
    shutil.copytree(source, target)
    with (target / "timeseries.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(AlphaLifecycleError, match="哈希不一致"):
        load_model_lifecycle(lifecycle_id, root=tmp_path)
