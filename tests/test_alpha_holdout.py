"""最终留出窗口注册与封存规则测试。"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from alpha.errors import AlphaConfigError, AlphaFinalHoldoutError
from alpha.final_holdout_data import FinalHoldoutSpec, build_final_holdout_dataset
from alpha.final_holdout_runner import run_final_holdout
from alpha.holdout import (
    FUTURE_HOLDOUT_WINDOW_ID,
    LEGACY_HOLDOUT_WINDOW_ID,
    get_holdout_window,
    holdout_run_ids,
    load_holdout_windows,
)
from common.paths import CONFIG_DIR


def test_registered_windows_are_ordered_and_disjoint():
    """注册表包含已封存窗口和不重叠的未来窗口。"""

    windows = load_holdout_windows()
    assert [item.window_id for item in windows] == [
        LEGACY_HOLDOUT_WINDOW_ID,
        FUTURE_HOLDOUT_WINDOW_ID,
    ]
    assert windows[0].end < windows[1].start
    assert get_holdout_window(LEGACY_HOLDOUT_WINDOW_ID).sealed is True
    assert get_holdout_window(FUTURE_HOLDOUT_WINDOW_ID).sealed is False


def test_overlapping_registry_is_rejected(tmp_path):
    """窗口重叠时注册表失败关闭。"""

    path = tmp_path / "holdouts.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "alpha_holdout_windows_v1",
                "windows": [
                    {
                        "window_id": "a",
                        "start": "2026-01-01",
                        "end": "2026-12-31",
                    },
                    {
                        "window_id": "b",
                        "start": "2026-12-31",
                        "end": "2027-12-31",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AlphaConfigError, match="重叠"):
        load_holdout_windows(path)


def test_sealed_window_rejects_new_run_before_reading_data():
    """已消耗窗口只读，即使调用 runner 也不会重新构建数据。"""

    spec = FinalHoldoutSpec(
        window_id=LEGACY_HOLDOUT_WINDOW_ID,
        plan_path=str(CONFIG_DIR / "alpha" / "final_holdout_2024_2026.json"),
    )
    with pytest.raises(AlphaFinalHoldoutError, match="已封存"):
        run_final_holdout(spec)


def test_future_window_requires_plan_binding():
    """新窗口可以登记，但未绑定评估计划时不能形成最终决策。"""

    spec = FinalHoldoutSpec(window_id=FUTURE_HOLDOUT_WINDOW_ID)
    assert spec.start == "2026-10-16"
    assert spec.end == "2027-10-15"
    with pytest.raises(AlphaFinalHoldoutError, match="尚未绑定"):
        _ = spec.evaluation_plan


def test_future_window_rejects_missing_market_data(monkeypatch):
    """未来窗口没有行情时不创建数据集目录。"""

    import alpha.final_holdout_data as final_data

    monkeypatch.setattr(
        final_data,
        "load_calendar",
        lambda start, end: pd.DatetimeIndex([]),
    )
    with pytest.raises(AlphaFinalHoldoutError, match="没有可用行情"):
        build_final_holdout_dataset(
            FinalHoldoutSpec(window_id=FUTURE_HOLDOUT_WINDOW_ID)
        )


def test_holdout_run_ids_match_exact_window(tmp_path):
    """运行 ID 只按窗口起止日精确匹配。"""

    runs = tmp_path / "final_holdout"
    target = runs / "final_holdout_demo"
    target.mkdir(parents=True)
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "holdout_id": "final_holdout_demo",
                "scope": {
                    "data_start": "2026-10-16",
                    "data_end": "2027-10-15",
                },
            }
        ),
        encoding="utf-8",
    )
    window = get_holdout_window(FUTURE_HOLDOUT_WINDOW_ID)
    assert holdout_run_ids(window, tmp_path) == ("final_holdout_demo",)
