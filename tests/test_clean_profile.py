"""清洗层剖析、隔离证据和 C++ 决策测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.clean import adjust, pipeline
from tools.monitor.clean_profile import (
    decide_cpp,
    run_profile,
)
from tools.monitor.clean_profile_contract import SCHEMA_VERSION
from tools.monitor.clean_profile_probe import (
    CallObservation,
    CleanProfileProbe,
    summarize_observations,
)
from tools.monitor.clean_profile_report import render_report
from tools.monitor.clean_profile_runtime import profile_year, stable_frame_digest


def _worker_result(
    *,
    elapsed: float,
    pure_share: float,
    digest: str = "a" * 64,
) -> dict:
    """构造一次 worker 剖析结果。"""

    pure_elapsed = elapsed * pure_share
    other_elapsed = max(elapsed - pure_elapsed, 0.0)
    return {
        "elapsed_s": elapsed,
        "peak_rss_mb": 100.0,
        "seed_elapsed_s": 10.0,
        "content_digest": {
            "rows": 10,
            "columns": [["close", "float64"]],
            "sha256": digest,
        },
        "profile": {
            "elapsed_s": elapsed,
            "stages": {
                "compute_return": {
                    "category": "numpy_rules",
                    "pure_array": True,
                    "count": 1,
                    "elapsed_s": pure_elapsed,
                    "share": pure_share,
                },
                "panel_assign": {
                    "category": "pandas_assembly",
                    "pure_array": False,
                    "count": 1,
                    "elapsed_s": other_elapsed,
                    "share": round(other_elapsed / elapsed, 6),
                },
            },
            "categories": {
                "numpy_rules": pure_elapsed,
                "pandas_assembly": other_elapsed,
                "unattributed": 0.0,
            },
            "unattributed_s": 0.0,
            "largest_pure_array_stage": "compute_return",
            "largest_pure_array_elapsed_s": pure_elapsed,
            "largest_pure_array_share": pure_share,
        },
        "duckdb_daily": {
            "elapsed_s": 2.0,
            "threads": 1,
            "row_counts": {"md.daily_bar": 10},
        },
        "stats": {"year": 2025, "rows": 10},
    }


def test_summary_attributes_unclassified_time_once():
    """阶段耗时只累计一次，剩余时间进入未归属桶。"""

    summary = summarize_observations(
        [
            CallObservation("compute_return", "numpy_rules", True, 0.2),
            CallObservation("compute_return", "numpy_rules", True, 0.3),
            CallObservation("panel_assign", "pandas_assembly", False, 0.1),
        ],
        1.0,
    )

    assert summary["stages"]["compute_return"]["count"] == 2
    assert summary["stages"]["compute_return"]["elapsed_s"] == 0.5
    assert summary["largest_pure_array_stage"] == "compute_return"
    assert summary["largest_pure_array_share"] == 0.5
    assert summary["unattributed_s"] == 0.4
    assert summary["categories"]["unattributed"] == 0.4


def test_probe_restores_functions_and_keeps_result():
    """剖析包装器不改变函数结果，并在退出后恢复原函数。"""

    original_return = adjust.compute_return
    original_to_frame = pipeline._to_frame
    pre_close = np.array([[10.0, 20.0]])
    change = np.array([[1.0, -2.0]])

    with CleanProfileProbe() as probe:
        result = adjust.compute_return(pre_close, change)
        assert adjust.compute_return is not original_return
        assert pipeline._to_frame is not original_to_frame
    summary = probe.summary(0.01)

    np.testing.assert_array_equal(result, change / pre_close)
    assert adjust.compute_return is original_return
    assert pipeline._to_frame is original_to_frame
    assert summary["stages"]["compute_return"]["count"] == 1


def test_stable_frame_digest_is_order_sensitive():
    """内容摘要忽略索引名称但保留行序和值。"""

    first = pd.DataFrame({"value": [1.0, 2.0]}, index=[10, 11])
    second = first.copy()
    second.index = [20, 21]
    reversed_frame = first.iloc[::-1]

    assert stable_frame_digest(first) == stable_frame_digest(second)
    assert stable_frame_digest(first)["sha256"] != stable_frame_digest(
        reversed_frame
    )["sha256"]


def test_decision_threshold_is_inclusive():
    """纯数组规则恰好占 20% 时登记为 C++ 候选。"""

    candidate = _worker_result(elapsed=10.0, pure_share=0.20)
    case = {
        "median": {
            "elapsed_s": candidate["elapsed_s"],
            "profile": candidate["profile"],
        }
    }

    assert decide_cpp(case)["status"] == "cpp_candidate"
    candidate["profile"]["largest_pure_array_share"] = 0.199999
    assert decide_cpp(case)["status"] == "no_cpp"


def test_run_profile_writes_isolated_evidence(tmp_path, monkeypatch):
    """证据写入指定临时目录并生成稳定结果和哈希清单。"""

    import tools.monitor.clean_profile as module

    counter = iter(
        [
            _worker_result(elapsed=10.0, pure_share=0.1),
            _worker_result(elapsed=12.0, pure_share=0.1),
            _worker_result(elapsed=11.0, pure_share=0.1),
        ]
    )
    monkeypatch.setattr(
        module,
        "environment_snapshot",
        lambda: {"schema_version": SCHEMA_VERSION, "git": {"commit": "test"}},
    )
    monkeypatch.setattr(
        module,
        "input_snapshot",
        lambda: {
            "schema_version": SCHEMA_VERSION,
            "files": [],
            "input_tree_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(module, "_run_worker", lambda case, repeat: next(counter))

    result = run_profile(("2025",), repeats=3, output=tmp_path)

    assert result["decision"]["status"] == "no_cpp"
    assert result["cases"]["2025"]["median"]["elapsed_s"] == 11.0
    assert result["cases"]["2025"]["stability"]["content_exact"] is True
    assert (tmp_path / "result.json").is_file()
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "manifest.json").is_file()
    assert "不实施清洗层 C++" in render_report(result)


def test_profile_year_restores_daily_dir_and_uses_temp_output(tmp_path, monkeypatch):
    """年度运行结束后恢复原路径，种子和目标年份都没有写正式目录。"""

    import tools.monitor.clean_profile_runtime as runtime

    production_dir = tmp_path / "production-clean"
    monkeypatch.setattr(runtime.pipeline, "DAILY_DIR", production_dir)
    visited = []

    def fake_process_year(year, start, end, carry):
        visited.append(runtime.pipeline.DAILY_DIR)
        directory = runtime.pipeline.DAILY_DIR / f"year={year}"
        directory.mkdir(parents=True)
        pd.DataFrame(
            {"stock_code": [f"S{year}"], "trade_date": [start]}
        ).to_parquet(directory / "part.parquet", index=False)
        return {"year": year, "rows": 1}, {f"S{year}": 10.0}

    class FakeProbe:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def summary(self, elapsed):
            return {
                "elapsed_s": elapsed,
                "stages": {},
                "categories": {},
                "unattributed_s": elapsed,
                "largest_pure_array_stage": None,
                "largest_pure_array_elapsed_s": 0.0,
                "largest_pure_array_share": 0.0,
            }

    monkeypatch.setattr(runtime.pipeline, "process_year", fake_process_year)
    monkeypatch.setattr(runtime, "CleanProfileProbe", FakeProbe)
    monkeypatch.setattr(
        runtime,
        "_build_duckdb_daily",
        lambda work_root: {
            "elapsed_s": 0.01,
            "threads": 1,
            "row_counts": {},
        },
    )

    result = profile_year(2025, seed_year=2024)

    assert runtime.pipeline.DAILY_DIR == production_dir
    assert not production_dir.exists()
    assert len(visited) == 2
    assert all(path != production_dir for path in visited)
    assert result["stats"]["year"] == 2025
    assert result["content_digest"]["rows"] == 1
