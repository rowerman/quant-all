"""清洗年度流程的临时目录剖析与稳定结果摘要。"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from scripts.clean import adjust, io, lifecycle, limits, pipeline, sources


@dataclass(frozen=True)
class StageTarget:
    """一个需要计时的清洗函数。"""

    module: Any
    name: str
    label: str
    category: str
    pure_array: bool


@dataclass(frozen=True)
class CallObservation:
    """一次函数调用的耗时。"""

    label: str
    category: str
    pure_array: bool
    elapsed_s: float


STAGE_TARGETS = (
    StageTarget(sources, "load_api", "source_parquet_load", "parquet_input", False),
    StageTarget(io, "_prepare", "panel_normalize", "pandas_assembly", False),
    StageTarget(io, "_assign", "panel_assign", "pandas_assembly", False),
    StageTarget(lifecycle, "build_reference", "build_reference", "pandas_assembly", False),
    StageTarget(lifecycle, "build_status", "build_status", "numpy_rules", True),
    StageTarget(io, "load_st_matrix", "st_matrix", "pandas_assembly", False),
    StageTarget(io, "load_suspend_matrix", "suspend_matrix", "pandas_assembly", False),
    StageTarget(lifecycle, "build_resume_flags", "resume_flags", "numpy_rules", True),
    StageTarget(adjust, "forward_fill_close", "forward_fill_close", "numpy_rules", True),
    StageTarget(adjust, "shift_down", "shift_down", "numpy_rules", True),
    StageTarget(adjust, "compute_pre_close", "compute_pre_close", "numpy_rules", True),
    StageTarget(adjust, "compute_return", "compute_return", "numpy_rules", True),
    StageTarget(adjust, "mark_ex_dividend", "mark_ex_dividend", "numpy_rules", True),
    StageTarget(adjust, "last_valid_close", "last_valid_close", "numpy_rules", True),
    StageTarget(limits, "compute_limit_ratios", "limit_ratios", "numpy_rules", True),
    StageTarget(limits, "compute_limit_prices", "limit_prices", "numpy_rules", True),
    StageTarget(limits, "prefer_official", "prefer_official", "numpy_rules", True),
    StageTarget(limits, "official_consistency", "official_consistency", "numpy_rules", True),
    StageTarget(limits, "mark_limits", "limit_marks", "numpy_rules", True),
    StageTarget(limits, "mark_tradeable", "tradeable_marks", "numpy_rules", True),
    StageTarget(pipeline, "_to_frame", "frame_assembly", "pandas_assembly", False),
)


def stage_definitions() -> list[dict[str, Any]]:
    """返回证据中记录的固定阶段和函数映射。"""

    return [
        {
            "function": f"{target.module.__name__}.{target.name}",
            "label": target.label,
            "category": target.category,
            "pure_array": target.pure_array,
        }
        for target in STAGE_TARGETS
    ]


def summarize_observations(
    observations: list[CallObservation],
    total_elapsed_s: float,
) -> dict[str, Any]:
    """汇总不重叠的阶段调用，并计算相对总耗时的占比。"""

    if total_elapsed_s <= 0:
        raise ValueError("总耗时必须大于 0")
    stages: dict[str, dict[str, Any]] = {}
    for item in observations:
        current = stages.setdefault(
            item.label,
            {
                "category": item.category,
                "pure_array": item.pure_array,
                "count": 0,
                "elapsed_s": 0.0,
            },
        )
        current["count"] += 1
        current["elapsed_s"] += item.elapsed_s

    for current in stages.values():
        current["elapsed_s"] = round(current["elapsed_s"], 6)
        current["share"] = round(current["elapsed_s"] / total_elapsed_s, 6)

    measured = round(sum(item["elapsed_s"] for item in stages.values()), 6)
    categories: dict[str, float] = {}
    for current in stages.values():
        category = str(current["category"])
        categories[category] = round(
            categories.get(category, 0.0) + float(current["elapsed_s"]),
            6,
        )
    categories["unattributed"] = round(
        max(total_elapsed_s - measured, 0.0),
        6,
    )
    pure_array_stages = [
        (name, current)
        for name, current in stages.items()
        if current["pure_array"]
    ]
    largest_name = None
    largest = {"elapsed_s": 0.0, "share": 0.0, "category": "numpy_rules"}
    if pure_array_stages:
        largest_name, largest_stage = max(
            pure_array_stages,
            key=lambda item: item[1]["elapsed_s"],
        )
        largest = {
            "elapsed_s": largest_stage["elapsed_s"],
            "share": largest_stage["share"],
            "category": largest_stage["category"],
        }
    return {
        "elapsed_s": round(total_elapsed_s, 6),
        "stages": dict(sorted(stages.items())),
        "categories": dict(sorted(categories.items())),
        "measured_elapsed_s": measured,
        "unattributed_s": categories["unattributed"],
        "pure_array_elapsed_s": round(
            sum(float(current["elapsed_s"]) for _, current in pure_array_stages),
            6,
        ),
        "largest_pure_array_stage": largest_name,
        "largest_pure_array_elapsed_s": largest["elapsed_s"],
        "largest_pure_array_share": largest["share"],
    }


class CleanProfileProbe:
    """在当前进程内包装清洗函数，退出时恢复所有原始属性。"""

    def __init__(self) -> None:
        self.observations: list[CallObservation] = []
        self._originals: list[tuple[Any, str, Callable[..., Any]]] = []

    def _wrap(self, target: StageTarget) -> Callable[..., Any]:
        original = getattr(target.module, target.name)

        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.observations.append(
                    CallObservation(
                        label=target.label,
                        category=target.category,
                        pure_array=target.pure_array,
                        elapsed_s=time.perf_counter() - started,
                    )
                )

        measured.__name__ = getattr(original, "__name__", target.name)
        measured.__doc__ = getattr(original, "__doc__", None)
        return measured

    def __enter__(self) -> CleanProfileProbe:
        for target in STAGE_TARGETS:
            original = getattr(target.module, target.name)
            self._originals.append((target.module, target.name, original))
            setattr(target.module, target.name, self._wrap(target))

        original_to_parquet = pd.DataFrame.to_parquet

        def measured_to_parquet(frame, *args, **kwargs):
            started = time.perf_counter()
            try:
                return original_to_parquet(frame, *args, **kwargs)
            finally:
                self.observations.append(
                    CallObservation(
                        label="parquet_write",
                        category="parquet_output",
                        pure_array=False,
                        elapsed_s=time.perf_counter() - started,
                    )
                )

        self._originals.append((pd.DataFrame, "to_parquet", original_to_parquet))
        pd.DataFrame.to_parquet = measured_to_parquet
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        pd.DataFrame.to_parquet = self._originals[-1][2]
        for module, name, original in reversed(self._originals[:-1]):
            setattr(module, name, original)
        self._originals.clear()

    def summary(self, total_elapsed_s: float) -> dict[str, Any]:
        """返回当前观测的汇总结果。"""

        return summarize_observations(self.observations, total_elapsed_s)
