"""最终留出窗口的注册、读取和范围校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from alpha.config import PROJECT_ROOT, RUNS_DIR
from alpha.errors import AlphaConfigError

HOLDOUT_WINDOWS_SCHEMA_VERSION = "alpha_holdout_windows_v1"
DEFAULT_HOLDOUT_WINDOWS_PATH = (
    PROJECT_ROOT / "configs" / "alpha" / "holdout_windows.json"
)
LEGACY_HOLDOUT_WINDOW_ID = "holdout_2024_2026"
FUTURE_HOLDOUT_WINDOW_ID = "holdout_2026_2027"


def _iso_date(value: str, name: str) -> str:
    """校验 ISO 日期。"""

    try:
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except ValueError as error:
        raise AlphaConfigError(f"{name} 必须是 YYYY-MM-DD：{value!r}") from error
    return value


@dataclass(frozen=True)
class HoldoutWindow:
    """一个不可复用的最终留出窗口。"""

    window_id: str
    start: str
    end: str
    min_label_coverage: float = 0.95
    min_signal_days: int = 100
    sealed: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.window_id, str)
            or not self.window_id
            or Path(self.window_id).name != self.window_id
        ):
            raise AlphaConfigError("留出窗口 ID 非法")
        start = _iso_date(self.start, "holdout.start")
        end = _iso_date(self.end, "holdout.end")
        if start > end:
            raise AlphaConfigError("留出窗口起始日晚于截止日")
        if not 0.0 < float(self.min_label_coverage) <= 1.0:
            raise AlphaConfigError("min_label_coverage 必须在 (0, 1]")
        if type(self.min_signal_days) is not int or self.min_signal_days <= 0:
            raise AlphaConfigError("min_signal_days 必须是正整数")
        if not isinstance(self.sealed, bool):
            raise AlphaConfigError("留出窗口 sealed 必须是布尔值")

    def to_dict(self) -> dict:
        """转换为稳定字典。"""

        return {
            "window_id": self.window_id,
            "start": self.start,
            "end": self.end,
            "min_label_coverage": self.min_label_coverage,
            "min_signal_days": self.min_signal_days,
            "sealed": self.sealed,
        }


def load_holdout_windows(
    path: Path | None = None,
) -> tuple[HoldoutWindow, ...]:
    """读取并校验留出窗口注册表。"""

    target = Path(path or DEFAULT_HOLDOUT_WINDOWS_PATH)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AlphaConfigError(f"无法读取留出窗口注册表：{target}") from error
    if not isinstance(payload, dict):
        raise AlphaConfigError("留出窗口注册表必须是 JSON 对象")
    if payload.get("schema_version") != HOLDOUT_WINDOWS_SCHEMA_VERSION:
        raise AlphaConfigError("留出窗口注册表 schema_version 不支持")
    records = payload.get("windows")
    if not isinstance(records, list) or not records:
        raise AlphaConfigError("留出窗口注册表没有窗口")
    windows = tuple(
        HoldoutWindow(
            window_id=str(item.get("window_id") or ""),
            start=str(item.get("start") or ""),
            end=str(item.get("end") or ""),
            min_label_coverage=float(item.get("min_label_coverage", 0.95)),
            min_signal_days=int(item.get("min_signal_days", 100)),
            sealed=bool(item.get("sealed", False)),
        )
        for item in records
    )
    identifiers = [item.window_id for item in windows]
    if len(set(identifiers)) != len(identifiers):
        raise AlphaConfigError("留出窗口 ID 重复")
    ordered = sorted(windows, key=lambda item: (item.start, item.window_id))
    for left, right in zip(ordered, ordered[1:]):
        if left.end >= right.start:
            raise AlphaConfigError(
                f"留出窗口重叠：{left.window_id} 与 {right.window_id}"
            )
    return windows


def get_holdout_window(
    window_id: str,
    path: Path | None = None,
) -> HoldoutWindow:
    """按 ID 返回留出窗口。"""

    for window in load_holdout_windows(path):
        if window.window_id == window_id:
            return window
    raise AlphaConfigError(f"未知留出窗口：{window_id}")


def validate_holdout_range(
    start: str,
    end: str,
    window: HoldoutWindow,
) -> None:
    """确认日期完整落在指定留出窗口内。"""

    start_text = _iso_date(start, "start")
    end_text = _iso_date(end, "end")
    if start_text > end_text:
        raise AlphaConfigError("留出区间起始日晚于截止日")
    if start_text < window.start or end_text > window.end:
        raise AlphaConfigError(
            f"留出区间超出 {window.window_id}："
            f"{window.start}～{window.end}"
        )


def holdout_run_ids(
    window: HoldoutWindow,
    root: Path | None = None,
) -> tuple[str, ...]:
    """返回已引用指定窗口的留出运行 ID。"""

    folder = Path(root or RUNS_DIR) / "final_holdout"
    if not folder.is_dir():
        return ()
    identifiers: list[str] = []
    for path in sorted(folder.iterdir()):
        manifest_path = path / "manifest.json"
        if not path.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise AlphaConfigError(
                f"无法读取留出运行 manifest：{manifest_path}"
            ) from error
        scope = manifest.get("scope") or {}
        start = scope.get("data_start") or scope.get("start")
        end = scope.get("data_end") or scope.get("end")
        if str(start) != window.start or str(end) != window.end:
            continue
        identifiers.append(
            str(
                manifest.get("holdout_id")
                or path.name
            )
        )
    return tuple(identifiers)


__all__ = [
    "DEFAULT_HOLDOUT_WINDOWS_PATH",
    "FUTURE_HOLDOUT_WINDOW_ID",
    "HOLDOUT_WINDOWS_SCHEMA_VERSION",
    "HoldoutWindow",
    "LEGACY_HOLDOUT_WINDOW_ID",
    "get_holdout_window",
    "holdout_run_ids",
    "load_holdout_windows",
    "validate_holdout_range",
]
