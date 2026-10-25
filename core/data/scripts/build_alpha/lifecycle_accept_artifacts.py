"""P5-3 验收产物写入、复用校验和基准对照。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from alpha.artifacts import file_sha256
from alpha.errors import AlphaLifecycleError
from backtest.specs import StrategySpec

CANDIDATE = "ML_LIFECYCLE"
CANDIDATE_SCHEMA_VERSION = "p5_3_candidate_v1"
CANDIDATE_FILES = (
    "metrics.json",
    "strategy.json",
    "yearly.csv",
    "daily.parquet",
    "orders.parquet",
    "holdings.parquet",
    "run_manifest.json",
)


def json_default(value):
    """把日期和 NumPy 标量转换成稳定 JSON 值。"""

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def atomic_json(path: Path, payload: dict) -> None:
    """原子写入 JSON。"""

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    """原子写入 CSV 或 Parquet。"""

    temporary = path.with_name(f".{path.name}.tmp")
    if path.suffix == ".csv":
        frame.to_csv(temporary, index=False)
    elif path.suffix == ".parquet":
        frame.reset_index(drop=True).to_parquet(temporary, index=False)
    else:
        raise AlphaLifecycleError(f"不支持的验收文件：{path}")
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    """原子写入文本。"""

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    """读取 JSON 对象。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AlphaLifecycleError(f"验收 JSON 必须是对象：{path}")
    return payload


def write_candidate(
    folder: Path,
    *,
    analysis_id: str,
    strategy: StrategySpec,
    result,
    metrics: dict,
    yearly: pd.DataFrame,
) -> dict:
    """写入单候选账本产物和完成标记。"""

    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "metrics.json", metrics)
    atomic_json(folder / "strategy.json", strategy.to_dict())
    atomic_frame(folder / "yearly.csv", yearly)
    daily = result.daily.copy()
    if not isinstance(daily.index, pd.RangeIndex):
        daily.index.name = "trade_date"
        daily = daily.reset_index()
    atomic_frame(folder / "daily.parquet", daily)
    atomic_frame(folder / "orders.parquet", result.orders)
    atomic_frame(folder / "holdings.parquet", result.holdings)
    atomic_json(folder / "run_manifest.json", result.metadata)
    complete = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "candidate": CANDIDATE,
        "files": {
            name: file_sha256(folder / name) for name in CANDIDATE_FILES
        },
    }
    atomic_json(folder / "complete.json", complete)
    return complete


def candidate_is_complete(folder: Path, analysis_id: str) -> bool:
    """校验已有候选产物是否完整。"""

    path = folder / "complete.json"
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
        if (
            payload.get("schema_version") != CANDIDATE_SCHEMA_VERSION
            or payload.get("analysis_id") != analysis_id
            or payload.get("candidate") != CANDIDATE
            or set(payload.get("files") or {}) != set(CANDIDATE_FILES)
        ):
            return False
        return all(
            (folder / name).is_file()
            and file_sha256(folder / name) == expected
            for name, expected in payload["files"].items()
        )
    except (OSError, ValueError, TypeError):
        return False


def comparison_frame(
    lifecycle_metrics: dict,
    lifecycle_manifest: dict,
    root: Path,
) -> pd.DataFrame:
    """生成生命周期策略与 P4 未启用规则候选的对照。"""

    sources = lifecycle_manifest["sources"]
    baseline_path = (
        root
        / "runs"
        / "acceptance"
        / sources["acceptance_id"]
        / sources["acceptance_candidate"]
        / "metrics.json"
    )
    baseline = read_json(baseline_path)["metrics"]
    current = lifecycle_metrics["metrics"]
    fields = (
        "annual_return",
        "excess_annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "turnover_two_way_annual",
        "total_cost",
    )
    return pd.DataFrame(
        [
            {
                "candidate": sources["acceptance_candidate"],
                "lifecycle_enabled": False,
                **{name: baseline.get(name) for name in fields},
            },
            {
                "candidate": CANDIDATE,
                "lifecycle_enabled": True,
                **{name: current.get(name) for name in fields},
            },
        ]
    )


__all__ = [
    "CANDIDATE",
    "CANDIDATE_FILES",
    "CANDIDATE_SCHEMA_VERSION",
    "atomic_frame",
    "atomic_json",
    "atomic_text",
    "candidate_is_complete",
    "comparison_frame",
    "json_default",
    "read_json",
    "write_candidate",
]
