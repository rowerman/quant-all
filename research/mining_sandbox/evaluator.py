"""沙箱代码限制和因子输出的确定性评估。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa

from mining.contract_types import sha256_json
from mining_sandbox.code_policy import (
    SandboxProposalPolicy,
    validate_proposal_code,
)
from mining_sandbox.errors import SandboxSnapshotError
from mining_sandbox.feedback import (
    SandboxCodeFeedback,
    SandboxFailureCategory,
    SandboxFeedbackError,
)
from mining_sandbox.metrics import (
    parse_research_metrics,
    read_research_metrics,
)
from mining_sandbox.qlib_snapshot import QlibSnapshotManifest

SANDBOX_EVALUATION_SCHEMA_VERSION = "mining_sandbox_evaluation_v1"
SANDBOX_FACTOR_COLUMNS = ("date", "instrument", "value")


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_file(
    snapshot_dir: Path,
    manifest: QlibSnapshotManifest,
    relative: str,
    *,
    expected_sha256: str,
) -> Path:
    root = snapshot_dir.resolve()
    path = snapshot_dir / relative
    if path.is_symlink() or not path.is_file():
        raise SandboxSnapshotError(f"冻结快照缺少文件：{relative}")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise SandboxSnapshotError(f"冻结快照路径越界：{relative}") from exc
    if _file_sha256(path) != expected_sha256:
        raise SandboxSnapshotError(f"冻结快照文件哈希变化：{relative}")
    expected = manifest.files.get(relative)
    if expected is not None:
        if int(expected.get("bytes", -1)) != path.stat().st_size:
            raise SandboxSnapshotError(f"冻结快照文件大小变化：{relative}")
        if str(expected.get("sha256")) != expected_sha256:
            raise SandboxSnapshotError(f"冻结快照清单哈希不一致：{relative}")
    return path


def _read_snapshot_index(
    snapshot_manifest: QlibSnapshotManifest,
    snapshot_dir: Path | str,
) -> tuple[set[str], dict[str, tuple[str, str]]]:
    root = Path(snapshot_dir)
    calendar_path = _snapshot_file(
        root,
        snapshot_manifest,
        "calendars/day.txt",
        expected_sha256=snapshot_manifest.calendar_sha256,
    )
    instruments_path = _snapshot_file(
        root,
        snapshot_manifest,
        "instruments/all.txt",
        expected_sha256=snapshot_manifest.instruments_sha256,
    )
    calendar = {
        line.strip()
        for line in calendar_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    instruments: dict[str, tuple[str, str]] = {}
    for line in instruments_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 3:
            raise SandboxSnapshotError("冻结股票索引行格式非法")
        code, start, end = parts
        if code in instruments:
            raise SandboxSnapshotError(f"冻结股票索引重复：{code}")
        instruments[code] = (start, end)
    if not calendar or not instruments:
        raise SandboxSnapshotError("冻结快照日历或股票索引为空")
    return calendar, instruments


def _normalized_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise _metrics_feedback(
            SandboxFailureCategory.INDEX,
            "因子明细包含无效日期",
            "把 date 写成冻结日历中的 ISO 交易日",
        )
    return parsed.dt.strftime("%Y-%m-%d")


def _normalized_instruments(series: pd.Series) -> pd.Series:
    if not series.map(lambda value: isinstance(value, str)).all():
        raise _metrics_feedback(
            SandboxFailureCategory.OUTPUT_FORMAT,
            "factor.instrument 类型不符合契约",
            "instrument 必须写成 Qlib 股票代码字符串",
        )
    normalized = series.str.strip()
    if (normalized == "").any():
        raise _metrics_feedback(
            SandboxFailureCategory.INDEX,
            "因子明细包含空股票代码",
            "每一行都必须绑定冻结快照中的股票代码",
        )
    return normalized


def _read_factor_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise _metrics_feedback(
            SandboxFailureCategory.OUTPUT_FORMAT,
            "缺少沙箱因子明细文件",
            "把研究结果写到 /workspace/factor.parquet",
        )
    try:
        frame = pd.read_parquet(path)
    except (OSError, TypeError, ValueError, pa.ArrowException) as exc:
        raise _metrics_feedback(
            SandboxFailureCategory.OUTPUT_FORMAT,
            f"沙箱因子明细不可读：{type(exc).__name__}",
            "写出合法 Parquet，固定列为 date、instrument、value",
        ) from exc
    if not isinstance(frame, pd.DataFrame) or set(frame.columns) != set(
        SANDBOX_FACTOR_COLUMNS
    ):
        raise _metrics_feedback(
            SandboxFailureCategory.OUTPUT_FORMAT,
            "沙箱因子明细字段不符合契约",
            "factor.parquet 必须且只能包含 date、instrument、value",
        )
    if len(frame.columns) != len(SANDBOX_FACTOR_COLUMNS):
        raise _metrics_feedback(
            SandboxFailureCategory.OUTPUT_FORMAT,
            "沙箱因子明细包含重复列",
            "factor.parquet 必须且只能包含 date、instrument、value",
        )
    if frame.empty:
        raise _metrics_feedback(
            SandboxFailureCategory.INDEX,
            "沙箱因子明细没有有效行",
            "至少输出一个属于冻结交易日和股票索引的有限因子值",
        )
    if not pd.api.types.is_numeric_dtype(frame["value"].dtype) or (
        pd.api.types.is_bool_dtype(frame["value"].dtype)
    ):
        raise _metrics_feedback(
            SandboxFailureCategory.OUTPUT_FORMAT,
            "factor.value 类型不符合契约",
            "value 必须是有限数值列",
        )
    values = frame["value"].to_numpy(dtype="float64", copy=False)
    if not np.isfinite(values).all():
        raise _metrics_feedback(
            SandboxFailureCategory.NUMERIC,
            "因子明细包含非有限值",
            "过滤 NaN 和正负无穷，缺失因子直接省略对应行",
        )
    frame = frame.copy()
    frame["date"] = _normalized_dates(frame["date"])
    frame["instrument"] = _normalized_instruments(frame["instrument"])
    return frame


@dataclass(frozen=True)
class SandboxFactorEvaluation:
    """一次因子明细确定性评估的稳定结果。"""

    snapshot_manifest_hash: str
    factor_sha256: str
    factor_bytes: int
    row_count: int
    finite_value_count: int
    valid_days: int
    instrument_count: int
    coverage: float
    schema_version: str = SANDBOX_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SANDBOX_EVALUATION_SCHEMA_VERSION:
            raise ValueError("沙箱评估结果版本不匹配")
        for name in ("snapshot_manifest_hash", "factor_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} 必须是 SHA-256")
        for name in (
            "factor_bytes",
            "row_count",
            "finite_value_count",
            "valid_days",
            "instrument_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if (
            isinstance(self.coverage, bool)
            or not isinstance(self.coverage, float)
            or not 0 <= self.coverage <= 1
        ):
            raise ValueError("coverage 必须是 0～1 的浮点数")

    def material(self) -> dict[str, Any]:
        """返回不包含宿主路径和时间的稳定评估内容。"""

        return {
            "schema_version": self.schema_version,
            "snapshot_manifest_hash": self.snapshot_manifest_hash,
            "factor_sha256": self.factor_sha256,
            "factor_bytes": self.factor_bytes,
            "row_count": self.row_count,
            "finite_value_count": self.finite_value_count,
            "valid_days": self.valid_days,
            "instrument_count": self.instrument_count,
            "coverage": self.coverage,
        }

    @property
    def evaluation_hash(self) -> str:
        """返回评估结果稳定哈希。"""

        return sha256_json(self.material())

    def to_dict(self) -> dict[str, Any]:
        return {**self.material(), "evaluation_hash": self.evaluation_hash}


def evaluate_factor_output(
    *,
    factor_file: Path | str,
    snapshot_manifest: QlibSnapshotManifest,
    snapshot_dir: Path | str,
    metrics: Mapping[str, Any],
) -> SandboxFactorEvaluation:
    """按冻结快照确定性校验因子明细和汇总指标。"""

    normalized_metrics = parse_research_metrics(metrics)
    factor_path = Path(factor_file)
    frame = _read_factor_frame(factor_path)
    if frame.duplicated(["date", "instrument"]).any():
        raise _metrics_feedback(
            SandboxFailureCategory.INDEX,
            "因子明细包含重复的日期和股票主键",
            "每个交易日和股票只能保留一行",
        )
    calendar, instruments = _read_snapshot_index(snapshot_manifest, snapshot_dir)
    if (~frame["date"].isin(calendar)).any():
        raise _metrics_feedback(
            SandboxFailureCategory.INDEX,
            "因子明细包含冻结日历外日期",
            "只输出 /snapshot 冻结交易日内的因子值",
        )
    if (~frame["instrument"].isin(instruments)).any():
        raise _metrics_feedback(
            SandboxFailureCategory.INDEX,
            "因子明细包含冻结股票索引外代码",
            "只输出 /snapshot 冻结股票索引中的代码",
        )
    for code, dates in frame.groupby("instrument", sort=True)["date"]:
        start, end = instruments[str(code)]
        if ((dates < start) | (dates > end)).any():
            raise _metrics_feedback(
                SandboxFailureCategory.INDEX,
                "因子明细包含股票有效期外日期",
                "按冻结股票索引的上市和失效区间过滤因子行",
            )
    if snapshot_manifest.row_count <= 0:
        raise SandboxSnapshotError("冻结快照行数必须大于零")
    coverage = float(len(frame) / snapshot_manifest.row_count)
    if coverage > 1:
        raise _metrics_feedback(
            SandboxFailureCategory.COVERAGE,
            "因子明细行数超过冻结快照",
            "去掉重复或快照外行后重新计算覆盖率",
        )
    valid_days = int(frame["date"].nunique())
    if normalized_metrics["valid_days"] != valid_days:
        raise _metrics_feedback(
            SandboxFailureCategory.INDEX,
            "metrics.valid_days 与因子明细不一致",
            "按 factor.parquet 中实际出现的交易日重新计算 valid_days",
        )
    if not math.isclose(
        normalized_metrics["coverage"],
        coverage,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise _metrics_feedback(
            SandboxFailureCategory.COVERAGE,
            "metrics.coverage 与因子明细不一致",
            "用有限因子行数除以冻结快照 row_count 重新计算 coverage",
        )
    return SandboxFactorEvaluation(
        snapshot_manifest_hash=snapshot_manifest.manifest_hash,
        factor_sha256=_file_sha256(factor_path),
        factor_bytes=factor_path.stat().st_size,
        row_count=snapshot_manifest.row_count,
        finite_value_count=len(frame),
        valid_days=valid_days,
        instrument_count=int(frame["instrument"].nunique()),
        coverage=coverage,
    )


__all__ = [
    "SANDBOX_EVALUATION_SCHEMA_VERSION",
    "SANDBOX_FACTOR_COLUMNS",
    "SandboxFactorEvaluation",
    "SandboxProposalPolicy",
    "evaluate_factor_output",
    "parse_research_metrics",
    "read_research_metrics",
    "validate_proposal_code",
]
