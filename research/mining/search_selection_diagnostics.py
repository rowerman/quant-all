"""选择期候选相关性计算和诊断公共入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from calculate import open_source
from kernels import get_backend
from mining.errors import MiningEvaluationError
from mining.search_incremental_value import (
    compute_incremental_value,
    formal_factor_incremental_value,
)
from mining.search_selection_contracts import (
    CLUSTER_CORRELATION_THRESHOLD,
    SELECTION_DIAGNOSTICS_SCHEMA_VERSION,
    SELECTION_DIAGNOSTICS_STAGE,
    CandidateCorrelationReport,
    IncrementalValueReport,
    SelectionDiagnosticsReport,
    build_selection_diagnostics,
)


def sample_trade_dates(
    conn,
    start: str,
    end: str,
    count: int,
) -> tuple[str, ...]:
    """在冻结窗口内均匀选择确定性交易日。"""

    rows = [
        str(row[0])
        for row in conn.execute(
            "SELECT trade_date FROM src.md.calendar "
            f"WHERE trade_date BETWEEN DATE '{start}' AND DATE '{end}' "
            "ORDER BY trade_date"
        ).fetchall()
    ]
    if count <= 0 or not rows:
        return ()
    if len(rows) <= count:
        return tuple(rows)
    if count == 1:
        return (rows[0],)
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return tuple(rows[int(index)] for index in indices)


def load_candidate_values(
    candidate_name: str,
    factor_glob: str | Path,
    dates: Iterable[str],
) -> pd.DataFrame:
    """读取一个候选在采样日的股票日取值。"""

    selected = tuple(str(item) for item in dates)
    if not selected:
        return pd.DataFrame(columns=["stock_code", "trade_date", "value"])
    date_sql = ", ".join(f"DATE '{day}'" for day in selected)
    quoted = str(factor_glob).replace("'", "''")
    conn = open_source()
    try:
        frame = conn.execute(
            f"""
            SELECT stock_code, trade_date,
                   CAST("{candidate_name}" AS DOUBLE) AS value
            FROM read_parquet('{quoted}', hive_partitioning=1)
            WHERE trade_date IN ({date_sql})
            """
        ).df()
    finally:
        conn.close()
    if frame.duplicated(["stock_code", "trade_date"]).any():
        raise MiningEvaluationError(
            f"候选 {candidate_name} 在采样日存在重复的股票日行"
        )
    return frame


def _candidate_grid(
    frame: pd.DataFrame,
    dates: pd.Index,
    stocks: pd.Index,
) -> np.ndarray:
    """把候选长表对齐到统一日期和股票网格。"""

    grid = np.full((dates.size, stocks.size), np.nan, dtype=np.float64)
    if frame.empty:
        return grid
    date_codes = dates.get_indexer(pd.to_datetime(frame["trade_date"]))
    stock_codes = stocks.get_indexer(frame["stock_code"])
    valid = (date_codes >= 0) & (stock_codes >= 0)
    if not valid.any():
        return grid
    values = frame["value"].to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    positions = date_codes[valid] * stocks.size + stock_codes[valid]
    grid.ravel()[positions] = values[valid]
    return grid


def _cluster_ids(
    candidate_ids: tuple[str, ...],
    peers: Mapping[str, tuple[Mapping[str, Any], ...]],
    threshold: float,
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """按绝对相关度阈值生成稳定连通分量。"""

    parent = {candidate_id: candidate_id for candidate_id in candidate_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    for candidate_id in candidate_ids:
        for peer in peers[candidate_id]:
            correlation = peer.get("correlation")
            if isinstance(correlation, (int, float)) and not isinstance(
                correlation,
                bool,
            ) and abs(float(correlation)) >= threshold:
                union(candidate_id, str(peer["candidate_id"]))
    groups: dict[str, list[str]] = {}
    for candidate_id in candidate_ids:
        groups.setdefault(find(candidate_id), []).append(candidate_id)
    ordered = sorted(tuple(sorted(items)) for items in groups.values())
    result: dict[str, tuple[str, tuple[str, ...]]] = {}
    for index, members in enumerate(ordered, start=1):
        cluster_id = f"cluster-{index:04d}"
        for candidate_id in members:
            result[candidate_id] = (cluster_id, members)
    return result


def compute_candidate_correlations(
    *,
    run_id: str,
    window_start: str,
    window_end: str,
    values: Mapping[str, pd.DataFrame],
    min_stocks: int,
    min_days: int,
    threshold: float = CLUSTER_CORRELATION_THRESHOLD,
) -> dict[str, CandidateCorrelationReport]:
    """计算同批候选两两逐日 Spearman 相关性和相关簇。"""

    candidate_ids = tuple(sorted(values))
    if not candidate_ids:
        return {}
    frames = {
        candidate_id: frame.assign(
            stock_code=frame["stock_code"].astype(str),
            trade_date=frame["trade_date"].astype(str),
        )
        for candidate_id, frame in values.items()
    }
    dates = sorted(
        {item for frame in frames.values() for item in frame["trade_date"]}
    )
    stocks = sorted(
        {item for frame in frames.values() for item in frame["stock_code"]}
    )
    if not dates or not stocks:
        return {
            candidate_id: CandidateCorrelationReport(
                run_id=run_id,
                candidate_id=candidate_id,
                window_start=window_start,
                window_end=window_end,
                sample_dates=(),
                peers=(),
                max_abs_correlation=None,
                cluster_id="cluster-0001",
                cluster_candidate_ids=(candidate_id,),
                status="not_computable",
                reason="采样日或股票日骨架为空",
            )
            for candidate_id in candidate_ids
        }
    date_index = pd.Index(pd.to_datetime(dates))
    stock_index = pd.Index(stocks)
    grids = {
        candidate_id: _candidate_grid(frame, date_index, stock_index)
        for candidate_id, frame in frames.items()
    }
    backend = get_backend().backend
    pair_values: dict[str, dict[str, tuple[float | None, int, str]]] = {
        candidate_id: {} for candidate_id in candidate_ids
    }
    for left_index, left_id in enumerate(candidate_ids):
        for right_id in candidate_ids[left_index + 1 :]:
            correlation, counts = backend.daily_row_spearman(
                np.ascontiguousarray(grids[left_id]),
                np.ascontiguousarray(grids[right_id]),
                min_stocks,
            )
            valid = np.isfinite(correlation) & (counts >= min_stocks)
            valid_days = int(valid.sum())
            if valid_days < min_days:
                mean = None
                status = "insufficient_days"
            else:
                mean = float(correlation[valid].mean())
                status = "available"
            pair_values[left_id][right_id] = (mean, valid_days, status)
            pair_values[right_id][left_id] = (mean, valid_days, status)

    peers: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for candidate_id in candidate_ids:
        peers[candidate_id] = tuple(
            {
                "candidate_id": peer_id,
                "correlation": pair_values[candidate_id][peer_id][0],
                "valid_days": pair_values[candidate_id][peer_id][1],
                "status": pair_values[candidate_id][peer_id][2],
            }
            for peer_id in candidate_ids
            if peer_id != candidate_id
        )
    clusters = _cluster_ids(candidate_ids, peers, threshold)
    result: dict[str, CandidateCorrelationReport] = {}
    for candidate_id in candidate_ids:
        available = [
            abs(float(item["correlation"]))
            for item in peers[candidate_id]
            if item["correlation"] is not None
        ]
        maximum = max(available) if available else None
        status = (
            "available"
            if any(item["status"] == "available" for item in peers[candidate_id])
            else ("not_computable" if len(candidate_ids) > 1 else "single_candidate")
        )
        cluster_id, members = clusters[candidate_id]
        result[candidate_id] = CandidateCorrelationReport(
            run_id=run_id,
            candidate_id=candidate_id,
            window_start=window_start,
            window_end=window_end,
            sample_dates=tuple(dates),
            peers=peers[candidate_id],
            max_abs_correlation=maximum,
            cluster_id=cluster_id,
            cluster_candidate_ids=members,
            status=status,
            reason=(
                None
                if status in {"available", "single_candidate"}
                else "没有达到最少有效日数的候选对"
            ),
        )
    return result


__all__ = [
    "CLUSTER_CORRELATION_THRESHOLD",
    "SELECTION_DIAGNOSTICS_SCHEMA_VERSION",
    "SELECTION_DIAGNOSTICS_STAGE",
    "CandidateCorrelationReport",
    "IncrementalValueReport",
    "SelectionDiagnosticsReport",
    "build_selection_diagnostics",
    "compute_candidate_correlations",
    "compute_incremental_value",
    "formal_factor_incremental_value",
    "load_candidate_values",
    "sample_trade_dates",
]
