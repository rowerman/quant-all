"""候选技术质量门禁。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import factors
from calculate.panel import open_source
from mining.candidate_materialization import CandidateComputation
from mining.evaluation_contract import QualityPolicy
from mining.logging_conf import format_event, get_mining_logger
from scripts.build_factors.sql import family_query, family_glob

_LOGGER = get_mining_logger("mining.evaluation.gates")
_POOL_FILTER = (
    "b.can_buy_close AND NOT b.is_st "
    "AND datediff('day', b.list_date, b.trade_date) >= 60"
)


def _quoted(path: Path | str) -> str:
    """转义 DuckDB 文件路径字面量。"""

    return str(path).replace("'", "''")


def _rowset_check(
    conn,
    parquet_glob: str,
    name: str,
    start: str,
    end: str,
) -> dict[str, Any]:
    """双向核对候选键集与研究期行情键集。"""

    glob = _quoted(parquet_glob)
    row = conn.execute(
        f"""
        WITH stored AS (
            SELECT stock_code, trade_date
            FROM read_parquet('{glob}', hive_partitioning=1)
            WHERE trade_date BETWEEN DATE '{start}' AND DATE '{end}'
        ), expected AS (
            SELECT stock_code, trade_date FROM src.md.daily_hfq
            WHERE trade_date BETWEEN DATE '{start}' AND DATE '{end}'
        )
        SELECT
          (SELECT count(*) FROM (SELECT * FROM stored EXCEPT SELECT * FROM expected)),
          (SELECT count(*) FROM (SELECT * FROM expected EXCEPT SELECT * FROM stored)),
          (SELECT count(*) FROM stored)
        """
    ).fetchone()
    extra, missing, rows = (int(value) for value in row)
    return {
        "passed": extra == 0 and missing == 0,
        "extra_rows": extra,
        "missing_rows": missing,
        "rows": rows,
        "start": start,
        "end": end,
        "name": name,
    }


def _value_checks(
    conn,
    parquet_glob: str,
    name: str,
    start: str,
    end: str,
    min_coverage: float,
    max_flat_day_share: float,
) -> dict[str, Any]:
    """检查覆盖率、非有限值和截面退化。"""

    glob = _quoted(parquet_glob)
    rows, non_null, non_finite = conn.execute(
        f"""
        SELECT count(*), count({name}),
               count(*) FILTER (WHERE isinf({name}) OR isnan({name}))
        FROM read_parquet('{glob}', hive_partitioning=1)
        WHERE trade_date BETWEEN DATE '{start}' AND DATE '{end}'
        """
    ).fetchone()
    eligible_days, flat_days = conn.execute(
        f"""
        SELECT count(*) FILTER (WHERE n >= 5),
               count(*) FILTER (WHERE n >= 5 AND sd = 0)
        FROM (
            SELECT trade_date, count({name}) AS n, stddev_pop({name}) AS sd
            FROM read_parquet('{glob}', hive_partitioning=1)
            WHERE trade_date BETWEEN DATE '{start}' AND DATE '{end}'
            GROUP BY trade_date
        )
        """
    ).fetchone()
    coverage = float(non_null) / int(rows) if rows else 0.0
    flat_share = float(flat_days) / int(eligible_days) if eligible_days else 0.0
    return {
        "passed": (
            int(non_finite) == 0
            and coverage >= min_coverage
            and int(eligible_days) > 0
            and flat_share <= max_flat_day_share
        ),
        "rows": int(rows),
        "non_null": int(non_null),
        "coverage": coverage,
        "min_coverage": min_coverage,
        "non_finite": int(non_finite),
        "eligible_days": int(eligible_days),
        "flat_days": int(flat_days),
        "flat_day_share": flat_share,
        "max_flat_day_share": max_flat_day_share,
    }


def _sample_dates(conn, start: str, end: str, count: int) -> list[str]:
    """在区间内确定性选取均匀交易日。"""

    rows = [
        str(row[0])
        for row in conn.execute(
            "SELECT trade_date FROM src.md.calendar "
            f"WHERE trade_date BETWEEN DATE '{start}' AND DATE '{end}' ORDER BY trade_date"
        ).fetchall()
    ]
    if len(rows) <= count:
        return rows
    if count == 1:
        return [rows[0]]
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[index] for index in indices]


def _forward_look_check(
    conn,
    computation: CandidateComputation,
    parquet_glob: str,
    start: str,
    end: str,
    policy: QualityPolicy,
) -> dict[str, Any]:
    """抽交易日截断重算，识别未来数据泄漏。"""

    sample_dates = _sample_dates(conn, start, end, policy.forward_look_samples)
    glob = _quoted(parquet_glob)
    worst = 0.0
    null_mismatch = 0
    for day in sample_dates:
        recomputed = conn.execute(
            family_query([computation], spine_start=day, spine_end=day)
        ).df()
        stored = conn.execute(
            f"SELECT stock_code, trade_date, {computation.name} "
            f"FROM read_parquet('{glob}', hive_partitioning=1) "
            f"WHERE trade_date = DATE '{day}'"
        ).df()
        merged = stored.merge(
            recomputed,
            on=["stock_code", "trade_date"],
            suffixes=("_stored", "_recomputed"),
            validate="one_to_one",
        )
        left = merged[f"{computation.name}_stored"].astype("float64")
        right = merged[f"{computation.name}_recomputed"].astype("float64")
        null_mismatch += int((left.isna() != right.isna()).sum())
        both = left.notna() & right.notna()
        if both.any():
            difference = (left[both] - right[both]).abs()
            scale = np.maximum(1.0, right[both].abs())
            worst = max(worst, float((difference / scale).max()))
    passed = (
        worst <= policy.forward_look_relative_tolerance
        and null_mismatch == 0
        and bool(sample_dates)
    )
    return {
        "passed": passed,
        "sample_dates": sample_dates,
        "max_relative_error": worst,
        "tolerance": policy.forward_look_relative_tolerance,
        "null_mismatch": null_mismatch,
    }


def _sample_date_sql(dates: list[str]) -> str:
    """把受控日期样本写为 SQL 列表。"""

    return ", ".join(f"DATE '{day}'" for day in dates)


def _target_correlation(
    conn,
    *,
    target_name: str,
    target_glob: str,
    peer_factor_dir: Path | str,
    start: str,
    end: str,
    policy: QualityPolicy,
) -> dict[str, Any]:
    """计算候选对正式因子的抽样逐日秩相关。"""

    dates = _sample_dates(conn, start, end, policy.correlation_sample_days)
    if not dates:
        return {
            "passed": False,
            "reason": "no_trade_dates",
            "sample_days": 0,
            "max_absolute": None,
            "peer": None,
        }
    date_sql = _sample_date_sql(dates)
    target = conn.execute(
        f"""
        SELECT b.stock_code, b.trade_date,
               CAST(f.{target_name} AS DOUBLE) AS {target_name}
        FROM read_parquet('{_quoted(target_glob)}', hive_partitioning=1) f
        JOIN src.md.daily_hfq b USING(stock_code, trade_date)
        WHERE b.trade_date IN ({date_sql}) AND {_POOL_FILTER}
        """
    ).df()
    if target.empty:
        return {
            "passed": False,
            "reason": "empty_target",
            "sample_days": len(dates),
            "max_absolute": None,
            "peer": None,
        }
    peers: list[str] = []
    frame = target
    for family in factors.ALL_FAMILIES:
        if family == "mined":
            specs = [
                item
                for item in factors.available_specs(peer_factor_dir)
                if item.family == family
            ]
        else:
            specs = list(factors.family_factors(family))
        specs = [
            spec
            for spec in specs
            if spec.name != target_name
        ]
        family_dir = Path(peer_factor_dir) / family
        glob = family_glob(peer_factor_dir, family)
        if not specs or not tuple(family_dir.glob("year=*/part.parquet")):
            continue
        columns = ", ".join(
            f'CAST(f."{spec.name}" AS DOUBLE) AS "{spec.name}"' for spec in specs
        )
        values = conn.execute(
            f"""
            SELECT b.stock_code, b.trade_date, {columns}
            FROM read_parquet('{_quoted(glob)}', hive_partitioning=1, union_by_name=1) f
            JOIN src.md.daily_hfq b USING(stock_code, trade_date)
            WHERE b.trade_date IN ({date_sql}) AND {_POOL_FILTER}
            """
        ).df()
        if values.empty:
            continue
        frame = frame.merge(
            values,
            on=["stock_code", "trade_date"],
            how="left",
            validate="one_to_one",
        )
        peers.extend(spec.name for spec in specs)
    if not peers:
        return {
            "passed": True,
            "reason": "no_formal_peers",
            "sample_days": len(dates),
            "max_absolute": None,
            "peer": None,
            "observed_days": 0,
        }
    sums = np.zeros(len(peers))
    days = np.zeros(len(peers), dtype=np.int64)
    for _, daily in frame.groupby("trade_date", sort=False):
        target_values = daily[[target_name, *peers]].to_numpy(dtype="float64")
        for index, peer in enumerate(peers, start=1):
            pair = target_values[:, [0, index]]
            pair = pair[np.isfinite(pair).all(axis=1)]
            if len(pair) < policy.correlation_min_stocks:
                continue
            left = pd.Series(pair[:, 0]).rank(method="average").to_numpy()
            right = pd.Series(pair[:, 1]).rank(method="average").to_numpy()
            if np.std(left) == 0 or np.std(right) == 0:
                continue
            correlation = float(np.corrcoef(left, right)[0, 1])
            if np.isfinite(correlation):
                sums[index - 1] += correlation
                days[index - 1] += 1
    means = np.divide(
        sums,
        days,
        out=np.full(sums.shape, np.nan),
        where=days >= policy.correlation_min_days,
    )
    finite = np.isfinite(means)
    if not finite.any():
        return {
            "passed": False,
            "reason": "insufficient_days",
            "sample_days": len(dates),
            "max_absolute": None,
            "peer": None,
            "min_days": policy.correlation_min_days,
        }
    positions = np.flatnonzero(finite)
    best = int(positions[np.argmax(np.abs(means[positions]))])
    maximum = float(abs(means[best]))
    return {
        "passed": maximum < policy.correlation_max_absolute,
        "reason": None,
        "sample_days": len(dates),
        "observed_days": int(days[best]),
        "min_days": policy.correlation_min_days,
        "peer": peers[best],
        "correlation": float(means[best]),
        "max_absolute": maximum,
        "threshold": policy.correlation_max_absolute,
    }


def evaluate_technical_checks(
    computation: CandidateComputation,
    *,
    parquet_glob: str,
    target_name: str,
    peer_factor_dir: Path | str,
    start: str,
    end: str,
    min_coverage: float,
    policy: QualityPolicy,
) -> tuple[bool, dict[str, Any], tuple[str, ...]]:
    """执行全部硬门禁并返回可供审批绑定的证据。"""

    conn = open_source()
    try:
        rowset = _rowset_check(conn, parquet_glob, computation.name, start, end)
        values = _value_checks(
            conn,
            parquet_glob,
            computation.name,
            start,
            end,
            min_coverage,
            policy.max_flat_day_share,
        )
        forward = _forward_look_check(
            conn, computation, parquet_glob, start, end, policy
        )
        correlation = _target_correlation(
            conn,
            target_name=target_name,
            target_glob=parquet_glob,
            peer_factor_dir=peer_factor_dir,
            start=start,
            end=end,
            policy=policy,
        )
    finally:
        conn.close()
    checks = {
        "rule_version": policy.schema_version,
        "rowset": rowset,
        "values": values,
        "forward_look": forward,
        "existing_correlation": correlation,
    }
    passed = all(
        bool(checks[key]["passed"])
        for key in ("rowset", "values", "forward_look", "existing_correlation")
    )
    warnings: list[str] = []
    if not rowset["passed"]:
        warnings.append("候选行集与行情库不一致")
    if not values["passed"]:
        warnings.append("候选覆盖率、非有限值或截面退化未通过")
    if not forward["passed"]:
        warnings.append("候选前视检查未通过")
    if not correlation["passed"]:
        warnings.append("候选与正式因子的相关性或相关样本未通过")
    _LOGGER.info(
        format_event(
            {
                "operation": "technical_gate",
                "candidate_id": computation.name,
                "passed": passed,
                "rule_version": policy.schema_version,
            }
        )
    )
    return passed, checks, tuple(warnings)
