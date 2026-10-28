"""候选与正式因子的抽样逐日秩相关。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

import factors
from kernels import get_backend
from mining.evaluation_contract import QualityPolicy
from mining.logging_conf import format_event, get_mining_logger
from scripts.build_factors.sql import family_glob

_LOGGER = get_mining_logger("mining.evaluation.correlation")
_POOL_FILTER = (
    "b.can_buy_close AND NOT b.is_st "
    "AND datediff('day', b.list_date, b.trade_date) >= 60"
)


@dataclass
class CandidateCorrelationCache:
    """单次挖掘筛选内复用正式 peer 宽表，不跨运行共享。"""

    peer_factor_dir: Path | str
    _key: tuple[Path, tuple[str, ...]] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _grid: _PeerGrid | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def load(
        self,
        conn,
        dates: list[str] | tuple[str, ...],
    ) -> _PeerGrid:
        """读取当前筛选窗口的正式 peer；窗口或目录变化时重新加载。"""

        key = (Path(self.peer_factor_dir).resolve(), tuple(dates))
        if self._key != key or self._grid is None:
            started = time.perf_counter()
            frame, peers = _load_peer_frame(
                conn,
                peer_factor_dir=self.peer_factor_dir,
                dates=dates,
            )
            grid = _dense_peer_grid(frame, peers)
            self._key = key
            self._grid = grid
            _LOGGER.info(
                format_event(
                    {
                        "operation": "candidate_peer_cache_load",
                        "peer_factor_dir": str(key[0]),
                        "sample_days": len(dates),
                        "rows": len(frame),
                        "peers": len(peers),
                        "elapsed_ms": round(
                            (time.perf_counter() - started) * 1000,
                            3,
                        ),
                    }
                )
            )
        return self._grid


@dataclass(frozen=True)
class _PeerGrid:
    """缓存后的正式 peer 三维矩阵，第一维为 peer。"""

    dates: pd.Index
    stocks: pd.Index
    peers: tuple[str, ...]
    values: np.ndarray


def _quoted(path: Path | str) -> str:
    """转义 DuckDB 文件路径字面量。"""

    return str(path).replace("'", "''")


def _sample_date_sql(dates: list[str] | tuple[str, ...]) -> str:
    """把受控日期样本写为 SQL 列表。"""

    return ", ".join(f"DATE '{day}'" for day in dates)


def _dense_peer_grid(
    frame: pd.DataFrame,
    peers: tuple[str, ...],
) -> _PeerGrid:
    """把正式 peer 宽表转换为连续的三维日期股票矩阵。"""

    dates = pd.Index(sorted(frame["trade_date"].unique()))
    stocks = pd.Index(sorted(frame["stock_code"].unique()))
    date_codes = dates.get_indexer(frame["trade_date"])
    stock_codes = stocks.get_indexer(frame["stock_code"])
    positions = date_codes * stocks.size + stock_codes
    values = np.full(
        (len(peers), dates.size, stocks.size),
        np.nan,
        dtype=np.float64,
    )
    for index, peer in enumerate(peers):
        values[index].ravel()[positions] = frame[peer].to_numpy(
            dtype=np.float64
        )
    return _PeerGrid(dates, stocks, peers, values)


def _correlate_matrices(
    target: np.ndarray,
    peer_values: np.ndarray,
    *,
    peers: tuple[str, ...],
    policy: QualityPolicy,
    row_count: int,
) -> dict[str, Any]:
    """用统一逐行核计算已对齐矩阵的目标与 peer 相关性。"""

    selection = get_backend()
    backend = selection.backend
    started = time.perf_counter()
    sums = np.zeros(len(peers), dtype=np.float64)
    days = np.zeros(len(peers), dtype=np.int64)
    for index, peer in enumerate(peers):
        correlation, count = backend.daily_row_spearman(
            target,
            peer_values[index],
            policy.correlation_min_stocks,
        )
        valid = np.isfinite(correlation) & (
            count >= policy.correlation_min_stocks
        )
        sums[index] = float(correlation[valid].sum())
        days[index] = int(valid.sum())
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    _LOGGER.info(
        format_event(
            {
                "operation": "candidate_correlation",
                "requested_backend": selection.requested,
                "actual_backend": selection.name,
                "fallback_reason": selection.fallback_reason,
                "rows": row_count,
                "days": target.shape[0],
                "stocks": target.shape[1],
                "peers": len(peers),
                "min_stocks": policy.correlation_min_stocks,
                "min_days": policy.correlation_min_days,
                "elapsed_ms": elapsed_ms,
            }
        )
    )

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
        "observed_days": int(days[best]),
        "min_days": policy.correlation_min_days,
        "peer": peers[best],
        "correlation": float(means[best]),
        "max_absolute": maximum,
        "threshold": policy.correlation_max_absolute,
    }


def _correlate_frame(
    frame: pd.DataFrame,
    *,
    target_name: str,
    peers: tuple[str, ...],
    policy: QualityPolicy,
) -> dict[str, Any]:
    """把宽表按日期×股票对齐后执行目标相关性。"""

    grid = _dense_peer_grid(frame, (target_name, *peers))
    return _correlate_matrices(
        grid.values[0],
        grid.values[1:],
        peers=peers,
        policy=policy,
        row_count=len(frame),
    )


def _target_grid(
    frame: pd.DataFrame,
    *,
    target_name: str,
    dates: pd.Index,
    stocks: pd.Index,
) -> np.ndarray:
    """把候选目标宽表对齐到缓存网格。"""

    target = np.full((dates.size, stocks.size), np.nan, dtype=np.float64)
    date_codes = dates.get_indexer(frame["trade_date"])
    stock_codes = stocks.get_indexer(frame["stock_code"])
    valid = (date_codes >= 0) & (stock_codes >= 0)
    positions = date_codes[valid] * stocks.size + stock_codes[valid]
    target.ravel()[positions] = frame.loc[valid, target_name].to_numpy(
        dtype=np.float64
    )
    return target


def _load_peer_frame(
    conn,
    *,
    peer_factor_dir: Path | str,
    dates: list[str] | tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """读取正式 peer 列和行情股票日骨架。"""

    date_sql = _sample_date_sql(dates)
    frame = conn.execute(
        f"""
        SELECT b.stock_code, b.trade_date
        FROM src.md.daily_hfq b
        WHERE b.trade_date IN ({date_sql}) AND {_POOL_FILTER}
        """
    ).df()
    peers: list[str] = []
    for family in factors.ALL_FAMILIES:
        if family == "mined":
            specs = [
                item
                for item in factors.available_specs(peer_factor_dir)
                if item.family == family
            ]
        else:
            specs = list(factors.family_factors(family))
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
    return frame, tuple(peers)


def _load_target_frame(
    conn,
    *,
    target_name: str,
    target_glob: str,
    dates: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """读取候选目标列并与行情股票日骨架内连接。"""

    date_sql = _sample_date_sql(dates)
    return conn.execute(
        f"""
        SELECT b.stock_code, b.trade_date,
               CAST(f.{target_name} AS DOUBLE) AS {target_name}
        FROM read_parquet('{_quoted(target_glob)}', hive_partitioning=1) f
        JOIN src.md.daily_hfq b USING(stock_code, trade_date)
        WHERE b.trade_date IN ({date_sql}) AND {_POOL_FILTER}
        """
    ).df()


def target_correlation(
    conn,
    *,
    target_name: str,
    target_glob: str,
    peer_factor_dir: Path | str,
    dates: list[str] | tuple[str, ...],
    policy: QualityPolicy,
    cache: CandidateCorrelationCache | None = None,
) -> dict[str, Any]:
    """计算候选对正式因子的抽样逐日秩相关。"""

    if not dates:
        return {
            "passed": False,
            "reason": "no_trade_dates",
            "sample_days": 0,
            "max_absolute": None,
            "peer": None,
        }
    target = _load_target_frame(
        conn,
        target_name=target_name,
        target_glob=target_glob,
        dates=dates,
    )
    if target.empty:
        return {
            "passed": False,
            "reason": "empty_target",
            "sample_days": len(dates),
            "max_absolute": None,
            "peer": None,
        }
    if cache is not None:
        grid = cache.load(conn, dates)
        peer_names = grid.peers
        selected = tuple(
            (index, name)
            for index, name in enumerate(peer_names)
            if name != target_name
        )
        peers = tuple(name for _, name in selected)
    else:
        peer_frame, peer_names = _load_peer_frame(
            conn,
            peer_factor_dir=peer_factor_dir,
            dates=dates,
        )
        peers = tuple(name for name in peer_names if name != target_name)
    if not peers:
        return {
            "passed": True,
            "reason": "no_formal_peers",
            "sample_days": len(dates),
            "max_absolute": None,
            "peer": None,
            "observed_days": 0,
        }
    if cache is not None:
        target_grid = _target_grid(
            target,
            target_name=target_name,
            dates=grid.dates,
            stocks=grid.stocks,
        )
        peer_values = grid.values[[index for index, _ in selected]]
    else:
        combined = target.merge(
            peer_frame[["stock_code", "trade_date", *peers]],
            on=["stock_code", "trade_date"],
            how="left",
            validate="one_to_one",
        )
        dense = _dense_peer_grid(combined, (target_name, *peers))
        target_grid = dense.values[0]
        peer_values = dense.values[1:]
    result = _correlate_matrices(
        target_grid,
        peer_values,
        peers=peers,
        policy=policy,
        row_count=len(target),
    )
    result["sample_days"] = len(dates)
    return result
