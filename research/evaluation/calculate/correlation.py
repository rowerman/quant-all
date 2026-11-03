"""按月读取因子，累计逐日成对 Spearman；不使用收益标签筛选样本。"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pandas as pd

import factors
from calculate.config import FACTOR_DIR
from calculate.errors import MetricError
from calculate.logging_conf import get_logger

if TYPE_CHECKING:
    from kernels import BackendSelection, NativeKernelBackend

logger = get_logger('calculate.correlation')


def available_factors(factor_dir: Path | str = FACTOR_DIR) -> dict[str, tuple[str, ...]]:
    """列出已注册且已落盘的因子，参数 factor_dir 为因子数据根目录。"""
    result = {}
    for family in factors.ALL_FAMILIES:
        files = sorted(Path(factor_dir).joinpath(family).glob('year=*/part.parquet'))
        if not files:
            continue
        with duckdb.connect() as conn:
            columns = {item[0] for item in conn.execute('SELECT * FROM read_parquet(?) LIMIT 0',
                                                       [str(files[-1])]).description}
        specs = (
            tuple(
                item
                for item in factors.available_specs(factor_dir)
                if item.family == family
            )
            if family == "mined"
            else factors.family_factors(family)
        )
        names = tuple(spec.name for spec in specs if spec.name in columns)
        if names:
            result[family] = names
    return result


def _backend_selection(
    backend: "NativeKernelBackend | None" = None,
) -> "BackendSelection":
    """返回带选择信息的后端，兼容调用方直接传入后端实例。"""

    from kernels import BackendSelection, get_backend

    if backend is None:
        return get_backend()
    if isinstance(backend, BackendSelection):
        return backend
    return BackendSelection(backend, backend.name, None)


def _kernel_spearman(
    values: np.ndarray,
    *,
    target_index: int | None,
    backend: "NativeKernelBackend | None" = None,
) -> np.ndarray:
    """调用批量核计算单个交易日的完整矩阵或目标行。"""

    if values.ndim != 2:
        raise MetricError('相关性输入必须是股票×因子二维数组')
    if target_index is not None and (
        type(target_index) is not int or not 0 <= target_index < values.shape[1]
    ):
        raise MetricError('目标因子位置超出相关性矩阵范围')
    array = np.ascontiguousarray(values, dtype=np.float64)
    selection = _backend_selection(backend)
    offsets = np.array([0, array.shape[0]], dtype=np.int64)
    correlation, _ = selection.backend.pairwise_spearman(
        array,
        offsets,
        5,
        target_index,
    )
    return correlation[0]


def accumulate_correlations(
    frame: pd.DataFrame,
    names: list[str],
    sums: np.ndarray,
    days: np.ndarray,
    backend: "NativeKernelBackend | None" = None,
) -> None:
    """累计每天成对有效截面的秩相关；参数为月表、列名及原地更新的和与计数矩阵。"""

    for _, daily in frame.groupby('trade_date', sort=False):
        matrix = daily_spearman_matrix(
            daily[names].to_numpy(dtype='float64'),
            backend=backend,
        )
        valid = np.isfinite(matrix)
        sums += np.where(valid, matrix, 0)
        days += valid


@dataclass(frozen=True)
class _MaskSelection:
    """一个缺失掩码键对应的布尔掩码、紧凑位置和有效样本数。"""

    mask: np.ndarray
    positions: np.ndarray
    count: int


@dataclass(frozen=True)
class _RankedColumn:
    """单因子在一种成对掩码下的中心化归一化秩和范数。"""

    ranks: np.ndarray
    norm: float


class _DailySpearmanWorkspace:
    """管理单个交易日内的有限值掩码、稳定排序和秩缓存。"""

    def __init__(self, values: np.ndarray):
        self.values = values
        self.factor_values = np.ascontiguousarray(values.T)
        self.finite = np.isfinite(values)
        self.packed = np.packbits(self.finite, axis=0)
        grouped: dict[bytes, list[int]] = {}
        for column in range(values.shape[1]):
            grouped.setdefault(self._column_key(column), []).append(column)
        self.groups = list(grouped.values())
        self._mask_cache: dict[bytes, _MaskSelection] = {}
        self._rank_cache: dict[tuple[int, bytes], _RankedColumn] = {}
        self.orders = [
            np.argsort(self.factor_values[column], kind='stable')
            for column in range(values.shape[1])
        ]

    def _column_key(self, column: int) -> bytes:
        """返回单因子有限值掩码的打包字节键。"""

        return self.packed[:, column].tobytes()

    def group_key(self, group: list[int]) -> bytes:
        """返回一组相同缺失掩码因子共用的键。"""

        return self._column_key(group[0])

    def intersection_key(self, left_column: int, right_column: int) -> bytes:
        """返回两个因子有效样本交集的打包字节键。"""

        return np.bitwise_and(
            self.packed[:, left_column],
            self.packed[:, right_column],
        ).tobytes()

    def selection(self, key: bytes) -> _MaskSelection:
        """按掩码键读取样本位置；同键只在当前工作日解包一次。"""

        selected = self._mask_cache.get(key)
        if selected is not None:
            return selected
        mask = np.unpackbits(
            np.frombuffer(key, dtype=np.uint8),
            count=self.values.shape[0],
        ).astype(bool)
        positions = np.full(self.values.shape[0], -1, dtype=np.intp)
        count = int(mask.sum())
        positions[mask] = np.arange(count, dtype=np.intp)
        selected = _MaskSelection(mask, positions, count)
        self._mask_cache[key] = selected
        return selected

    def ranked(self, columns: list[int], key: bytes) -> tuple[np.ndarray, np.ndarray]:
        """返回指定因子在掩码样本内的中心化归一化秩和范数。"""

        selected = self.selection(key)
        ranked = np.empty((selected.count, len(columns)), dtype=np.float64)
        norms = np.empty(len(columns), dtype=np.float64)
        for local, column in enumerate(columns):
            cache_key = (column, key)
            cached = self._rank_cache.get(cache_key)
            if cached is None:
                cached = _RankedColumn(
                    *self._build_ranked_column(column, selected)
                )
                self._rank_cache[cache_key] = cached
            ranked[:, local] = cached.ranks
            norms[local] = cached.norm
        return ranked, norms

    def _build_ranked_column(
        self,
        column: int,
        selected: _MaskSelection,
    ) -> tuple[np.ndarray, float]:
        """按稳定排序索引生成交集样本的并列平均名次并直接归一化。"""

        order = self.orders[column][selected.mask[self.orders[column]]]
        ordered = self.factor_values[column, order]
        starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        ends = np.r_[starts[1:], selected.count]
        sizes = ends - starts
        averages = (starts + ends - 1) / 2.0 + 1.0
        ranks = np.empty(selected.count, dtype=np.float64)
        ranks[selected.positions[order]] = np.repeat(averages, sizes)

        mean = (selected.count + 1.0) / 2.0
        deviations = averages - mean
        norm = float(np.sqrt(np.sum(sizes * deviations * deviations)))
        ranks -= mean
        if norm > 0:
            ranks /= norm
        return ranks, norm


def _daily_spearman(values: np.ndarray, target_index: int | None = None) -> np.ndarray:
    """计算完整矩阵或单目标行；参数values为股票×因子矩阵。"""

    if values.ndim != 2:
        raise MetricError('相关性输入必须是股票×因子二维数组')
    if target_index is not None and (
        type(target_index) is not int or not 0 <= target_index < values.shape[1]
    ):
        raise MetricError('目标因子位置超出相关性矩阵范围')
    workspace = _DailySpearmanWorkspace(values)
    grouped = workspace.groups
    result = np.full((values.shape[1], values.shape[1]), np.nan)

    if target_index is not None:
        for group in grouped:
            key = (
                workspace.group_key(group)
                if target_index in group
                else workspace.intersection_key(target_index, group[0])
            )
            if workspace.selection(key).count < 5:
                continue
            columns = group if target_index in group else [target_index, *group]
            normalized, norms = workspace.ranked(columns, key)
            target_local = columns.index(target_index)
            peer_local = [
                index for index in range(len(columns)) if index != target_local
            ]
            correlation = np.clip(
                normalized[:, target_local] @ normalized[:, peer_local],
                -1,
                1,
            )
            correlation[
                ~(norms[target_local] > 0)
                | ~(norms[peer_local] > 0)
            ] = np.nan
            peers = [columns[index] for index in peer_local]
            result[target_index, peers] = correlation
            result[peers, target_index] = correlation
            if target_index in group and norms[target_local] > 0:
                result[target_index, target_index] = np.clip(
                    normalized[:, target_local] @ normalized[:, target_local],
                    -1,
                    1,
                )
        return result

    for i, left in enumerate(grouped):
        for right in grouped[i:]:
            key = (
                workspace.group_key(left)
                if left is right
                else workspace.intersection_key(left[0], right[0])
            )
            if workspace.selection(key).count < 5:
                continue
            columns = left if left is right else left + right
            normalized, norms = workspace.ranked(columns, key)
            left_ranks = normalized[:, :len(left)]
            right_ranks = left_ranks if left is right else normalized[:, len(left):]
            matrix = np.clip(left_ranks.T @ right_ranks, -1, 1)
            valid_left = norms[:len(left)] > 0
            valid_right = valid_left if left is right else norms[len(left):] > 0
            matrix[~(valid_left[:, None] & valid_right[None, :])] = np.nan
            result[np.ix_(left, right)] = matrix
            result[np.ix_(right, left)] = matrix.T
    return result


def daily_spearman_matrix(
    values: np.ndarray,
    *,
    target_index: int | None = None,
    backend: "NativeKernelBackend | None" = None,
) -> np.ndarray:
    """按配置后端计算完整 Spearman 矩阵或单目标行；成对最少 5 只。"""

    return _kernel_spearman(values, target_index=target_index, backend=backend)


def _accumulate_target_correlations(
    frame: pd.DataFrame,
    names: list[str],
    target: str,
    sums: np.ndarray,
    days: np.ndarray,
    backend: "NativeKernelBackend | None" = None,
) -> None:
    """只累计目标因子行和列；参数为月表、列名、目标名及原地累计矩阵。"""

    if target not in names:
        raise MetricError(f'目标因子 {target!r} 不在相关性列中')
    position = names.index(target)
    peers = np.arange(len(names)) != position
    for _, daily in frame.groupby('trade_date', sort=False):
        matrix = daily_spearman_matrix(
            daily[names].to_numpy(dtype='float64'),
            target_index=position,
            backend=backend,
        )
        row = matrix[position]
        valid = peers & np.isfinite(row)
        columns = np.flatnonzero(valid)
        sums[position, columns] += row[valid]
        days[position, columns] += 1
        sums[columns, position] += row[valid]
        days[columns, position] += 1
        diagonal = matrix[position, position]
        if np.isfinite(diagonal):
            sums[position, position] += diagonal
            days[position, position] += 1


def correlation_result(names: list[str], sums: np.ndarray, days: np.ndarray,
                       factor: str, top_n: int = 10) -> dict:
    """生成均值矩阵与 Top-N；参数为累计统计、目标因子和展示数量。"""
    if type(top_n) is not int or not 1 <= top_n <= 100:
        raise MetricError('top_n 必须是 1～100 的整数')
    means = np.divide(sums, days, out=np.full(sums.shape, np.nan), where=days >= 20)
    position = names.index(factor)
    peers = [{'factor': name, 'correlation': float(means[position, i]),
              'days': int(days[position, i])} for i, name in enumerate(names)
             if name != factor and np.isfinite(means[position, i])]
    peers.sort(key=lambda item: (-abs(item['correlation']), item['factor']))
    return {'names': names, 'matrix': means, 'days': days, 'top': peers[:top_n],
            'top_n': top_n, 'min_days': 20, 'min_stocks': 5,
            'basis': '同池逐日成对 Spearman 的时间等权均值；成对剔除非有限值后重新排名，不筛收益标签',
            'note': '不足20个有效日返回空值；不同因子对样本可不同，均值矩阵不保证半正定',
            'insufficient': not peers}


def compute_correlation(context, options) -> dict:
    """读取上下文所在数据目录的所有因子，按月对齐池内键并计算相关性。"""
    if type(options['top_n']) is not int or not 1 <= options['top_n'] <= 100:
        raise MetricError('top_n 必须是 1～100 的整数')
    selection = _backend_selection()
    if not getattr(getattr(context, "spec", None), "formal", True):
        return compute_target_correlation(context, options, backend=selection)
    from kernels import native_threads

    started = time.perf_counter()
    rows = 0
    groups = 0
    threads = native_threads() if selection.name == "native" else 1
    grouped = available_factors(context.factor_dir)
    names = [name for family_names in grouped.values() for name in family_names]
    if context.params.factor not in names:
        raise MetricError('目标因子没有可用于相关性计算的落盘列')
    sums = np.zeros((len(names), len(names)))
    days = np.zeros(sums.shape, dtype=np.int64)
    pool = context.panel.loc[context.panel.in_universe, ['stock_code', 'trade_date']]
    with duckdb.connect() as conn:
        conn.execute("SET memory_limit='512MB'")
        conn.execute('SET threads=2')
        for month_key, keys in pool.groupby(pool.trade_date.dt.to_period('M'), sort=True):
            conn.register('pool_keys', keys)
            frame = keys.copy()
            for family, family_names in grouped.items():
                path = str(Path(context.factor_dir) / family / 'year=*' / 'part.parquet')
                selections = ','.join('f."' + name.replace('"', '""') + '"' for name in family_names)
                month = conn.execute(f'''SELECT f.stock_code, f.trade_date, {selections}
                    FROM read_parquet(?, hive_partitioning=1, union_by_name=1) f
                    JOIN pool_keys k USING(stock_code, trade_date)
                    WHERE f.trade_date BETWEEN ? AND ?''',
                    [path, keys.trade_date.min(), keys.trade_date.max()]).df()
                frame = frame.merge(month, on=['stock_code', 'trade_date'], how='left',
                                    validate='one_to_one')
            accumulate_correlations(frame, names, sums, days, backend=selection)
            rows += len(frame)
            groups += int(frame['trade_date'].nunique())
            conn.unregister('pool_keys')
            logger.info('相关性月块完成：factor=%s month=%s rows=%d factors=%d',
                        context.params.factor, month_key, len(frame), len(names))
    result = correlation_result(
        names,
        sums,
        days,
        context.params.factor,
        options['top_n'],
    )
    logger.info(
        "相关性后端完成 scope=full factor=%s requested=%s backend=%s "
        "fallback_reason=%s threads=%d rows=%d groups=%d factors=%d elapsed_ms=%.3f",
        context.params.factor,
        selection.requested,
        selection.name,
        selection.fallback_reason or "none",
        threads,
        rows,
        groups,
        len(names),
        (time.perf_counter() - started) * 1000,
    )
    return result


def compute_target_correlation(
    context,
    options,
    backend: "NativeKernelBackend | None" = None,
) -> dict:
    """只计算候选对正式因子的相关性，避免构建正式因子两两全矩阵。"""

    selection = _backend_selection(backend)
    from kernels import native_threads

    started = time.perf_counter()
    rows = 0
    groups = 0
    threads = native_threads() if selection.name == "native" else 1
    grouped = available_factors(context.factor_dir)
    formal_names = [
        name
        for family_names in grouped.values()
        for name in family_names
        if name != context.params.factor
    ]
    names = [context.params.factor, *formal_names]
    sums = np.zeros((len(names), len(names)))
    days = np.zeros(sums.shape, dtype=np.int64)
    pool = context.panel.loc[context.panel.in_universe, ['stock_code', 'trade_date']]
    target_column = 'f."' + context.params.factor.replace('"', '""') + '"'
    with duckdb.connect() as conn:
        conn.execute("SET memory_limit='512MB'")
        conn.execute('SET threads=2')
        for month_key, keys in pool.groupby(pool.trade_date.dt.to_period('M'), sort=True):
            conn.register('pool_keys', keys)
            frame = keys.copy()
            target = conn.execute(
                "SELECT f.stock_code, f.trade_date, "
                + target_column
                + " FROM read_parquet(?, hive_partitioning=1) f "
                "JOIN pool_keys k USING(stock_code, trade_date) "
                "WHERE f.trade_date BETWEEN ? AND ?",
                [context.spec.parquet_glob, keys.trade_date.min(), keys.trade_date.max()],
            ).df()
            frame = frame.merge(
                target, on=['stock_code', 'trade_date'], how='left',
                validate='one_to_one',
            )
            for family, family_names in grouped.items():
                selected = [name for name in family_names if name != context.params.factor]
                if not selected:
                    continue
                path = str(Path(context.factor_dir) / family / 'year=*' / 'part.parquet')
                selections = ','.join('f."' + name.replace('"', '""') + '"' for name in selected)
                month = conn.execute(f'''SELECT f.stock_code, f.trade_date, {selections}
                    FROM read_parquet(?, hive_partitioning=1, union_by_name=1) f
                    JOIN pool_keys k USING(stock_code, trade_date)
                    WHERE f.trade_date BETWEEN ? AND ?''',
                    [path, keys.trade_date.min(), keys.trade_date.max()]).df()
                frame = frame.merge(
                    month, on=['stock_code', 'trade_date'], how='left',
                    validate='one_to_one',
                )
            _accumulate_target_correlations(
                frame,
                names,
                context.params.factor,
                sums,
                days,
                backend=selection,
            )
            rows += len(frame)
            groups += int(frame['trade_date'].nunique())
            conn.unregister('pool_keys')
            logger.info('候选相关性月块完成：factor=%s month=%s rows=%d factors=%d',
                        context.params.factor, month_key, len(frame), len(names))
    result = correlation_result(
        names,
        sums,
        days,
        context.params.factor,
        options['top_n'],
    )
    logger.info(
        "相关性后端完成 scope=target factor=%s requested=%s backend=%s "
        "fallback_reason=%s threads=%d rows=%d groups=%d factors=%d elapsed_ms=%.3f",
        context.params.factor,
        selection.requested,
        selection.name,
        selection.fallback_reason or "none",
        threads,
        rows,
        groups,
        len(names),
        (time.perf_counter() - started) * 1000,
    )
    return result
