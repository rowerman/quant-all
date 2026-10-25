"""按月读取因子，累计逐日成对 Spearman；不使用收益标签筛选样本。"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import factors
from calculate.config import FACTOR_DIR
from calculate.errors import MetricError
from calculate.logging_conf import get_logger

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


def accumulate_correlations(frame: pd.DataFrame, names: list[str],
                            sums: np.ndarray, days: np.ndarray) -> None:
    """累计每天成对有效截面的秩相关；参数为月表、列名及原地更新的和与计数矩阵。"""
    for _, daily in frame.groupby('trade_date', sort=False):
        matrix = daily_spearman_matrix(daily[names].to_numpy(dtype='float64'))
        valid = np.isfinite(matrix)
        sums += np.where(valid, matrix, 0)
        days += valid


def daily_spearman_matrix(values: np.ndarray) -> np.ndarray:
    """按相同缺失掩码分组复用排名；参数values为股票×因子矩阵，成对最少5只。"""
    finite = np.isfinite(values)
    groups = {}
    for column in range(values.shape[1]):
        key = np.packbits(finite[:, column]).tobytes()
        groups.setdefault(key, []).append(column)
    grouped = list(groups.values())
    # 每列只排序一次；成对删除后保留原顺序，再按剩余并列块重算平均名次。
    orders = [np.argsort(values[:, column], kind='stable') for column in range(values.shape[1])]
    rank_cache = {}
    result = np.full((values.shape[1], values.shape[1]), np.nan)
    for i, left in enumerate(grouped):
        for right in grouped[i:]:
            mask = finite[:, left[0]] & finite[:, right[0]]
            if mask.sum() < 5:
                continue
            columns = left if left is right else left + right
            key = np.packbits(mask).tobytes()
            positions = np.full(values.shape[0], -1, dtype=int)
            positions[mask] = np.arange(mask.sum())
            ranked = []
            for column in columns:
                cache_key = (column, key)
                if cache_key not in rank_cache:
                    order = orders[column][mask[orders[column]]]
                    ordered = values[order, column]
                    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
                    ends = np.r_[starts[1:], len(order)]
                    ranks = np.empty(len(order))
                    ranks[positions[order]] = np.repeat((starts + ends - 1)/2 + 1, ends-starts)
                    rank_cache[cache_key] = ranks
                ranked.append(rank_cache[cache_key])
            ranks = np.column_stack(ranked)
            ranks -= ranks.mean(axis=0)
            norms = np.sqrt(np.sum(ranks*ranks, axis=0))
            normalized = np.divide(ranks, norms, out=np.zeros_like(ranks), where=norms > 0)
            left_ranks = normalized[:, :len(left)]
            right_ranks = left_ranks if left is right else normalized[:, len(left):]
            matrix = np.clip(left_ranks.T @ right_ranks, -1, 1)
            valid_left = norms[:len(left)] > 0
            valid_right = valid_left if left is right else norms[len(left):] > 0
            matrix[~(valid_left[:, None] & valid_right[None, :])] = np.nan
            result[np.ix_(left, right)] = matrix
            result[np.ix_(right, left)] = matrix.T
    return result


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
    if not getattr(getattr(context, "spec", None), "formal", True):
        return compute_target_correlation(context, options)
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
            accumulate_correlations(frame, names, sums, days)
            conn.unregister('pool_keys')
            logger.info('相关性月块完成：factor=%s month=%s rows=%d factors=%d',
                        context.params.factor, month_key, len(frame), len(names))
    return correlation_result(names, sums, days, context.params.factor, options['top_n'])


def compute_target_correlation(context, options) -> dict:
    """只计算候选对正式因子的相关性，避免构建正式因子两两全矩阵。"""

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
            accumulate_correlations(frame, names, sums, days)
            conn.unregister('pool_keys')
            logger.info('候选相关性月块完成：factor=%s month=%s rows=%d factors=%d',
                        context.params.factor, month_key, len(frame), len(names))
    return correlation_result(names, sums, days, context.params.factor, options['top_n'])
