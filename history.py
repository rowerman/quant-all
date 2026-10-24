"""全历史 IC 取数：只在 DuckDB 内计算逐日统计，不物化完整评估面板。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd

from calculate.config import DB_FILE, FACTOR_DIR
from calculate.daily import MIN_CROSS_SECTION
from calculate.errors import EvalParamError
from calculate.logging_conf import get_logger
from calculate.normalize import MAD_SCALE
from calculate.panel import (
    FactorView, check_label_columns, count_trade_days, open_source, resolve_factor,
)
from calculate.params import EvalParams
from calculate.universe import membership_cte, resolve_universe, TRADABLE_FILTER
logger = get_logger('calculate.history')


def ic_sql(
    params: EvalParams,
    *,
    factor_dir: Path | str = FACTOR_DIR,
    factor: FactorView | None = None,
) -> str:
    """生成逐日 IC 的 SQL：有效配对内平均并列排名，Pearson 可选 MAD 去极值。

    参数:
        params: 因子、股票池、区间、标签与去极值参数。
        factor_dir: 正式因子对照根目录。
        factor: 显式因子来源；留空时按名称从正式注册表解析。
    """
    spec = factor or resolve_factor(params.factor, factor_dir)
    if spec.name != params.factor:
        raise EvalParamError(
            f"显式因子名 {spec.name!r} 与评估参数 {params.factor!r} 不一致"
        )
    universe = resolve_universe(params.universe)
    membership = membership_cte(universe, params.start, params.end) + ', ' if universe.is_index else ''
    member_join = ('JOIN member m ON m.stock_code = f.stock_code AND m.trade_date = f.trade_date'
                   if universe.is_index else '')
    path = spec.parquet_glob.replace("'", "''")
    normalise = ''
    pearson_x, pearson_y = 'factor', 'label'
    if params.winsorize:
        limit = 3 * MAD_SCALE
        normalise = f''', medians AS (
            SELECT *, median(factor) OVER (PARTITION BY trade_date) AS mx,
                      median(label) OVER (PARTITION BY trade_date) AS my FROM ranked
        ), deviations AS (
            SELECT *, median(abs(factor-mx)) OVER (PARTITION BY trade_date) AS dx,
                      median(abs(label-my)) OVER (PARTITION BY trade_date) AS dy FROM medians
        ), clipped AS (
            SELECT *, CASE WHEN dx>0 THEN greatest(mx-{limit}*dx, least(factor,mx+{limit}*dx))
                           ELSE factor END AS px,
                      CASE WHEN dy>0 THEN greatest(my-{limit}*dy, least(label,my+{limit}*dy))
                           ELSE label END AS py FROM deviations
        )'''
        pearson_x, pearson_y = 'px', 'py'
    source = 'clipped' if params.winsorize else 'ranked'
    return f'''WITH {membership} joined AS (
        SELECT f.trade_date, CAST(f.{spec.name} AS DOUBLE) AS factor,
               CAST(y.{params.label_column} AS DOUBLE) AS label
        FROM read_parquet('{path}', hive_partitioning=1) f
        JOIN src.md.fwd_return y ON y.stock_code=f.stock_code AND y.trade_date=f.trade_date
        JOIN src.md.daily_hfq b ON b.stock_code=f.stock_code AND b.trade_date=f.trade_date
        {member_join}
        WHERE b.trade_date BETWEEN DATE '{params.start}' AND DATE '{params.end}'
          AND f.trade_date BETWEEN DATE '{params.start}' AND DATE '{params.end}'
          AND {TRADABLE_FILTER} AND isfinite(f.{spec.name}) AND isfinite(y.{params.label_column})
    ), ranked AS (
        SELECT *, rank() OVER (PARTITION BY trade_date ORDER BY factor)
                    + (count(*) OVER (PARTITION BY trade_date,factor)-1)/2.0 AS rx,
                  rank() OVER (PARTITION BY trade_date ORDER BY label)
                    + (count(*) OVER (PARTITION BY trade_date,label)-1)/2.0 AS ry
        FROM joined
    ){normalise}
    SELECT trade_date, CASE WHEN count(*) >= {MIN_CROSS_SECTION}
                          THEN corr({pearson_x},{pearson_y}) END AS ic,
                       CASE WHEN count(*) >= {MIN_CROSS_SECTION} THEN corr(rx,ry) END AS rank_ic,
                       count(*) AS n
    FROM {source} GROUP BY trade_date ORDER BY trade_date'''


def load_daily_ic(
    params: EvalParams,
    *,
    db_file: Path | str = DB_FILE,
    factor_dir: Path | str = FACTOR_DIR,
    factor: FactorView | None = None,
) -> pd.DataFrame:
    """读取任意区间（含全历史）的日度 IC、RankIC 与配对数。

    参数:
        params: 同 build_panel 的评估参数；仅使用因子、日期、股票池、标签与 winsorize。
        db_file: 行情主库路径。
        factor_dir: 因子 parquet 根目录。
        factor: 显式因子来源；留空时按名称从正式注册表解析。

    返回:
        交易日索引的 DataFrame，列 ic、rank_ic、n。无配对的日期不出现；
        不设完整面板的行数/天数上限。DuckDB 工作内存限制 1GB，溢出临时文件自动清理。
        attrs['elapsed_ms'] 为本次含校验和取数的耗时，结果不持久化。
    """
    started = time.perf_counter()
    sql = ic_sql(params, factor_dir=factor_dir, factor=factor)
    conn = open_source(db_file)
    try:
        if count_trade_days(conn, params.start, params.end) == 0:
            raise EvalParamError(f'区间 {params.start} ~ {params.end} 内没有交易日')
        check_label_columns(conn, [params.label_column])
        with TemporaryDirectory(prefix='calculate-ic-') as scratch:
            conn.execute("SET memory_limit='1GB'")
            conn.execute('SET threads=4')
            conn.execute('SET temp_directory=?', [scratch])
            frame = conn.execute(sql).df().set_index('trade_date')
    finally:
        conn.close()
    # corr 在零方差时可能返回 NaN/inf，统一为无定义，不计入有效日。
    frame[['ic', 'rank_ic']] = frame[['ic', 'rank_ic']].replace([np.inf, -np.inf], np.nan)
    frame.attrs['elapsed_ms'] = (time.perf_counter() - started) * 1000
    logger.info('全历史 IC 就绪：因子=%s 池=%s 区间=%s~%s 配对=%d 日期=%d 耗时=%.0fms',
                params.factor, params.universe, params.start, params.end,
                frame['n'].sum(), len(frame), frame.attrs['elapsed_ms'])
    return frame
