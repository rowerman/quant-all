"""口径一致性：评估包算出的 RankIC 必须与既有自检脚本对得上。

两套实现走的是完全不同的路径（一个是 pandas + numpy 的分位相关，
一个是 DuckDB 的 percent_rank + corr），如果哪天口径漂了，这条测试会先炸。
"""

import re

import pytest

from calculate.metrics import ic as ic_metric
from calculate.panel import EvalParams, build_panel, open_source
from scripts.build_factors import validate
from calculate.config import DB_FILE

pytestmark = pytest.mark.skipif(not DB_FILE.exists(), reason="需要真实主库 db/quant.duckdb")

START, END = "2024-01-01", "2024-12-31"


def test_rank_ic_matches_validate_direction_check(momentum_dir):
    """同区间同股票池下，RankIC 均值与有效交易日数都与 validate.py 一致。"""
    context = build_panel(
        EvalParams(factor="MOM_20", start=START, end=END), factor_dir=momentum_dir
    )
    mine = ic_metric.compute_stats(context, {})

    conn = open_source()
    try:
        checks = validate.check_direction(conn, "momentum", START, END, factor_dir=momentum_dir)
    finally:
        conn.close()
    row = next(item for item in checks if item["name"] == "MOM_20")

    # 两边都先取有效配对，再用平均并列名次，恢复原定一致性验收。
    assert row["value"] == pytest.approx(mine["rank_ic"]["mean"], abs=1e-6)
    days = int(re.search(r"(\d+) 个交易日", row["detail"]).group(1))
    assert days == mine["valid_days"]
