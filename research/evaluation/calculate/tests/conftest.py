"""评估相关测试共用的夹具。"""

from __future__ import annotations

import pytest

from scripts.build_factors import build as build_module
from calculate.config import DB_FILE

YEAR = 2024


@pytest.fixture(scope="session")
def momentum_dir(tmp_path_factory):
    """把动量族 2024 年的因子构建到临时目录，供评估相关测试复用。"""
    if not DB_FILE.exists():
        pytest.skip("需要真实主库 data/databases/quant.duckdb")
    root = tmp_path_factory.mktemp("calculate_momentum")
    build_module.build(
        ["momentum"], f"{YEAR}-01-01", f"{YEAR}-12-31",
        factor_dir=root, catalog_file=root / "factors.duckdb", write_catalog=False,
    )
    return root
