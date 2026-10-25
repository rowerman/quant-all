"""业务域目录、公共路径和日志路由的回归测试。"""

from __future__ import annotations

import importlib

import pytest

from common.paths import DB_FILE
from common.paths import FACTOR_DIR
from common.paths import MINING_CATALOG_FILE
from common.paths import MINING_DATA_DIR
from common.paths import MINING_DB_FILE
from common.paths import MINING_EVALUATION_FILE
from common.paths import MINING_WORKER_CONFIG_FILE
from common.paths import MODEL_DATA_DIR
from common.paths import PROJECT_ROOT
from common.paths import log_root_for


@pytest.mark.parametrize(
    "module_name, expected",
    [
        ("scripts.build_db.build", "core/data"),
        ("build_factors.catalog", "core/factors"),
        ("alpha.monitor", "research/modeling"),
        ("scripts.build_alpha.train", "research/modeling"),
        ("calculate.panel", "research/evaluation"),
        ("mining.storage", "research/mining"),
        ("backtest.engine", "trading/backtest"),
        ("backend.app", "apps/monitor"),
    ],
)
def test_log_root_follows_business_domain(module_name, expected):
    """模块日志写入与源码业务域对应的集中目录。"""

    assert log_root_for(module_name) == PROJECT_ROOT / "logs" / expected


@pytest.mark.parametrize(
    "module_name",
    [
        "common",
        "quant",
        "factors",
        "scripts",
        "alpha",
        "calculate",
        "backtest",
        "backend",
        "mining",
        "mining_worker",
    ],
)
def test_public_packages_remain_importable(module_name):
    """迁移后调用方仍可使用原包名。"""

    assert importlib.import_module(module_name)


def test_canonical_data_paths_exist():
    """公共路径入口指向迁移后的真实数据产物。"""

    assert DB_FILE == PROJECT_ROOT / "data/databases/quant.duckdb"
    assert FACTOR_DIR == PROJECT_ROOT / "data/factors"
    assert MODEL_DATA_DIR == PROJECT_ROOT / "data/models"
    assert DB_FILE.is_file()
    assert FACTOR_DIR.is_dir()
    assert MODEL_DATA_DIR.is_dir()
    assert MINING_DB_FILE == PROJECT_ROOT / "data/databases/factor_mining.duckdb"
    assert MINING_DATA_DIR == PROJECT_ROOT / "data/factor_mining"
    assert MINING_CATALOG_FILE == PROJECT_ROOT / "configs/mining/catalog.json"
    assert MINING_CATALOG_FILE.is_file()
    assert MINING_EVALUATION_FILE == PROJECT_ROOT / "configs/mining/evaluation.json"
    assert MINING_EVALUATION_FILE.is_file()
    assert MINING_WORKER_CONFIG_FILE == PROJECT_ROOT / "configs/mining/worker.json"
    assert MINING_WORKER_CONFIG_FILE.is_file()
