"""项目统一路径入口。

所有模块都从这里解析数据、配置、文档和日志目录，避免在业务代码里重复拼接
项目根路径。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"
LOGS_DIR = PROJECT_ROOT / "logs"
EVIDENCE_DIR = PROJECT_ROOT / "AiRecord" / "evidence"

DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
TUSHARE_DIR = SOURCE_DIR / "tushare"
CLEAN_DIR = DATA_DIR / "clean"
DAILY_DIR = CLEAN_DIR / "daily"
DB_DIR = DATA_DIR / "databases"
DB_FILE = DB_DIR / "quant.duckdb"
FACTOR_DB_FILE = DB_DIR / "factors.duckdb"
BACKTEST_DB_FILE = DB_DIR / "backtest.duckdb"
MINING_DB_FILE = DB_DIR / "factor_mining.duckdb"
FACTOR_DIR = DATA_DIR / "factors"
MODEL_DATA_DIR = DATA_DIR / "models"
MINING_DATA_DIR = DATA_DIR / "factor_mining"
MINING_CACHE_DIR = MINING_DATA_DIR / "cache"
MINING_CATALOG_FILE = CONFIG_DIR / "mining" / "catalog.json"
MINING_EVALUATION_FILE = CONFIG_DIR / "mining" / "evaluation.json"
MINING_WORKER_CONFIG_FILE = CONFIG_DIR / "mining" / "worker.json"
MINING_ORCHESTRATOR_FILE = CONFIG_DIR / "mining" / "orchestrator.json"


def log_root_for(module_name: str) -> Path:
    """返回模块所属业务域的日志根目录。

    参数:
        module_name: Python 模块名。
    """
    if module_name == "calculate" or module_name.startswith("calculate."):
        return LOGS_DIR / "research" / "evaluation"
    if module_name == "alpha" or module_name.startswith("alpha."):
        return LOGS_DIR / "research" / "modeling"
    if module_name == "mining" or module_name.startswith("mining."):
        return LOGS_DIR / "research" / "mining"
    if module_name.startswith(("scripts.build_alpha", "build_alpha")):
        return LOGS_DIR / "research" / "modeling"
    if module_name.startswith(("scripts.build_factors", "build_factors")):
        return LOGS_DIR / "core" / "factors"
    if module_name.startswith("backtest"):
        return LOGS_DIR / "trading" / "backtest"
    if module_name.startswith("backend"):
        return LOGS_DIR / "apps" / "monitor"
    return LOGS_DIR / "core" / "data"


def ensure_layout_dirs() -> None:
    """创建运行时可写的集中目录。"""
    for path in (
        SOURCE_DIR,
        CLEAN_DIR,
        DAILY_DIR,
        DB_DIR,
        FACTOR_DIR,
        MODEL_DATA_DIR,
        MINING_DATA_DIR,
        MINING_CACHE_DIR,
        LOGS_DIR / "core" / "data",
        LOGS_DIR / "core" / "factors",
        LOGS_DIR / "research" / "modeling",
        LOGS_DIR / "research" / "evaluation",
        LOGS_DIR / "research" / "mining",
        LOGS_DIR / "trading" / "backtest",
        LOGS_DIR / "apps" / "monitor",
    ):
        path.mkdir(parents=True, exist_ok=True)
