"""计算包的路径与常量。

计算包只认数据产物（`db/` 与 `factor_data/`），不引用数据项目内部的 `scripts` 模块，
所以路径在这里独立定义；需要换数据位置时改这里或给调用方传参。
"""

from __future__ import annotations

from pathlib import Path

# calculate/ 的上一级就是数据项目根目录，库与因子产物都在那里
CALCULATE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CALCULATE_ROOT.parent

DB_DIR = PROJECT_ROOT / "db"
DB_FILE = DB_DIR / "quant.duckdb"
FACTOR_DB_FILE = DB_DIR / "factors.duckdb"
FACTOR_DIR = PROJECT_ROOT / "factor_data"

# 计算包自己的日志目录，与数据项目的 logs/ 互不干扰
LOG_DIR = CALCULATE_ROOT / "logs"
