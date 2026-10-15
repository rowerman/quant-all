"""回测模块的路径与资源上限。

回测只认数据产物（``db/`` 与 ``factor_data/``），路径在这里独立定义，
不引用数据项目与计算包内部的配置模块。
"""

from __future__ import annotations

from pathlib import Path

BACKTEST_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKTEST_ROOT.parent

DB_DIR = PROJECT_ROOT / "db"
DEFAULT_BACKTEST_DB = DB_DIR / "backtest.duckdb"
# 因子目录库：只用来读因子版本，回测本身不写因子
FACTOR_DB = DB_DIR / "factors.duckdb"
FACTOR_DIR = PROJECT_ROOT / "factor_data"

# 回测模块自己的日志目录，与数据项目的 logs/ 和计算包的 calculate/logs/ 分开
LOG_DIR = BACKTEST_ROOT / "logs"

# 资源上限：交易日上限按约十年设计，单元格上限按实测内存定。
# 实测全 A 三年（664×5671≈377 万格）峰值约 3.2GB，十年全 A（约 1540 万格）会到十几 GB，
# 因此全 A 池最长约五年；十年区间请改用指数成分池（沪深300 并集约 950 只、中证500 约 1900 只）。
# 超出时明确报错并给出建议，避免把机器打爆。
MAX_TRADE_DAYS = 2700
MAX_PANEL_CELLS = 8_000_000

