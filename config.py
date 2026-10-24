"""计算包的路径与常量。

计算包只认数据产物（`data/databases/` 与 `data/factors/`），不引用数据项目内部的 `scripts` 模块，
所以路径在这里独立定义；需要换数据位置时改这里或给调用方传参。
"""

from __future__ import annotations

from pathlib import Path

from common.paths import DB_DIR
from common.paths import DB_FILE
from common.paths import FACTOR_DB_FILE
from common.paths import FACTOR_DIR
from common.paths import LOGS_DIR
from common.paths import PROJECT_ROOT

CALCULATE_ROOT = Path(__file__).resolve().parent

# 计算包写入集中日志目录的业务域子目录
LOG_DIR = LOGS_DIR / "research" / "evaluation"
