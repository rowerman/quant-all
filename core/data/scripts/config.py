"""项目级配置：路径、口径常量、环境变量读取。

所有模块只从这里取路径与口径常量，避免散落在各处。
"""

from __future__ import annotations

import os
from pathlib import Path

from common.paths import CLEAN_DIR
from common.paths import DAILY_DIR
from common.paths import DB_DIR
from common.paths import DB_FILE
from common.paths import DOCS_DIR
from common.paths import FACTOR_DB_FILE
from common.paths import FACTOR_DIR
from common.paths import LOGS_DIR
from common.paths import PROJECT_ROOT
from common.paths import SOURCE_DIR
from common.paths import ensure_layout_dirs

EXTRA_DIR = SOURCE_DIR
LOG_DIR = LOGS_DIR / "core" / "data"

ENV_FILE = PROJECT_ROOT / ".env"

# 数据区间：全 A 日线自 2000 年起；截止日由数据本身决定（daily 分片里最新的交易日）
START_DATE = "2000-01-01"
EARLIEST_START = "2000-01-04"

# 涨跌停判定的半价位容差（元）
LIMIT_EPS = 0.005
# 收益双口径允许的相对差异
RET_TOLERANCE = 1e-4


def load_env_file(path: Path | None = None) -> None:
    """把 .env 里的键值对注入环境变量，已存在的环境变量不覆盖。

    参数:
        path: .env 文件路径，默认取项目根目录的 .env。
    """
    env_path = path or ENV_FILE
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_tushare_token() -> str:
    """读取 Tushare token，缺失时抛错而不是静默降级。

    返回:
        token 字符串。
    """
    load_env_file()
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("未找到 TUSHARE_TOKEN，请在项目根目录的 .env 中配置")
    return token


def ensure_dirs() -> None:
    """创建本项目需要的全部输出目录。"""
    ensure_layout_dirs()
