"""取数缓存：把外部接口的返回原样落盘，便于断点续跑与审计。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.config import EXTRA_DIR
from scripts.logging_conf import get_logger

logger = get_logger("sources.cache")


def source_dir(source: str) -> Path:
    """返回某个数据源的缓存目录，不存在则创建。

    参数:
        source: 数据源名，例如 "tushare"。
    """
    path = EXTRA_DIR / source
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_frame(source: str, name: str, frame: pd.DataFrame) -> Path:
    """把一张表写入缓存。

    参数:
        source: 数据源名。
        name: 表名，作为文件名。
        frame: 待写入的数据。

    返回:
        写入的文件路径。
    """
    path = source_dir(source) / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    logger.info("写入缓存 %s 行=%d", path, len(frame))
    return path


def read_frame(source: str, name: str) -> pd.DataFrame | None:
    """读取缓存表，不存在时返回 None。

    参数:
        source: 数据源名。
        name: 表名。
    """
    path = source_dir(source) / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)
