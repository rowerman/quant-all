"""把计算结果转成 JSON 可序列化的值。

指标函数里会用到 numpy 标量、numpy 数组与时间戳，直接塞进 SSE 会序列化失败；
统一在这里转成 python 原生类型，并保证 NaN / inf 变成 None（前端按空值处理）。
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    """递归地把值转成 JSON 可序列化形式。

    参数:
        value: 任意计算结果，支持 dict / list / ndarray / Series / 标量 / 时间戳。

    返回:
        只含 dict、list、str、int、float、bool、None 的结构；
        非有限浮点数一律转成 None。
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return [json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64, dt.datetime, dt.date)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return str(value)
