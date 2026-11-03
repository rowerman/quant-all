"""回测剖析中的稳定调用签名和结果表指纹。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.data import MarketData
from backtest.store import BacktestStore, RESULT_TABLES


def call_signature(name: str, args: tuple, kwargs: dict) -> tuple[str, dict[str, Any]]:
    """生成归一化调用指纹，并把大参数压缩成摘要。"""
    normalized = {
        "args": [_json_value(value) for value in args],
        "kwargs": {
            str(key): _json_value(value)
            for key, value in sorted(kwargs.items())
        },
    }
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    example = {"name": name, "signature_sha256": digest, **normalized}
    return f"{name}:{digest}", example


def _json_value(value: Any) -> Any:
    """把大列表、DataFrame 和路径转换成稳定且紧凑的摘要。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, pd.DataFrame):
        return {"frame_shape": [int(value.shape[0]), int(value.shape[1])]}
    if isinstance(value, (pd.Series, pd.Index)):
        return {
            "series_length": int(len(value)),
            "dtype": str(value.dtype),
            "name": str(value.name),
        }
    if isinstance(value, np.ndarray):
        return {
            "array_shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, (list, tuple)):
        if len(value) <= 8:
            return [_json_value(item) for item in value]
        return {
            "count": len(value),
            "sha256": _sequence_digest(value),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MarketData):
        return {"market_shape": [int(len(value.index)), int(len(value.columns))]}
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in list(value.items())[:20]
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _sequence_digest(values: list | tuple) -> str:
    """对大序列内容做稳定摘要。"""
    text = "\n".join(str(item) for item in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_frame_digest(frame: pd.DataFrame) -> dict[str, Any]:
    """计算结果表的稳定内容指纹，不包含数据库行号。"""
    normalized = frame.copy().drop(columns=["run_id"], errors="ignore")
    normalized.columns = [str(name) for name in normalized.columns]
    row_hashes = pd.util.hash_pandas_object(
        normalized, index=True, categorize=False
    ).to_numpy(dtype="uint64", copy=False)
    digest = hashlib.sha256(row_hashes.tobytes()).hexdigest()
    schema = [
        [str(name), str(dtype)]
        for name, dtype in zip(normalized.columns, normalized.dtypes, strict=True)
    ]
    return {
        "rows": int(len(normalized)),
        "columns": schema,
        "sha256": digest,
    }


def result_fingerprints(store: BacktestStore, run_id: str) -> dict[str, Any]:
    """读取一次运行的全部标准结果表，返回行数、列和内容摘要。"""
    return {
        table: stable_frame_digest(pd.DataFrame(store.get_result_table(run_id, table)))
        for table in RESULT_TABLES
    }
