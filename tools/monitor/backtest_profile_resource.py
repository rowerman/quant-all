"""回测剖析中的资源调用和 MarketData 规模统计。

该模块只在剖析命令内启用，生产回测不会导入或安装这些包装器。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
import time
from typing import Any, Callable
from unittest import mock

import pandas as pd

import quant
from backtest import engine
from backtest import pipeline
from backtest.data import MarketData
from backtest.store import BacktestStore
from backtest import chunked
from tools.monitor.backtest_profile_fingerprint import call_signature


READ_FUNCTIONS = (
    "load_panel",
    "load_factor_panel",
    "load_stock_basic",
    "load_calendar",
    "load_sw_industry_member",
    "load_index_daily",
    "load_index_basic",
    "load_corporate_actions",
)


@dataclass
class _CallStats:
    """一个被包装函数的调用次数、耗时和返回规模。"""

    count: int = 0
    elapsed_s: float = 0.0
    max_elapsed_s: float = 0.0
    returned_mb: float = 0.0


@dataclass
class ResourceProfiler:
    """统计回测取数、切片和执行调用，供单个剖析进程汇总。"""

    calls: dict[str, _CallStats] = field(
        default_factory=lambda: defaultdict(_CallStats)
    )
    signatures: Counter = field(default_factory=Counter)
    signature_examples: dict[str, dict[str, Any]] = field(default_factory=dict)
    loaded_data: list[dict[str, Any]] = field(default_factory=list)
    crop_sizes_mb: list[float] = field(default_factory=list)
    _exit_stack: ExitStack | None = field(default=None, init=False)

    def __enter__(self) -> "ResourceProfiler":
        self._exit_stack = ExitStack()
        for name in READ_FUNCTIONS:
            function = getattr(quant, name, None)
            if function is not None:
                self._patch(self._exit_stack, quant, name, self._wrap_reader(name, function))
        self._patch(
            self._exit_stack,
            chunked,
            "load_market_data",
            self._wrap_market_loader("chunked.load_market_data", chunked.load_market_data),
        )
        self._patch(
            self._exit_stack,
            pipeline,
            "load_market_data",
            self._wrap_market_loader("pipeline.load_market_data", pipeline.load_market_data),
        )
        self._patch(
            self._exit_stack,
            MarketData,
            "crop",
            self._wrap_crop(MarketData.crop),
        )
        self._patch(
            self._exit_stack,
            engine,
            "run_backtest",
            self._wrap_instance("engine.run_backtest", engine.run_backtest),
        )
        self._patch(
            self._exit_stack,
            engine,
            "_run_portfolio",
            self._wrap_instance("engine._run_portfolio", engine._run_portfolio),
        )
        self._patch(
            self._exit_stack,
            engine,
            "cap_target_weights",
            self._wrap_instance("engine.cap_target_weights", engine.cap_target_weights),
        )
        self._patch(
            self._exit_stack,
            BacktestStore,
            "finish_run",
            self._wrap_instance("store.finish_run", BacktestStore.finish_run),
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._exit_stack is not None:
            self._exit_stack.close()
            self._exit_stack = None

    @staticmethod
    def _patch(
        stack: ExitStack,
        owner: object,
        name: str,
        replacement: Callable,
    ) -> None:
        """临时替换对象属性，并在剖析结束后恢复。"""
        stack.enter_context(mock.patch.object(owner, name, replacement))

    def _record(
        self,
        name: str,
        elapsed: float,
        returned_mb: float,
        signature: str,
        example: dict[str, Any],
    ) -> None:
        stats = self.calls[name]
        stats.count += 1
        stats.elapsed_s += elapsed
        stats.max_elapsed_s = max(stats.max_elapsed_s, elapsed)
        stats.returned_mb += returned_mb
        self.signatures[signature] += 1
        self.signature_examples.setdefault(signature, example)

    def _wrap_reader(self, name: str, function: Callable) -> Callable:
        """包装量化读取函数，记录调用参数和耗时。"""

        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            frame = function(*args, **kwargs)
            elapsed = time.perf_counter() - started
            size = frame_size_mb(frame)
            signature, example = call_signature(name, args, kwargs)
            self._record(name, elapsed, size, signature, example)
            return frame

        return wrapped

    def _wrap_market_loader(self, name: str, function: Callable) -> Callable:
        """包装完整 MarketData 装载，并保留宽表规模快照。"""

        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            data = function(*args, **kwargs)
            elapsed = time.perf_counter() - started
            snapshot = market_data_snapshot(data)
            self.loaded_data.append(snapshot)
            signature, example = call_signature(name, args, kwargs)
            self._record(
                name,
                elapsed,
                float(snapshot["total_frame_mb"]),
                signature,
                example,
            )
            return data

        return wrapped

    def _wrap_crop(self, function: Callable) -> Callable:
        """包装 MarketData.crop，记录分段宽表复制规模。"""

        def wrapped(data, *args, **kwargs):
            started = time.perf_counter()
            result = function(data, *args, **kwargs)
            elapsed = time.perf_counter() - started
            size = market_data_memory_mb(result)
            self.crop_sizes_mb.append(size)
            signature, example = call_signature("MarketData.crop", args, kwargs)
            self._record("MarketData.crop", elapsed, size, signature, example)
            return result

        return wrapped

    def _wrap_instance(self, name: str, function: Callable) -> Callable:
        """包装实例方法或普通函数，记录成功调用耗时。"""

        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            result = function(*args, **kwargs)
            elapsed = time.perf_counter() - started
            signature, example = call_signature(name, args, kwargs)
            self._record(name, elapsed, 0.0, signature, example)
            return result

        return wrapped

    def summary(self) -> dict[str, Any]:
        """返回可 JSON 序列化的调用统计。"""
        calls = {
            name: {
                "count": stats.count,
                "elapsed_s": round(stats.elapsed_s, 6),
                "max_elapsed_s": round(stats.max_elapsed_s, 6),
                "returned_mb": round(stats.returned_mb, 3),
            }
            for name, stats in sorted(self.calls.items())
        }
        repeated = [
            {
                **self.signature_examples[signature],
                "count": count,
            }
            for signature, count in self.signatures.most_common()
            if count > 1
        ]
        return {
            "calls": calls,
            "repeated_calls": repeated[:30],
            "loaded_data": self.loaded_data,
            "crop_calls": len(self.crop_sizes_mb),
            "crop_max_mb": round(max(self.crop_sizes_mb, default=0.0), 3),
            "crop_total_mb": round(sum(self.crop_sizes_mb), 3),
        }


def frame_size_mb(frame: pd.DataFrame) -> float:
    """用浅层内存口径估算 DataFrame 当前引用规模。"""
    if frame is None or frame.empty:
        return 0.0
    return round(float(frame.memory_usage(index=True, deep=False).sum()) / 2**20, 3)


def market_data_memory_mb(data: MarketData) -> float:
    """汇总 MarketData 中所有宽表的浅层内存。"""
    frames = [getattr(data, name) for name in (
        "open",
        "close",
        "can_buy_close",
        "can_buy_open_next",
        "can_sell_open_next",
        "is_suspended",
        "is_st",
        "is_limit_up",
        "is_limit_down",
        "total_mv",
        "amount_cny",
        "is_delisting",
    )]
    frames.extend(data.factors.values())
    for name in ("membership", "industry_l1", "impact_volatility"):
        frame = getattr(data, name)
        if frame is not None:
            frames.append(frame)
    return round(sum(frame_size_mb(frame) for frame in frames), 3)


def market_data_snapshot(data: MarketData) -> dict[str, Any]:
    """记录统一面板的形状、dtype 和单表内存。"""
    fields = (
        "open",
        "close",
        "can_buy_close",
        "can_buy_open_next",
        "can_sell_open_next",
        "is_suspended",
        "is_st",
        "is_limit_up",
        "is_limit_down",
        "total_mv",
        "amount_cny",
        "is_delisting",
    )
    frame_stats = {}
    for name in fields:
        frame = getattr(data, name)
        frame_stats[name] = _frame_snapshot(frame)
    frame_stats["factors"] = {
        name: _frame_snapshot(frame) for name, frame in data.factors.items()
    }
    for name in ("membership", "industry_l1", "impact_volatility"):
        frame = getattr(data, name)
        if frame is not None:
            frame_stats[name] = _frame_snapshot(frame)
    return {
        "shape": [int(len(data.index)), int(len(data.columns))],
        "total_frame_mb": market_data_memory_mb(data),
        "fields": frame_stats,
    }


def _frame_snapshot(frame: pd.DataFrame) -> dict[str, Any]:
    """描述一张宽表的形状、dtype 和浅层内存。"""
    return {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "dtypes": {str(dtype): int(count) for dtype, count in frame.dtypes.value_counts().items()},
        "memory_mb": frame_size_mb(frame),
    }
