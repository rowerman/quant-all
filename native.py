"""串行推进原生计算阶段，交付事件和等待客户端时释放锁。"""

from contextlib import contextmanager
from functools import wraps
from threading import RLock

from calculate.jsonutil import json_safe

CALCULATION_LOCK = RLock()


class CalculationCancelled(Exception):
    """等待原生阶段的任务已取消，不再进入计算。"""


@contextmanager
def native_section(cancelled=None):
    """保护一次原生操作；cancelled为可选的跨线程取消事件。"""
    if cancelled is None:
        with CALCULATION_LOCK:
            yield
        return
    while not cancelled.is_set():
        if CALCULATION_LOCK.acquire(timeout=.1):
            try:
                if cancelled.is_set():
                    raise CalculationCancelled()
                yield
            finally:
                CALCULATION_LOCK.release()
            return
    raise CalculationCancelled()


def serialized(function):
    """保护同步元数据函数；参数function为被调用函数。"""
    @wraps(function)
    def call(*args, **kwargs):
        """在共享锁内执行函数，参数原样传递。"""
        with native_section():
            return function(*args, **kwargs)
    return call


def serialized_events(function):
    """保护生成器推进、JSON转换与关闭；参数function为事件生成器工厂。"""
    @wraps(function)
    def events(*args, **kwargs):
        """逐阶段返回纯Python事件，参数原样传递给工厂。"""
        iterator = None
        try:
            with native_section():
                iterator = iter(function(*args, **kwargs))
            while True:
                with native_section():
                    try:
                        stage, payload = next(iterator)
                    except StopIteration:
                        break
                    try:
                        event = stage, json_safe(payload)
                    finally:
                        payload = None
                yield event
        finally:
            with native_section():
                try:
                    if iterator is not None and hasattr(iterator, 'close'):
                        iterator.close()
                finally:
                    iterator = None
    return events
