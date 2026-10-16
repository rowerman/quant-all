"""回测模块的可预期错误。"""

from __future__ import annotations

from scripts.errors import QuantError


class BacktestError(QuantError):
    """回测模块所有可预期异常的基类。"""

    code = "BT-000"


class SpecError(BacktestError):
    """策略配置非法。"""

    code = "BT-101"


class DataAlignmentError(BacktestError):
    """行情、因子或交易规则数据无法对齐。"""

    code = "BT-201"


class ExecutionError(BacktestError):
    """VectorBT 执行或结果提取失败。"""

    code = "BT-301"


class ResourceLimitError(BacktestError):
    """回测区间或面板规模超过资源上限。"""

    code = "BT-401"


class RunCancelled(BacktestError):
    """调用方主动取消回测。"""

    code = "BT-501"


class RunNotFoundError(BacktestError):
    """指定的回测运行不存在。"""

    code = "BT-601"


class RunConflictError(BacktestError):
    """回测运行当前状态不允许执行请求的操作。"""

    code = "BT-602"
