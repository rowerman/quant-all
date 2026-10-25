"""统一错误码与异常类。

错误码分段：
    SRC-1xx    外部数据源取数
    CLEAN-1xx  原始数据读取与对齐
    CLEAN-2xx  复权与收益
    CLEAN-3xx  生命周期与涨跌停标记
    CLEAN-4xx  入库
    CLEAN-5xx  校验
    FACTOR-1xx 因子定义与注册
    FACTOR-2xx 因子构建与产物
    FACTOR-3xx 因子自检
    FACTOR-4xx 新因子函数化接口

因子质量监测项目（`calculate/` 与 `apps/monitor/` 下的后端、前端）的错误码不在这里，
见 ``calculate/errors.py`` 与后续的 ``backend/errors.py``。
"""

from __future__ import annotations


class QuantError(Exception):
    """项目内所有可预期异常的基类，携带错误码便于从日志中定位。"""

    code = "QUANT-000"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(f"[{self.code}] {message}")


class SourceError(QuantError):
    """外部数据源取数失败。"""

    code = "SRC-100"


class SourcePermissionError(SourceError):
    """接口无权限。"""

    code = "SRC-101"


class SourceRateLimitError(SourceError):
    """接口触发限速。"""

    code = "SRC-102"


class SourceEmptyError(SourceError):
    """接口返回空数据。"""

    code = "SRC-103"


class DataReadError(QuantError):
    """原始数据读取或对齐失败。"""

    code = "CLEAN-101"


class CalendarError(DataReadError):
    """交易日历异常。"""

    code = "CLEAN-102"


class AdjustError(QuantError):
    """复权或收益计算异常。"""

    code = "CLEAN-201"


class MissingFactorError(AdjustError):
    """有效区间内复权因子缺失。"""

    code = "CLEAN-202"


class LifecycleError(QuantError):
    """生命周期与停牌状态异常。"""

    code = "CLEAN-301"


class LimitRuleError(QuantError):
    """涨跌停规则或标记异常。"""

    code = "CLEAN-302"


class DbWriteError(QuantError):
    """数据库写入失败。"""

    code = "CLEAN-401"


class ValidationError(QuantError):
    """验收校验未通过。"""

    code = "CLEAN-501"


class FactorRegistryError(QuantError):
    """因子定义非法或与已注册因子冲突。"""

    code = "FACTOR-101"


class FactorBuildError(QuantError):
    """因子构建失败。"""

    code = "FACTOR-201"


class FactorDataError(QuantError):
    """因子产物缺失或结构不符合约定。"""

    code = "FACTOR-202"


class FactorCheckError(QuantError):
    """因子自检未通过。"""

    code = "FACTOR-301"


class FactorApiError(QuantError):
    """新因子函数化接口调用失败。"""

    code = "FACTOR-401"
