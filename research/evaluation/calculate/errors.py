"""计算包的错误码与异常类。

错误码分段（后端接口的 MONITOR-1xx 用到时在 backend 里定义）：

    EVAL-1xx  评估参数与评估面板
    EVAL-2xx  评估指标
    EVAL-3xx  拆分与风格因子

类名与错误码沿用项目早期的约定，避免已经写在文档与记录里的码失效。
"""

from __future__ import annotations


class CalculateError(Exception):
    """计算包所有可预期异常的基类，携带错误码便于从日志里定位。"""

    code = "EVAL-000"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(f"[{self.code}] {message}")


class EvalParamError(CalculateError):
    """评估参数非法或超出上限。"""

    code = "EVAL-101"


class EvalPanelError(CalculateError):
    """评估面板过大、结构异常或构建失败。"""

    code = "EVAL-102"


class UniverseError(CalculateError):
    """股票池标识非法或成分数据缺失。"""

    code = "EVAL-103"


class EvalFactorError(CalculateError):
    """要评估的因子未知或没有落盘数据。"""

    code = "EVAL-104"


class MetricError(CalculateError):
    """指标注册或计算失败。"""

    code = "EVAL-201"


class DecomposeError(CalculateError):
    """拆分与风格因子相关失败。"""

    code = "EVAL-301"


class StyleFactorError(CalculateError):
    """风格因子取数或计算失败（指数收益缺失、表结构不符等）。"""

    code = "EVAL-302"


class IndustryError(CalculateError):
    """行业数据源缺失或不可用。"""

    code = "EVAL-303"
