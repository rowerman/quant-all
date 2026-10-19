"""Alpha 模型层的可预期错误。

错误码用于从日志中快速定位模块：

    ALPHA-1xx  配置与数据
    ALPHA-2xx  预处理与切分
    ALPHA-3xx  模型与评估
    ALPHA-4xx  产物与登记
"""

from __future__ import annotations


class AlphaError(Exception):
    """Alpha 模型层所有可预期异常的基类。"""

    code = "ALPHA-000"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(f"[{self.code}] {message}")


class AlphaConfigError(AlphaError):
    """研究契约或运行配置非法。"""

    code = "ALPHA-101"


class AlphaDataError(AlphaError):
    """数据读取、行集或标签对齐失败。"""

    code = "ALPHA-102"


class AlphaSplitError(AlphaError):
    """时间切分、purge 或 embargo 非法。"""

    code = "ALPHA-201"


class AlphaPreprocessError(AlphaError):
    """截面预处理失败或输入结构不符。"""

    code = "ALPHA-202"


class AlphaModelError(AlphaError):
    """模型训练或预测失败。"""

    code = "ALPHA-301"


class AlphaEvaluationError(AlphaError):
    """模型评估输入或指标计算失败。"""

    code = "ALPHA-302"


class AlphaArtifactError(AlphaError):
    """模型、数据集或预测产物缺失或损坏。"""

    code = "ALPHA-401"


class AlphaRegistryError(AlphaError):
    """模型登记冲突或状态非法。"""

    code = "ALPHA-402"
