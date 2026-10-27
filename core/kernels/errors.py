"""批量计算核的错误类型与错误码。"""

from __future__ import annotations


class KernelError(Exception):
    """批量计算核可预期错误的基类。"""

    code = "KERNEL-000"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(f"[{self.code}] {message}")


class KernelArrayError(KernelError):
    """数组形状、dtype 或内存连续性不符合契约。"""

    code = "KERNEL-001"


class KernelOffsetError(KernelError):
    """分组 offset 不是合法、单调的边界数组。"""

    code = "KERNEL-002"


class KernelParameterError(KernelError):
    """最小样本数等计算参数非法。"""

    code = "KERNEL-003"


class KernelBackendError(KernelError):
    """后端名称非法或请求的后端不可用。"""

    code = "KERNEL-004"
