"""量化批量数值核的统一入口。"""

from kernels.backend import (
    BackendSelection,
    DailyOlsResult,
    NativeKernelBackend,
    get_backend,
    native_threads,
    validate_target_index,
)
from kernels.errors import (
    KernelArrayError,
    KernelBackendError,
    KernelError,
    KernelOffsetError,
    KernelParameterError,
)
from kernels.fallback import PythonKernelBackend
from kernels.native import CppKernelBackend

__all__ = [
    "BackendSelection",
    "DailyOlsResult",
    "KernelArrayError",
    "KernelBackendError",
    "KernelError",
    "KernelOffsetError",
    "KernelParameterError",
    "NativeKernelBackend",
    "CppKernelBackend",
    "PythonKernelBackend",
    "get_backend",
    "native_threads",
    "validate_target_index",
]
