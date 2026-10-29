"""隔离代码研究沙箱的稳定错误。"""

from __future__ import annotations


class SandboxError(Exception):
    """沙箱可预期异常基类。"""

    code = "MINING-133"
    status = 400

    def __init__(self, message: str) -> None:
        self.message = str(message)
        super().__init__(f"[{self.code}] {self.message}")


class SandboxPolicyError(SandboxError):
    """沙箱政策缺失、字段未知或安全边界不成立。"""


class SandboxRunnerError(SandboxError):
    """容器启动、执行、超时或清理失败。"""

    code = "MINING-134"
    status = 500


class SandboxSnapshotError(SandboxError):
    """冻结快照导出、校验或重复构建失败。"""

    code = "MINING-135"
    status = 422


class SandboxIsolationError(SandboxError):
    """路径、挂载或容器隔离边界被破坏。"""

    code = "MINING-136"
    status = 403


class SandboxArtifactError(SandboxError):
    """沙箱产物缺失、篡改或封存失败。"""

    code = "MINING-137"
    status = 422
