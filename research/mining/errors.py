"""自动因子挖掘领域的可预期错误。

错误码固定为 ``MINING-1xx``，调用方可据错误码从日志定位模块和失败阶段。
"""

from __future__ import annotations


class MiningError(Exception):
    """挖掘模块可预期异常基类。

    参数:
        message: 面向用户的简短说明。
        code: 稳定错误码。
        status: 建议的 HTTP 状态码。
    """

    code = "MINING-100"
    status = 400

    def __init__(self, message: str) -> None:
        self.message = str(message)
        super().__init__(f"[{self.code}] {self.message}")


class MiningContractError(MiningError):
    """请求、上下文或领域对象不符合契约。"""

    code = "MINING-101"


class MiningBusyError(MiningError):
    """已有运行占用唯一执行槽位。"""

    code = "MINING-102"
    status = 429

    def __init__(self, message: str = "已有自动因子挖掘任务正在运行") -> None:
        super().__init__(message)


class MiningWorkerError(MiningError):
    """Worker 不可用或协议失败。"""

    code = "MINING-103"
    status = 503


class CandidateError(MiningError):
    """候选定义不合法。"""

    code = "MINING-104"


class MiningDuplicateError(MiningError):
    """幂等键或候选内容发生冲突。"""

    code = "MINING-105"
    status = 409


class MiningEvaluationError(MiningError):
    """候选评估失败。"""

    code = "MINING-106"
    status = 422


class HoldoutError(MiningError):
    """模块测试段状态不允许当前操作。"""

    code = "MINING-107"
    status = 409


class ApprovalConflictError(MiningError):
    """审批依据过期或重复审批结论冲突。"""

    code = "MINING-108"
    status = 409


class MiningPublicationError(MiningError):
    """候选暂存、构建或正式发布失败。"""

    code = "MINING-109"
    status = 500


class MiningNotFoundError(MiningError):
    """运行、候选、审批或其他记录不存在。"""

    code = "MINING-110"
    status = 404


class MiningStateError(MiningError):
    """状态迁移或并发写入冲突。"""

    code = "MINING-111"
    status = 409


class MiningStorageError(MiningError):
    """候选库不可用、损坏或事务失败。"""

    code = "MINING-112"
    status = 503


class MiningRecoveryError(MiningError):
    """中断运行不能按请求恢复。"""

    code = "MINING-113"
    status = 409


class ExpressionValidationError(MiningError):
    """表达式语法、字段、算子或安全规则不合法。"""

    code = "MINING-114"
    status = 400

    def __init__(self, message: str, *, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(message)


class CatalogDriftError(MiningError):
    """冻结目录损坏或与实际数据 schema 漂移。"""

    code = "MINING-115"
    status = 500


class WorkerTimeoutError(MiningWorkerError):
    """Worker 作业超过调用方或服务端时限。"""

    code = "MINING-116"
    status = 504


class WorkerCancelledError(MiningWorkerError):
    """Worker 作业被取消或在重启后中断。"""

    code = "MINING-117"
    status = 409


class WorkerResponseError(MiningWorkerError):
    """Worker 返回了不符合协议的响应。"""

    code = "MINING-118"
    status = 502


class WorkerConfigurationError(MiningWorkerError):
    """Worker 鉴权、模型或运行配置不可用。"""

    code = "MINING-119"
    status = 503


class SnapshotStaleError(MiningError):
    """运行依赖的数据或规则快照已经变化。"""

    code = "MINING-120"
    status = 409


class PublicationConflictError(MiningError):
    """发布资格、批次组成或并发状态不允许当前操作。"""

    code = "MINING-121"
    status = 409


class PublicationValidationError(MiningError):
    """全量正式因子未通过行集、取值、前视或冲突校验。"""

    code = "MINING-122"
    status = 422


class PublicationFilesystemError(MiningError):
    """正式因子提交、回滚或重启恢复失败。"""

    code = "MINING-123"
    status = 503


class BatchMaterializationError(MiningEvaluationError):
    """候选批量物化的共享计算或批次边界失败。"""

    code = "MINING-124"
    status = 422


class MiningVisibilityError(MiningError):
    """搜索角色请求访问未授权的数据窗口或证据字段。"""

    code = "MINING-125"
    status = 400


class MiningMigrationError(MiningError):
    """候选库显式迁移失败或源快照不允许迁移。"""

    code = "MINING-126"
    status = 503


class AnalysisToolNotFoundError(MiningError):
    """请求了未登记的分析工具。"""

    code = "MINING-127"
    status = 400


class AnalysisToolArgumentError(MiningError):
    """工具参数缺失、未知或超过限制。"""

    code = "MINING-128"
    status = 400


class AnalysisToolVisibilityError(MiningError):
    """工具请求尝试访问当前角色不可见的数据。"""

    code = "MINING-129"
    status = 403


class AnalysisToolTimeoutError(MiningError):
    """工具执行超时或被取消。"""

    code = "MINING-130"
    status = 504


class AnalysisToolResultError(MiningError):
    """工具结果为空、损坏、无法计算或超过大小限制。"""

    code = "MINING-131"
    status = 422


class AnalysisToolExecutionError(MiningError):
    """工具执行过程中发生业务失败。"""

    code = "MINING-132"
    status = 500


class ResearchMaterialContractError(MiningContractError):
    """资料清单或解析记录不符合契约。"""

    code = "MINING-147"
    status = 400


class ResearchMaterialSecurityError(MiningError):
    """资料路径或输入越过允许的安全边界。"""

    code = "MINING-148"
    status = 403


class ResearchMaterialReadError(MiningError):
    """资料文件缺失、不可读或无法计算指纹。"""

    code = "MINING-149"
    status = 422


class ResearchMaterialParseError(MiningError):
    """资料格式、正文或加密状态不允许解析。"""

    code = "MINING-150"
    status = 422


class ResearchMaterialParserError(MiningError):
    """解析器自身发生未预期失败。"""

    code = "MINING-151"
    status = 500
