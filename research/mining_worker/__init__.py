"""主服务与独立 RD-Agent Worker 共享的受控协议。"""

from mining_worker.protocol import (
    JOB_STATUSES,
    WORKER_SCHEMA_VERSION,
    WorkerCandidateDraft,
    WorkerGenerateRequest,
    WorkerGenerationResponse,
    WorkerJobView,
    WorkerProtocolError,
)
from mining_worker.worker_config import WorkerConfig, WorkerConfigError, load_worker_config

__all__ = [
    "JOB_STATUSES",
    "WORKER_SCHEMA_VERSION",
    "WorkerCandidateDraft",
    "WorkerGenerateRequest",
    "WorkerGenerationResponse",
    "WorkerJobView",
    "WorkerProtocolError",
    "WorkerConfig",
    "WorkerConfigError",
    "load_worker_config",
]
from mining_worker.protocol_v2 import (
    WORKER_SEARCH_SCHEMA_VERSION,
    WorkerDecisionProposal,
    WorkerSearchCandidate,
    WorkerSearchHypothesis,
    WorkerSearchRequest,
    WorkerSearchResponse,
)

__all__ = [
    "WORKER_SEARCH_SCHEMA_VERSION",
    "WorkerDecisionProposal",
    "WorkerSearchCandidate",
    "WorkerSearchHypothesis",
    "WorkerSearchRequest",
    "WorkerSearchResponse",
]
