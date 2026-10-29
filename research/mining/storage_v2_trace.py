"""挖掘 V2 幂等轨迹组合入口。"""

from mining.storage_v2_candidate_trace import V2CandidateTraceMixin
from mining.storage_v2_trace_common import V2TraceCommonMixin
from mining.storage_v2_trace_validation import V2TraceValidationMixin
from mining.storage_v2_turns import V2TurnStoreMixin


class V2TraceStoreMixin(
    V2TurnStoreMixin,
    V2CandidateTraceMixin,
    V2TraceValidationMixin,
    V2TraceCommonMixin,
):
    """组合 turn、候选和完整性校验的 V2 轨迹。"""
