"""挖掘 V2 独立存储公共入口。"""

from mining.storage_v2_base import V2StorageBase
from mining.storage_v2_legacy import V2LegacyStoreMixin
from mining.storage_v2_runs import V2RunStoreMixin
from mining.storage_v2_trace import V2TraceStoreMixin
from mining.storage_v2_trials import V2TrialStoreMixin
from mining.storage_v2_trial_queries import V2TrialQueryMixin


class MiningV2Store(
    V2RunStoreMixin,
    V2TraceStoreMixin,
    V2TrialStoreMixin,
    V2TrialQueryMixin,
    V2LegacyStoreMixin,
    V2StorageBase,
):
    """组合 V2 运行、轨迹、台账和 V1 只读快照。"""
