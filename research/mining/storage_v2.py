"""挖掘 V2 独立存储公共入口。"""

from mining.storage_v2_base import V2StorageBase
from mining.storage_v2_legacy import V2LegacyStoreMixin
from mining.storage_v2_runs import V2RunStoreMixin


class MiningV2Store(
    V2RunStoreMixin,
    V2LegacyStoreMixin,
    V2StorageBase,
):
    """组合 V2 运行、轨迹、台账和 V1 只读快照。"""
