"""V2 正式发布仓储组合入口。"""

from mining.storage_v2_publication_read import V2PublicationReadMixin
from mining.storage_v2_publication_results import V2PublicationResultMixin
from mining.storage_v2_publication_write import V2PublicationWriteMixin


class V2PublicationStoreMixin(
    V2PublicationReadMixin,
    V2PublicationWriteMixin,
    V2PublicationResultMixin,
):
    """组合 V2 发布输入、批次写入和结果收口。"""
