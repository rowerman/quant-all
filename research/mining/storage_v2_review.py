"""V2 审核候选审批的读写存储入口。"""

from mining.storage_v2_review_read import V2ReviewReadMixin
from mining.storage_v2_review_write import V2ReviewWriteMixin


class V2ReviewStoreMixin(V2ReviewReadMixin, V2ReviewWriteMixin):
    """组合 V2 候选审批的只读查询和原子写入。"""
