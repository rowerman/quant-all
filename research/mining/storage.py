"""挖掘候选存储的公开组合入口。"""

from __future__ import annotations

from mining.storage_base import StorageBase
from mining.storage_candidates import CandidateStoreMixin
from mining.storage_events import EventStoreMixin
from mining.storage_evidence import EvidenceStoreMixin
from mining.storage_publications import PublicationStoreMixin
from mining.storage_publication_results import PublicationResultStoreMixin
from mining.storage_review import ReviewStoreMixin
from mining.storage_run_control import RunControlStoreMixin
from mining.storage_runs import RunStoreMixin


class MiningStore(
    RunStoreMixin,
    CandidateStoreMixin,
    EvidenceStoreMixin,
    ReviewStoreMixin,
    PublicationStoreMixin,
    PublicationResultStoreMixin,
    EventStoreMixin,
    RunControlStoreMixin,
    StorageBase,
):
    """组合运行、候选、证据和事件的独立候选存储。"""

