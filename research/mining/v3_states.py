"""V3 研究闭环对既有运行、候选和发布状态机的只读门面。"""

from __future__ import annotations

from mining.search_states import (
    SEARCH_RUN_ACTIVE_STATUSES,
    SEARCH_RUN_STATUSES,
    SEARCH_RUN_TERMINAL_STATUSES,
    SearchRunStateMachine,
)
from mining.states import (
    CANDIDATE_ACTIVE_STATUSES,
    CANDIDATE_STATUSES,
    CANDIDATE_TERMINAL_STATUSES,
    PUBLICATION_STATUSES,
    CandidateStateMachine,
    PublicationStateMachine,
)

V3_RESEARCH_RUN_STATUSES = SEARCH_RUN_STATUSES
V3_RESEARCH_RUN_ACTIVE_STATUSES = SEARCH_RUN_ACTIVE_STATUSES
V3_RESEARCH_RUN_TERMINAL_STATUSES = SEARCH_RUN_TERMINAL_STATUSES
V3_CANDIDATE_STATUSES = CANDIDATE_STATUSES
V3_CANDIDATE_ACTIVE_STATUSES = CANDIDATE_ACTIVE_STATUSES
V3_CANDIDATE_TERMINAL_STATUSES = CANDIDATE_TERMINAL_STATUSES
V3_PUBLICATION_STATUSES = PUBLICATION_STATUSES


class V3ResearchStateMachine:
    """委托既有状态机，防止 V3 状态表与 V2 漂移。"""

    @staticmethod
    def ensure_run(
        current: str,
        target: str,
        *,
        interrupted_from: str | None = None,
    ) -> None:
        """校验 V3 研究运行迁移。"""

        SearchRunStateMachine.ensure(
            current,
            target,
            interrupted_from=interrupted_from,
        )

    @staticmethod
    def can_interrupt_run(status: str) -> bool:
        """判断 V3 研究运行能否中断。"""

        return SearchRunStateMachine.can_interrupt(status)

    @staticmethod
    def ensure_candidate(
        current: str,
        target: str,
        *,
        interrupted_from: str | None = None,
    ) -> None:
        """校验 V3 候选迁移。"""

        CandidateStateMachine.ensure(
            current,
            target,
            interrupted_from=interrupted_from,
        )

    @staticmethod
    def ensure_publication(current: str, target: str) -> None:
        """校验 V3 共享发布批次迁移。"""

        PublicationStateMachine.ensure(current, target)


__all__ = [
    "V3_CANDIDATE_ACTIVE_STATUSES",
    "V3_CANDIDATE_STATUSES",
    "V3_CANDIDATE_TERMINAL_STATUSES",
    "V3_PUBLICATION_STATUSES",
    "V3_RESEARCH_RUN_ACTIVE_STATUSES",
    "V3_RESEARCH_RUN_STATUSES",
    "V3_RESEARCH_RUN_TERMINAL_STATUSES",
    "V3ResearchStateMachine",
]
