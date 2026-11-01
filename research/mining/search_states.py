"""V2 证据驱动搜索的运行状态机。"""

from __future__ import annotations

from mining.errors import MiningStateError

SEARCH_RUN_STATUSES = (
    "queued",
    "searching",
    "selecting",
    "evaluating",
    "holding_out",
    "review_ready",
    "interrupted",
    "failed",
    "cancelled",
    "stale",
)
SEARCH_RUN_ACTIVE_STATUSES = (
    "searching",
    "selecting",
    "evaluating",
    "holding_out",
)
SEARCH_RUN_TERMINAL_STATUSES = (
    "review_ready",
    "failed",
    "cancelled",
    "stale",
)


class SearchRunStateMachine:
    """限制 V2 运行只能沿证据链向前推进。"""

    _ALLOWED = {
        "queued": {"searching", "failed", "cancelled", "stale"},
        "searching": {"selecting", "interrupted", "failed", "cancelled", "stale"},
        "selecting": {"evaluating", "interrupted", "failed", "cancelled", "stale"},
        "evaluating": {"holding_out", "interrupted", "failed", "cancelled", "stale"},
        "holding_out": {"review_ready", "interrupted", "failed", "cancelled", "stale"},
        "interrupted": {"failed", "cancelled", "stale"},
        "review_ready": {"stale"},
        "failed": set(),
        "cancelled": set(),
        "stale": set(),
    }

    @classmethod
    def ensure(
        cls,
        current: str,
        target: str,
        *,
        interrupted_from: str | None = None,
    ) -> None:
        """校验一次 V2 运行状态迁移。"""

        if current not in SEARCH_RUN_STATUSES:
            raise MiningStateError(f"未知 V2 运行状态：{current!r}")
        if target not in SEARCH_RUN_STATUSES:
            raise MiningStateError(f"未知 V2 运行状态：{target!r}")
        if current == "interrupted" and target in SEARCH_RUN_ACTIVE_STATUSES:
            if interrupted_from != target:
                raise MiningStateError(
                    f"中断运行只能恢复到原阶段 {interrupted_from!r}，"
                    f"不能进入 {target!r}"
                )
            return
        if target not in cls._ALLOWED[current]:
            raise MiningStateError(f"V2 运行不能从 {current} 变为 {target}")

    @classmethod
    def can_interrupt(cls, status: str) -> bool:
        """判断状态是否可以标记中断。"""

        return status in SEARCH_RUN_ACTIVE_STATUSES
