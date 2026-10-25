"""挖掘运行与候选的唯一状态迁移表。"""

from __future__ import annotations

from mining.errors import MiningStateError

RUN_STATUSES = (
    "queued",
    "generating",
    "screening",
    "evaluating",
    "holding_out",
    "review_ready",
    "completed",
    "interrupted",
    "failed",
    "cancelled",
    "stale",
)
RUN_ACTIVE_STATUSES = (
    "generating",
    "screening",
    "evaluating",
    "holding_out",
)
RUN_TERMINAL_STATUSES = ("completed", "failed", "cancelled", "stale")

CANDIDATE_STATUSES = (
    "proposed",
    "validating",
    "technical_failed",
    "screening",
    "screened_out",
    "evaluating",
    "evaluated",
    "holdout_running",
    "holdout_failed",
    "review_ready",
    "approved",
    "building",
    "build_failed",
    "published",
    "rejected",
    "interrupted",
    "failed",
    "cancelled",
    "stale",
)
CANDIDATE_ACTIVE_STATUSES = (
    "validating",
    "screening",
    "evaluating",
    "holdout_running",
    "building",
)
CANDIDATE_TERMINAL_STATUSES = (
    "technical_failed",
    "screened_out",
    "holdout_failed",
    "published",
    "rejected",
    "failed",
    "cancelled",
    "stale",
)

PUBLICATION_STATUSES = (
    "staging",
    "validating",
    "published",
    "failed",
    "rolled_back",
)


def _validate_known(status: str, allowed: tuple[str, ...], kind: str) -> None:
    """拒绝未知状态。"""

    if status not in allowed:
        raise MiningStateError(f"未知{kind}状态：{status!r}")


class RunStateMachine:
    """研究运行状态机。"""

    _ALLOWED = {
        "queued": {"generating", "failed", "cancelled", "stale"},
        "generating": {
            "screening",
            "interrupted",
            "failed",
            "cancelled",
            "stale",
        },
        "screening": {
            "evaluating",
            "review_ready",
            "interrupted",
            "failed",
            "cancelled",
            "stale",
        },
        "evaluating": {
            "holding_out",
            "review_ready",
            "interrupted",
            "failed",
            "cancelled",
            "stale",
        },
        "holding_out": {
            "review_ready",
            "interrupted",
            "failed",
            "cancelled",
            "stale",
        },
        "review_ready": {"completed", "stale"},
        "completed": set(),
        "interrupted": {"failed", "cancelled", "stale"},
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
        """校验一次运行状态迁移。

        参数:
            current: 当前状态。
            target: 目标状态。
            interrupted_from: 恢复中断运行时，原活动阶段。
        """

        _validate_known(current, RUN_STATUSES, "运行")
        _validate_known(target, RUN_STATUSES, "运行")
        if current == "interrupted" and target in RUN_ACTIVE_STATUSES:
            if interrupted_from != target:
                raise MiningStateError(
                    f"中断运行只能恢复到原阶段 {interrupted_from!r}，不能进入 {target!r}"
                )
            return
        if target not in cls._ALLOWED[current]:
            raise MiningStateError(f"运行状态不能从 {current} 变为 {target}")

    @classmethod
    def can_interrupt(cls, status: str) -> bool:
        """判断状态是否可在重启时转成中断。"""

        return status in RUN_ACTIVE_STATUSES


class CandidateStateMachine:
    """候选因子状态机。"""

    _ALLOWED = {
        "proposed": {"validating", "failed", "cancelled", "stale"},
        "validating": {
            "technical_failed",
            "screening",
            "interrupted",
            "failed",
            "cancelled",
            "stale",
        },
        "screening": {
            "screened_out",
            "evaluating",
            "interrupted",
            "failed",
            "cancelled",
            "stale",
        },
        "evaluating": {
            "evaluated",
            "interrupted",
            "failed",
            "cancelled",
            "stale",
        },
        "evaluated": {
            "holdout_running",
            "interrupted",
            "failed",
            "cancelled",
            "stale",
        },
        "holdout_running": {
            "holdout_failed",
            "review_ready",
            "interrupted",
            "failed",
            "cancelled",
            "stale",
        },
        "review_ready": {"approved", "rejected", "cancelled", "stale"},
        "approved": {"building", "stale"},
        "building": {"published", "build_failed", "stale"},
        "build_failed": {"building", "cancelled", "stale"},
        "interrupted": {"failed", "cancelled", "stale"},
        "technical_failed": set(),
        "screened_out": set(),
        "holdout_failed": set(),
        "published": set(),
        "rejected": set(),
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
        """校验一次候选状态迁移。"""

        _validate_known(current, CANDIDATE_STATUSES, "候选")
        _validate_known(target, CANDIDATE_STATUSES, "候选")
        if current == "interrupted" and target in CANDIDATE_ACTIVE_STATUSES:
            if interrupted_from != target:
                raise MiningStateError(
                    f"中断候选只能恢复到原阶段 {interrupted_from!r}，不能进入 {target!r}"
                )
            return
        if target not in cls._ALLOWED[current]:
            raise MiningStateError(f"候选状态不能从 {current} 变为 {target}")

    @classmethod
    def can_interrupt(cls, status: str) -> bool:
        """判断候选是否处于可中断的活动阶段。"""

        return status in CANDIDATE_ACTIVE_STATUSES


class PublicationStateMachine:
    """正式发布尝试的状态机；S2 只约束存储，不执行发布。"""

    _ALLOWED = {
        "staging": {"validating", "failed", "rolled_back"},
        "validating": {"published", "failed", "rolled_back"},
        "failed": {"rolled_back"},
        "published": set(),
        "rolled_back": set(),
    }

    @classmethod
    def ensure(cls, current: str, target: str) -> None:
        """校验发布尝试的状态迁移。"""

        _validate_known(current, PUBLICATION_STATUSES, "发布")
        _validate_known(target, PUBLICATION_STATUSES, "发布")
        if target not in cls._ALLOWED[current]:
            raise MiningStateError(f"发布状态不能从 {current} 变为 {target}")
