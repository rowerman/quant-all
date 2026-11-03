"""挖掘运行和候选状态机测试。"""

from __future__ import annotations

import pytest

from mining.errors import MiningStateError
from mining.states import (
    CANDIDATE_STATUSES,
    PUBLICATION_STATUSES,
    RUN_STATUSES,
    CandidateStateMachine,
    PublicationStateMachine,
    RunStateMachine,
)


@pytest.mark.parametrize(
    "current,target",
    [
        ("queued", "generating"),
        ("generating", "screening"),
        ("screening", "evaluating"),
        ("evaluating", "holding_out"),
        ("holding_out", "review_ready"),
        ("review_ready", "completed"),
        ("generating", "interrupted"),
        ("interrupted", "failed"),
    ],
)
def test_run_state_machine_accepts_declared_path(current, target):
    """主路径和中断终态迁移合法。"""

    RunStateMachine.ensure(current, target)


@pytest.mark.parametrize(
    "current,target",
    [
        ("queued", "evaluating"),
        ("failed", "generating"),
        ("completed", "review_ready"),
        ("review_ready", "generating"),
    ],
)
def test_run_state_machine_rejects_skips_and_terminal_restart(current, target):
    """跨阶段和终态重启被拒绝。"""

    with pytest.raises(MiningStateError):
        RunStateMachine.ensure(current, target)


def test_interrupted_run_resumes_only_to_original_stage():
    """中断恢复不能借机跨阶段。"""

    RunStateMachine.ensure("interrupted", "screening", interrupted_from="screening")
    with pytest.raises(MiningStateError, match="原阶段"):
        RunStateMachine.ensure("interrupted", "evaluating", interrupted_from="screening")


@pytest.mark.parametrize(
    "current,target",
    [
        ("proposed", "validating"),
        ("validating", "screening"),
        ("screening", "evaluating"),
        ("evaluating", "evaluated"),
        ("evaluated", "holdout_running"),
        ("holdout_running", "review_ready"),
        ("review_ready", "approved"),
        ("approved", "building"),
        ("building", "published"),
        ("building", "build_failed"),
        ("build_failed", "building"),
    ],
)
def test_candidate_state_machine_accepts_declared_path(current, target):
    """候选完整主路径合法。"""

    CandidateStateMachine.ensure(current, target)


@pytest.mark.parametrize(
    "current,target",
    [
        ("proposed", "evaluating"),
        ("validating", "review_ready"),
        ("failed", "approved"),
        ("stale", "published"),
        ("rejected", "approved"),
        ("published", "building"),
    ],
)
def test_candidate_state_machine_rejects_gate_bypass(current, target):
    """失败、过期和终态不能绕过审批门禁。"""

    with pytest.raises(MiningStateError):
        CandidateStateMachine.ensure(current, target)


def test_unknown_status_is_rejected_for_both_machines():
    """拼写错误不能创建隐式状态。"""

    with pytest.raises(MiningStateError):
        RunStateMachine.ensure("queued", "unknown")
    with pytest.raises(MiningStateError):
        CandidateStateMachine.ensure("proposed", "unknown")
    assert "unknown" not in RUN_STATUSES
    assert "unknown" not in CANDIDATE_STATUSES


def test_publication_state_machine_rejects_publication_without_validation():
    """发布只能从暂存、校验到正式结果，失败后只能回滚。"""

    PublicationStateMachine.ensure("staging", "validating")
    PublicationStateMachine.ensure("validating", "published")
    PublicationStateMachine.ensure("validating", "failed")
    PublicationStateMachine.ensure("failed", "rolled_back")
    with pytest.raises(MiningStateError):
        PublicationStateMachine.ensure("staging", "published")
    with pytest.raises(MiningStateError):
        PublicationStateMachine.ensure("published", "rolled_back")
    assert set(PUBLICATION_STATUSES) == {
        "staging",
        "validating",
        "published",
        "failed",
        "rolled_back",
    }
