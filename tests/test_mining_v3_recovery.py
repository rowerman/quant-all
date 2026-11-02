"""V3 手工假设、自由探索和恢复校验测试。"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from mining.errors import V3RunStateError
from tests.mining_v3_recovery_support import (
    CrashOnSecondFreeTurnWorker,
    CrashOnSecondTurnWorker,
    approve_first_candidate,
    build_hypothesis_fixture,
    build_free_fixture,
    open_runtime,
    search_orchestrator,
)
from tests.test_mining_free_exploration import FreeBranchWorker
from tests.test_mining_search_orchestrator import SequenceWorker


def test_hypothesis_restart_reuses_completed_turn(tmp_path: Path):
    runtime, contract, request, context = build_hypothesis_fixture(tmp_path)
    first = search_orchestrator(runtime, CrashOnSecondTurnWorker())
    run = first.create_run(request, context, background=False)
    try:
        with pytest.raises(KeyboardInterrupt):
            first.execute(run["run_id"])
    finally:
        first.close()

    runtime, _contract = open_runtime(tmp_path)
    interrupted = runtime.interrupt_orphaned_search_runs()
    assert interrupted[0]["interrupted_from"] == "searching"
    recovered = runtime.recover_v3_run(run["run_id"])
    assert recovered.request == request
    assert recovered.context == context
    assert recovered.input_envelope.input_mode == "hypothesis"

    worker = SequenceWorker()
    second = search_orchestrator(runtime, worker)
    try:
        second.resume(run["run_id"], background=False)
        result = second.execute(run["run_id"])
    finally:
        second.close()

    assert result["status"] == "review_ready"
    assert [item.turn_index for item in worker.requests] == [2, 3]
    assert contract.load_input(run["run_id"]).input_mode == "hypothesis"


def test_free_exploration_restart_preserves_seed_and_direction(tmp_path: Path):
    runtime, contract, request, context = build_free_fixture(tmp_path)
    first = search_orchestrator(
        runtime,
        CrashOnSecondFreeTurnWorker(("branch", "promote")),
    )
    run = first.create_run(request, context, background=False)
    try:
        with pytest.raises(KeyboardInterrupt):
            first.execute(run["run_id"])
    finally:
        first.close()

    runtime, _contract = open_runtime(tmp_path)
    runtime.interrupt_orphaned_search_runs()
    recovered = runtime.recover_v3_run(run["run_id"])
    assert recovered.request.effective_exploration_seed == 53

    worker = FreeBranchWorker(("branch", "promote"))
    second = search_orchestrator(runtime, worker)
    try:
        second.resume(run["run_id"], background=False)
        result = second.execute(run["run_id"])
    finally:
        second.close()

    assert result["status"] == "review_ready"
    assert [item.turn_index for item in worker.requests] == [2]
    assert contract.load_input(run["run_id"]).input_mode == "free_exploration"


def test_recovery_rejects_changed_context_without_state_change(tmp_path: Path):
    runtime, _contract, request, context = build_hypothesis_fixture(tmp_path)
    orchestrator = search_orchestrator(runtime, SequenceWorker())
    run = orchestrator.create_run(request, context, background=False)
    orchestrator.close()
    runtime.interrupt_orphaned_search_runs()

    with duckdb.connect(str(tmp_path / "factor_mining.duckdb")) as conn:
        conn.execute(
            "UPDATE mining_v3.run_context SET context_hash=? WHERE run_id=?",
            ["f" * 64, run["run_id"]],
        )
    resumed = search_orchestrator(runtime, SequenceWorker())
    try:
        with pytest.raises(V3RunStateError, match="恢复事实校验失败"):
            resumed.resume(run["run_id"], background=False)
    finally:
        resumed.close()
    assert runtime.search_control_state(run["run_id"])["status"] == "interrupted"


def test_recovery_rejects_changed_candidate_evidence(tmp_path: Path):
    runtime, contract, request, context = build_hypothesis_fixture(tmp_path)
    first = search_orchestrator(runtime, CrashOnSecondTurnWorker())
    run = first.create_run(request, context, background=False)
    try:
        with pytest.raises(KeyboardInterrupt):
            first.execute(run["run_id"])
    finally:
        first.close()
    runtime.interrupt_orphaned_search_runs()

    envelopes = contract.list_candidate_envelopes(run["run_id"])
    assert envelopes
    candidate_id = envelopes[0].candidate.candidate_id
    with duckdb.connect(str(tmp_path / "factor_mining.duckdb")) as conn:
        conn.execute(
            "UPDATE mining_v3.evaluation_evidence SET status='failed' "
            "WHERE run_id=? AND candidate_id=?",
            [run["run_id"], candidate_id],
        )

    resumed = search_orchestrator(runtime, SequenceWorker())
    try:
        with pytest.raises(V3RunStateError, match="恢复事实校验失败"):
            resumed.resume(run["run_id"], background=False)
    finally:
        resumed.close()
    assert runtime.search_control_state(run["run_id"])["status"] == "interrupted"


def test_recovery_rejects_changed_approval_evidence(tmp_path: Path):
    runtime, contract, request, context = build_hypothesis_fixture(tmp_path)
    orchestrator = search_orchestrator(runtime, SequenceWorker())
    run = orchestrator.create_run(request, context, background=False)
    try:
        result = orchestrator.execute(run["run_id"])
    finally:
        orchestrator.close()
    assert result["status"] == "review_ready"
    candidate_id = approve_first_candidate(contract, run["run_id"])

    with duckdb.connect(str(tmp_path / "factor_mining.duckdb")) as conn:
        conn.execute(
            "UPDATE mining_v3.candidate_approval SET evidence_hash=? "
            "WHERE run_id=? AND candidate_id=?",
            ["f" * 64, run["run_id"], candidate_id],
        )
    with pytest.raises(V3RunStateError, match="审批证据"):
        runtime.recover_v3_run(run["run_id"])
