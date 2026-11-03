"""V3 CoSTEER 人工提升候选的重启恢复测试。"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from mining.errors import V3RunStateError
from tests.mining_v3_recovery_support import (
    NoWorker,
    build_costeer_fixture,
    open_runtime,
    search_orchestrator,
)


def test_costeer_queued_restart_restores_seed_without_worker(tmp_path: Path):
    (
        runtime,
        contract,
        request,
        context,
        candidate,
    ) = build_costeer_fixture(tmp_path, "0001")
    first = search_orchestrator(runtime, NoWorker())
    run = first.create_run(request, context, background=False)
    first.close()

    runtime, _contract = open_runtime(tmp_path)
    interrupted = runtime.interrupt_orphaned_search_runs()
    assert interrupted[0]["interrupted_from"] == "queued"
    recovered = runtime.recover_v3_run(run["run_id"])
    assert recovered.seed_candidate == candidate

    second = search_orchestrator(runtime, NoWorker())
    try:
        second.resume(run["run_id"], background=False)
        result = second.execute(run["run_id"])
    finally:
        second.close()

    assert result["status"] == "review_ready", result
    assert contract.load_input(run["run_id"]).input_mode == "costeer"


def test_recovery_rejects_tampered_seed_without_state_change(
    tmp_path: Path,
):
    (
        runtime,
        contract,
        request,
        context,
        _candidate,
    ) = build_costeer_fixture(tmp_path, "0002")
    orchestrator = search_orchestrator(runtime, NoWorker())
    run = orchestrator.create_run(request, context, background=False)
    orchestrator.close()

    runtime, _contract = open_runtime(tmp_path)
    runtime.interrupt_orphaned_search_runs()
    with duckdb.connect(str(tmp_path / "factor_mining.duckdb")) as conn:
        conn.execute(
            "UPDATE mining_v3.runtime_seed_candidate SET content_hash=? "
            "WHERE run_id=?",
            ["f" * 64, run["run_id"]],
        )

    resumed = search_orchestrator(runtime, NoWorker())
    try:
        with pytest.raises(V3RunStateError, match="预置候选"):
            resumed.resume(run["run_id"], background=False)
    finally:
        resumed.close()
    state = runtime.search_control_state(run["run_id"])
    assert state["status"] == "interrupted"
    assert state["interrupted_from"] == "queued"
    assert contract.load_input(run["run_id"]).input_mode == "costeer"
