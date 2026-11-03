"""使用真实行情短区间的 S6 评估链路测试。"""

from __future__ import annotations

from pathlib import Path

from mining.catalog import load_catalog
from mining.contract_requests import FrozenContext, MiningConstraints, MiningRunRequest
from mining.orchestrator import MiningOrchestrator
from mining.storage import MiningStore
from mining.worker_client import MockWorkerClient
from mining_worker.protocol import WorkerCandidateDraft, WorkerGenerationResponse


def test_real_short_interval_reaches_review_ready(tmp_path: Path):
    """真实数据只读短链路完成 technical/quick/full/holdout。"""

    catalog = load_catalog()
    constraints = MiningConstraints(
        max_rounds=1,
        max_candidates=1,
        full_eval_top_k=1,
        max_llm_calls=1,
        max_failure_rate=1.0,
        min_quality_checks_for_failure_stop=1,
    )
    request = MiningRunRequest(
        "s6-real-short-20261024",
        "短期价格趋势可能带来未来五日超额收益",
        constraints=constraints,
    )
    context = FrozenContext(
        data_fingerprint="a" * 64,
        research_start="2024-03-01",
        research_end="2024-12-31",
        full_eval_start="2024-09-01",
        full_eval_end="2024-12-31",
        holdout_start="2025-01-01",
        holdout_end="2025-03-31",
        field_catalog=catalog.frozen_field_mappings(),
        operator_catalog=catalog.frozen_operator_mappings(),
        budget=constraints,
    )
    store = MiningStore(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "candidate-data",
    )
    run = store.create_run(request, context, worker_mode="mock")
    worker = MockWorkerClient(
        WorkerGenerationResponse(
            candidates=(
                WorkerCandidateDraft(
                    expression="mean(close,5)",
                    mechanism="价格趋势",
                    description="收盘价五日均值",
                    direction=1,
                    steps=("计算收盘价五日均值",),
                ),
            ),
            usage={"calls": 1, "mode": "mock"},
        )
    )
    orchestrator = MiningOrchestrator(store=store, worker=worker)
    orchestrator._assert_snapshot = lambda _context: None

    final = orchestrator.execute(run["run_id"])
    orchestrator.close()

    assert final["status"] == "review_ready"
    assert [
        (item["stage"], item["status"])
        for item in store.list_evaluations(run["run_id"])
    ] == [
        ("technical", "passed"),
        ("quick", "passed"),
        ("full", "passed"),
        ("holdout", "passed"),
    ]
    assert store.list_candidates(run["run_id"])[0]["status"] == "review_ready"
    assert len(store.get_holdout_usage()["candidate_ids"]) == 1
