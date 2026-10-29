"""V2 选择期相关性、正式因子暴露和增量价值测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mining.search_budget import TrialCounts
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_context import FrozenSearchContext
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_run import SearchRunRequest
from mining.search_scorecard import build_candidate_scorecard
from mining.search_selection_diagnostics import (
    CandidateCorrelationReport,
    IncrementalValueReport,
    build_selection_diagnostics,
    compute_candidate_correlations,
    formal_factor_incremental_value,
)
from mining.storage_v2 import MiningV2Store


def _values(
    candidate_id: str,
    *,
    days: int = 20,
    sign: int = 1,
    constant: bool = False,
) -> tuple[str, pd.DataFrame]:
    rows = []
    for day in range(days):
        for stock in range(6):
            value = 1.0 if constant else sign * (stock + day)
            rows.append(
                {
                    "stock_code": f"S{stock:03d}",
                    "trade_date": f"2020-01-{day + 1:02d}",
                    "value": value,
                }
            )
    return candidate_id, pd.DataFrame(rows)


def _correlations(values):
    return compute_candidate_correlations(
        run_id="searchrun_diagnostics",
        window_start="2020-01-01",
        window_end="2020-01-31",
        values=dict(values),
        min_stocks=3,
        min_days=5,
    )


def test_same_batch_correlation_is_signed_and_clusters_absolute_correlation():
    reports = _correlations(
        [
            _values("cand_a", sign=1),
            _values("cand_b", sign=1),
            _values("cand_c", sign=-1),
        ]
    )

    assert reports["cand_a"].peers[0]["correlation"] == 1.0
    assert reports["cand_a"].peers[1]["correlation"] == -1.0
    assert reports["cand_a"].cluster_id == reports["cand_b"].cluster_id
    assert reports["cand_a"].cluster_id == reports["cand_c"].cluster_id
    assert reports["cand_a"].cluster_candidate_ids == (
        "cand_a",
        "cand_b",
        "cand_c",
    )
    assert reports["cand_a"].max_abs_correlation == 1.0
    assert reports["cand_a"].report_hash == _correlations(
        [
            _values("cand_a", sign=1),
            _values("cand_b", sign=1),
            _values("cand_c", sign=-1),
        ]
    )["cand_a"].report_hash


def test_constant_and_short_pairs_are_not_treated_as_zero_correlation():
    reports = _correlations(
        [
            _values("cand_a", sign=1),
            _values("cand_constant", constant=True),
            _values("cand_short", sign=-1, days=3),
        ]
    )

    peers = {
        item["candidate_id"]: item for item in reports["cand_a"].peers
    }
    assert peers["cand_constant"]["correlation"] is None
    assert peers["cand_constant"]["status"] == "insufficient_days"
    assert peers["cand_short"]["correlation"] is None
    assert reports["cand_a"].status == "not_computable"
    assert reports["cand_a"].max_abs_correlation is None


def test_formal_factor_metrics_distinguish_missing_peer_and_available():
    assert formal_factor_incremental_value({}) == (
        "not_computable",
        None,
        None,
        None,
    )
    assert formal_factor_incremental_value(
        {"existing_correlation": {"reason": "no_peers"}}
    ) == ("no_formal_factors", None, None, None)
    assert formal_factor_incremental_value(
        {
            "existing_correlation": {
                "peer": "MOM_20",
                "correlation": -0.31,
                "max_absolute": 0.31,
            }
        }
    ) == ("available", "MOM_20", -0.31, 0.31)


def test_selection_diagnostics_are_hashed_and_read_by_scorecard(tmp_path: Path):
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    run = store.create_run(
        SearchRunRequest(
            "selection-diagnostics-0001",
            "短期反转可能带来开发期超额收益",
        ),
        FrozenSearchContext(data_fingerprint="a" * 64),
    )
    card = HypothesisCard(
        run_id=run["run_id"],
        turn_index=1,
        branch_id="branch-0001",
        statement="短期反转可能带来开发期超额收益",
        mechanism="短期价格修复",
        expected_direction="positive",
        falsification_criteria=("年度同号率低于 0.6",),
    )
    batch = ExperimentBatch(
        run_id=run["run_id"],
        hypothesis_id=card.hypothesis_id,
        turn_index=1,
        branch_id=card.branch_id,
        candidate_ids=("cand_0001",),
        parent_candidate_ids=(),
        development_start="2012-01-01",
        development_end="2019-12-31",
        input_hash="b" * 64,
    )
    store.begin_turn(run["run_id"], card, batch)
    store.record_candidate(
        run["run_id"],
        candidate_id="cand_0001",
        candidate_hash="c" * 64,
        semantic_hash="d" * 64,
        name="MINE_SELECTION_0001",
        status="screening",
        hypothesis_id=card.hypothesis_id,
        batch_id=batch.batch_id,
        payload={"canonical_expression": "mean(close, 5)"},
    )
    correlation = _correlations([_values("cand_0001")])["cand_0001"]
    incremental = IncrementalValueReport(
        formal_factor_status="available",
        formal_factor_peer="MOM_20",
        formal_factor_correlation=0.1,
        formal_factor_max_absolute=0.1,
        raw_rank_ic=0.04,
        incremental_rank_ic=0.03,
        survivor_share=0.75,
        valid_days=100,
        status="available",
    )
    diagnostics = build_selection_diagnostics(
        run_id=run["run_id"],
        candidate_id="cand_0001",
        input_material={
            "factor_sha256": "e" * 64,
            "sample_dates": list(correlation.sample_dates),
        },
        correlation=correlation,
        incremental=incremental,
    )
    store.record_evaluation_evidence(diagnostics.to_evaluation_evidence())

    scorecard = build_candidate_scorecard(
        store,
        run["run_id"],
        "cand_0001",
    )

    payload = scorecard.to_dict()["selection_diagnostics"]
    assert payload["evidence_hash"] == diagnostics.evidence_hash
    assert payload["incremental"]["incremental_rank_ic"] == 0.03
    assert scorecard.trial_counts == TrialCounts()
    assert len(scorecard.scorecard_hash) == 64


def test_diagnostic_evidence_accepts_new_stage():
    report = CandidateCorrelationReport(
        run_id="searchrun_diagnostics",
        candidate_id="cand_0001",
        window_start="2020-01-01",
        window_end="2022-12-31",
        sample_dates=(),
        peers=(),
        max_abs_correlation=None,
        cluster_id="cluster-0001",
        cluster_candidate_ids=("cand_0001",),
        status="single_candidate",
    )
    incremental = IncrementalValueReport(
        formal_factor_status="no_formal_factors",
        formal_factor_peer=None,
        formal_factor_correlation=None,
        formal_factor_max_absolute=None,
        raw_rank_ic=0.04,
        incremental_rank_ic=0.03,
        survivor_share=0.75,
        valid_days=100,
        status="available",
    )
    diagnostics = build_selection_diagnostics(
        run_id="searchrun_diagnostics",
        candidate_id="cand_0001",
        input_material={"context_hash": "a" * 64},
        correlation=report,
        incremental=incremental,
    )
    evidence = diagnostics.to_evaluation_evidence()

    assert isinstance(evidence, SearchEvaluationEvidence)
    assert evidence.stage == "selection_diagnostics"
