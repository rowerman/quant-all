"""V2 统一候选评分卡测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mining.search_budget import TrialCounts
from mining.search_cards import ExperimentBatch, HypothesisCard
from mining.search_context import FrozenSearchContext
from mining.search_evaluation_evidence import SearchEvaluationEvidence
from mining.search_orchestrator_support import trial_record
from mining.search_run import SearchRunRequest
from mining.search_scorecard import (
    build_candidate_scorecard,
    build_run_scorecards,
)
from mining.search_statistical_diagnostics import (
    build_multiple_testing_report,
)
from mining.storage_v2 import MiningV2Store


def _store(tmp_path: Path) -> tuple[MiningV2Store, str, HypothesisCard, ExperimentBatch]:
    store = MiningV2Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )
    run = store.create_run(
        SearchRunRequest(
            "scorecard-test-0001",
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
        name="MINE_SCORE_0001",
        status="screening",
        hypothesis_id=card.hypothesis_id,
        batch_id=batch.batch_id,
        payload={"canonical_expression": "mean(close, 5)"},
    )
    store.record_trial(
        trial_record(
            run_id=run["run_id"],
            trial_key="score-proposed",
            kind="proposed",
            turn_index=1,
            hypothesis_id=card.hypothesis_id,
            batch_id=batch.batch_id,
            candidate_id="cand_0001",
        )
    )
    store.record_trial(
        trial_record(
            run_id=run["run_id"],
            trial_key="score-evaluated",
            kind="evaluated",
            turn_index=1,
            hypothesis_id=card.hypothesis_id,
            batch_id=batch.batch_id,
            candidate_id="cand_0001",
        )
    )
    return store, run["run_id"], card, batch


def _evidence(
    run_id: str,
    stage: str,
    status: str,
    metrics: dict,
    suffix: str,
) -> SearchEvaluationEvidence:
    return SearchEvaluationEvidence(
        run_id=run_id,
        candidate_id="cand_0001",
        stage=stage,
        input_hash=suffix * 64,
        status=status,
        metrics=metrics,
        warnings=(),
    )


def test_scorecard_combines_registered_stage_evidence(tmp_path: Path):
    store, run_id, _card, _batch = _store(tmp_path)
    store.record_evaluation_evidence(
        _evidence(
            run_id,
            "technical",
            "passed",
            {"values": {"coverage": 0.8, "rows": 10000}},
            "1",
        )
    )
    store.record_evaluation_evidence(
        _evidence(
            run_id,
            "development",
            "passed",
            {
                "rank_ic_mean": 0.03,
                "rank_ic_ir": 0.4,
                "year_sign_consistency": 0.75,
                "valid_days": 100,
            },
            "2",
        )
    )
    store.record_evaluation_evidence(
        _evidence(
            run_id,
            "full",
            "passed",
            {
                "events": {
                    "overview": {
                        "metrics": {
                            "ic_stats": {
                                "rank_ic": {"mean": 0.04, "ir": 0.5},
                                "valid_days": 500,
                            }
                        }
                    },
                    "turnover": {
                        "metrics": {
                            "turnover": {
                                "daily_turnover": {"mean": [0.2, 0.4]}
                            }
                        }
                    },
                    "decay": {
                        "metrics": {
                            "decay": {
                                "primary_horizon": 5,
                                "curve": [
                                    {
                                        "horizon": 5,
                                        "rank_ic_mean": 0.035,
                                        "rank_ic_ir": 0.3,
                                    }
                                ],
                            }
                        }
                    },
                }
            },
            "3",
        )
    )

    card = build_candidate_scorecard(store, run_id, "cand_0001")

    assert card.decision_stage == "full"
    assert card.metrics["rank_ic"] == 0.04
    assert card.metrics["icir"] == 0.5
    assert card.metrics["year_sign_consistency"] == 0.75
    assert card.metrics["coverage"] == 0.8
    assert card.metrics["turnover"] == pytest.approx(0.3)
    assert card.metrics["ic_decay"] == 0.035
    assert card.metrics["sample_size"] == 10000
    assert card.metrics["valid_days"] == 500
    assert card.gate_passed is True
    assert card.trial_counts == TrialCounts(proposed=1, evaluated=1)
    assert len(card.evidence_ids) == 3
    assert len(card.scorecard_hash) == 64
    assert build_run_scorecards(store, run_id) == (card,)


def test_scorecard_marks_missing_and_failed_evidence_without_zero_fill(
    tmp_path: Path,
):
    store, run_id, _card, _batch = _store(tmp_path)
    store.record_evaluation_evidence(
        _evidence(
            run_id,
            "technical",
            "failed",
            {"values": {"coverage": 0.1, "rows": 100}},
            "4",
        )
    )

    card = build_candidate_scorecard(store, run_id, "cand_0001")

    assert card.gate_passed is False
    assert card.metrics["rank_ic"] is None
    assert card.metrics["icir"] is None
    assert {"rank_ic", "icir", "turnover", "ic_decay"} <= set(
        card.missing_fields
    )
    assert card.failure_reasons
    assert len(card.scorecard_hash) == 64


def test_scorecard_exposes_multiple_testing_diagnostics(tmp_path: Path):
    store, run_id, _card, _batch = _store(tmp_path)
    values = tuple(0.01 * ((index % 5) - 2) for index in range(25))
    dates = tuple(f"2020-01-{index + 1:02d}" for index in range(25))
    report = build_multiple_testing_report(
        run_id=run_id,
        series={"cand_0001": (dates, values)},
        attempt_count=3,
        evaluated_count=1,
    )
    store.record_evaluation_evidence(
        report.to_evaluation_evidence("cand_0001")
    )

    scorecard = build_candidate_scorecard(store, run_id, "cand_0001")

    assert scorecard.to_dict()["multiple_testing"]["evidence_hash"] == (
        report.evidence_hash
    )
