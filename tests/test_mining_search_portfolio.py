"""V2 Pareto 入围与独立研究质量结论测试。"""

from __future__ import annotations

from mining.search_budget import TrialCounts
from mining.search_control import SearchControlPolicy
from mining.search_portfolio import select_portfolio_finalists
from mining.search_research_quality import evaluate_research_quality
from mining.search_scorecard import CandidateScoreCard


def _scorecard(
    candidate_id: str,
    *,
    rank_ic: float | None = 0.04,
    icir: float | None = 0.4,
    year_sign: float | None = 0.75,
    coverage: float | None = 0.8,
    incremental: float | None = 0.03,
    stability: str = "stable",
    ratio: float | None = 0.8,
    cluster: str = "cluster-0001",
    candidate_hash: str | None = None,
    status: str = "passed",
) -> CandidateScoreCard:
    selection_metrics = {
        "rank_ic": rank_ic,
        "icir": icir,
        "year_sign_consistency": year_sign,
        "coverage": coverage,
    }
    analysis = {
        "selection_diagnostics": {
            "correlation": {"cluster_id": cluster},
            "incremental": {"incremental_rank_ic": incremental},
        },
        "stability": {
            "status": stability,
            "rank_ic_ratio": ratio,
        },
    }
    return CandidateScoreCard(
        run_id="searchrun_portfolio",
        candidate_id=candidate_id,
        candidate_hash=candidate_hash or (candidate_id[-1] * 64),
        semantic_hash=candidate_id[-1] * 64,
        hypothesis_id="hyp_portfolio_0001",
        batch_id="batch_portfolio_0001",
        context_hash="a" * 64,
        status="screening",
        windows={
            "development_start": "2012-01-01",
            "development_end": "2019-12-31",
            "selection_start": "2020-01-01",
            "selection_end": "2022-12-31",
        },
        decision_stage="selection",
        metrics=selection_metrics,
        metrics_by_stage={"selection": selection_metrics},
        evidence_status_by_stage={"selection": status},
        analysis_by_stage=analysis,
        gate_passed=True,
        failure_reasons=(),
        evidence_ids=(),
        evidence_hashes=(),
        trial_counts=TrialCounts(proposed=2, evaluated=2),
        candidate_trial_counts=TrialCounts(proposed=1, evaluated=1),
    )


def test_portfolio_uses_tradeoffs_and_is_independent_of_input_order():
    first = _scorecard(
        "cand_a",
        rank_ic=0.04,
        incremental=0.03,
        coverage=0.7,
        cluster="cluster-0001",
    )
    second = _scorecard(
        "cand_b",
        rank_ic=0.03,
        incremental=0.04,
        coverage=0.8,
        cluster="cluster-0002",
    )

    left = select_portfolio_finalists(
        (first, second),
        SearchControlPolicy(),
        max_finalists=2,
    )
    right = select_portfolio_finalists(
        (second, first),
        SearchControlPolicy(),
        max_finalists=2,
    )

    assert left.to_dict() == right.to_dict()
    assert left.fronts == (("cand_a", "cand_b"),)
    assert left.finalists == ("cand_a", "cand_b")


def test_portfolio_limits_one_candidate_per_correlation_cluster():
    first = _scorecard(
        "cand_a",
        rank_ic=0.05,
        cluster="cluster-0001",
    )
    correlated = _scorecard(
        "cand_b",
        rank_ic=0.04,
        cluster="cluster-0001",
    )
    diverse = _scorecard(
        "cand_c",
        rank_ic=0.03,
        cluster="cluster-0002",
    )

    result = select_portfolio_finalists(
        (first, correlated, diverse),
        SearchControlPolicy(),
        max_finalists=2,
    )

    assert result.finalists[0] == "cand_a"
    assert "cand_b" not in result.finalists
    assert "cand_c" in result.finalists


def test_missing_incremental_or_stability_evidence_is_not_zero_filled():
    incomplete = _scorecard(
        "cand_a",
        incremental=None,
        stability="not_computable",
        ratio=None,
    )

    result = select_portfolio_finalists(
        (incomplete,),
        SearchControlPolicy(),
        max_finalists=1,
    )

    assert result.finalists == ()
    assert result.candidates[0].eligible is False
    assert "增量价值" in result.candidates[0].exclusion_reason
    assert "稳定性" in result.candidates[0].exclusion_reason


def test_quality_conclusion_separates_execution_and_research_quality():
    policy = SearchControlPolicy()

    passed = evaluate_research_quality(
        _scorecard("cand_a"),
        policy,
        stage="selection",
    )
    failed = evaluate_research_quality(
        _scorecard("cand_b", rank_ic=0.01),
        policy,
        stage="selection",
    )
    execution = evaluate_research_quality(
        _scorecard("cand_c", status="error"),
        policy,
        stage="selection",
    )
    missing = evaluate_research_quality(
        _scorecard("cand_d", icir=None),
        policy,
        stage="selection",
    )

    assert passed.status == "quality_passed"
    assert failed.status == "quality_failed"
    assert execution.status == "execution_failed"
    assert missing.status == "evidence_insufficient"
    assert missing.missing_fields == ("icir",)
    assert passed.context["attempts"]["proposed"] == 2
