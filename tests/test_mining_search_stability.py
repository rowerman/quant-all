"""V2 邻域变体生成和稳定性报告测试。"""

from __future__ import annotations

import pytest

from mining.expression_validation import validate_expression
from mining.search_stability import generate_primary_window_variants
from mining.search_stability_contracts import (
    VariantPlan,
    WindowVariantObservation,
    build_stability_report,
)


def _candidate(expression: str = "mean(close, 5)"):
    return validate_expression(
        expression,
        direction=1,
    ).to_candidate(
        mechanism="短期价格修复",
        description="测试候选",
        steps=("读取收盘价", "计算窗口均值"),
        discovery_round=1,
        discovery_index=1,
    )


def _observation(value: float, delta: int) -> WindowVariantObservation:
    return WindowVariantObservation(
        candidate_id=f"cand_variant_{delta}",
        candidate_hash=("a" if delta < 0 else "b") * 64,
        window=5 + delta,
        delta=delta,
        status="passed",
        gate_passed=True,
        rank_ic=value,
        rank_ic_ir=0.4,
    )


def test_primary_window_variants_are_deterministic_and_limited():
    candidate = _candidate("mean(mean(close, 5), 3)")

    first = generate_primary_window_variants(candidate)
    second = generate_primary_window_variants(candidate)

    assert first == second
    assert first.status == "ready"
    assert first.base_window == 3
    assert [item.variant_window for item in first.variants] == [2, 4]
    assert len({item.candidate.candidate_id for item in first.variants}) == 2


def test_window_boundary_only_generates_valid_neighbor():
    plan = generate_primary_window_variants(_candidate("mean(close, 1)"))

    assert plan.base_window == 1
    assert [item.variant_window for item in plan.variants] == [2]


def test_non_window_expression_is_not_applicable():
    plan = generate_primary_window_variants(_candidate("close + open"))

    assert plan.status == "not_applicable"
    assert plan.variants == ()


def test_stable_versions_require_direction_consistency_and_half_rank_ic():
    candidate = _candidate()
    plan = VariantPlan(
        base_candidate_id=candidate.candidate_id,
        base_window=5,
        variants=(),
        status="ready",
    )

    stable = build_stability_report(
        run_id="searchrun_stability",
        candidate=candidate,
        plan=plan,
        base_rank_ic=0.04,
        variants=(_observation(0.03, -1), _observation(0.02, 1)),
    )
    direction_failed = build_stability_report(
        run_id="searchrun_stability",
        candidate=candidate,
        plan=plan,
        base_rank_ic=0.04,
        variants=(_observation(-0.03, -1), _observation(-0.02, 1)),
    )
    ratio_failed = build_stability_report(
        run_id="searchrun_stability",
        candidate=candidate,
        plan=plan,
        base_rank_ic=0.04,
        variants=(_observation(0.01, -1), _observation(0.01, 1)),
    )

    assert stable.status == "stable"
    assert stable.aligned_versions == 3
    assert stable.rank_ic_ratio == pytest.approx(0.625)
    assert direction_failed.status == "unstable"
    assert direction_failed.aligned_versions == 1
    assert ratio_failed.status == "unstable"
    assert ratio_failed.rank_ic_ratio == pytest.approx(0.25)


def test_missing_variant_metrics_are_not_zero_filled():
    candidate = _candidate()
    plan = VariantPlan(
        base_candidate_id=candidate.candidate_id,
        base_window=5,
        variants=(),
        status="ready",
    )
    failed = WindowVariantObservation(
        candidate_id="cand_failed",
        candidate_hash="c" * 64,
        window=4,
        delta=-1,
        status="error",
        gate_passed=False,
        rank_ic=None,
        rank_ic_ir=None,
        error_code="MINING-106",
    )

    report = build_stability_report(
        run_id="searchrun_stability",
        candidate=candidate,
        plan=plan,
        base_rank_ic=0.04,
        variants=(failed, _observation(0.03, 1)),
    )

    assert report.status == "not_computable"
    assert report.rank_ic_ratio == pytest.approx(0.75)
    assert report.variants[0].rank_ic is None
