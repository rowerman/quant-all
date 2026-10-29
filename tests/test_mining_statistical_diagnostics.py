"""V2 Newey-West 和 BH 多重检验诊断测试。"""

from __future__ import annotations

import math
from statistics import NormalDist

import pytest

from mining.search_statistical_diagnostics import (
    build_multiple_testing_report,
)


def _series(scale: float, *, days: int = 25):
    values = tuple(
        scale * ((index % 5) - 2) / 10
        for index in range(days)
    )
    dates = tuple(f"2020-01-{index + 1:02d}" for index in range(days))
    return dates, values


def _expected_hac(values):
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    lag = int(math.floor(4 * (len(values) / 100) ** (2 / 9)))
    variance = sum(value * value for value in centered) / len(values)
    for offset in range(1, lag + 1):
        weight = 1 - offset / (lag + 1)
        covariance = sum(
            centered[index] * centered[index - offset]
            for index in range(offset, len(values))
        ) / len(values)
        variance += 2 * weight * covariance
    standard_error = math.sqrt(max(variance, 0) / len(values))
    t_stat = mean / standard_error
    p_value = 2 * (1 - NormalDist().cdf(abs(t_stat)))
    return mean, standard_error, t_stat, p_value, lag


def test_newey_west_matches_independent_formula_and_is_reproducible():
    series = {"cand_a": _series(1.0)}
    first = build_multiple_testing_report(
        run_id="searchrun_stats",
        series=series,
        attempt_count=4,
        evaluated_count=1,
    )
    second = build_multiple_testing_report(
        run_id="searchrun_stats",
        series=series,
        attempt_count=4,
        evaluated_count=1,
    )
    row = first.candidates[0]
    expected = _expected_hac(series["cand_a"][1])

    assert first.evidence_hash == second.evidence_hash
    assert row.mean_rank_ic == pytest.approx(expected[0])
    assert row.standard_error == pytest.approx(expected[1])
    assert row.t_stat == pytest.approx(expected[2])
    assert row.p_value == pytest.approx(expected[3])
    assert row.lag == expected[4]
    assert row.bh_q_value is not None


def test_bh_uses_conservative_attempt_denominator():
    report = build_multiple_testing_report(
        run_id="searchrun_stats",
        series={
            "cand_a": _series(1.0),
            "cand_b": _series(0.5),
        },
        attempt_count=8,
        evaluated_count=2,
    )

    assert report.denominator == 8
    assert report.tested_count == 2
    assert all(
        item.bh_q_value is not None
        and item.p_value is not None
        and item.bh_q_value >= item.p_value
        for item in report.candidates
    )
    assert "不能替代人工" in report.warnings[0]


def test_short_or_constant_series_are_not_zero_filled():
    report = build_multiple_testing_report(
        run_id="searchrun_stats",
        series={
            "cand_short": _series(1.0, days=10),
            "cand_constant": (
                tuple(f"2020-02-{index + 1:02d}" for index in range(25)),
                (0.03,) * 25,
            ),
        },
        attempt_count=2,
        evaluated_count=2,
    )

    assert report.tested_count == 0
    assert all(item.status == "not_computable" for item in report.candidates)
    assert all(item.p_value is None for item in report.candidates)
    assert all(item.bh_q_value is None for item in report.candidates)
