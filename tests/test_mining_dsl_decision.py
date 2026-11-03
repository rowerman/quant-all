"""P2-03 受控 DSL 原生执行器决策的确定性测试。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from mining.catalog import load_catalog
from tools.monitor.mining_dsl_analysis import (
    OPERATOR_COVERAGE_EXPRESSIONS,
    classify_profile_operator,
    decision_from_measurements,
    summarize_corpus,
    summarize_profile,
)
from tools.monitor.mining_dsl_report import summarize_runs


def test_corpus_covers_every_frozen_operator_and_is_deduplicated():
    """覆盖语料必须包含全部算子，别名和重复项只计一次。"""

    catalog = load_catalog()
    corpus = summarize_corpus(
        {
            "coverage": OPERATOR_COVERAGE_EXPRESSIONS,
            "aliases": ("delay(close, 2)", "mean(close,5)"),
        },
        catalog,
    )

    assert corpus["coverage"]["missing"] == []
    assert corpus["coverage"]["unknown"] == []
    assert corpus["coverage"]["covered_operator_count"] == len(catalog.operators)
    assert corpus["candidate_count"] == len(OPERATOR_COVERAGE_EXPRESSIONS)
    assert corpus["operator_counts"]["lag"] == 1
    assert corpus["depth"]["max"] <= 6


def test_profile_operator_classification_and_summary():
    """Profile 分类须区分扫描、表达式、物化和未归属耗时。"""

    profile = {
        "latency": 10.0,
        "cpu_time": 8.0,
        "children": [
            {
                "operator_type": "COPY_TO_FILE",
                "operator_timing": 1.0,
                "extra_info": {},
                "children": [
                    {
                        "operator_type": "PROJECTION",
                        "operator_timing": 2.0,
                        "extra_info": {
                            "Projections": ["stock_code", "close + open"]
                        },
                        "children": [
                            {
                                "operator_type": "WINDOW",
                                "operator_timing": 3.0,
                                "extra_info": {},
                                "children": [
                                    {
                                        "operator_type": "TABLE_SCAN",
                                        "operator_timing": 1.0,
                                        "extra_info": {},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    summary = summarize_profile(profile)

    assert classify_profile_operator("TABLE_SCAN", {}) == "scan"
    assert classify_profile_operator("WINDOW", {}) == "expression"
    assert classify_profile_operator(
        "PROJECTION",
        {"Projections": ["#0", "__internal_decompress_string(#1)"]},
    ) == "materialization"
    assert summary["query_wall_s"] == 10.0
    assert summary["query_cpu_s"] == 8.0
    assert summary["category_cpu_s"] == {
        "scan": 1.0,
        "expression": 5.0,
        "materialization": 1.0,
        "other": 0.0,
        "unattributed": 1.0,
    }
    assert summary["category_wall_equivalent_s"] == {
        "scan": 1.25,
        "expression": 6.25,
        "materialization": 1.25,
        "other": 0.0,
        "unattributed": 1.25,
    }


@pytest.mark.parametrize(
    "share, decision, reasons",
    [
        (0.50, "create_followup_design", []),
        (0.4999, "do_not_implement", ["expression_share_below_50pct"]),
        (
            0.1999,
            "do_not_implement",
            [
                "expression_share_below_50pct",
                "zero_cost_improvement_below_20pct",
            ],
        ),
    ],
)
def test_decision_threshold_boundaries(share, decision, reasons):
    """表达式占比和零成本改善门槛必须按约定值执行。"""

    result = decision_from_measurements(share, share)

    assert result["decision"] == decision
    assert result["reasons"] == reasons
    assert result["cpp_modules_replaced"] == 0


def _run(
    *,
    total: float,
    expression: float,
    stable_hash: str = "a" * 64,
) -> dict:
    """构造一轮可汇总的最小测量结果。"""

    return {
        "threads": {"duckdb": 4, "native": 1},
        "stable_payload_sha256": stable_hash,
        "peak_rss_mb": 100.0,
        "stages_s": {
            "plan": 0.01,
            "compile": 0.02,
            "query_copy": 5.0,
            "split_publish": 0.5,
            "materialization": 5.53,
            "evaluation": total - 5.53,
            "total_candidate": total,
        },
        "profile": {
            "query_wall_s": 5.0,
            "query_cpu_s": 5.0,
            "category_cpu_s": {
                "scan": 0.5,
                "expression": expression,
                "materialization": 1.0,
                "other": 0.0,
                "unattributed": 5.0 - expression - 1.5,
            },
            "category_wall_equivalent_s": {
                "scan": 0.5,
                "expression": expression,
                "materialization": 1.0,
                "other": 0.0,
                "unattributed": 5.0 - expression - 1.5,
            },
        },
    }


def test_summarize_runs_uses_medians_and_rejects_payload_drift():
    """三轮耗时取中位数，稳定 payload 不一致时明确失败。"""

    runs = [
        _run(total=40.0, expression=4.0),
        _run(total=50.0, expression=5.0),
        _run(total=60.0, expression=6.0),
    ]

    summary = summarize_runs(runs)

    assert summary["stage_s"]["total_candidate"] == 50.0
    assert summary["profile"]["category_wall_equivalent_s"]["expression"] == 5.0
    assert summary["profile"]["expression_share_of_pipeline"] == pytest.approx(
        0.1
    )
    assert summary["decision"]["decision"] == "do_not_implement"

    drifted = deepcopy(runs)
    drifted[-1]["stable_payload_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="稳定 payload"):
        summarize_runs(drifted)
