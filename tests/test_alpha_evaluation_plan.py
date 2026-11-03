"""通用评估计划、融合来源和候选排序测试。"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from alpha.artifacts import file_sha256, write_json
from alpha.errors import AlphaEvaluationError
from alpha.evaluation_plan import EvaluationPlan
from common.paths import CONFIG_DIR
from common.paths import MODEL_DATA_DIR
from scripts.build_alpha.evaluation_executor import (
    EVALUATION_SCHEMA_VERSION,
    _load_reusable_evaluation,
    candidate_table,
    evaluation_analysis_id,
    select_candidate,
)


def _plan_payload() -> dict:
    """构造不依赖具体模型名称的 v2 计划。"""

    return {
        "schema_version": "alpha_model_evaluation_plan_v2",
        "reference": "REFERENCE",
        "gate": {
            "expected_years": [2022, 2023],
            "required_improvements": 2,
            "turnover_multiple_limit": 2.0,
        },
        "selection_metrics": ["annual_return", "sharpe"],
        "candidates": [
            {
                "name": "REFERENCE",
                "source": {"kind": "model", "id": "arbitrary_reference_id"},
                "score_policy": {"kind": "raw", "smoothing_weeks": 1},
                "holding_policy": {"kind": "top_n", "top_n": 30},
            },
            {
                "name": "CHALLENGER",
                "source": {"kind": "model", "id": "arbitrary_challenger_id"},
                "score_policy": {
                    "kind": "smooth_rank",
                    "smoothing_weeks": 4,
                },
                "holding_policy": {
                    "kind": "rank_buffer",
                    "top_n": 30,
                    "buffer_rank": 45,
                },
                "priority": 2,
            },
            {
                "name": "BLEND",
                "source": {
                    "kind": "blend",
                    "id": "arbitrary_blend",
                    "members": [
                        "arbitrary_reference_id",
                        "arbitrary_challenger_id",
                    ],
                    "weights": [3.0, 1.0],
                },
                "score_policy": {"kind": "raw", "smoothing_weeks": 1},
                "holding_policy": {"kind": "top_n", "top_n": 30},
                "selectable": False,
            },
        ],
        "controls": [
            {
                "name": "BASELINE",
                "source": {"kind": "baseline", "id": "arbitrary_baseline"},
                "score_policy": {"kind": "raw", "smoothing_weeks": 1},
                "holding_policy": {"kind": "top_n", "top_n": 30},
                "selectable": False,
            }
        ],
    }


def test_plan_v2_supports_arbitrary_sources_and_normalizes_blend():
    """计划可以引用任意 ID，并把融合权重归一化。"""

    plan = EvaluationPlan.from_dict(_plan_payload())
    assert plan.by_name("CHALLENGER").source.id == "arbitrary_challenger_id"
    assert plan.by_name("BLEND").source.weights == pytest.approx((0.75, 0.25))
    assert plan.by_name("BLEND").selectable is False
    assert plan.model_ids() == (
        "arbitrary_reference_id",
        "arbitrary_challenger_id",
    )


def test_plan_v3_reads_risk_gate():
    """新风险门槛进入计划内容并参与分析 ID。"""

    payload = _plan_payload()
    payload["schema_version"] = "alpha_model_evaluation_plan_v3"
    payload["risk_gate"] = {
        "rolling_weeks": 13,
        "min_periods": 8,
        "max_size_abs_exposure": 2.0,
        "max_liquidity_abs_exposure": 2.0,
        "max_industry_active_weight": 0.35,
    }
    plan = EvaluationPlan.from_dict(payload)
    assert plan.risk_gate is not None
    assert plan.risk_gate.rolling_weeks == 13
    assert plan.to_dict()["schema_version"] == "alpha_model_evaluation_plan_v3"


def test_plan_rejects_selectable_blend_and_duplicate_names():
    """融合不能伪装成可自动选中的单一模型，重复名称失败。"""

    payload = _plan_payload()
    payload["candidates"][2]["selectable"] = True
    with pytest.raises(AlphaEvaluationError, match="selectable=false"):
        EvaluationPlan.from_dict(payload)

    duplicate = _plan_payload()
    duplicate["candidates"][1]["name"] = "REFERENCE"
    with pytest.raises(AlphaEvaluationError, match="候选名重复"):
        EvaluationPlan.from_dict(duplicate)


def test_legacy_plan_remains_readable():
    """v1 最终留出配置不要求改写。"""

    plan = EvaluationPlan.from_file(
        CONFIG_DIR / "alpha" / "final_holdout_2024_2026.json"
    )
    assert plan.reference == "RIDGE_RAW"
    assert plan.by_name("RIDGE_RAW").source.kind == "model"
    assert plan.controls[0].source.kind == "baseline"


def test_selection_uses_plan_order_and_ignores_non_selectable():
    """可配置指标顺序决定胜者，融合和失败候选不进入选择。"""

    plan = EvaluationPlan.from_dict(_plan_payload())
    metrics = {
        "REFERENCE": {"annual_return": 0.10, "sharpe": 0.8},
        "CHALLENGER": {"annual_return": 0.18, "sharpe": 0.7},
        "BLEND": {"annual_return": 0.30, "sharpe": 1.2},
        "BASELINE": {"annual_return": 0.05, "sharpe": 0.3},
    }
    gates = {
        "CHALLENGER": {"passed": True},
        "BLEND": {"passed": True},
    }
    result = select_candidate(plan, metrics, gates)
    assert result["selected"] == "CHALLENGER"
    assert result["ranked"] == ["CHALLENGER"]


def test_candidate_table_and_analysis_id_follow_plan_content():
    """候选表保留计划规则，计划内容变化会改变分析 ID。"""

    plan = EvaluationPlan.from_dict(_plan_payload())
    metrics = {
        name: {
            "rank_ic_mean": 0.1,
            "annual_return": 0.15,
            "max_drawdown": -0.2,
            "turnover_two_way_annual": 50.0,
        }
        for name in ("REFERENCE", "CHALLENGER", "BLEND", "BASELINE")
    }
    result = {
        "plan": plan,
        "metrics": metrics,
        "gates": {"CHALLENGER": {"status": "pass"}},
        "selected_candidate": "CHALLENGER",
    }
    table = candidate_table(result)
    assert len(table) == 4
    assert table.loc[table["name"].eq("CHALLENGER"), "buffer_rank"].item() == 45

    left = evaluation_analysis_id(plan, {"m": {"sha256": "a"}}, {"id": "d"}, "c")
    changed = deepcopy(_plan_payload())
    changed["candidates"][1]["priority"] = 5
    right = evaluation_analysis_id(
        EvaluationPlan.from_dict(changed),
        {"m": {"sha256": "a"}},
        {"id": "d"},
        "c",
    )
    assert left != right


def test_formal_evaluation_artifact_is_complete():
    """正式通用评估包含十候选、风险时序和不可变候选分数。"""

    folders = sorted(
        path
        for path in (MODEL_DATA_DIR / "runs" / "evaluation").glob("evaluation_*")
        if (path / "manifest.json").is_file()
    )
    assert folders
    manifest = json.loads(
        (folders[-1] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["candidate_count"] == 10
    assert manifest["selected_candidate"] == "LIGHTGBM_SMOOTH_4W"
    assert manifest["status"] == "pass_evaluation"
    assert {
        "candidates.csv",
        "decision.json",
        "orders.parquet",
        "plan.json",
        "report.md",
        "weekly_results.parquet",
        "yearly.csv",
        "scores/SKLEARN_HIST_RAW.parquet",
        "risk/SKLEARN_HIST_RAW/summary.json",
    } <= set(manifest["files"])


def test_existing_evaluation_reuse_validates_every_output(tmp_path: Path):
    """已有评估目录只有输入和全部输出哈希一致时才可复用。"""

    plan = EvaluationPlan.from_dict(_plan_payload())
    sources = {"arbitrary_reference_id": {"sha256": "a" * 64}}
    dataset_manifest = {"dataset_id": "dataset_demo"}
    code_hash = "b" * 64
    output = tmp_path / "evaluation_demo"
    output.mkdir()
    artifact = output / "candidates.csv"
    artifact.write_text("name\nDEMO\n", encoding="utf-8")
    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "analysis_id": "evaluation_demo",
        "status": "pass_evaluation",
        "plan": plan.to_dict(),
        "sources": sources,
        "dataset_manifest": dataset_manifest,
        "code_sha256": code_hash,
        "files": {"candidates.csv": file_sha256(artifact)},
    }
    write_json(output / "manifest.json", manifest)

    loaded = _load_reusable_evaluation(
        output,
        analysis_id="evaluation_demo",
        plan=plan,
        sources=sources,
        dataset_manifest=dataset_manifest,
        code_sha256=code_hash,
    )
    assert loaded == manifest

    artifact.write_text("name\nCHANGED\n", encoding="utf-8")
    with pytest.raises(AlphaEvaluationError, match="哈希不一致"):
        _load_reusable_evaluation(
            output,
            analysis_id="evaluation_demo",
            plan=plan,
            sources=sources,
            dataset_manifest=dataset_manifest,
            code_sha256=code_hash,
        )
