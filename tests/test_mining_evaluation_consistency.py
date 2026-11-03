"""候选临时路径与正式因子路径的全指标一致性测试。"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import factors
from calculate import (
    EvalParams,
    build_panel,
    evaluate_context,
)
from calculate.config import DB_FILE
from mining import CandidateEvaluationInput, FrozenContext, MiningRunRequest, evaluate_candidate, validate_expression
from mining.candidate_materialization import materialize_candidate

pytestmark = pytest.mark.skipif(
    not DB_FILE.exists(), reason="需要真实主库 data/databases/quant.duckdb"
)

START, END = "2024-05-06", "2024-06-28"


def _candidate():
    """用受控 DSL 表达与 LN_MV 完全相同的总市值对数。"""

    return validate_expression(
        "ln(total_mv)",
        direction=-1,
    ).to_candidate(
        mechanism="规模因子",
        description="总市值对数",
        steps=("取总市值", "计算自然对数"),
        discovery_round=1,
        discovery_index=1,
    )


def _normalize(value, *, key: str | None = None):
    """去掉运行耗时，并把相关性矩阵收敛到目标对正式因子的结果。"""

    if isinstance(value, dict):
        ignored = {"elapsed_ms", "total_elapsed_ms", "generated_at"}
        return {
            item_key: _normalize(item, key=item_key)
            for item_key, item in value.items()
            if item_key not in ignored
        }
    if isinstance(value, list):
        return [_normalize(item, key=key) for item in value]
    return value


def _assert_nested_equal(left, right, path: str = "") -> None:
    """递归比较 JSON 载荷，浮点数按 1e-6 容差。"""

    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right), f"{path}: keys differ"
        for key in left:
            _assert_nested_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right), f"{path}: lengths differ"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_nested_equal(left_item, right_item, f"{path}[{index}]")
        return
    if isinstance(left, float) or isinstance(right, float):
        assert float(left) == pytest.approx(float(right), abs=1e-6, rel=1e-6), path
        return
    assert left == right, f"{path}: {left!r} != {right!r}"


def _comparable_correlation(payload: dict, formal_target: str) -> dict:
    """候选相关结果移除同名正式因子后应与正式目标对等。"""

    metrics = payload["metrics"]["correlation"]
    peers = {
        item["factor"]: {
            "correlation": item["correlation"],
            "days": item["days"],
        }
        for item in metrics["top"]
        if item["factor"] != formal_target
    }
    return {"peers": peers, "min_days": metrics["min_days"], "min_stocks": metrics["min_stocks"]}


def test_candidate_and_formal_paths_share_all_metrics():
    """同一公式走正式与临时路径时，全部指标和面板值一致。"""

    candidate = _candidate()
    request = MiningRunRequest("evaluation-consistency-001", "测试候选与正式因子口径一致")
    context = FrozenContext(data_fingerprint="b" * 64)
    with TemporaryDirectory(prefix="mining-consistency-") as root:
        materialized = materialize_candidate(
            candidate,
            context,
            Path(root),
            start=START,
            end=END,
        )
        params = EvalParams(
            materialized.factor_view.name,
            START,
            END,
            horizon=5,
            price="open",
            quantiles=10,
        )
        candidate_context = build_panel(params, factor=materialized.factor_view)
        formal_context = build_panel(
            EvalParams("LN_MV", START, END, horizon=5, price="open", quantiles=10)
        )
        candidate_panel = candidate_context.panel
        formal_panel = formal_context.panel
        aligned = candidate_panel[
            ["stock_code", "trade_date", "factor", "in_universe", "label"]
        ].merge(
            formal_panel[["stock_code", "trade_date", "factor", "label"]],
            on=["stock_code", "trade_date"],
            suffixes=("_candidate", "_formal"),
            validate="one_to_one",
        )
        assert len(aligned) == len(candidate_panel) == len(formal_panel)
        assert np.allclose(
            aligned["factor_candidate"], aligned["factor_formal"], equal_nan=True
        )
        assert np.allclose(
            aligned["label_candidate"], aligned["label_formal"], equal_nan=True
        )

        options = {"correlation": {"top_n": 100}}
        candidate_events = dict(
            evaluate_context(candidate_context, metric_options=options)
        )
        formal_events = dict(evaluate_context(formal_context, metric_options=options))
        for stage in (
            "overview",
            "ic_series",
            "ic_dist",
            "bins",
            "bin_curves",
            "spread",
            "bin_dist",
            "turnover",
            "decay",
            "exposure",
            "incremental_ic",
        ):
            _assert_nested_equal(
                _normalize(candidate_events[stage]),
                _normalize(formal_events[stage]),
                stage,
            )
        _assert_nested_equal(
            _comparable_correlation(candidate_events["correlation"], "LN_MV"),
            _comparable_correlation(formal_events["correlation"], "LN_MV"),
            "correlation",
        )

        evaluation = evaluate_candidate(
            CandidateEvaluationInput(
                run_id=request.run_id,
                candidate_id=candidate.candidate_id,
                candidate=candidate,
                context=context,
                candidate_root=Path(root),
                run_full=False,
                quick_window=(START, END),
            )
        )
        assert not evaluation.gate.passed
        assert evaluation.technical.status == "failed"
        assert evaluation.quick is None
        assert evaluation.full is None
        assert (
            evaluation.technical.metrics["existing_correlation"]["peer"]
            == "LN_MV"
        )
        assert (
            evaluation.technical.metrics["existing_correlation"]["max_absolute"]
            >= 0.90
        )


def test_candidate_quick_evaluation_creates_only_quick_report():
    """通过质量门禁的候选只生成快速筛选报告，不自动进入完整评估。"""

    candidate = validate_expression("mean(close, 5)", direction=1).to_candidate(
        mechanism="短期价格均值",
        description="五日收盘价均值",
        steps=("计算五日收盘价均值",),
        discovery_round=1,
        discovery_index=1,
    )
    request = MiningRunRequest("evaluation-quick-001", "测试候选快速筛选报告")
    context = FrozenContext(data_fingerprint="c" * 64)
    with TemporaryDirectory(prefix="mining-quick-") as root:
        registered_before = tuple(item.name for item in factors.all_factors())
        evaluation = evaluate_candidate(
            CandidateEvaluationInput(
                run_id=request.run_id,
                candidate_id=candidate.candidate_id,
                candidate=candidate,
                context=context,
                candidate_root=Path(root),
                run_full=False,
                quick_window=(START, END),
            )
        )
        assert evaluation.gate.passed
        assert evaluation.technical.status == "passed"
        assert evaluation.quick is not None
        assert evaluation.quick.status == "passed"
        assert evaluation.quick.metrics["rows"] > 0
        assert evaluation.quick.metrics["coverage"] == pytest.approx(
            evaluation.technical.metrics["values"]["coverage"]
        )
        assert evaluation.quick.metrics["elapsed_ms"] >= 0
        assert evaluation.technical.metrics["elapsed_ms"] >= 0
        assert evaluation.full is None
        assert Path(root, "evaluations", "quick.json").is_file()
        assert not Path(root, "evaluations", "full.json").exists()
        assert tuple(item.name for item in factors.all_factors()) == registered_before


def test_candidate_full_evaluation_runs_every_existing_stage():
    """通过快速筛选后，完整评估复用现有全部 stage 并保存独立报告。"""

    candidate = validate_expression("mean(close, 5)", direction=1).to_candidate(
        mechanism="短期价格均值",
        description="五日收盘价均值",
        steps=("计算五日收盘价均值",),
        discovery_round=1,
        discovery_index=1,
    )
    request = MiningRunRequest("evaluation-full-001", "测试候选完整评估流程")
    context = FrozenContext(data_fingerprint="d" * 64)
    with TemporaryDirectory(prefix="mining-full-") as root:
        evaluation = evaluate_candidate(
            CandidateEvaluationInput(
                run_id=request.run_id,
                candidate_id=candidate.candidate_id,
                candidate=candidate,
                context=context,
                candidate_root=Path(root),
                run_full=True,
                quick_window=(START, END),
                full_window=(START, END),
            )
        )
        assert evaluation.gate.passed
        assert evaluation.quick is not None and evaluation.quick.status == "passed"
        assert evaluation.full is not None and evaluation.full.status == "passed"
        assert evaluation.full.metrics["elapsed_ms"] >= 0
        stages = set(evaluation.full.metrics["events"])
        assert {
            "overview",
            "ic_series",
            "ic_dist",
            "bins",
            "bin_curves",
            "spread",
            "bin_dist",
            "turnover",
            "decay",
            "exposure",
            "incremental_ic",
            "correlation",
            "done",
        } <= stages
        assert Path(root, "evaluations", "full.json").is_file()
