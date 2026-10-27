"""候选批量物化的产物一致性、隔离和回退测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from calculate.config import DB_FILE
from mining.candidate_batch import (
    CandidateBatchItem,
    materialize_candidate_batch,
)
from mining.candidate_materialization import MaterializedCandidate, materialize_candidate
from mining.catalog import load_catalog
from mining.contract_requests import FrozenContext, MiningConstraints
from mining.errors import BatchMaterializationError
from mining.expression_validation import validate_expression

pytestmark = pytest.mark.skipif(
    not DB_FILE.exists(),
    reason="需要真实主库 data/databases/quant.duckdb",
)

START, END = "2024-05-06", "2024-06-28"


def _context() -> FrozenContext:
    catalog = load_catalog()
    return FrozenContext(
        data_fingerprint="a" * 64,
        field_catalog=catalog.frozen_field_mappings(),
        operator_catalog=catalog.frozen_operator_mappings(),
        budget=MiningConstraints(
            max_rounds=1,
            max_candidates=4,
            full_eval_top_k=1,
            max_llm_calls=1,
            max_failure_rate=1.0,
            min_quality_checks_for_failure_stop=1,
        ),
    )


def _candidates():
    expressions = (
        "mean(close, 5) + rank(open)",
        "mean(close, 5) - open",
        "rank(mean(close, 5))",
        "mean(close, 5) + close",
    )
    return tuple(
        validate_expression(expression, direction=1).to_candidate(
            mechanism="批量物化测试",
            description=f"测试候选 {index}",
            steps=("执行受控表达式",),
            discovery_round=1,
            discovery_index=index,
        )
        for index, expression in enumerate(expressions, start=1)
    )


def _factor_root(materialized: MaterializedCandidate) -> Path:
    return Path(materialized.factor_view.parquet_glob).parent.parent


def _tree_bytes(materialized: MaterializedCandidate) -> dict[str, bytes]:
    root = _factor_root(materialized)
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in materialized.files
    }


def test_batch_materialization_matches_single_and_is_idempotent(tmp_path: Path):
    """批量拆分与串行物化的文件字节和 artifact hash 完全一致。"""

    context = _context()
    candidates = _candidates()
    serial = []
    for candidate in candidates:
        serial.append(
            materialize_candidate(
                candidate,
                context,
                tmp_path / "serial" / candidate.candidate_id,
                start=START,
                end=END,
            )
        )
    items = [
        CandidateBatchItem(candidate, tmp_path / "batch" / candidate.candidate_id)
        for candidate in candidates
    ]
    result = materialize_candidate_batch(
        items,
        context,
        start=START,
        end=END,
    )

    assert result.failures == ()
    assert result.diagnostics.shared_node_count >= 1
    assert result.diagnostics.analysis_sql_count == 1
    assert result.diagnostics.daily_source_reference_count == 2
    assert result.diagnostics.written_count == 4
    assert result.diagnostics.fallback_reason is None
    for serialized, batched in zip(serial, result.materialized, strict=True):
        assert serialized.artifact_sha256 == batched.artifact_sha256
        assert _tree_bytes(serialized) == _tree_bytes(batched)
    assert not tuple(tmp_path.rglob(".batch-materialization.staging-*"))

    repeated = materialize_candidate_batch(
        items,
        context,
        start=START,
        end=END,
    )
    assert repeated.failures == ()
    assert repeated.diagnostics.reused_count == 4
    assert repeated.diagnostics.written_count == 0
    assert repeated.diagnostics.fallback_reason == "insufficient_pending_candidates"


def test_candidate_validation_failure_does_not_stop_batch(tmp_path: Path):
    """一个候选身份漂移只让该候选失败。"""

    context = _context()
    first, second, bad = _candidates()[:3]
    bad = replace(bad, lookback=bad.lookback + 1)
    items = [
        CandidateBatchItem(first, tmp_path / first.candidate_id),
        CandidateBatchItem(second, tmp_path / second.candidate_id),
        CandidateBatchItem(bad, tmp_path / bad.candidate_id),
    ]

    result = materialize_candidate_batch(
        items,
        context,
        start=START,
        end=END,
    )

    assert len(result.materialized) == 2
    assert [failure.candidate_id for failure in result.failures] == [bad.candidate_id]
    assert not (tmp_path / bad.candidate_id / "factor").exists()


def test_batch_query_failure_falls_back_to_single_candidates(
    tmp_path: Path,
    monkeypatch,
):
    """共享查询失败时，同批健康候选仍可通过单候选路径物化。"""

    from mining import candidate_batch

    context = _context()
    candidates = _candidates()[:2]
    items = [
        CandidateBatchItem(candidate, tmp_path / candidate.candidate_id)
        for candidate in candidates
    ]

    def fail_batch(*args, **kwargs):
        raise RuntimeError("synthetic batch failure")

    monkeypatch.setattr(
        candidate_batch,
        "_copy_combined_batch_query",
        fail_batch,
    )
    result = materialize_candidate_batch(
        items,
        context,
        start=START,
        end=END,
    )

    assert result.failures == ()
    assert len(result.materialized) == 2
    assert result.diagnostics.fallback_reason == (
        "batch_query_failed:RuntimeError"
    )
    assert result.diagnostics.analysis_sql_count == 2
    assert all((tmp_path / item.candidate.candidate_id / "factor").is_dir() for item in items)


def test_shared_failure_and_all_fallbacks_raise_batch_error(
    tmp_path: Path,
    monkeypatch,
):
    """共享计算和全部逐候选回退都失败时返回明确批级错误。"""

    from mining import candidate_batch

    context = _context()
    candidates = _candidates()[:2]
    items = [
        CandidateBatchItem(candidate, tmp_path / candidate.candidate_id)
        for candidate in candidates
    ]

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(
        candidate_batch,
        "_copy_combined_batch_query",
        fail,
    )
    monkeypatch.setattr(candidate_batch, "materialize_candidate", fail)

    with pytest.raises(BatchMaterializationError, match="逐候选回退"):
        materialize_candidate_batch(
            items,
            context,
            start=START,
            end=END,
        )

    assert not tuple(tmp_path.rglob("factor"))
    assert not tuple(tmp_path.rglob(".batch-materialization.staging-*"))


def test_no_shared_computation_uses_single_path(tmp_path: Path):
    """只有重复字段而没有共享计算时，不生成批量宽查询。"""

    context = _context()
    candidates = (
        validate_expression("mean(close, 5)", direction=1).to_candidate(
            mechanism="无共享计算",
            description="五日均值",
            steps=("计算五日均值",),
            discovery_round=1,
            discovery_index=1,
        ),
        validate_expression("mean(close, 10)", direction=1).to_candidate(
            mechanism="无共享计算",
            description="十日均值",
            steps=("计算十日均值",),
            discovery_round=1,
            discovery_index=2,
        ),
    )
    items = [
        CandidateBatchItem(candidate, tmp_path / candidate.candidate_id)
        for candidate in candidates
    ]

    result = materialize_candidate_batch(
        items,
        context,
        start=START,
        end=END,
    )

    assert len(result.materialized) == 2
    assert result.diagnostics.fallback_reason == "no_shared_nodes"
    assert result.diagnostics.analysis_sql_count == 2


def test_batch_materialization_joins_extra_sources_once(tmp_path: Path):
    """财务附加源只加入一次，批量与串行产物保持一致。"""

    context = _context()
    candidates = (
        validate_expression("mean(fi_roe, 5) + 1", direction=1).to_candidate(
            mechanism="财务时点测试",
            description="ROE 均值加一",
            steps=("计算 ROE 均值", "加一"),
            discovery_round=1,
            discovery_index=1,
        ),
        validate_expression("mean(fi_roe, 5) - 1", direction=1).to_candidate(
            mechanism="财务时点测试",
            description="ROE 均值减一",
            steps=("计算 ROE 均值", "减一"),
            discovery_round=1,
            discovery_index=2,
        ),
    )
    serial = [
        materialize_candidate(
            candidate,
            context,
            tmp_path / "serial" / candidate.candidate_id,
            start=START,
            end=END,
        )
        for candidate in candidates
    ]
    result = materialize_candidate_batch(
        [
            CandidateBatchItem(
                candidate,
                tmp_path / "batch" / candidate.candidate_id,
            )
            for candidate in candidates
        ],
        context,
        start=START,
        end=END,
    )

    assert result.failures == ()
    assert {
        item.artifact_sha256 for item in serial
    } == {
        item.artifact_sha256 for item in result.materialized
    }
