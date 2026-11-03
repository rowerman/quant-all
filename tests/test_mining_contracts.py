"""挖掘领域契约、稳定哈希和错误语义测试。"""

from __future__ import annotations

import pytest

from mining.contracts import (
    ApprovalDecision,
    CandidateDefinition,
    FrozenContext,
    MiningConstraints,
    MiningRunRequest,
    canonical_json,
    sha256_json,
    validate_utc_timestamp,
)
from mining.errors import ApprovalConflictError, CandidateError, MiningContractError


def _context() -> FrozenContext:
    """构造符合首版冻结口径的上下文。"""

    return FrozenContext(data_fingerprint="a" * 64)


def _candidate(**changes) -> CandidateDefinition:
    """构造最小合法候选，并按需要覆盖字段。"""

    payload = {
        "name": "MINE_REVERSAL_00000001",
        "mechanism": "short reversal",
        "description": "五日价格变化取负",
        "canonical_expression": "neg(sub(close,lag(close,5)))",
        "direction": 1,
        "lookback": 5,
        "dependencies": ("close",),
        "output_type": "number",
        "steps": ("计算五日变化", "取负"),
        "discovery_round": 1,
        "discovery_index": 1,
    }
    payload.update(changes)
    return CandidateDefinition(**payload)


def test_canonical_json_is_key_order_independent_and_rejects_non_finite():
    """稳定 JSON 不受字典顺序影响，且不会静默写入 NaN。"""

    assert canonical_json({"b": 2, "a": {"d": 4, "c": 3}}) == canonical_json(
        {"a": {"c": 3, "d": 4}, "b": 2}
    )
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})
    with pytest.raises(MiningContractError, match="NaN"):
        canonical_json({"value": float("nan")})


def test_timestamps_require_timezone_and_normalize_to_utc():
    """持久化时间不接受缺少时区的本机时间。"""

    assert validate_utc_timestamp("2026-10-24T12:00:00+08:00", "time") == (
        "2026-10-24T04:00:00.000000+00:00"
    )
    with pytest.raises(MiningContractError, match="时区"):
        validate_utc_timestamp("2026-10-24T12:00:00", "time")


def test_request_validates_hypothesis_profile_and_generates_stable_run_id():
    """请求字段受限，幂等键相同则运行 ID 稳定。"""

    first = MiningRunRequest("client-run-0001", "短期价格反转假设")
    second = MiningRunRequest("client-run-0001", "短期价格反转假设")
    assert first.run_id == second.run_id
    assert first.request_hash == second.request_hash

    with pytest.raises(MiningContractError, match="hypothesis"):
        MiningRunRequest("client-run-0001", "太短")
    with pytest.raises(MiningContractError, match="llm_profile"):
        MiningRunRequest("client-run-0001", "短期价格反转假设", llm_profile="../bad")


@pytest.mark.parametrize(
    "changes",
    [
        {"max_rounds": 0},
        {"max_candidates": 2, "full_eval_top_k": 3},
        {"min_coverage": 1.1},
        {"direction": "up"},
    ],
)
def test_constraints_reject_invalid_budget(changes):
    """资源约束在进入存储前失败。"""

    with pytest.raises(MiningContractError):
        MiningConstraints(**changes)


def test_frozen_context_hash_covers_catalog_and_budget():
    """字段目录和预算变化都会改变冻结上下文。"""

    base = _context()
    changed_catalog = FrozenContext(
        data_fingerprint="a" * 64,
        field_catalog=({"name": "close", "available_at": "close"},),
    )
    changed_budget = FrozenContext(
        data_fingerprint="a" * 64,
        budget=MiningConstraints(max_rounds=4),
    )
    assert base.context_id.startswith("ctx_")
    assert len({base.context_hash, changed_catalog.context_hash, changed_budget.context_hash}) == 3


def test_candidate_hashes_separate_semantics_from_review_text():
    """说明文字变化只改变内容哈希，计算语义变化同时改变两种哈希。"""

    first = _candidate()
    described = _candidate(description="更清楚的说明")
    changed_formula = _candidate(canonical_expression="neg(close)")

    assert first.semantic_hash == described.semantic_hash
    assert first.content_hash != described.content_hash
    assert first.candidate_id != described.candidate_id
    assert first.semantic_hash != changed_formula.semantic_hash
    assert first.content_hash != changed_formula.content_hash


def test_candidate_normalizes_dependency_order():
    """同一依赖集合换顺序不改变两种候选哈希。"""

    first = _candidate(dependencies=("close", "amount"))
    second = _candidate(dependencies=("amount", "close"))
    assert first.semantic_hash == second.semantic_hash
    assert first.content_hash == second.content_hash


def test_candidate_rejects_invalid_name_and_missing_dependency():
    """候选名和依赖字段在落库前校验。"""

    with pytest.raises(CandidateError, match="候选名"):
        _candidate(name="mine lower")
    with pytest.raises(CandidateError, match="至少依赖"):
        _candidate(dependencies=())


def test_approval_requires_reviewer_and_note():
    """人工决定必须包含审核人和备注。"""

    with pytest.raises(MiningContractError, match="reviewer"):
        ApprovalDecision(
            run_id="run_a",
            candidate_id="cand_a",
            decision="approved",
            reviewer=" ",
            note="通过",
            candidate_content_hash="a" * 64,
            semantic_hash="b" * 64,
            context_hash="c" * 64,
            evidence_hash="d" * 64,
        )
    with pytest.raises(ApprovalConflictError, match="approved"):
        ApprovalDecision(
            run_id="run_a",
            candidate_id="cand_a",
            decision="maybe",
            reviewer="local",
            note="通过",
            candidate_content_hash="a" * 64,
            semantic_hash="b" * 64,
            context_hash="c" * 64,
            evidence_hash="d" * 64,
        )
