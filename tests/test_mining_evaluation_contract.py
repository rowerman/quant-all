"""S5 候选评估输入、配置和路径隔离测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from common.paths import FACTOR_DIR, MINING_EVALUATION_FILE
from mining import (
    CandidateEvaluationInput,
    FrozenContext,
    MiningRunRequest,
    QualityPolicy,
    load_quality_policy,
    validate_expression,
)
from mining.candidate_materialization import materialize_candidate
from mining.errors import CandidateError, MiningContractError, MiningEvaluationError


def _candidate():
    """构造一个由 S3 产生的合法候选。"""

    return validate_expression("mean(close, 5)", direction=1).to_candidate(
        mechanism="短期价格趋势",
        description="五日收盘价均值",
        steps=("计算五日收盘价均值",),
        discovery_round=1,
        discovery_index=1,
    )


def test_quality_policy_is_versioned_and_loads_from_default_config():
    """质量阈值从集中配置读取，版本和首版硬门禁固定。"""

    policy = load_quality_policy()
    assert MINING_EVALUATION_FILE.is_file()
    assert isinstance(policy, QualityPolicy)
    assert policy.schema_version == "mining_quality_v1"
    assert policy.forward_look_samples == 3
    assert policy.max_flat_day_share == pytest.approx(0.01)
    assert policy.correlation_max_absolute == pytest.approx(0.90)


def test_evaluation_input_uses_frozen_windows_and_rejects_drift(tmp_path: Path):
    """评估窗口默认取上下文，且不能越出研究期或完整评估期。"""

    candidate = _candidate()
    request = MiningRunRequest("evaluation-input-001", "测试候选评估输入契约")
    base = {
        "run_id": request.run_id,
        "candidate_id": candidate.candidate_id,
        "candidate": candidate,
        "context": FrozenContext(data_fingerprint="a" * 64),
        "candidate_root": tmp_path,
    }
    evaluation = CandidateEvaluationInput(**base)
    assert evaluation.quick_window == ("2012-01-01", "2025-10-15")
    assert evaluation.full_window == ("2020-01-01", "2025-10-15")

    with pytest.raises(MiningContractError, match="研究期"):
        CandidateEvaluationInput(
            **base,
            quick_window=("2011-01-01", "2024-01-01"),
        )
    with pytest.raises(MiningContractError, match="candidate_id"):
        CandidateEvaluationInput(**{**base, "candidate_id": "cand_wrong"})


def test_candidate_materialization_rejects_formal_factor_directory():
    """候选根目录不能位于正式因子目录内。"""

    candidate = _candidate()
    with pytest.raises(MiningEvaluationError, match="正式因子目录"):
        materialize_candidate(
            candidate,
            FrozenContext(data_fingerprint="a" * 64),
            FACTOR_DIR / "forbidden-candidate",
            start="2024-01-02",
            end="2024-01-03",
        )


def test_candidate_materialization_rejects_semantic_drift(tmp_path: Path):
    """候选名与表达式重新校验结果不一致时，在执行 SQL 前失败。"""

    original = _candidate()
    drifted = type(original)(
        name=original.name,
        mechanism=original.mechanism,
        description=original.description,
        canonical_expression="mean(close, 10)",
        direction=original.direction,
        lookback=original.lookback,
        dependencies=original.dependencies,
        output_type=original.output_type,
        steps=original.steps,
        discovery_round=original.discovery_round,
        discovery_index=original.discovery_index,
    )
    with pytest.raises(CandidateError, match="语义哈希"):
        materialize_candidate(
            drifted,
            FrozenContext(data_fingerprint="a" * 64),
            tmp_path,
            start="2024-01-02",
            end="2024-01-03",
        )


def test_materialization_failure_removes_staging(tmp_path: Path, monkeypatch):
    """SQL 失败时不留下 staging 或半完成年分区。"""

    module = __import__(
        "mining.candidate_materialization",
        fromlist=["family_query"],
    )

    def fail_query(*args, **kwargs):
        raise RuntimeError("synthetic build failure")

    class FakeConnection:
        def close(self):
            return None

    monkeypatch.setattr(module, "open_source", lambda: FakeConnection())
    monkeypatch.setattr(module, "family_query", fail_query)
    with pytest.raises(RuntimeError, match="synthetic"):
        materialize_candidate(
            _candidate(),
            FrozenContext(data_fingerprint="a" * 64),
            tmp_path,
            start="2024-01-02",
            end="2024-01-03",
        )
    assert not (tmp_path / "factor").exists()
    assert not tuple(tmp_path.glob(".factor.staging-*"))
