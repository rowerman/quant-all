"""V2/V3 候选详情投影测试。"""

from __future__ import annotations

from mining.search_evaluation_evidence import SearchEvaluationEvidence

from backend.mining_candidate_detail import (
    evaluation_view,
    technical_gate_view,
)


def _row(
    *,
    evaluation_id: str,
    stage: str,
    status: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    evidence = SearchEvaluationEvidence(
        run_id="researchrun_111111111111111111111111",
        candidate_id="cand_" + "1" * 24,
        stage=stage,
        input_hash="a" * 64,
        status=status,
        metrics=metrics,
        created_at="2026-11-02T00:00:00+00:00",
    )
    return {
        "evaluation_id": evaluation_id,
        "evidence": evidence.to_dict(),
    }


def test_evaluation_and_technical_gate_use_latest_technical_evidence():
    first = evaluation_view(
        _row(
            evaluation_id="eval_first",
            stage="technical",
            status="passed",
            metrics={"checks": ["first"]},
        )
    )
    latest = evaluation_view(
        _row(
            evaluation_id="eval_latest",
            stage="technical",
            status="failed",
            metrics={"checks": ["latest"]},
        )
    )

    assert first["evaluation_id"] == "eval_first"
    assert technical_gate_view([first, latest]) == {
        "passed": False,
        "rule_version": "mining_search_approval_v1",
        "checks": {"checks": ["latest"]},
    }
