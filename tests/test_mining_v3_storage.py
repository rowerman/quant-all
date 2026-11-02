"""V3 统一输入、候选来源和证据存储测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import duckdb
import pytest

from mining.contract_candidates import CandidateDefinition
from mining.contract_types import sha256_json
from mining.errors import (
    MiningContractError,
    MiningDuplicateError,
    MiningMigrationError,
    MiningStorageError,
)
from mining.search_run import SearchRunRequest
from mining.storage_v2 import MiningV2Store
from mining.storage_v3 import MiningV3Store
from mining.v3_contracts import (
    V3_REQUIRED_EVIDENCE_STAGES,
    CandidateEnvelope,
    EvidenceReference,
    ResearchInputEnvelope,
    ResearchSourceReference,
)
from mining.v3_sources import hypothesis_input_envelope


CREATED_AT = "2026-11-02T00:00:00+00:00"


def _store(tmp_path: Path) -> MiningV3Store:
    return MiningV3Store(
        tmp_path / "factor_mining.duckdb",
        tmp_path / "data",
    )


def _request(key: str = "v3-storage-0001") -> SearchRunRequest:
    return SearchRunRequest(
        key,
        "价格趋势可能带来未来超额收益",
    )


def _input(
    request: SearchRunRequest,
    *,
    context_hash: str = "a" * 64,
):
    return hypothesis_input_envelope(request, context_hash=context_hash)


def _candidate() -> CandidateDefinition:
    return CandidateDefinition(
        name="V3_STORAGE_FACTOR",
        mechanism="价格趋势",
        description="验证 V3 来源绑定",
        canonical_expression="rank(mean(close, 20))",
        direction=1,
        lookback=20,
        dependencies=("close",),
        output_type="score",
        steps=("计算二十日价格均值", "按交易日做截面排名"),
        discovery_round=1,
        discovery_index=1,
    )


def _evidence(candidate_id: str) -> tuple[EvidenceReference, ...]:
    return tuple(
        EvidenceReference(
            candidate_id=candidate_id,
            stage=stage,
            evaluation_id=f"eval_{stage}",
            input_hash=f"{index + 1:x}" * 64,
            evidence_hash=f"{index + 5:x}" * 64,
            status="passed",
        )
        for index, stage in enumerate(V3_REQUIRED_EVIDENCE_STAGES)
    )


def test_v3_store_initializes_only_on_fresh_database(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    store = _store(tmp_path)

    with duckdb.connect(str(path), read_only=True) as conn:
        schemas = {
            row[0]
            for row in conn.execute(
                "SELECT schema_name FROM duckdb_schemas()"
            ).fetchall()
        }
        version = conn.execute(
            "SELECT value FROM mining_v3.schema_meta "
            "WHERE key='schema_version'"
        ).fetchone()[0]
        assert "mining_v3" in schemas
        assert "mining_v2" not in schemas
        assert version == "mining_v3"
        assert conn.execute(
            "SELECT count(*) FROM mining_v3.research_input"
        ).fetchone()[0] == 0
    assert store.data_root.is_dir()


def test_v3_store_rejects_existing_v2_database(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    MiningV2Store(path, tmp_path / "data")

    with pytest.raises(MiningMigrationError, match="显式 V3 迁移"):
        MiningV3Store(path, tmp_path / "data")

    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute(
            "SELECT count(*) FROM duckdb_schemas() "
            "WHERE schema_name='mining_v3'"
        ).fetchone()[0] == 0


def test_input_registration_is_idempotent_and_detects_changed_payload(
    tmp_path: Path,
):
    store = _store(tmp_path)
    request = _request()
    envelope = _input(request)

    first = store.register_input(envelope)
    second = store.register_input(envelope)

    assert first["reused"] is False
    assert second["reused"] is True
    assert store.load_input(envelope.run_id) == envelope
    assert (
        store.get_source_reference(
            envelope.run_id,
            envelope.source.source_id,
        )
        == envelope.source
    )

    changed = _input(request, context_hash="b" * 64)
    with pytest.raises(MiningDuplicateError, match="不同输入"):
        store.register_input(changed)


def test_candidate_source_allows_multiple_sources_and_append_only_evidence(
    tmp_path: Path,
):
    store = _store(tmp_path)
    request = _request()
    input_envelope = _input(request)
    store.register_input(input_envelope)
    candidate = _candidate()
    first_source = ResearchSourceReference(
        source_kind="hypothesis",
        source_id="source_first",
        source_hash="1" * 64,
    )
    second_source = ResearchSourceReference(
        source_kind="hypothesis",
        source_id="source_second",
        source_hash="2" * 64,
    )
    initial = CandidateEnvelope(
        candidate=candidate,
        source=first_source,
        evidence=(_evidence(candidate.candidate_id)[0],),
    )

    first = store.record_candidate_envelope(
        input_envelope.run_id,
        initial,
    )
    reused = store.record_candidate_envelope(
        input_envelope.run_id,
        initial,
    )
    completed = CandidateEnvelope(
        candidate=candidate,
        source=first_source,
        evidence=_evidence(candidate.candidate_id),
    )
    appended = store.record_candidate_envelope(
        input_envelope.run_id,
        completed,
    )
    second = CandidateEnvelope(
        candidate=candidate,
        source=second_source,
        evidence=_evidence(candidate.candidate_id),
    )
    store.record_candidate_envelope(input_envelope.run_id, second)

    assert first["reused"] is False
    assert reused["reused"] is True
    assert appended["reused"] is False
    loaded = store.get_candidate_envelope(
        input_envelope.run_id,
        candidate.candidate_id,
        first_source.source_id,
    )
    assert loaded == completed
    envelopes = store.list_candidate_envelopes(input_envelope.run_id)
    assert len(envelopes) == 2
    assert {item.source.source_id for item in envelopes} == {
        "source_first",
        "source_second",
    }
    assert completed.approval_evidence_hash("a" * 64)


def test_candidate_source_rejects_mode_mismatch_and_evidence_conflict(
    tmp_path: Path,
):
    store = _store(tmp_path)
    request = _request()
    input_envelope = _input(request)
    store.register_input(input_envelope)
    candidate = _candidate()
    mismatched_source = ResearchSourceReference(
        source_kind="research_report",
        source_id="source_report",
        source_hash="3" * 64,
    )
    with pytest.raises(MiningContractError, match="输入模式"):
        store.record_candidate_envelope(
            input_envelope.run_id,
            CandidateEnvelope(
                candidate=candidate,
                source=mismatched_source,
            ),
        )

    source = ResearchSourceReference(
        source_kind="hypothesis",
        source_id="source_hypothesis",
        source_hash="4" * 64,
    )
    original = CandidateEnvelope(
        candidate=candidate,
        source=source,
        evidence=(_evidence(candidate.candidate_id)[0],),
    )
    store.record_candidate_envelope(input_envelope.run_id, original)
    changed_evidence = replace(
        original.evidence[0],
        evidence_hash="f" * 64,
    )
    with pytest.raises(MiningDuplicateError, match="不同证据"):
        store.record_candidate_envelope(
            input_envelope.run_id,
            CandidateEnvelope(
                candidate=candidate,
                source=source,
                evidence=(changed_evidence,),
            ),
        )


def test_v3_store_reopens_and_rejects_unknown_schema(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    store = _store(tmp_path)
    envelope = _input(_request())
    store.register_input(envelope)

    reopened = MiningV3Store(path, tmp_path / "data")
    assert reopened.load_input(envelope.run_id) == envelope

    with duckdb.connect(str(path)) as conn:
        conn.execute(
            "UPDATE mining_v3.schema_meta SET value='mining_v4' "
            "WHERE key='schema_version'"
        )
    with pytest.raises(MiningMigrationError, match="schema"):
        MiningV3Store(path, tmp_path / "data")


def test_v3_read_rejects_tampered_input_column(tmp_path: Path):
    path = tmp_path / "factor_mining.duckdb"
    store = _store(tmp_path)
    envelope = _input(_request())
    store.register_input(envelope)

    with duckdb.connect(str(path)) as conn:
        conn.execute(
            "UPDATE mining_v3.research_input SET request_hash=? "
            "WHERE run_id=?",
            ["f" * 64, envelope.run_id],
        )

    with pytest.raises(MiningStorageError, match="不一致"):
        store.load_input(envelope.run_id)


@pytest.mark.parametrize(
    "source_kind",
    ("hypothesis", "free_exploration", "research_report", "costeer"),
)
def test_store_accepts_every_v3_source_with_one_candidate_contract(
    tmp_path: Path,
    source_kind: str,
):
    store = _store(tmp_path)
    source = ResearchSourceReference(
        source_kind=source_kind,
        source_id=f"source_{source_kind}",
        source_hash=sha256_json({"source_kind": source_kind}),
        artifact_manifest_hash=("1" * 64 if source_kind == "costeer" else None),
        promotion_hash=("2" * 64 if source_kind == "costeer" else None),
    )
    input_envelope = ResearchInputEnvelope(
        run_id=f"researchrun_{source_kind[:8]:0<24}",
        idempotency_key=f"v3-source-{source_kind}",
        input_mode=source_kind,
        context_hash="a" * 64,
        request_hash=sha256_json({"request": source_kind}),
        source=source,
    )
    candidate = _candidate()
    envelope = CandidateEnvelope(
        candidate=candidate,
        source=source,
        evidence=_evidence(candidate.candidate_id),
    )
    store.register_input(input_envelope)
    store.record_candidate_envelope(input_envelope.run_id, envelope)

    loaded = store.get_candidate_envelope(
        input_envelope.run_id,
        candidate.candidate_id,
        source.source_id,
    )
    assert loaded.candidate.candidate_id == candidate.candidate_id
    assert loaded.source.source_kind == source_kind
    assert loaded == envelope
