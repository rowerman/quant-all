"""资料驱动输入、指纹、来源和安全边界契约测试。"""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from mining import (
    RESEARCH_MATERIAL_SCHEMA_VERSION,
    ResearchDocumentRecord,
    ResearchMaterialContractError,
    ResearchMaterialFailure,
    ResearchMaterialInput,
    ResearchMaterialManifest,
    ResearchMaterialReadError,
    ResearchMaterialSecurityError,
    SourceReference,
    build_research_material_manifest,
    fingerprint_research_file,
    redact_research_material_text,
    safe_research_material_log_fields,
)

PARSER_VERSION = "pypdf-6.21.2.page-v1"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _input(**changes) -> ResearchMaterialInput:
    payload = {
        "relative_path": "reports/momentum.pdf",
        "media_type": "application/pdf",
        "size_bytes": 40,
        "sha256": _sha(b"research report content"),
        "parser_version": PARSER_VERSION,
    }
    payload.update(changes)
    return ResearchMaterialInput(**payload)


def _source(**changes) -> SourceReference:
    payload = {
        "document_id": _input().document_id,
        "page_number": 3,
        "excerpt_sha256": _sha(b"quoted report text"),
        "locator": "paragraph:2",
        "created_at": "2026-11-01T08:00:00+08:00",
    }
    payload.update(changes)
    return SourceReference(**payload)


def _record(**changes) -> ResearchDocumentRecord:
    material = _input()
    payload = {
        "document_id": material.document_id,
        "material_hash": material.material_hash,
        "parser_version": material.parser_version,
        "status": "parsed",
        "page_count": 12,
        "text_sha256": _sha(b"full document text"),
        "source_references": (_source(),),
        "created_at": "2026-11-01T08:05:00+08:00",
    }
    payload.update(changes)
    return ResearchDocumentRecord(**payload)


def test_material_identity_uses_content_and_manifest_is_order_independent():
    first = _input()
    moved = _input(relative_path="archive/momentum.pdf")
    assert first.document_id.startswith("doc_")
    assert first.document_id == moved.document_id
    assert first.material_hash != moved.material_hash

    other = _input(
        relative_path="reports/value.pdf",
        sha256=_sha(b"another report"),
    )
    left = ResearchMaterialManifest((first, other))
    right = ResearchMaterialManifest((other, first))
    assert left.manifest_hash == right.manifest_hash
    assert left.manifest_id == right.manifest_id
    assert left.to_dict() != right.to_dict()


def test_contract_round_trip_and_time_fields_excluded_from_stable_hashes():
    material = _input()
    assert ResearchMaterialInput.from_dict(material.to_dict()) == material

    failure = ResearchMaterialFailure(
        category="unsupported_format",
        message="仅支持文本型 PDF",
        retryable=False,
        occurred_at="2026-11-01T08:00:00+08:00",
    )
    changed_failure = replace(
        failure,
        occurred_at="2026-11-02T08:00:00+08:00",
    )
    assert failure.failure_hash == changed_failure.failure_hash
    assert ResearchMaterialFailure.from_dict(failure.to_dict()) == failure

    reference = _source()
    changed_reference = replace(
        reference,
        created_at="2026-11-02T08:00:00+08:00",
    )
    assert reference.source_reference_id == changed_reference.source_reference_id
    assert SourceReference.from_dict(reference.to_dict()) == reference

    record = _record()
    changed_record = replace(
        record,
        created_at="2026-11-02T08:00:00+08:00",
    )
    assert record.record_hash == changed_record.record_hash
    assert ResearchDocumentRecord.from_dict(record.to_dict()) == record


@pytest.mark.parametrize(
    ("contract", "payload", "name"),
    [
        (
            ResearchMaterialInput,
            {
                "relative_path": "reports/a.pdf",
                "media_type": "application/pdf",
                "size_bytes": 1,
                "sha256": "a" * 64,
                "parser_version": PARSER_VERSION,
                "schema_version": RESEARCH_MATERIAL_SCHEMA_VERSION,
            },
            "research_material_input",
        ),
        (
            ResearchMaterialManifest,
            {"materials": [_input().to_dict()], "schema_version": RESEARCH_MATERIAL_SCHEMA_VERSION},
            "research_material_manifest",
        ),
        (
            ResearchMaterialFailure,
            {
                "category": "parser_error",
                "message": "解析器内部错误",
                "retryable": True,
                "error_code": "MINING-151",
                "occurred_at": "2026-11-01T00:00:00+00:00",
                "schema_version": RESEARCH_MATERIAL_SCHEMA_VERSION,
            },
            "research_material_failure",
        ),
        (
            SourceReference,
            _source().to_dict(),
            "source_reference",
        ),
        (
            ResearchDocumentRecord,
            _record().to_dict(),
            "research_document_record",
        ),
    ],
)
def test_contracts_reject_missing_and_unknown_fields(contract, payload, name):
    missing = dict(payload)
    missing.pop("schema_version")
    with pytest.raises(ResearchMaterialContractError, match="缺少"):
        contract.from_dict(missing)
    unknown = {**payload, "unexpected": True}
    with pytest.raises(ResearchMaterialContractError, match="未知"):
        contract.from_dict(unknown)


def test_manifest_rejects_duplicate_content():
    second_path = _input(relative_path="reports/momentum-copy.pdf")
    with pytest.raises(ResearchMaterialContractError, match="重复"):
        ResearchMaterialManifest((_input(), second_path))


@pytest.mark.parametrize(
    "relative_path",
    [
        "/tmp/report.pdf",
        "../report.pdf",
        "reports/../../report.pdf",
        r"reports\report.pdf",
        "C:/reports/report.pdf",
        "~/report.pdf",
        "reports/./report.pdf",
        "reports//report.pdf",
    ],
)
def test_material_path_rejects_host_and_traversal_forms(relative_path):
    with pytest.raises(ResearchMaterialSecurityError):
        _input(relative_path=relative_path)


def test_failure_records_have_stable_codes_and_redacted_messages():
    failure = ResearchMaterialFailure(
        category="parser_error",
        message=(
            "parser failed at /home/hanwen/quant/report.pdf "
            "token=plain-secret C:\\Users\\owner\\report.pdf"
        ),
        retryable=True,
    )
    assert failure.error_code == "MINING-151"
    assert "/home/hanwen" not in failure.message
    assert "plain-secret" not in failure.message
    assert "C:\\Users" not in failure.message
    with pytest.raises(ResearchMaterialContractError, match="错误码"):
        ResearchMaterialFailure(
            category="parser_error",
            message="内部错误",
            retryable=True,
            error_code="MINING-150",
        )


def test_document_record_enforces_success_failure_and_reference_ownership():
    failure = ResearchMaterialFailure(
        category="no_extractable_text",
        message="没有可提取正文",
        retryable=False,
    )
    failed = _record(
        status="failed",
        page_count=None,
        text_sha256=None,
        source_references=(),
        failure=failure,
    )
    assert failed.failure == failure

    with pytest.raises(ResearchMaterialContractError, match="failure"):
        _record(failure=failure)
    with pytest.raises(ResearchMaterialContractError, match="必须包含 failure"):
        _record(status="failed", page_count=None, text_sha256=None)
    with pytest.raises(ResearchMaterialContractError, match="当前资料"):
        _record(
            source_references=(
                _source(document_id=_input(sha256=_sha(b"other")).document_id),
            )
        )


def test_fingerprint_and_manifest_stream_file_metadata_without_host_path(
    tmp_path: Path,
):
    first = tmp_path / "reports" / "momentum.pdf"
    second = tmp_path / "reports" / "value.pdf"
    first.parent.mkdir()
    first.write_bytes(b"%PDF-1.7 momentum")
    second.write_bytes(b"%PDF-1.7 value")

    manifest = build_research_material_manifest(
        tmp_path,
        ["reports/value.pdf", "reports/momentum.pdf"],
        parser_version=PARSER_VERSION,
    )
    assert len(manifest.materials) == 2
    assert {item.relative_path for item in manifest.materials} == {
        "reports/momentum.pdf",
        "reports/value.pdf",
    }
    momentum = fingerprint_research_file(
        tmp_path,
        "reports/momentum.pdf",
        parser_version=PARSER_VERSION,
    )
    assert momentum.media_type == "application/pdf"
    assert momentum.size_bytes == first.stat().st_size
    assert momentum.sha256 == _sha(first.read_bytes())
    assert str(tmp_path) not in repr(momentum.to_dict())


def test_fingerprint_rejects_symlinks_and_reports_missing_files(tmp_path: Path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.pdf"
    outside.write_bytes(b"%PDF-1.7 outside")
    link = tmp_path / "outside.pdf"
    link.symlink_to(outside)

    with pytest.raises(ResearchMaterialSecurityError, match="符号链接"):
        fingerprint_research_file(
            tmp_path,
            "outside.pdf",
            parser_version=PARSER_VERSION,
        )
    with pytest.raises(ResearchMaterialReadError, match="不存在"):
        fingerprint_research_file(
            tmp_path,
            "missing.pdf",
            parser_version=PARSER_VERSION,
        )


def test_text_redaction_and_safe_log_fields_exclude_raw_content():
    raw = (
        "api_key=abcdef123456 Bearer abcdefgh123456 sk-abcdefgh123456 "
        "/home/hanwen/quant/reports/a.pdf C:\\Users\\owner\\a.pdf"
    )
    redacted = redact_research_material_text(raw)
    assert "abcdef123456" not in redacted
    assert "abcdefgh123456" not in redacted
    assert "/home/hanwen" not in redacted
    assert "C:\\Users" not in redacted

    fields = safe_research_material_log_fields(
        {
            "document_id": _input().document_id,
            "manifest_hash": "a" * 64,
            "status": "parsed",
            "relative_path": "reports/a.pdf",
            "message": raw,
            "source_text": raw,
        }
    )
    assert fields == {
        "document_id": _input().document_id,
        "manifest_hash": "a" * 64,
        "status": "parsed",
    }
