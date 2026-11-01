"""文本型 PDF 解析、研报分类、失败隔离和日志安全测试。"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import logging
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
import pytest

from mining import (
    RESEARCH_CLASSIFICATION_RULE_VERSION,
    RESEARCH_PDF_PARSER_VERSION,
    ResearchMaterialBatchResult,
    ResearchMaterialClassification,
    ResearchMaterialContractError,
    ResearchMaterialParseLimits,
    ResearchMaterialParseResult,
    ResearchMaterialReadError,
    ResearchMaterialSecurityError,
    ResearchPageText,
    build_research_material_manifest,
    classify_research_material,
    discover_research_material_paths,
    parse_research_material,
    parse_research_material_target,
    research_pages_text_sha256,
)


def _write_pdf(
    path: Path,
    pages: tuple[str, ...],
    *,
    title: str | None = None,
    password: str | None = None,
) -> None:
    """写入包含真实 Helvetica 文本内容的最小 PDF。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    for content in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_reference}
                )
            }
        )
        stream = DecodedStreamObject()
        stream.set_data(
            f"BT /F1 12 Tf 72 720 Td ({content}) Tj ET".encode("latin-1")
        )
        page.replace_contents(stream)
    if title:
        writer.add_metadata({"/Title": title})
    if password:
        writer.encrypt(password, algorithm="RC4-128")
    with path.open("wb") as stream:
        writer.write(stream)


def _material(root: Path, relative_path: str):
    """为测试文件生成当前解析器版本的资料输入。"""

    return build_research_material_manifest(
        root,
        [relative_path],
        parser_version=RESEARCH_PDF_PARSER_VERSION,
    ).materials[0]


def _classification(**changes) -> ResearchMaterialClassification:
    """构造便于校验的稳定分类结果。"""

    payload = {
        "document_id": f"doc_{'a' * 24}",
        "text_sha256": "b" * 64,
        "kind": "financial_research",
        "rule_version": RESEARCH_CLASSIFICATION_RULE_VERSION,
        "report_signals": ("equity research",),
        "finance_signals": ("stock", "valuation"),
        "created_at": "2026-11-01T00:00:00+00:00",
    }
    payload.update(changes)
    return ResearchMaterialClassification(**payload)


def test_classification_requires_report_and_two_finance_signals():
    text_hash = hashlib.sha256(b"report").hexdigest()
    document_id = f"doc_{'c' * 24}"
    target = classify_research_material(
        document_id=document_id,
        text_sha256=text_hash,
        metadata_title="ＥＱＵＩＴＹ   ＲＥＳＥＡＲＣＨ",
        text="Stock valuation and factor evidence.",
        created_at="2026-11-01T00:00:00+00:00",
    )
    assert target.kind == "financial_research"
    assert target.report_signals == ("equity research",)
    assert target.finance_signals == (
        "equity",
        "factor",
        "stock",
        "valuation",
    )

    one_finance_term = classify_research_material(
        document_id=document_id,
        text_sha256=text_hash,
        text="Research report about stocks.",
        created_at="2026-11-01T00:00:00+00:00",
    )
    assert one_finance_term.kind == "not_target"

    no_report_signal = classify_research_material(
        document_id=document_id,
        text_sha256=text_hash,
        text="Stock valuation factor portfolio.",
        created_at="2026-11-01T00:00:00+00:00",
    )
    assert no_report_signal.kind == "not_target"

    ascii_boundary = classify_research_material(
        document_id=document_id,
        text_sha256=text_hash,
        text="Equity research report alphabet stock valuation.",
        created_at="2026-11-01T00:00:00+00:00",
    )
    assert "alpha" not in ascii_boundary.finance_signals

    plural_signals = classify_research_material(
        document_id=document_id,
        text_sha256=text_hash,
        text="Research report covering stocks and valuations.",
        created_at="2026-11-01T00:00:00+00:00",
    )
    assert plural_signals.kind == "financial_research"
    assert plural_signals.finance_signals == ("stock", "valuation")


def test_classification_contract_hash_is_stable_across_creation_time():
    first = _classification()
    later = replace(first, created_at="2026-11-02T00:00:00+00:00")
    assert first.classification_hash == later.classification_hash
    assert ResearchMaterialClassification.from_dict(first.to_dict()) == first

    invalid_payload = first.to_dict()
    invalid_payload["report_signals"] = "equity research"
    with pytest.raises(ResearchMaterialContractError, match="字符串数组"):
        ResearchMaterialClassification.from_dict(invalid_payload)
    with pytest.raises(ResearchMaterialContractError, match="不一致"):
        _classification(finance_signals=("stock",))
    with pytest.raises(ResearchMaterialContractError, match="未知固定信号"):
        _classification(report_signals=("unknown report",))
    with pytest.raises(ResearchMaterialContractError, match="rule_version"):
        _classification(rule_version="mining_research_report_v2")
    with pytest.raises(ResearchMaterialContractError, match="排序"):
        _classification(
            report_signals=("research report", "equity research"),
        )


def test_parse_target_pdf_returns_pages_hash_sources_and_classification(
    tmp_path: Path,
):
    path = tmp_path / "reports" / "momentum.pdf"
    _write_pdf(
        path,
        (
            "Equity Research Report",
            "Stock valuation factor evidence",
            "",
        ),
        title="Factor Strategy",
    )
    material = _material(tmp_path, "reports/momentum.pdf")
    result = parse_research_material(tmp_path, material)

    assert result.record.status == "parsed"
    assert result.record.page_count == 3
    assert [page.text for page in result.pages] == [
        "Equity Research Report",
        "Stock valuation factor evidence",
        "",
    ]
    assert result.record.text_sha256 == research_pages_text_sha256(result.pages)
    assert len(result.record.source_references) == 2
    assert result.classification is not None
    assert result.classification.kind == "financial_research"
    assert result.is_financial_research is True
    assert result.is_filtered is False
    assert str(tmp_path) not in str(result.to_dict())
    assert (
        ResearchMaterialParseResult.from_dict(result.to_dict()) == result
    )


def test_parse_non_target_pdf_is_successful_but_filtered(tmp_path: Path):
    path = tmp_path / "accounting-memo.pdf"
    _write_pdf(
        path,
        (
            "Quarterly accounting policy memo",
            "Revenue recognition notes",
            "Equity Research Report stock valuation factor",
        ),
    )
    result = parse_research_material(
        tmp_path,
        _material(tmp_path, "accounting-memo.pdf"),
    )
    assert result.record.status == "parsed"
    assert result.classification is not None
    assert result.classification.kind == "not_target"
    assert result.is_filtered is True
    assert result.is_financial_research is False


def test_directory_batch_isolates_corrupt_pdf_and_filters_non_pdf(
    tmp_path: Path,
):
    reports = tmp_path / "reports"
    _write_pdf(
        reports / "a-report.pdf",
        ("Equity Research Report", "Stock valuation factor"),
    )
    _write_pdf(reports / "b-memo.PDF", ("Accounting memo",))
    (reports / "c-corrupt.pdf").write_bytes(b"%PDF-1.7\ninvalid")
    (reports / "notes.txt").write_text("not a pdf", encoding="utf-8")
    outside = tmp_path / "outside.pdf"
    _write_pdf(outside, ("Equity Research Report stock valuation",))
    (reports / "linked.pdf").symlink_to(outside)

    paths = discover_research_material_paths(tmp_path, "reports")
    assert paths == (
        "reports/a-report.pdf",
        "reports/b-memo.PDF",
        "reports/c-corrupt.pdf",
    )
    result = parse_research_material_target(tmp_path, "reports")
    assert [item.material.relative_path for item in result.documents] == list(
        paths
    )
    assert len(result.financial_research) == 1
    assert len(result.filtered) == 1
    assert len(result.failures) == 1
    assert result.failures[0].record.failure.category == "corrupt_document"
    assert ResearchMaterialBatchResult.from_dict(result.to_dict()) == result


def test_directory_discovery_respects_recursive_and_rejects_unsafe_target(
    tmp_path: Path,
):
    _write_pdf(tmp_path / "top.pdf", ("Top",))
    _write_pdf(tmp_path / "nested" / "child.pdf", ("Child",))
    assert discover_research_material_paths(
        tmp_path,
        ".",
        recursive=False,
    ) == ("top.pdf",)
    assert discover_research_material_paths(tmp_path, ".") == (
        "nested/child.pdf",
        "top.pdf",
    )
    link = tmp_path / "target-link"
    link.symlink_to(tmp_path / "nested", target_is_directory=True)
    with pytest.raises(ResearchMaterialSecurityError, match="符号链接"):
        discover_research_material_paths(tmp_path, "target-link")
    with pytest.raises(ResearchMaterialSecurityError):
        discover_research_material_paths(tmp_path, "../")


def test_explicit_non_pdf_is_reported_as_unsupported(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("stock valuation factor", encoding="utf-8")
    result = parse_research_material_target(tmp_path, "notes.txt")
    assert len(result.failures) == 1
    assert result.failures[0].record.failure.category == "unsupported_format"


def test_encrypted_and_blank_pdfs_have_distinct_failures(tmp_path: Path):
    encrypted = tmp_path / "encrypted.pdf"
    _write_pdf(encrypted, ("Equity Research Report",), password="secret")
    encrypted_result = parse_research_material(
        tmp_path,
        _material(tmp_path, "encrypted.pdf"),
    )
    assert encrypted_result.record.failure.category == "encrypted_document"

    blank = tmp_path / "blank.pdf"
    _write_pdf(blank, ("",))
    blank_result = parse_research_material(
        tmp_path,
        _material(tmp_path, "blank.pdf"),
    )
    assert blank_result.record.failure.category == "no_extractable_text"


def test_changed_file_and_resource_limits_are_isolated(tmp_path: Path):
    path = tmp_path / "report.pdf"
    _write_pdf(path, ("Equity Research Report", "Stock valuation factor"))
    material = _material(tmp_path, "report.pdf")
    _write_pdf(path, ("Changed",))
    changed = parse_research_material(tmp_path, material)
    assert changed.record.failure.category == "input_changed"
    assert changed.record.failure.retryable is True

    current = _material(tmp_path, "report.pdf")
    too_small = parse_research_material(
        tmp_path,
        current,
        limits=ResearchMaterialParseLimits(max_file_bytes=10),
    )
    assert too_small.record.failure.category == "resource_limit"
    assert too_small.record.failure.error_code == "MINING-152"

    _write_pdf(path, ("First page", "Second page"))
    limited_pages = parse_research_material(
        tmp_path,
        _material(tmp_path, "report.pdf"),
        limits=ResearchMaterialParseLimits(max_pages=1),
    )
    assert limited_pages.record.failure.category == "resource_limit"

    limited_page_text = parse_research_material(
        tmp_path,
        _material(tmp_path, "report.pdf"),
        limits=ResearchMaterialParseLimits(max_page_chars=1),
    )
    assert limited_page_text.record.failure.category == "resource_limit"

    limited_total_text = parse_research_material(
        tmp_path,
        _material(tmp_path, "report.pdf"),
        limits=ResearchMaterialParseLimits(max_total_text_chars=5),
    )
    assert limited_total_text.record.failure.category == "resource_limit"


def test_missing_and_replaced_symlink_files_are_isolated(tmp_path: Path):
    missing_path = tmp_path / "missing.pdf"
    _write_pdf(missing_path, ("Equity Research Report stock valuation",))
    missing_material = _material(tmp_path, "missing.pdf")
    missing_path.unlink()
    missing = parse_research_material(tmp_path, missing_material)
    assert missing.record.failure.category == "unreadable"

    linked_path = tmp_path / "linked.pdf"
    _write_pdf(linked_path, ("Equity Research Report stock valuation",))
    linked_material = _material(tmp_path, "linked.pdf")
    original = tmp_path / "original.pdf"
    linked_path.rename(original)
    linked_path.symlink_to(original)
    linked = parse_research_material(tmp_path, linked_material)
    assert linked.record.failure.category == "unsafe_input"


def test_unexpected_parser_error_does_not_stop_following_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    reports = tmp_path / "reports"
    _write_pdf(reports / "a-bad.pdf", ("Bad",))
    _write_pdf(
        reports / "b-good.pdf",
        ("Equity Research Report stock valuation",),
    )
    real_reader = PdfReader

    def flaky_reader(stream, *, strict=False):
        if Path(stream.name).name == "a-bad.pdf":
            raise ValueError("provider failed /home/hanwen/secret")
        return real_reader(stream, strict=strict)

    monkeypatch.setattr(
        "mining.research_material_parser.PdfReader",
        flaky_reader,
    )
    result = parse_research_material_target(tmp_path, "reports")
    assert result.documents[0].record.failure.category == "parser_error"
    assert result.documents[1].record.status == "parsed"


def test_parse_contracts_reject_tampered_hashes_and_page_order(
    tmp_path: Path,
):
    path = tmp_path / "report.pdf"
    _write_pdf(path, ("Equity Research Report stock valuation",))
    result = parse_research_material(
        tmp_path,
        _material(tmp_path, "report.pdf"),
    )
    with pytest.raises(ResearchMaterialContractError, match="哈希"):
        ResearchMaterialParseResult(
            material=result.material,
            record=replace(result.record, text_sha256="f" * 64),
            pages=result.pages,
            classification=result.classification,
        )
    with pytest.raises(ResearchMaterialContractError, match="连续编号"):
        ResearchMaterialParseResult(
            material=result.material,
            record=result.record,
            pages=(ResearchPageText(page_number=2, text="changed"),),
            classification=result.classification,
        )
    with pytest.raises(ResearchMaterialContractError, match="整数"):
        ResearchMaterialParseLimits(max_pages=True)


def test_logs_exclude_raw_pdf_text_paths_and_parser_errors(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    path = tmp_path / "sensitive-secret.pdf"
    secret_text = "api_key=plain-secret /home/hanwen/quant/internal"
    _write_pdf(path, (secret_text,))
    with caplog.at_level(logging.INFO):
        result = parse_research_material(
            tmp_path,
            _material(tmp_path, "sensitive-secret.pdf"),
        )
    assert result.record.status == "parsed"
    assert "sensitive-secret.pdf" not in caplog.text
    assert "api_key=plain-secret" not in caplog.text
    assert "/home/hanwen/quant" not in caplog.text
    assert result.material.document_id in caplog.text


def test_directory_without_pdf_reports_read_error(tmp_path: Path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "notes.txt").write_text("notes", encoding="utf-8")
    with pytest.raises(ResearchMaterialReadError, match="没有"):
        discover_research_material_paths(tmp_path, "reports")
