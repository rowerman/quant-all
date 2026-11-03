"""金融研报的确定性标题与领域词分类。"""

from __future__ import annotations

import re
import unicodedata

from mining.contract_types import utc_now_iso
from mining.research_material_parse_contracts import (
    RESEARCH_CLASSIFICATION_RULE_VERSION,
    RESEARCH_FINANCE_SIGNALS,
    RESEARCH_REPORT_SIGNALS,
    ResearchMaterialClassification,
)

RESEARCH_CLASSIFICATION_TEXT_CHAR_LIMIT = 20000

_ASCII_WORD = re.compile(r"^[a-z0-9][a-z0-9 _-]*$")


def normalize_research_text(value: str) -> str:
    """统一全半角、大小写和连续空白。"""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_signal(text: str, signal: str) -> bool:
    """匹配固定短语，英文信号要求完整词边界。"""

    normalized_signal = normalize_research_text(signal)
    if _ASCII_WORD.fullmatch(normalized_signal):
        pattern = (
            rf"(?<![a-z0-9]){re.escape(normalized_signal)}"
            rf"(?:s)?(?![a-z0-9])"
        )
        return re.search(pattern, text) is not None
    return normalized_signal in text


def classify_research_material(
    *,
    document_id: str,
    text_sha256: str,
    text: str,
    metadata_title: str | None = None,
    created_at: str | None = None,
) -> ResearchMaterialClassification:
    """按固定规则判断资料是否为金融研报。

    参数:
        document_id: 当前资料的内容稳定 ID。
        text_sha256: 完整页结构哈希。
        text: 已提取的正文。
        metadata_title: PDF 元数据标题，可为空。
        created_at: 分类创建时间；为空时取当前 UTC 时间。
    """

    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    if metadata_title is not None and not isinstance(metadata_title, str):
        raise TypeError("metadata_title 必须是字符串或 None")
    source = "\n".join(
        [metadata_title or "", text[:RESEARCH_CLASSIFICATION_TEXT_CHAR_LIMIT]]
    )
    normalized = normalize_research_text(
        source[:RESEARCH_CLASSIFICATION_TEXT_CHAR_LIMIT]
    )
    report_signals = tuple(
        sorted(
            signal
            for signal in RESEARCH_REPORT_SIGNALS
            if _contains_signal(normalized, signal)
        )
    )
    finance_signals = tuple(
        sorted(
            signal
            for signal in RESEARCH_FINANCE_SIGNALS
            if _contains_signal(normalized, signal)
        )
    )
    kind = (
        "financial_research"
        if report_signals and len(finance_signals) >= 2
        else "not_target"
    )
    return ResearchMaterialClassification(
        document_id=document_id,
        text_sha256=text_sha256,
        kind=kind,
        rule_version=RESEARCH_CLASSIFICATION_RULE_VERSION,
        report_signals=report_signals,
        finance_signals=finance_signals,
        created_at=created_at or utc_now_iso(),
    )


__all__ = [
    "RESEARCH_CLASSIFICATION_RULE_VERSION",
    "RESEARCH_CLASSIFICATION_TEXT_CHAR_LIMIT",
    "classify_research_material",
    "normalize_research_text",
]
