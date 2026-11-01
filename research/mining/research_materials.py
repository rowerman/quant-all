"""资料驱动输入、PDF 解析、研报分类和来源引用契约的公开入口。"""

from mining.research_material_classification import (
    RESEARCH_CLASSIFICATION_TEXT_CHAR_LIMIT,
    classify_research_material,
    normalize_research_text,
)
from mining.research_material_contracts import (
    RESEARCH_DOCUMENT_STATUSES,
    RESEARCH_FAILURE_CATEGORIES,
    RESEARCH_MATERIAL_SCHEMA_VERSION,
    ResearchDocumentRecord,
    ResearchMaterialFailure,
    ResearchMaterialInput,
    ResearchMaterialManifest,
    SourceReference,
)
from mining.research_material_io import (
    build_research_material_manifest,
    discover_research_material_paths,
    fingerprint_research_file,
    resolve_research_file,
)
from mining.research_material_parse_contracts import (
    RESEARCH_CLASSIFICATION_KINDS,
    RESEARCH_CLASSIFICATION_RULE_VERSION,
    RESEARCH_CLASSIFICATION_SCHEMA_VERSION,
    RESEARCH_FINANCE_SIGNALS,
    RESEARCH_MATERIAL_PARSE_SCHEMA_VERSION,
    RESEARCH_PAGE_SCHEMA_VERSION,
    RESEARCH_REPORT_SIGNALS,
    ResearchMaterialBatchResult,
    ResearchMaterialClassification,
    ResearchMaterialParseLimits,
    ResearchMaterialParseResult,
    ResearchPageText,
    page_text_material,
    research_pages_text_sha256,
)
from mining.research_material_parser import (
    RESEARCH_PDF_PARSER_VERSION,
    parse_research_material,
    parse_research_material_target,
    parse_research_materials,
)
from mining.research_material_safety import (
    redact_research_material_text,
    safe_research_material_log_fields,
)

__all__ = [
    "RESEARCH_CLASSIFICATION_KINDS",
    "RESEARCH_CLASSIFICATION_RULE_VERSION",
    "RESEARCH_CLASSIFICATION_SCHEMA_VERSION",
    "RESEARCH_CLASSIFICATION_TEXT_CHAR_LIMIT",
    "RESEARCH_DOCUMENT_STATUSES",
    "RESEARCH_FAILURE_CATEGORIES",
    "RESEARCH_FINANCE_SIGNALS",
    "RESEARCH_MATERIAL_PARSE_SCHEMA_VERSION",
    "RESEARCH_MATERIAL_SCHEMA_VERSION",
    "RESEARCH_PAGE_SCHEMA_VERSION",
    "RESEARCH_PDF_PARSER_VERSION",
    "RESEARCH_REPORT_SIGNALS",
    "ResearchDocumentRecord",
    "ResearchMaterialBatchResult",
    "ResearchMaterialClassification",
    "ResearchMaterialFailure",
    "ResearchMaterialInput",
    "ResearchMaterialManifest",
    "ResearchMaterialParseLimits",
    "ResearchMaterialParseResult",
    "ResearchPageText",
    "SourceReference",
    "build_research_material_manifest",
    "classify_research_material",
    "discover_research_material_paths",
    "fingerprint_research_file",
    "normalize_research_text",
    "page_text_material",
    "parse_research_material",
    "parse_research_material_target",
    "parse_research_materials",
    "redact_research_material_text",
    "research_pages_text_sha256",
    "resolve_research_file",
    "safe_research_material_log_fields",
]
