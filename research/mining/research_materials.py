"""资料驱动输入、文件指纹、来源引用和解析失败契约的公开入口。"""

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
    fingerprint_research_file,
)
from mining.research_material_safety import (
    redact_research_material_text,
    safe_research_material_log_fields,
)

__all__ = [
    "RESEARCH_DOCUMENT_STATUSES",
    "RESEARCH_FAILURE_CATEGORIES",
    "RESEARCH_MATERIAL_SCHEMA_VERSION",
    "ResearchDocumentRecord",
    "ResearchMaterialFailure",
    "ResearchMaterialInput",
    "ResearchMaterialManifest",
    "SourceReference",
    "build_research_material_manifest",
    "fingerprint_research_file",
    "redact_research_material_text",
    "safe_research_material_log_fields",
]
