"""V1/V2 到 V3 只读迁移快照的稳定约定。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from mining.contract_types import sha256_json

LEGACY_SOURCE_SCHEMAS = ("mining", "mining_v2")
LEGACY_SOURCE_VERSIONS = {
    "mining": "mining_v1",
    "mining_v2": "mining_v2",
}
LEGACY_CONTEXT_FIELDS = (
    "context_id",
    "context_hash",
    "payload_json",
    "created_at",
)
LEGACY_RUN_FIELDS = (
    "run_id",
    "context_id",
    "idempotency_key",
    "request_hash",
    "input_mode",
    "status",
    "hypothesis",
    "request_json",
    "created_at",
    "updated_at",
)
LEGACY_CANDIDATE_FIELDS = (
    "run_id",
    "candidate_id",
    "semantic_hash",
    "content_hash",
    "name",
    "status",
    "definition_json",
    "created_at",
    "updated_at",
)


def _material(
    rows: Iterable[Mapping[str, Any]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """按约定字段顺序规范化一行，供稳定哈希和快照复制共用。"""

    return [{name: row[name] for name in fields} for row in rows]


def legacy_snapshot_hash(
    source_schema: str,
    contexts: Iterable[Mapping[str, Any]],
    runs: Iterable[Mapping[str, Any]],
    candidates: Iterable[Mapping[str, Any]],
) -> str:
    """计算一个旧 schema 的上下文、运行和候选快照哈希。"""

    if source_schema not in LEGACY_SOURCE_SCHEMAS:
        raise ValueError(f"未知 legacy 来源：{source_schema}")
    return sha256_json(
        {
            "source_schema": source_schema,
            "source_schema_version": LEGACY_SOURCE_VERSIONS[source_schema],
            "contexts": _material(contexts, LEGACY_CONTEXT_FIELDS),
            "runs": _material(runs, LEGACY_RUN_FIELDS),
            "candidates": _material(candidates, LEGACY_CANDIDATE_FIELDS),
        }
    )


__all__ = [
    "LEGACY_CANDIDATE_FIELDS",
    "LEGACY_CONTEXT_FIELDS",
    "LEGACY_RUN_FIELDS",
    "LEGACY_SOURCE_SCHEMAS",
    "LEGACY_SOURCE_VERSIONS",
    "legacy_snapshot_hash",
]
