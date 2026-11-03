"""数据、目录和正式因子快照核对。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from calculate.correlation import available_factors
from common.paths import DB_FILE, FACTOR_DIR, PROJECT_ROOT
from mining.catalog import load_catalog
from mining.contract_requests import FrozenContext
from mining.contract_types import canonical_json, sha256_json
from mining.errors import SnapshotStaleError

DEFAULT_DATA_MANIFEST = PROJECT_ROOT / "data" / "clean" / "manifest.json"


def _file_sha256(path: Path) -> str:
    """计算小型清单文件的 SHA-256。"""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SnapshotStaleError(f"数据清单不可用：{path}") from exc
    return digest.hexdigest()


def data_snapshot_fingerprint(
    db_file: Path | str = DB_FILE,
    manifest_file: Path | str = DEFAULT_DATA_MANIFEST,
) -> str:
    """用清单哈希和行情库元数据生成轻量数据快照指纹。"""

    database = Path(db_file)
    manifest = Path(manifest_file)
    try:
        before = database.stat()
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotStaleError("无法读取数据快照元信息") from exc
    after = database.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise SnapshotStaleError("读取数据快照期间行情库发生变化")
    if not isinstance(manifest_payload, dict):
        raise SnapshotStaleError("数据清单根节点必须是对象")
    return sha256_json(
        {
            "schema_version": "mining_data_snapshot_v1",
            "data_version": manifest_payload.get("data_version"),
            "manifest_sha256": _file_sha256(manifest),
            "database_size": before.st_size,
            "database_mtime_ns": before.st_mtime_ns,
        }
    )


def assert_context_snapshot(
    context: FrozenContext,
    *,
    db_file: Path | str = DB_FILE,
    factor_dir: Path | str = FACTOR_DIR,
    manifest_file: Path | str = DEFAULT_DATA_MANIFEST,
) -> None:
    """确认冻结上下文仍与当前数据、目录和正式因子一致。"""

    current_fingerprint = data_snapshot_fingerprint(db_file, manifest_file)
    if current_fingerprint != context.data_fingerprint:
        raise SnapshotStaleError("行情数据快照已变化")
    catalog = load_catalog()
    if context.field_catalog and _catalog_hash(
        context.field_catalog
    ) != _catalog_hash(catalog.frozen_field_mappings()):
        raise SnapshotStaleError("字段目录已变化")
    if context.operator_catalog and _catalog_hash(
        context.operator_catalog
    ) != _catalog_hash(catalog.frozen_operator_mappings()):
        raise SnapshotStaleError("算子目录已变化")
    if context.existing_factors:
        current = tuple(
            sorted(
                name
                for names in available_factors(factor_dir).values()
                for name in names
            )
        )
        if current != context.existing_factors:
            raise SnapshotStaleError("正式因子清单已变化")


def _catalog_hash(values) -> str:
    """忽略输入顺序比较目录内容。"""

    normalized = sorted(
        (dict(item) for item in values),
        key=canonical_json,
    )
    return sha256_json(normalized)
