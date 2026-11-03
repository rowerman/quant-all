"""正式发布产物契约、原子提交和回滚。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from factors.published import MINED_FAMILY
from factors.registry import FactorSpec
from mining.contract_types import canonical_json
from mining.errors import PublicationFilesystemError
from scripts.build_factors import catalog
from scripts.build_factors.sql import family_glob


@dataclass(frozen=True)
class StagedPublication:
    """已构建并校验、等待原子提交的完整 mined 族。"""

    staging_root: Path
    family_root: Path
    definitions_path: Path
    catalog_path: Path
    manifest_hash: str
    artifact_sha256: str
    factor_names: tuple[str, ...]
    specs: tuple[FactorSpec, ...]
    coverage: tuple[dict[str, Any], ...]


def tree_hash(files: tuple[Path, ...], root: Path) -> str:
    """按相对路径与文件摘要计算产物树哈希。"""

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """原子写入规范化 JSON。"""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(dict(payload)), encoding="utf-8")
    os.replace(temporary, path)


def commit_publication(
    *,
    staged: StagedPublication,
    batch_id: str,
    factor_dir: Path,
    catalog_file: Path,
) -> None:
    """用符号链接和目录库替换完成原子可见性切换。"""

    version_root = factor_dir / ".mined-versions" / batch_id
    version_root.parent.mkdir(parents=True, exist_ok=True)
    if version_root.exists():
        raise PublicationFilesystemError("发布版本目录已存在")
    os.replace(staged.family_root, version_root)
    link = factor_dir / MINED_FAMILY
    previous_link = os.readlink(link) if link.is_symlink() else None
    previous_catalog = staged.staging_root / "factors.previous.duckdb"
    if catalog_file.exists():
        shutil.copy2(catalog_file, previous_catalog)
    journal = {
        "batch_id": batch_id,
        "state": "prepared",
        "previous_link": previous_link,
        "new_link": version_root.relative_to(factor_dir).as_posix(),
        "previous_catalog": str(previous_catalog) if previous_catalog.exists() else None,
    }
    journal_path = staged.staging_root / "commit.json"
    write_json(journal_path, journal)
    temporary_link = factor_dir / f".mined.link-{batch_id}"
    try:
        if temporary_link.exists() or temporary_link.is_symlink():
            temporary_link.unlink()
        os.symlink(version_root.relative_to(factor_dir), temporary_link)
        os.replace(temporary_link, link)
        with catalog.connect(staged.catalog_path) as conn:
            conn.execute(
                "CREATE OR REPLACE VIEW v.mined AS "
                f"SELECT * FROM read_parquet('{family_glob(factor_dir, MINED_FAMILY)}', "
                "hive_partitioning = 1)"
            )
        os.replace(staged.catalog_path, catalog_file)
        journal["state"] = "committed"
        write_json(journal_path, journal)
    except BaseException as error:
        rollback_publication(
            staging_root=staged.staging_root,
            factor_dir=factor_dir,
            catalog_file=catalog_file,
        )
        raise PublicationFilesystemError(f"正式因子提交失败：{error}") from error


def rollback_publication(
    *,
    staging_root: Path,
    factor_dir: Path,
    catalog_file: Path,
) -> None:
    """根据提交日志恢复旧链接和旧目录库。"""

    journal_path = staging_root / "commit.json"
    if not journal_path.is_file():
        return
    try:
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        link = factor_dir / MINED_FAMILY
        previous_link = journal.get("previous_link")
        if previous_link is None:
            if link.is_symlink():
                link.unlink()
        else:
            temporary = factor_dir / f".mined.rollback-{journal['batch_id']}"
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
            os.symlink(previous_link, temporary)
            os.replace(temporary, link)
        previous_catalog = journal.get("previous_catalog")
        if previous_catalog and Path(previous_catalog).is_file():
            os.replace(previous_catalog, catalog_file)
    except (OSError, ValueError, KeyError) as error:
        raise PublicationFilesystemError(f"发布回滚失败：{error}") from error
