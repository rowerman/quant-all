"""资料文件读取、指纹和清单构建。"""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path, PurePosixPath
from typing import Iterable

from mining.errors import (
    ResearchMaterialContractError,
    ResearchMaterialReadError,
    ResearchMaterialSecurityError,
)
from mining.research_material_contracts import (
    ResearchMaterialInput,
    ResearchMaterialManifest,
)
from mining.research_material_safety import validate_relative_material_path


def fingerprint_research_file(
    root: Path,
    relative_path: str,
    *,
    parser_version: str,
    media_type: str | None = None,
) -> ResearchMaterialInput:
    """校验单份资料路径并生成流式 SHA-256 指纹。"""

    safe_path = validate_relative_material_path(relative_path)
    try:
        root_path = Path(root).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ResearchMaterialReadError("资料根目录不存在或不可读") from error
    if not root_path.is_dir():
        raise ResearchMaterialReadError("资料根目录不是目录")

    candidate = root_path.joinpath(*PurePosixPath(safe_path).parts)
    cursor = root_path
    for part in PurePosixPath(safe_path).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ResearchMaterialSecurityError(
                f"资料路径不能包含符号链接：{safe_path}"
            )
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ResearchMaterialReadError(f"资料文件不存在：{safe_path}") from error
    except RuntimeError as error:
        raise ResearchMaterialSecurityError(
            f"资料路径包含无法安全解析的链接：{safe_path}"
        ) from error
    except OSError as error:
        raise ResearchMaterialReadError(f"资料路径不可读：{safe_path}") from error
    try:
        resolved.relative_to(root_path)
    except ValueError as error:
        raise ResearchMaterialSecurityError(
            f"资料路径必须在根目录内：{safe_path}"
        ) from error
    if not resolved.is_file():
        raise ResearchMaterialSecurityError(f"资料必须是普通文件：{safe_path}")

    digest = hashlib.sha256()
    try:
        size = 0
        with resolved.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise ResearchMaterialReadError(f"资料文件不可读：{safe_path}") from error

    resolved_media_type = media_type
    if resolved_media_type is None:
        resolved_media_type = (
            mimetypes.guess_type(safe_path)[0] or "application/octet-stream"
        )
    return ResearchMaterialInput(
        relative_path=safe_path,
        media_type=resolved_media_type,
        size_bytes=size,
        sha256=digest.hexdigest(),
        parser_version=parser_version,
    )


def build_research_material_manifest(
    root: Path,
    relative_paths: Iterable[str],
    *,
    parser_version: str,
) -> ResearchMaterialManifest:
    """为一批相对路径生成稳定资料清单。"""

    if isinstance(relative_paths, (str, bytes)):
        raise ResearchMaterialContractError("relative_paths 必须是路径序列")
    paths = list(relative_paths)
    if not 1 <= len(paths) <= 1000:
        raise ResearchMaterialContractError("relative_paths 数量必须在 1～1000")
    materials = tuple(
        fingerprint_research_file(
            root,
            path,
            parser_version=parser_version,
        )
        for path in paths
    )
    return ResearchMaterialManifest(materials=materials)


__all__ = [
    "build_research_material_manifest",
    "fingerprint_research_file",
]
