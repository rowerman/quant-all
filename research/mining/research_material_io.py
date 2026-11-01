"""资料文件读取、指纹和清单构建。"""

from __future__ import annotations

import hashlib
import mimetypes
import os
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


def _resolve_material_root(root: Path) -> Path:
    """解析并校验资料根目录。"""

    try:
        root_path = Path(root).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ResearchMaterialReadError("资料根目录不存在或不可读") from error
    if not root_path.is_dir():
        raise ResearchMaterialReadError("资料根目录不是目录")
    return root_path


def _resolve_relative_target(root_path: Path, relative_path: str) -> Path:
    """解析根目录内的普通文件或目录，并逐段拒绝符号链接。"""

    if relative_path == ".":
        return root_path
    safe_path = validate_relative_material_path(relative_path)
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
        raise ResearchMaterialReadError(f"资料路径不存在：{safe_path}") from error
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
    return resolved


def resolve_research_file(root: Path, relative_path: str) -> Path:
    """返回根目录内经过安全校验的普通文件绝对路径。"""

    root_path = _resolve_material_root(root)
    resolved = _resolve_relative_target(root_path, relative_path)
    if not resolved.is_file():
        raise ResearchMaterialSecurityError(
            f"资料必须是普通文件：{relative_path}"
        )
    return resolved


def _collect_pdf_paths(
    root_path: Path,
    directory: Path,
    *,
    recursive: bool,
) -> tuple[str, ...]:
    """按稳定顺序收集目录中的普通 PDF，跳过符号链接。"""

    result: list[str] = []
    try:
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
    except OSError as error:
        raise ResearchMaterialReadError("资料目录不可读") from error
    for entry in ordered:
        if entry.is_symlink():
            continue
        if entry.is_dir(follow_symlinks=False):
            if recursive:
                result.extend(
                    _collect_pdf_paths(
                        root_path,
                        Path(entry.path),
                        recursive=True,
                    )
                )
            continue
        if (
            entry.is_file(follow_symlinks=False)
            and Path(entry.name).suffix.casefold() == ".pdf"
        ):
            result.append(
                Path(entry.path).relative_to(root_path).as_posix()
            )
    return tuple(result)


def discover_research_material_paths(
    root: Path,
    target: str,
    *,
    recursive: bool = True,
) -> tuple[str, ...]:
    """发现单文件或目录中的资料路径。

    参数:
        root: 资料根目录。
        target: 根目录内的相对文件或目录；``.`` 表示根目录本身。
        recursive: 目录是否递归扫描。
    """

    if not isinstance(recursive, bool):
        raise ResearchMaterialContractError("recursive 必须是布尔值")
    if not isinstance(target, str):
        raise ResearchMaterialContractError("target 必须是字符串路径")
    root_path = _resolve_material_root(root)
    resolved = _resolve_relative_target(root_path, target)
    if resolved.is_file():
        return (resolved.relative_to(root_path).as_posix(),)
    if not resolved.is_dir():
        raise ResearchMaterialSecurityError("资料目标必须是普通文件或目录")
    paths = _collect_pdf_paths(
        root_path,
        resolved,
        recursive=recursive,
    )
    if not paths:
        raise ResearchMaterialReadError("资料目录中没有可解析的 PDF 文件")
    if len(paths) > 1000:
        raise ResearchMaterialContractError("资料目录中的 PDF 数量超过 1000")
    return paths


def fingerprint_research_file(
    root: Path,
    relative_path: str,
    *,
    parser_version: str,
    media_type: str | None = None,
) -> ResearchMaterialInput:
    """校验单份资料路径并生成流式 SHA-256 指纹。"""

    safe_path = validate_relative_material_path(relative_path)
    resolved = resolve_research_file(root, safe_path)

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
    "discover_research_material_paths",
    "fingerprint_research_file",
    "resolve_research_file",
]
