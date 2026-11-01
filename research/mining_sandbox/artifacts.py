"""只读沙箱研究产物清单和人工提升记录。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from mining.contract_types import identifier, sha256_json, text, utc_now_iso
from mining_sandbox.errors import SandboxArtifactError
from mining_sandbox.policy import SandboxPolicy

SANDBOX_ARTIFACT_SCHEMA_VERSION = "mining_sandbox_artifact_v1"
SANDBOX_PROMOTION_SCHEMA_VERSION = "mining_sandbox_promotion_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_mapping(value: Any, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SandboxArtifactError(f"{name} 必须是 JSON 对象")
    if set(value) != fields:
        raise SandboxArtifactError(f"{name} 字段不完整或包含未知字段")
    return dict(value)


def _tree_files(root: Path) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise SandboxArtifactError("沙箱产物根目录不可用")
    files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise SandboxArtifactError(f"沙箱产物包含符号链接：{relative}")
        if path.is_file():
            files.add(relative)
        elif not path.is_dir():
            raise SandboxArtifactError(f"沙箱产物包含非普通文件：{relative}")
    return files


@dataclass(frozen=True)
class SandboxArtifactManifest:
    """一次沙箱研究的全部尝试、文件和环境证据。"""

    sandbox_run_id: str
    request_hash: str
    context_hash: str
    snapshot_manifest_hash: str
    hypothesis_hash: str
    image_id: str
    status: str
    attempts: tuple[Mapping[str, Any], ...]
    environment: Mapping[str, Any]
    files: Mapping[str, Mapping[str, Any]]
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SANDBOX_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SANDBOX_ARTIFACT_SCHEMA_VERSION:
            raise SandboxArtifactError("沙箱产物清单版本不匹配")
        object.__setattr__(
            self,
            "sandbox_run_id",
            identifier(self.sandbox_run_id, "sandbox_run_id"),
        )
        for name in (
            "request_hash",
            "context_hash",
            "snapshot_manifest_hash",
            "hypothesis_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise SandboxArtifactError(f"{name} 必须是 SHA-256")
        if not self.image_id.startswith("sha256:"):
            raise SandboxArtifactError("image_id 必须是 Docker 镜像 ID")
        object.__setattr__(self, "status", identifier(self.status, "status"))
        if not isinstance(self.attempts, (list, tuple)) or not self.attempts:
            raise SandboxArtifactError("attempts 不能为空")
        object.__setattr__(
            self,
            "attempts",
            tuple(dict(item) for item in self.attempts),
        )
        object.__setattr__(self, "environment", dict(self.environment))
        object.__setattr__(
            self,
            "files",
            {
                key: dict(value)
                for key, value in sorted(self.files.items())
            },
        )

    def material(self) -> dict[str, Any]:
        """返回稳定清单内容。"""

        return {
            "schema_version": self.schema_version,
            "sandbox_run_id": self.sandbox_run_id,
            "request_hash": self.request_hash,
            "context_hash": self.context_hash,
            "snapshot_manifest_hash": self.snapshot_manifest_hash,
            "hypothesis_hash": self.hypothesis_hash,
            "image_id": self.image_id,
            "status": self.status,
            "attempts": [dict(item) for item in self.attempts],
            "environment": dict(self.environment),
            "files": {key: dict(value) for key, value in self.files.items()},
        }

    @property
    def manifest_hash(self) -> str:
        """返回产物清单稳定哈希。"""

        return sha256_json(self.material())

    @classmethod
    def from_dict(cls, value: Any) -> SandboxArtifactManifest:
        """严格解析产物清单。"""

        payload = _strict_mapping(
            value,
            "sandbox_artifact_manifest",
            set(cls.__dataclass_fields__),
        )
        payload["attempts"] = tuple(payload["attempts"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回持久化清单。"""

        return {**self.material(), "created_at": self.created_at}


def seal_artifact_tree(
    artifact_dir: Path | str,
    manifest: SandboxArtifactManifest,
) -> None:
    """写入清单并把整棵产物树封存为只读。"""

    root = Path(artifact_dir)
    _tree_files(root)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        except PermissionError:
            mode = path.stat().st_mode & 0o777
            if path.stat().st_uid == os.geteuid() or mode & 0o022:
                raise
    os.chmod(root, 0o555)


def verify_artifact_tree(
    artifact_dir: Path | str,
) -> SandboxArtifactManifest:
    """读取清单并验证所有文件未被修改。"""

    root = Path(artifact_dir)
    manifest_path = root / "manifest.json"
    try:
        manifest = SandboxArtifactManifest.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SandboxArtifactError(f"沙箱产物清单不可用：{exc}") from exc
    actual_files = _tree_files(root)
    expected_files = set(manifest.files) | {"manifest.json"}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise SandboxArtifactError(
            f"沙箱产物文件集合变化：缺失={missing}，额外={extra}"
        )
    for relative, expected in manifest.files.items():
        path = root / relative
        if not path.is_file():
            raise SandboxArtifactError(f"沙箱产物缺失：{relative}")
        if int(path.stat().st_size) != int(expected["bytes"]):
            raise SandboxArtifactError(f"沙箱产物大小变化：{relative}")
        if _sha256(path) != expected["sha256"]:
            raise SandboxArtifactError(f"沙箱产物哈希变化：{relative}")
    return manifest


@dataclass(frozen=True)
class PromotionRequest:
    """人工审核后转写沙箱研究思想的可审计请求。"""

    sandbox_run_id: str
    artifact_manifest_hash: str
    reviewer: str
    reason: str
    target_type: str
    target_reference: str
    schema_version: str = SANDBOX_PROMOTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SANDBOX_PROMOTION_SCHEMA_VERSION:
            raise SandboxArtifactError("人工提升记录版本不匹配")
        object.__setattr__(
            self,
            "sandbox_run_id",
            identifier(self.sandbox_run_id, "sandbox_run_id"),
        )
        if (
            not isinstance(self.artifact_manifest_hash, str)
            or len(self.artifact_manifest_hash) != 64
        ):
            raise SandboxArtifactError("artifact_manifest_hash 必须是 SHA-256")
        object.__setattr__(
            self,
            "reviewer",
            text(self.reviewer, "reviewer", maximum=100),
        )
        object.__setattr__(
            self,
            "reason",
            text(self.reason, "reason", maximum=2000),
        )
        if self.target_type not in {"dsl", "operator"}:
            raise SandboxArtifactError("target_type 必须是 dsl 或 operator")
        object.__setattr__(
            self,
            "target_reference",
            text(self.target_reference, "target_reference", maximum=4000),
        )

    def material(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sandbox_run_id": self.sandbox_run_id,
            "artifact_manifest_hash": self.artifact_manifest_hash,
            "reviewer": self.reviewer,
            "reason": self.reason,
            "target_type": self.target_type,
            "target_reference": self.target_reference,
        }

    @property
    def promotion_id(self) -> str:
        return f"promotion_{sha256_json(self.material())[:24]}"


def record_manual_promotion(
    *,
    policy: SandboxPolicy,
    artifact_dir: Path | str,
    request: PromotionRequest,
) -> dict[str, Any]:
    """验证只读产物并记录人工提升，不创建正式候选。"""

    artifacts = verify_artifact_tree(artifact_dir)
    if artifacts.manifest_hash != request.artifact_manifest_hash:
        raise SandboxArtifactError("人工提升请求绑定了不同产物清单")
    if artifacts.sandbox_run_id != request.sandbox_run_id:
        raise SandboxArtifactError("人工提升请求绑定了不同沙箱运行")
    review_root = policy.review_root_path / artifacts.sandbox_run_id
    review_root.mkdir(parents=True, exist_ok=True)
    target = review_root / f"{request.promotion_id}.json"
    payload = {
        **request.material(),
        "promotion_id": request.promotion_id,
        "recorded_at": utc_now_iso(),
        "formal_candidate_created": False,
    }
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        comparable = {
            key: value
            for key, value in existing.items()
            if key not in {"recorded_at", "reused"}
        }
        expected = {
            key: value
            for key, value in payload.items()
            if key != "recorded_at"
        }
        if comparable != expected:
            raise SandboxArtifactError("人工提升记录已存在且内容不同")
        return {**existing, "reused": True}
    target.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    os.chmod(target, 0o444)
    return {**payload, "reused": False}


def copy_artifacts_for_review(
    *,
    policy: SandboxPolicy,
    artifact_dir: Path | str,
    destination_name: str | None = None,
) -> Path:
    """把已校验产物复制到独立只读审核目录。"""

    source = Path(artifact_dir)
    manifest = verify_artifact_tree(source)
    name = destination_name or manifest.sandbox_run_id
    destination = policy.review_root_path / "artifacts" / identifier(
        name,
        "destination_name",
    )
    if destination.exists():
        if verify_artifact_tree(destination).manifest_hash == manifest.manifest_hash:
            return destination
        raise SandboxArtifactError("审核目录已有不同产物")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    os.replace(temporary, destination)
    return destination


__all__ = [
    "PromotionRequest",
    "SANDBOX_ARTIFACT_SCHEMA_VERSION",
    "SANDBOX_PROMOTION_SCHEMA_VERSION",
    "SandboxArtifactManifest",
    "copy_artifacts_for_review",
    "record_manual_promotion",
    "seal_artifact_tree",
    "verify_artifact_tree",
]
