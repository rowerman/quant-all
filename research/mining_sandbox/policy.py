"""隔离代码研究沙箱的冻结政策和威胁边界。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from common.paths import MINING_SANDBOX_CONFIG_FILE, PROJECT_ROOT
from mining.contract_types import canonical_json, sha256_json, text
from mining.search_contract_types import bounded_integer, strict_fields
from mining_sandbox.errors import SandboxPolicyError

SANDBOX_SCHEMA_VERSION = "mining_sandbox_v1"
_RELATIVE_PATH = re.compile(r"^[A-Za-z0-9._/-]+$")


def _positive_float(value: Any, name: str, *, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SandboxPolicyError(f"{name} 必须是数值")
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= maximum:
        raise SandboxPolicyError(f"{name} 必须在 0～{maximum}")
    return result


def _path_text(value: Any, name: str, project_root: Path) -> Path:
    result = text(value, name, maximum=500)
    if not _RELATIVE_PATH.fullmatch(result):
        raise SandboxPolicyError(f"{name} 包含非法字符")
    candidate = Path(result)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SandboxPolicyError(f"{name} 必须是项目内相对路径")
    resolved = (project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise SandboxPolicyError(f"{name} 超出项目目录") from exc
    return resolved


@dataclass(frozen=True)
class SandboxPolicy:
    """代码研究沙箱不可放宽的运行时边界。"""

    enabled: bool = False
    image: str = "quant-mining-sandbox:s5"
    timeout_seconds: int = 120
    cpu_limit: float = 2.0
    memory_mb: int = 2048
    pids_limit: int = 64
    workspace_mb: int = 512
    output_mb: int = 10
    network: bool = False
    read_only_root: bool = True
    user: str = "65532:65532"
    snapshot_root: str = "data/mining_sandbox/snapshots"
    workspace_root: str = "data/mining_sandbox/work"
    schema_version: str = SANDBOX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SANDBOX_SCHEMA_VERSION:
            raise SandboxPolicyError(
                f"schema_version 必须为 {SANDBOX_SCHEMA_VERSION}"
            )
        if not isinstance(self.enabled, bool):
            raise SandboxPolicyError("enabled 必须是布尔值")
        object.__setattr__(
            self,
            "image",
            text(self.image, "image", maximum=300),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            bounded_integer(
                self.timeout_seconds,
                "timeout_seconds",
                minimum=1,
                maximum=900,
            ),
        )
        object.__setattr__(
            self,
            "cpu_limit",
            _positive_float(self.cpu_limit, "cpu_limit", maximum=8),
        )
        for name, value, maximum in (
            ("memory_mb", self.memory_mb, 8192),
            ("pids_limit", self.pids_limit, 256),
            ("workspace_mb", self.workspace_mb, 8192),
            ("output_mb", self.output_mb, 100),
        ):
            object.__setattr__(
                self,
                name,
                bounded_integer(value, name, minimum=1, maximum=maximum),
            )
        if self.network is not False:
            raise SandboxPolicyError("沙箱网络必须关闭")
        if self.read_only_root is not True:
            raise SandboxPolicyError("沙箱根文件系统必须只读")
        user = text(self.user, "user", maximum=32)
        if user in {"0", "0:0", "root"}:
            raise SandboxPolicyError("沙箱不得使用 root 用户")
        object.__setattr__(self, "user", user)
        _path_text(self.snapshot_root, "snapshot_root", PROJECT_ROOT)
        _path_text(self.workspace_root, "workspace_root", PROJECT_ROOT)

    @classmethod
    def from_dict(cls, value: Any) -> SandboxPolicy:
        """严格解析政策 JSON。"""

        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping):
            raise SandboxPolicyError("沙箱政策必须是 JSON 对象")
        payload = dict(value)
        payload.setdefault("schema_version", SANDBOX_SCHEMA_VERSION)
        return cls(**strict_fields(payload, "sandbox_policy", fields))

    def to_dict(self) -> dict[str, Any]:
        """返回稳定政策字典。"""

        return asdict(self)

    @property
    def policy_hash(self) -> str:
        """返回冻结政策哈希。"""

        return sha256_json(self.to_dict())

    @property
    def snapshot_root_path(self) -> Path:
        return _path_text(self.snapshot_root, "snapshot_root", PROJECT_ROOT)

    @property
    def workspace_root_path(self) -> Path:
        return _path_text(self.workspace_root, "workspace_root", PROJECT_ROOT)


def load_sandbox_policy(
    path: Path | str = MINING_SANDBOX_CONFIG_FILE,
) -> SandboxPolicy:
    """从版本化配置读取沙箱政策。"""

    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SandboxPolicyError(f"沙箱政策不存在：{target}") from exc
    except json.JSONDecodeError as exc:
        raise SandboxPolicyError(f"沙箱政策 JSON 非法：{exc}") from exc
    payload.setdefault("schema_version", SANDBOX_SCHEMA_VERSION)
    return SandboxPolicy.from_dict(payload)


__all__ = ["SANDBOX_SCHEMA_VERSION", "SandboxPolicy", "load_sandbox_policy"]
