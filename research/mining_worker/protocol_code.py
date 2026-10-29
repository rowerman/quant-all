"""Worker 代码研究提案协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping

from mining_worker.protocol import WorkerProtocolError
from mining_worker.protocol_v2 import ensure_agent_payload

WORKER_CODE_SCHEMA_VERSION = "mining_worker_code_v1"
CODE_PROMPT_VERSION = "mining_qlib_code_prompt_v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")


def _strict_mapping(
    value: Any,
    name: str,
    fields: set[str],
    *,
    maximum_bytes: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkerProtocolError(f"{name} 必须是 JSON 对象")
    actual = set(value)
    missing = fields - actual
    unknown = actual - fields
    if missing or unknown:
        raise WorkerProtocolError(
            f"{name} 字段不匹配：缺少 {sorted(missing)}，未知 {sorted(unknown)}"
        )
    result = dict(value)
    encoded = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > maximum_bytes:
        raise WorkerProtocolError(f"{name} 超过 {maximum_bytes} 字节")
    return result


@dataclass(frozen=True)
class WorkerCodeProposalRequest:
    """一轮 Qlib 代码研究提案请求。"""

    request_id: str
    run_id: str
    iteration: int
    hypothesis: str
    snapshot: Mapping[str, Any]
    previous_attempt: Mapping[str, Any] | None
    max_code_bytes: int
    llm_model: str
    prompt_version: str = CODE_PROMPT_VERSION
    schema_version: str = WORKER_CODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_CODE_SCHEMA_VERSION:
            raise WorkerProtocolError(
                f"schema_version 必须为 {WORKER_CODE_SCHEMA_VERSION}"
            )
        for name in ("request_id", "run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
                raise WorkerProtocolError(f"{name} 包含非法字符")
        if (
            isinstance(self.iteration, bool)
            or not isinstance(self.iteration, int)
            or not 1 <= self.iteration <= 3
        ):
            raise WorkerProtocolError("iteration 必须在 1～3")
        if not isinstance(self.hypothesis, str) or not 8 <= len(
            self.hypothesis.strip()
        ) <= 4000:
            raise WorkerProtocolError("hypothesis 长度必须在 8～4000")
        object.__setattr__(self, "hypothesis", self.hypothesis.strip())
        object.__setattr__(
            self,
            "snapshot",
            _strict_mapping(
                self.snapshot,
                "snapshot",
                {
                    "context_hash",
                    "manifest_hash",
                    "start_date",
                    "end_date",
                    "fields",
                    "image_id",
                },
                maximum_bytes=8192,
            ),
        )
        if self.previous_attempt is not None:
            object.__setattr__(
                self,
                "previous_attempt",
                _strict_mapping(
                    self.previous_attempt,
                    "previous_attempt",
                    {
                        "iteration",
                        "code_hash",
                        "status",
                        "metrics",
                        "error_code",
                    },
                    maximum_bytes=16384,
                ),
            )
        if (
            isinstance(self.max_code_bytes, bool)
            or not isinstance(self.max_code_bytes, int)
            or not 1024 <= self.max_code_bytes <= 65536
        ):
            raise WorkerProtocolError("max_code_bytes 必须在 1024～65536")
        if not isinstance(self.llm_model, str) or not self.llm_model.strip():
            raise WorkerProtocolError("llm_model 不能为空")
        if self.prompt_version != CODE_PROMPT_VERSION:
            raise WorkerProtocolError("prompt_version 不匹配")
        ensure_agent_payload(self.to_dict(), path="worker_code_request")

    @classmethod
    def from_dict(cls, value: Any) -> WorkerCodeProposalRequest:
        """严格解析代码提案请求。"""

        return cls(
            **_strict_mapping(
                value,
                "worker_code_request",
                set(cls.__dataclass_fields__),
                maximum_bytes=131072,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """返回稳定请求字典。"""

        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "hypothesis": self.hypothesis,
            "snapshot": dict(self.snapshot),
            "previous_attempt": (
                dict(self.previous_attempt)
                if self.previous_attempt is not None
                else None
            ),
            "max_code_bytes": self.max_code_bytes,
            "llm_model": self.llm_model,
            "prompt_version": self.prompt_version,
        }

    @property
    def payload_hash(self) -> str:
        """返回请求稳定哈希。"""

        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @property
    def job_id(self) -> str:
        """返回稳定作业 ID。"""

        encoded = json.dumps(
            {"request_id": self.request_id},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"cjob_{hashlib.sha256(encoded).hexdigest()[:24]}"


@dataclass(frozen=True)
class WorkerCodeProposalResponse:
    """RD-Agent 返回的未信任 Qlib 研究代码。"""

    proposal_id: str
    code: str
    rationale: str
    model: str
    prompt_version: str = CODE_PROMPT_VERSION
    usage: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = WORKER_CODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_CODE_SCHEMA_VERSION:
            raise WorkerProtocolError(
                f"schema_version 必须为 {WORKER_CODE_SCHEMA_VERSION}"
            )
        if not isinstance(self.proposal_id, str) or not _IDENTIFIER.fullmatch(
            self.proposal_id
        ):
            raise WorkerProtocolError("proposal_id 包含非法字符")
        if (
            not isinstance(self.model, str)
            or not self.model.strip()
            or len(self.model) > 200
        ):
            raise WorkerProtocolError("model 必须是 1～200 字符")
        object.__setattr__(self, "model", self.model.strip())
        if not isinstance(self.code, str):
            raise WorkerProtocolError("code 必须是字符串")
        code = self.code.strip()
        if not 1 <= len(code.encode("utf-8")) <= 65536:
            raise WorkerProtocolError("code 长度必须在 1～65536 字节")
        object.__setattr__(self, "code", code)
        if not isinstance(self.rationale, str):
            raise WorkerProtocolError("rationale 必须是字符串")
        object.__setattr__(
            self,
            "rationale",
            self.rationale.strip()[:2000],
        )
        if self.prompt_version != CODE_PROMPT_VERSION:
            raise WorkerProtocolError("prompt_version 不匹配")
        if not isinstance(self.usage, Mapping):
            raise WorkerProtocolError("usage 必须是 JSON 对象")
        usage = dict(self.usage)
        if len(json.dumps(usage, ensure_ascii=False).encode("utf-8")) > 16000:
            raise WorkerProtocolError("usage 超过 16000 字节")
        object.__setattr__(self, "usage", usage)
        ensure_agent_payload(self.to_dict(), path="worker_code_response")

    @classmethod
    def from_dict(cls, value: Any) -> WorkerCodeProposalResponse:
        """严格解析代码提案响应。"""

        return cls(
            **_strict_mapping(
                value,
                "worker_code_response",
                set(cls.__dataclass_fields__),
                maximum_bytes=131072,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """返回稳定响应字典。"""

        return {
            "schema_version": self.schema_version,
            "proposal_id": self.proposal_id,
            "code": self.code,
            "rationale": self.rationale,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "usage": dict(self.usage),
        }


__all__ = [
    "CODE_PROMPT_VERSION",
    "WORKER_CODE_SCHEMA_VERSION",
    "WorkerCodeProposalRequest",
    "WorkerCodeProposalResponse",
]
