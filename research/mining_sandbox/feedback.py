"""沙箱代码研究的结构化失败反馈。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Iterable, Mapping

from mining.contract_types import sha256_json
from mining_sandbox.errors import SandboxError

SANDBOX_FEEDBACK_SCHEMA_VERSION = "mining_sandbox_feedback_v1"

_SECRET_PATTERN = re.compile(
    r"(?i)(sk-[A-Za-z0-9_-]{8,}|bearer\s+[A-Za-z0-9._~+/=-]{8,})"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|authorization|credential|password|secret|token)"
    r"\s*[:=]\s*(['\"]?)[^,\s'\"]+\2"
)
_POSIX_PATH = re.compile(r"(?<![\w.-])(?:/[A-Za-z0-9._-]+){2,}")
_WINDOWS_PATH = re.compile(
    r"(?i)(?<![\w.-])[A-Z]:\\(?:[^\\\s]+\\)*[^\\\s]+"
)
_WHITESPACE = re.compile(r"\s+")


class SandboxFailureCategory(str, Enum):
    """可供下一轮代码修订使用的失败类型。"""

    COMPILE = "compile"
    EXECUTION = "execution"
    OUTPUT_FORMAT = "output_format"
    INDEX = "index"
    COVERAGE = "coverage"
    NUMERIC = "numeric"
    CODE_POLICY = "code_policy"
    RESOURCE = "resource"


class SandboxFeedbackStage(str, Enum):
    """失败反馈适用的研究阶段。"""

    COMPILE = "compile"
    EXECUTION = "execution"
    OUTPUT = "output"
    EVALUATION = "evaluation"


_CATEGORY_RULES = {
    SandboxFailureCategory.COMPILE: ("MINING-138", SandboxFeedbackStage.COMPILE),
    SandboxFailureCategory.EXECUTION: (
        "MINING-139",
        SandboxFeedbackStage.EXECUTION,
    ),
    SandboxFailureCategory.OUTPUT_FORMAT: (
        "MINING-140",
        SandboxFeedbackStage.OUTPUT,
    ),
    SandboxFailureCategory.INDEX: (
        "MINING-141",
        SandboxFeedbackStage.EVALUATION,
    ),
    SandboxFailureCategory.COVERAGE: (
        "MINING-142",
        SandboxFeedbackStage.EVALUATION,
    ),
    SandboxFailureCategory.NUMERIC: (
        "MINING-143",
        SandboxFeedbackStage.EVALUATION,
    ),
    SandboxFailureCategory.CODE_POLICY: (
        "MINING-144",
        SandboxFeedbackStage.EVALUATION,
    ),
    SandboxFailureCategory.RESOURCE: (
        "MINING-145",
        SandboxFeedbackStage.EXECUTION,
    ),
}


def _coerce_category(
    category: SandboxFailureCategory | str,
) -> SandboxFailureCategory:
    try:
        return SandboxFailureCategory(category)
    except (TypeError, ValueError) as exc:
        raise SandboxError("沙箱反馈分类非法") from exc


def sandbox_error_code(category: SandboxFailureCategory | str) -> str:
    """返回失败分类对应的稳定错误码。"""

    return _CATEGORY_RULES[_coerce_category(category)][0]


def sandbox_feedback_stage(
    category: SandboxFailureCategory | str,
) -> SandboxFeedbackStage:
    """返回失败分类默认所属的研究阶段。"""

    return _CATEGORY_RULES[_coerce_category(category)][1]


def sanitize_feedback_text(
    value: Any,
    *,
    maximum: int = 500,
    known_secrets: Iterable[str] = (),
) -> str:
    """移除密钥、主机路径和多余空白，返回可持久化的短文本。"""

    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise SandboxError("反馈文本上限必须是正整数")
    result = str(value)
    for secret in sorted(
        {item for item in known_secrets if isinstance(item, str) and len(item) >= 4},
        key=len,
        reverse=True,
    ):
        result = result.replace(secret, "[REDACTED]")
    result = _SECRET_PATTERN.sub("[REDACTED]", result)
    result = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", result)
    result = _WINDOWS_PATH.sub("[PATH]", result)
    result = _POSIX_PATH.sub("[PATH]", result)
    result = _WHITESPACE.sub(" ", result).strip()
    return result[:maximum]


@dataclass(frozen=True)
class SandboxCodeFeedback:
    """一次代码研究失败的可审计、可反馈摘要。"""

    error_code: str
    category: SandboxFailureCategory
    stage: SandboxFeedbackStage
    summary: str
    repair_hint: str
    retryable: bool
    schema_version: str = SANDBOX_FEEDBACK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SANDBOX_FEEDBACK_SCHEMA_VERSION:
            raise SandboxError("沙箱反馈版本不匹配")
        category = _coerce_category(self.category)
        try:
            stage = SandboxFeedbackStage(self.stage)
        except (TypeError, ValueError) as exc:
            raise SandboxError("沙箱反馈阶段非法") from exc
        expected_code, expected_stage = _CATEGORY_RULES[category]
        if self.error_code != expected_code:
            raise SandboxError("沙箱反馈错误码与分类不一致")
        if stage != expected_stage:
            raise SandboxError("沙箱反馈阶段与分类不一致")
        if not isinstance(self.retryable, bool):
            raise SandboxError("retryable 必须是布尔值")
        summary = sanitize_feedback_text(self.summary)
        repair_hint = sanitize_feedback_text(self.repair_hint)
        if not summary or not repair_hint:
            raise SandboxError("沙箱反馈摘要和修复提示不能为空")
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "repair_hint", repair_hint)

    @classmethod
    def create(
        cls,
        category: SandboxFailureCategory,
        *,
        summary: str,
        repair_hint: str,
        retryable: bool = True,
        known_secrets: Iterable[str] = (),
    ) -> SandboxCodeFeedback:
        """按分类创建反馈，并统一执行脱敏。"""

        selected = _coerce_category(category)
        return cls(
            error_code=sandbox_error_code(selected),
            category=selected,
            stage=sandbox_feedback_stage(selected),
            summary=sanitize_feedback_text(
                summary,
                known_secrets=known_secrets,
            ),
            repair_hint=sanitize_feedback_text(
                repair_hint,
                known_secrets=known_secrets,
            ),
            retryable=retryable,
        )

    @classmethod
    def from_dict(cls, value: Any) -> SandboxCodeFeedback:
        """严格解析持久化反馈。"""

        if not isinstance(value, Mapping):
            raise SandboxError("沙箱反馈必须是 JSON 对象")
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise SandboxError("沙箱反馈字段不完整或包含未知字段")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        """返回稳定持久化字典。"""

        return {
            "schema_version": self.schema_version,
            "error_code": self.error_code,
            "category": self.category.value,
            "stage": self.stage.value,
            "summary": self.summary,
            "repair_hint": self.repair_hint,
            "retryable": self.retryable,
        }

    @property
    def feedback_hash(self) -> str:
        """返回反馈内容稳定哈希。"""

        return sha256_json(self.to_dict())


class SandboxFeedbackError(SandboxError):
    """携带结构化反馈的沙箱失败。"""

    def __init__(self, feedback: SandboxCodeFeedback) -> None:
        if not isinstance(feedback, SandboxCodeFeedback):
            raise SandboxError("feedback 必须是 SandboxCodeFeedback")
        self.feedback = feedback
        self.code = feedback.error_code
        super().__init__(feedback.summary)


__all__ = [
    "SANDBOX_FEEDBACK_SCHEMA_VERSION",
    "SandboxCodeFeedback",
    "SandboxFailureCategory",
    "SandboxFeedbackError",
    "SandboxFeedbackStage",
    "sandbox_error_code",
    "sandbox_feedback_stage",
    "sanitize_feedback_text",
]
