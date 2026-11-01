"""沙箱结构化反馈、脱敏和日志边界测试。"""

from __future__ import annotations

import pytest

from mining_sandbox.errors import SandboxError
from mining_sandbox.feedback import (
    SandboxCodeFeedback,
    SandboxFailureCategory,
    SandboxFeedbackError,
    SandboxFeedbackStage,
    sandbox_error_code,
    sandbox_feedback_stage,
    sanitize_feedback_text,
)
from mining_sandbox.logging_conf import log_sandbox_feedback


class CaptureLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def _feedback(
    category: SandboxFailureCategory = SandboxFailureCategory.EXECUTION,
) -> SandboxCodeFeedback:
    return SandboxCodeFeedback.create(
        category,
        summary="沙箱执行失败",
        repair_hint="修复运行时异常",
    )


def test_feedback_categories_have_unique_stable_codes_and_stages():
    codes = {sandbox_error_code(category) for category in SandboxFailureCategory}
    assert len(codes) == len(SandboxFailureCategory)
    assert codes == {f"MINING-{number}" for number in range(138, 146)}
    assert sandbox_feedback_stage(SandboxFailureCategory.COMPILE) is (
        SandboxFeedbackStage.COMPILE
    )
    assert sandbox_feedback_stage(SandboxFailureCategory.RESOURCE) is (
        SandboxFeedbackStage.EXECUTION
    )


def test_feedback_serialization_and_hash_are_strict_and_stable():
    feedback = _feedback()
    payload = feedback.to_dict()

    assert SandboxCodeFeedback.from_dict(payload) == feedback
    assert feedback.feedback_hash == SandboxCodeFeedback.from_dict(
        payload
    ).feedback_hash
    assert SandboxCodeFeedback.from_dict(
        {
            **payload,
            "summary": "不同的失败摘要",
        }
    ).feedback_hash != feedback.feedback_hash

    with pytest.raises(SandboxError):
        SandboxCodeFeedback.from_dict({**payload, "unknown": True})
    with pytest.raises(SandboxError):
        SandboxCodeFeedback.from_dict(
            {
                **payload,
                "error_code": "MINING-138",
            }
        )
    with pytest.raises(SandboxError):
        SandboxCodeFeedback.from_dict(
            {
                **payload,
                "stage": "compile",
            }
        )


def test_feedback_sanitizes_secrets_paths_and_long_text():
    text = sanitize_feedback_text(
        "api_key=tenant-secret "
        "token=sk-abcdefgh1234 "
        "Bearer abcdefghijk "
        "/home/hanwen/quant/data/private "
        "C:\\Users\\alice\\secret.py\n"
        + "x" * 800,
        known_secrets=("tenant-secret",),
    )

    assert "tenant-secret" not in text
    assert "sk-abcdefgh1234" not in text
    assert "abcdefghijk" not in text
    assert "/home/hanwen" not in text
    assert "C:\\Users" not in text
    assert "[REDACTED]" in text
    assert "[PATH]" in text
    assert len(text) == 500


def test_feedback_error_uses_feedback_code_and_safe_log_metadata():
    feedback = _feedback(SandboxFailureCategory.INDEX)
    error = SandboxFeedbackError(feedback)
    logger = CaptureLogger()

    log_sandbox_feedback(
        logger,
        run_id="sandboxrun_feedback_0001",
        iteration=2,
        code_hash="a" * 64,
        feedback=feedback,
    )

    assert error.code == "MINING-141"
    message = logger.messages[0]
    assert "MINING-141" in message
    assert "category='index'" in message
    assert "feedback_hash=" in message
    assert "修复运行时异常" not in message
    assert "沙箱执行失败" not in message
