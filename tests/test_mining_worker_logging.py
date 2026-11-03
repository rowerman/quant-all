"""Worker 事件日志脱敏测试。"""

from __future__ import annotations

from mining_worker.logging_conf import log_worker_event


class CaptureLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def test_worker_event_log_drops_raw_research_text():
    logger = CaptureLogger()

    log_worker_event(
        logger,
        "worker_job_created",
        job_id="wjob_123",
        hypothesis="不应进入日志的完整研究假设",
        expression="close",
        prompt="原始 Prompt",
        round_index=1,
    )

    message = logger.messages[0]
    assert "wjob_123" in message
    assert "round_index=1" in message
    assert "研究假设" not in message
    assert "Prompt" not in message
