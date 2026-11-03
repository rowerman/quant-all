"""挖掘日志路由和敏感字段过滤测试。"""

from __future__ import annotations

from common.paths import PROJECT_ROOT, log_root_for
from mining.logging_conf import format_event, safe_event_fields


def test_mining_logs_use_research_mining_domain():
    """所有挖掘子模块写入独立业务域日志目录。"""

    module_name = "mining.storage"
    assert log_root_for(module_name) == PROJECT_ROOT / "logs" / "research" / "mining"
    assert format_event({"run_id": "run_123", "candidate_id": "cand_456"}) == (
        "candidate_id='cand_456' run_id='run_123'"
    )


def test_sensitive_and_raw_research_text_is_not_logged():
    """Key、原始假设和公式不会进入状态日志。"""

    fields = safe_event_fields(
        {
            "run_id": "run_123",
            "api_key": "sk-secret-value-1234",
            "hypothesis": "用户的完整研究假设",
            "formula": "close-lag(close,5)",
            "message": "token=sk-secret-value-1234",
        }
    )
    assert fields == {"run_id": "run_123", "message": "token=[REDACTED]"}
