"""在 Worker 子进程中调用 RD-Agent 的受控 JSON 生成后端。"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

from mining_worker.protocol import (
    WorkerCandidateDraft,
    WorkerGenerateRequest,
    WorkerGenerationResponse,
)
from mining_worker.worker_config import load_worker_config

_SECRET_PATTERN = re.compile(r"(?i)(sk-[A-Za-z0-9_-]{8,}|bearer\s+\S+)")


def _sanitize(message: str) -> str:
    result = _SECRET_PATTERN.sub("[REDACTED]", str(message))
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name, "")
        if value:
            result = result.replace(value, "[REDACTED]")
    return result[:500]


def _prompt(request: WorkerGenerateRequest) -> tuple[str, str]:
    system = (
        "你是量化因子研究员。只能使用用户给出的字段和算子生成候选因子。"
        "只返回 JSON 对象，不输出 SQL、Python、文件路径、命令或额外说明。"
        '{"candidates":[{"expression":"受控表达式","mechanism":"经济机制",'
        '"description":"候选说明","direction":1,"steps":["计算步骤"]}]}'
        "direction 只能是 1 或 -1；表达式必须是单一受控 DSL；不得引用未给出的字段或算子。"
    )
    user = json.dumps(
        {
            "hypothesis": request.hypothesis,
            "round_index": request.round_index,
            "max_candidates": request.max_candidates,
            "max_expression_depth": request.max_expression_depth,
            "direction": request.direction,
            "excluded_fields": list(request.excluded_fields),
            "fields": list(request.fields),
            "operators": list(request.operators),
            "feedback": list(request.feedback),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return system, user


def generate(request: WorkerGenerateRequest) -> WorkerGenerationResponse:
    """调用 RD-Agent LiteLLM 后端并解析严格候选 JSON。"""

    expected_model = os.environ.get("CHAT_MODEL", load_worker_config().default_model)
    if request.llm_model != expected_model:
        raise RuntimeError("请求模型与 Worker 固定模型不一致")
    from rdagent.log import rdagent_logger
    from rdagent.oai.llm_conf import LLM_SETTINGS
    from rdagent.oai.llm_utils import APIBackend

    class _DiscardStorage:
        """丢弃 RD-Agent trace，避免原始假设和模型响应落盘。"""

        def log(self, *args, **kwargs) -> None:
            return None

    rdagent_logger.storage = _DiscardStorage()
    LLM_SETTINGS.log_llm_chat_content = False
    LLM_SETTINGS.chat_stream = False
    LLM_SETTINGS.max_retry = 1
    system, user = _prompt(request)
    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user,
        system_prompt=system,
        json_mode=True,
    )
    payload: Any = json.loads(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError("RD-Agent 响应缺少 candidates 数组")
    candidates = tuple(
        WorkerCandidateDraft.from_dict(item) for item in payload["candidates"]
    )
    if len(candidates) > request.max_candidates:
        raise ValueError("RD-Agent 返回候选数超过本次预算")
    return WorkerGenerationResponse(
        candidates=candidates,
        usage={
            "calls": 1,
            "candidates": len(candidates),
            "mode": "rdagent",
            "model": expected_model,
        },
    )


def main() -> int:
    """读取一个请求，向 stdout 输出一行严格 JSON。"""

    try:
        request = WorkerGenerateRequest.from_dict(json.loads(sys.stdin.read()))
        response = generate(request)
        output = {"ok": True, "response": response.to_dict()}
    except Exception as exc:
        output = {
            "ok": False,
            "error_code": "MINING-118",
            "error_message": _sanitize(str(exc)),
        }
        print(json.dumps(output, ensure_ascii=False))
        return 1
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
