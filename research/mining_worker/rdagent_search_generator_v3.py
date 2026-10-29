"""在 Worker 子进程中执行工具感知的 RD-Agent 搜索建议。"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from mining_worker.generator import _sanitize
from mining_worker.protocol_v3 import (
    WorkerAnalysisToolCall,
    WorkerSearchTurnRequest,
    WorkerSearchTurnResponse,
)
from mining_worker.protocol_v2 import WorkerSearchResponse
from mining_worker.worker_config import load_worker_config


def _prompt(request: WorkerSearchTurnRequest) -> tuple[str, str]:
    base = request.base_request
    if request.phase == "discover":
        system = (
            "你是量化因子研究员。只能使用请求提供的字段、算子和固定分析工具。"
            "如需额外证据，只返回 JSON 工具批次；不需要工具时直接返回最终搜索响应。"
            '工具批次格式为 {"kind":"tool_batch","tool_calls":['
            '{"call_id":"c1","tool_name":"工具名","arguments":{}}]}。'
            "工具调用最多 4 个，不允许 SQL、Python、路径、日期切片或任意代码。"
            '最终响应格式为 {"kind":"decision","response":{'
            '"schema_version":"mining_worker_v2","decision":{"action":'
            '"refine|branch|stop|promote","reason":"理由"},'
            '"next_hypothesis":null或假设对象,"candidates":[]'
            ",\"usage\":{}}}。refine 或 branch 必须提供下一假设和至少一个受控 DSL "
            "候选；stop 或 promote 不提供候选。"
        )
    else:
        system = (
            "你是量化因子研究员。工具结果仅来自开发期且只是补充证据。"
            "现在必须返回最终 JSON，不能再请求工具。"
            '格式为 {"kind":"decision","response":{"schema_version":'
            '"mining_worker_v2","decision":{"action":"refine|branch|stop|promote",'
            '"reason":"理由"},"next_hypothesis":null或假设对象,'
            '"candidates":[],"usage":{}}}。refine 或 branch 必须提供下一假设和至少'
            "一个受控 DSL 候选；不得输出 SQL、Python、路径或非开发期信息。"
        )
    user = json.dumps(
        {
            "phase": request.phase,
            "hypothesis": base.hypothesis.to_dict(),
            "lineage": [item.to_dict() for item in base.lineage],
            "evidence": (
                dict(base.evidence) if base.evidence is not None else None
            ),
            "budget": dict(base.budget),
            "allowed_actions": list(base.allowed_actions),
            "max_candidates": base.max_candidates,
            "max_expression_depth": base.max_expression_depth,
            "excluded_fields": list(base.excluded_fields),
            "fields": list(base.fields),
            "operators": list(base.operators),
            "tool_catalog": [dict(item) for item in request.tool_catalog],
            "tool_results": [dict(item) for item in request.tool_results],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return system, user


def _parse_final(payload: MappingLike) -> WorkerSearchResponse:
    raw = payload.get("response")
    if not isinstance(raw, dict):
        raise ValueError("最终响应缺少 response 对象")
    allowed = {"schema_version", "decision", "next_hypothesis", "candidates", "usage"}
    if set(raw) != allowed:
        raise ValueError("最终搜索响应字段不符合 Worker v2 契约")
    return WorkerSearchResponse.from_dict(raw)


MappingLike = dict[str, Any]


def generate(request: WorkerSearchTurnRequest) -> WorkerSearchTurnResponse:
    """调用固定模型并严格解析工具批次或最终搜索响应。"""

    expected_model = os.environ.get(
        "CHAT_MODEL",
        load_worker_config().default_model,
    )
    if request.base_request.llm_model != expected_model:
        raise RuntimeError("请求模型与 Worker 固定模型不一致")
    from rdagent.log import rdagent_logger
    from rdagent.oai.llm_conf import LLM_SETTINGS
    from rdagent.oai.llm_utils import APIBackend

    class _DiscardStorage:
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
    payload = json.loads(response)
    if not isinstance(payload, dict):
        raise ValueError("RD-Agent 响应根节点不是 JSON 对象")
    kind = payload.get("kind")
    if kind == "tool_batch":
        expected = {"kind", "tool_calls"}
        if request.phase != "discover" or set(payload) != expected:
            raise ValueError("工具批次响应字段或阶段非法")
        raw_calls = payload["tool_calls"]
        if not isinstance(raw_calls, list):
            raise ValueError("tool_calls 必须是数组")
        calls = tuple(
            WorkerAnalysisToolCall.from_dict(item) for item in raw_calls
        )
        return WorkerSearchTurnResponse(
            request_id=request.request_id,
            kind="tool_batch",
            tool_calls=calls,
            usage={"calls": 1, "mode": "rdagent", "model": expected_model},
        )
    if kind == "decision":
        expected = {"kind", "response"}
        if set(payload) != expected:
            raise ValueError("最终响应字段非法")
        final = _parse_final(payload)
        if len(final.candidates) > request.base_request.max_candidates:
            raise ValueError("RD-Agent 返回候选数超过本次预算")
        return WorkerSearchTurnResponse(
            request_id=request.request_id,
            kind="decision",
            response=final,
            usage={
                "calls": 1,
                "candidates": len(final.candidates),
                "mode": "rdagent",
                "model": expected_model,
            },
        )
    raise ValueError("RD-Agent 响应的 kind 必须是 tool_batch 或 decision")


def main() -> int:
    """读取一个 v3 请求并向 stdout 输出严格 JSON。"""

    try:
        request = WorkerSearchTurnRequest.from_dict(json.loads(sys.stdin.read()))
        response = generate(request)
        output = {"ok": True, "response": response.to_dict()}
    except Exception as exc:
        output = {
            "ok": False,
            "error_code": "MINING-118",
            "error_message": _sanitize(f"{type(exc).__name__}: {exc}"),
        }
        print(json.dumps(output, ensure_ascii=False))
        return 1
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
