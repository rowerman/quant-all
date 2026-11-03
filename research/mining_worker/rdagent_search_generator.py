"""在 Worker 子进程中调用 RD-Agent 生成搜索建议。"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from mining_worker.generator import _sanitize
from mining_worker.protocol_v2 import (
    WorkerDecisionProposal,
    WorkerSearchHypothesis,
    WorkerSearchRequest,
    WorkerSearchResponse,
)
from mining_worker.worker_config import load_worker_config


def _prompt(request: WorkerSearchRequest) -> tuple[str, str]:
    system = (
        "你是量化因子研究员。只能根据开发期证据判断下一步，并只返回 JSON 对象。"
        "不得输出 SQL、Python、路径、命令、选择期或测试段信息。"
        '输出格式为 {"decision":{"action":"refine|branch|stop|promote",'
        '"reason":"理由"},"next_hypothesis":null或假设对象,'
        '"candidates":[]}。refine 或 branch 必须提供下一假设和至少一个受控 DSL '
        "候选；stop 或 promote 的 next_hypothesis 必须为 null 且 candidates 为空。"
        "假设对象字段为 statement、mechanism、expected_direction、"
        "falsification_criteria、parent_hypothesis_id。候选字段为 expression、"
        "mechanism、description、direction、steps。"
    )
    user = json.dumps(
        {
            "hypothesis": request.hypothesis.to_dict(),
            "lineage": [item.to_dict() for item in request.lineage],
            "evidence": (
                dict(request.evidence) if request.evidence is not None else None
            ),
            "budget": dict(request.budget),
            "allowed_actions": list(request.allowed_actions),
            "max_candidates": request.max_candidates,
            "max_expression_depth": request.max_expression_depth,
            "excluded_fields": list(request.excluded_fields),
            "fields": list(request.fields),
            "operators": list(request.operators),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return system, user


def generate(request: WorkerSearchRequest) -> WorkerSearchResponse:
    """调用固定 RD-Agent 模型并解析严格搜索响应。"""

    expected_model = os.environ.get("CHAT_MODEL", load_worker_config().default_model)
    if request.llm_model != expected_model:
        raise RuntimeError("请求模型与 Worker 固定模型不一致")
    from rdagent.log import rdagent_logger
    from rdagent.oai.llm_conf import LLM_SETTINGS
    from rdagent.oai.llm_utils import APIBackend

    class _DiscardStorage:
        """丢弃 RD-Agent trace，避免原始假设和响应落盘。"""

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
    if not isinstance(payload, dict):
        raise ValueError("RD-Agent 响应根节点不是 JSON 对象")
    expected = {"decision", "next_hypothesis", "candidates"}
    if set(payload) != expected:
        raise ValueError("RD-Agent 响应字段不符合 Worker v2 契约")
    decision = WorkerDecisionProposal.from_dict(payload["decision"])
    if decision.action not in request.allowed_actions:
        raise ValueError("RD-Agent 返回了本轮不允许的搜索动作")
    next_hypothesis = (
        WorkerSearchHypothesis.from_dict(payload["next_hypothesis"])
        if payload["next_hypothesis"] is not None
        else None
    )
    raw_candidates = payload["candidates"]
    if not isinstance(raw_candidates, list):
        raise ValueError("RD-Agent 候选必须是数组")
    candidates = WorkerSearchResponse.from_dict(
        {
            "schema_version": request.schema_version,
            "decision": decision.to_dict(),
            "next_hypothesis": (
                next_hypothesis.to_dict()
                if next_hypothesis is not None
                else None
            ),
            "candidates": raw_candidates,
            "usage": {},
        }
    )
    if len(candidates.candidates) > request.max_candidates:
        raise ValueError("RD-Agent 返回候选数超过本次预算")
    return WorkerSearchResponse(
        decision=candidates.decision,
        next_hypothesis=candidates.next_hypothesis,
        candidates=candidates.candidates,
        usage={
            "calls": 1,
            "candidates": len(candidates.candidates),
            "mode": "rdagent",
            "model": expected_model,
        },
    )


def main() -> int:
    """读取一个搜索请求，向 stdout 输出一行严格 JSON。"""

    try:
        request = WorkerSearchRequest.from_dict(json.loads(sys.stdin.read()))
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
