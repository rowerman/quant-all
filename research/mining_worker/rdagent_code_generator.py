"""使用 RD-Agent 生成在 Qlib 沙箱中运行的代码研究提案。"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from mining_worker.generator import _sanitize
from mining_worker.protocol_code import (
    WorkerCodeProposalRequest,
    WorkerCodeProposalResponse,
)
from mining_worker.worker_config import load_worker_config


def _prompt(request: WorkerCodeProposalRequest) -> tuple[str, str]:
    system = (
        "你是量化因子代码研究员。生成一段可独立运行的 Python 3.11 研究脚本，"
        "使用 Qlib 0.9.7 从只读目录 /snapshot 读取数据。代码不能联网，不能读取其他"
        "路径，不能启动 shell，不能读取环境变量或密钥。脚本必须把有效因子值写到 "
        "/workspace/factor.parquet，列只能是 date、instrument、value；date 和 "
        "instrument 必须属于 /snapshot 冻结索引，value 必须有限，缺失值直接省略行。"
        "脚本还必须把汇总写到 /workspace/metrics.json，JSON 只包含 rank_ic、icir、"
        "coverage、valid_days；coverage 等于有限因子行数除以快照总行数，valid_days "
        "等于 factor.parquet 中不同交易日的数量。"
        "迭代请求包含当前运行的历史知识证据时，必须优先依据上一轮结构化反馈中的"
        " error_code、summary 和 repair_hint 修正代码，并避免重复已记录的失败；"
        "成功和修复证据只用于理解已验证路径，不能扩大数据窗口或绕过宿主评估。"
        "输出严格 JSON，字段为 proposal_id、code、rationale、model、prompt_version、"
        "usage。只能研究开发期冻结数据，不能把结果表述为正式候选或审批结论。"
    )
    user = json.dumps(
        {
            "run_id": request.run_id,
            "iteration": request.iteration,
            "task_hash": request.task_hash,
            "hypothesis": request.hypothesis,
            "snapshot": dict(request.snapshot),
            "previous_attempt": (
                dict(request.previous_attempt)
                if request.previous_attempt is not None
                else None
            ),
            "knowledge_evidence": (
                request.knowledge_evidence.to_dict()
                if request.knowledge_evidence is not None
                else None
            ),
            "model": request.llm_model,
            "prompt_version": request.prompt_version,
            "max_code_bytes": request.max_code_bytes,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return system, user


def generate(request: WorkerCodeProposalRequest) -> WorkerCodeProposalResponse:
    """调用固定 RD-Agent 模型并严格解析代码提案。"""

    expected_model = os.environ.get(
        "CHAT_MODEL",
        load_worker_config().default_model,
    )
    if request.llm_model != expected_model:
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
    raw = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user,
        system_prompt=system,
        json_mode=True,
    )
    payload: Any = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("RD-Agent 代码响应根节点不是 JSON 对象")
    expected = {
        "proposal_id",
        "code",
        "rationale",
        "model",
        "prompt_version",
        "usage",
    }
    if set(payload) != expected:
        raise ValueError("RD-Agent 代码响应字段不符合契约")
    if payload["model"] != expected_model:
        raise ValueError("RD-Agent 返回模型与固定模型不一致")
    if payload["prompt_version"] != request.prompt_version:
        raise ValueError("RD-Agent 返回提示版本不匹配")
    response = WorkerCodeProposalResponse.from_dict(payload)
    if len(response.code.encode("utf-8")) > request.max_code_bytes:
        raise ValueError("RD-Agent 代码超过本轮限制")
    return response


def main() -> int:
    """读取一个代码提案请求并向 stdout 输出严格 JSON。"""

    try:
        request = WorkerCodeProposalRequest.from_dict(
            json.loads(sys.stdin.read())
        )
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
