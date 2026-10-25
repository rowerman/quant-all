"""挖掘编排的默认约束和可恢复进度。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from common.paths import MINING_ORCHESTRATOR_FILE
from mining.contract_requests import MiningConstraints
from mining.contract_types import sha256_json
from mining.errors import MiningContractError


@dataclass(frozen=True)
class MiningProgress:
    """从候选库和事件恢复出的运行进度。"""

    run_id: str
    status: str
    current_stage: str
    rounds_requested: int
    rounds_completed: int
    llm_calls_used: int
    candidates_stored: int
    quality_checks_used: int
    full_evaluations_used: int
    failed_candidates: int
    holdout_claimed: bool
    budget: MiningConstraints

    def to_dict(self) -> dict[str, Any]:
        """返回适合接口展示的稳定结构。"""

        return {
            "run_id": self.run_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "rounds_requested": self.rounds_requested,
            "rounds_completed": self.rounds_completed,
            "llm_calls_used": self.llm_calls_used,
            "candidates_stored": self.candidates_stored,
            "quality_checks_used": self.quality_checks_used,
            "full_evaluations_used": self.full_evaluations_used,
            "failed_candidates": self.failed_candidates,
            "holdout_claimed": self.holdout_claimed,
            "budget": self.budget.to_dict(),
        }


def load_orchestration_constraints(
    path: Path | str = MINING_ORCHESTRATOR_FILE,
) -> MiningConstraints:
    """读取并校验编排默认预算。"""

    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MiningContractError(f"编排配置不存在：{target}") from exc
    except json.JSONDecodeError as exc:
        raise MiningContractError(f"编排配置 JSON 无法解析：{exc}") from exc
    if not isinstance(payload, dict):
        raise MiningContractError("编排配置根节点必须是对象")
    schema_version = payload.pop("schema_version", None)
    if schema_version != "mining_orchestrator_v1":
        raise MiningContractError("编排配置 schema_version 必须为 mining_orchestrator_v1")
    return MiningConstraints(**payload)


def orchestration_policy_hash(constraints: MiningConstraints) -> str:
    """返回冻结编排预算的稳定哈希。"""

    return sha256_json(
        {
            "schema_version": "mining_orchestrator_v1",
            "budget": constraints.to_dict(),
        }
    )
