"""候选评估的输入、结果和质量规则契约。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from common.paths import MINING_EVALUATION_FILE
from mining.contract_candidates import CandidateDefinition
from mining.contract_evidence import EvaluationReport, TechnicalGate
from mining.contract_requests import FrozenContext
from mining.contract_types import (
    identifier,
    sha256_json,
    validate_date,
)
from mining.errors import MiningContractError


@dataclass(frozen=True)
class QualityPolicy:
    """版本化技术门禁阈值。"""

    schema_version: str
    forward_look_samples: int
    forward_look_relative_tolerance: float
    max_flat_day_share: float
    correlation_sample_days: int
    correlation_min_stocks: int
    correlation_min_days: int
    correlation_max_absolute: float
    quick_min_valid_days: int
    warning_rank_ic: float
    warning_icir: float
    warning_year_sign_ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            identifier(self.schema_version, "schema_version"),
        )
        for name, minimum, maximum in (
            ("forward_look_samples", 1, 20),
            ("correlation_sample_days", 20, 500),
            ("correlation_min_stocks", 5, 100),
            ("correlation_min_days", 20, 500),
            ("quick_min_valid_days", 20, 1000),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise MiningContractError(f"{name} 必须是 {minimum}～{maximum} 的整数")
        for name in (
            "forward_look_relative_tolerance",
            "max_flat_day_share",
            "correlation_max_absolute",
            "warning_rank_ic",
            "warning_icir",
            "warning_year_sign_ratio",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise MiningContractError(f"{name} 必须在 0～1")
        if not 0 <= self.max_flat_day_share < 1:
            raise MiningContractError("max_flat_day_share 必须小于 1")
        if not 0 < self.correlation_max_absolute <= 1:
            raise MiningContractError("correlation_max_absolute 必须在 (0, 1]")

    def material(self) -> dict[str, Any]:
        """返回参与输入哈希的完整门禁规则。"""

        return {
            "schema_version": self.schema_version,
            "forward_look_samples": self.forward_look_samples,
            "forward_look_relative_tolerance": self.forward_look_relative_tolerance,
            "max_flat_day_share": self.max_flat_day_share,
            "correlation_sample_days": self.correlation_sample_days,
            "correlation_min_stocks": self.correlation_min_stocks,
            "correlation_min_days": self.correlation_min_days,
            "correlation_max_absolute": self.correlation_max_absolute,
            "quick_min_valid_days": self.quick_min_valid_days,
            "warning_rank_ic": self.warning_rank_ic,
            "warning_icir": self.warning_icir,
            "warning_year_sign_ratio": self.warning_year_sign_ratio,
        }


def load_quality_policy(path: Path | str = MINING_EVALUATION_FILE) -> QualityPolicy:
    """读取并校验冻结质量门禁配置。"""

    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MiningContractError(f"质量门禁配置不存在：{target}") from exc
    except json.JSONDecodeError as exc:
        raise MiningContractError(f"质量门禁配置 JSON 无法解析：{exc}") from exc
    if not isinstance(payload, dict):
        raise MiningContractError("质量门禁配置根节点必须是对象")
    expected = {
        "schema_version",
        "forward_look_samples",
        "forward_look_relative_tolerance",
        "max_flat_day_share",
        "correlation_sample_days",
        "correlation_min_stocks",
        "correlation_min_days",
        "correlation_max_absolute",
        "quick_min_valid_days",
        "warning_rank_ic",
        "warning_icir",
        "warning_year_sign_ratio",
    }
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown or missing:
        raise MiningContractError(
            f"质量门禁配置字段不完整：缺少 {sorted(missing)}，未知 {sorted(unknown)}"
        )
    return QualityPolicy(**payload)


@dataclass(frozen=True)
class CandidateEvaluationInput:
    """一次单候选评估的完整输入。"""

    run_id: str
    candidate_id: str
    candidate: CandidateDefinition
    context: FrozenContext
    candidate_root: Path | str
    run_full: bool = False
    quick_window: tuple[str, str] | None = None
    full_window: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", identifier(self.run_id, "run_id"))
        object.__setattr__(self, "candidate_id", identifier(self.candidate_id, "candidate_id"))
        if not isinstance(self.candidate, CandidateDefinition):
            raise MiningContractError("candidate 必须是 CandidateDefinition")
        if not isinstance(self.context, FrozenContext):
            raise MiningContractError("context 必须是 FrozenContext")
        if self.candidate_id != self.candidate.candidate_id:
            raise MiningContractError("candidate_id 与候选内容哈希不一致")
        if not isinstance(self.run_full, bool):
            raise MiningContractError("run_full 必须是布尔值")
        object.__setattr__(self, "candidate_root", Path(self.candidate_root))
        quick = self.quick_window or (self.context.research_start, self.context.research_end)
        full = self.full_window or (self.context.full_eval_start, self.context.full_eval_end)
        object.__setattr__(self, "quick_window", _window(quick, "quick_window"))
        object.__setattr__(self, "full_window", _window(full, "full_window"))
        if not (
            self.context.research_start <= self.quick_window[0]
            <= self.quick_window[1] <= self.context.research_end
        ):
            raise MiningContractError("quick_window 必须落在研究期内")
        if not (
            self.context.full_eval_start <= self.full_window[0]
            <= self.full_window[1] <= self.context.full_eval_end
        ):
            raise MiningContractError("full_window 必须落在完整评估期内")

    @property
    def quick_params_material(self) -> dict[str, Any]:
        """快速筛选的稳定参数输入。"""

        return _params_material(self.context, self.quick_window)

    @property
    def full_params_material(self) -> dict[str, Any]:
        """完整评估的稳定参数输入。"""

        return _params_material(self.context, self.full_window)


def _window(value: tuple[str, str], name: str) -> tuple[str, str]:
    """校验升序日期窗口。"""

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise MiningContractError(f"{name} 必须是两个日期")
    start = validate_date(value[0], f"{name}.start")
    end = validate_date(value[1], f"{name}.end")
    if start > end:
        raise MiningContractError(f"{name} 起始日不能晚于截止日")
    return start, end


def _params_material(context: FrozenContext, window: tuple[str, str]) -> dict[str, Any]:
    """返回影响评估结果的冻结参数。"""

    return {
        "start": window[0],
        "end": window[1],
        "universe": context.universe,
        "label": context.label,
        "price_channel": context.price_channel,
        "horizon": 5,
        "quantiles": 10,
        "winsorize": True,
        "data_version": context.data_version,
        "data_fingerprint": context.data_fingerprint,
        "min_coverage": context.budget.min_coverage,
    }


def stage_input_hash(
    *,
    stage: str,
    evaluation_input: CandidateEvaluationInput,
    factor_sha256: str,
    params: dict[str, Any],
    policy: QualityPolicy,
) -> str:
    """计算阶段报告的稳定输入哈希。"""

    return sha256_json(
        {
            "schema_version": "mining_evaluation_input_v1",
            "stage": stage,
            "run_id": evaluation_input.run_id,
            "candidate_id": evaluation_input.candidate_id,
            "candidate_content_hash": evaluation_input.candidate.content_hash,
            "context_hash": evaluation_input.context.context_hash,
            "factor_sha256": factor_sha256,
            "params": params,
            "quality_policy": policy.material(),
        }
    )


@dataclass(frozen=True)
class CandidateEvaluationResult:
    """S5 返回给编排层的完整评估结果。"""

    gate: TechnicalGate
    technical: EvaluationReport
    quick: EvaluationReport | None
    full: EvaluationReport | None
    warnings: tuple[str, ...]
    factor_glob: str
    factor_sha256: str


@dataclass(frozen=True)
class FullEvaluationResult:
    """研究期完整评估的单阶段结果。"""

    report: EvaluationReport
    warnings: tuple[str, ...]
    factor_glob: str
    factor_sha256: str


@dataclass(frozen=True)
class HoldoutEvaluationResult:
    """模块测试段评估结果。"""

    report: EvaluationReport
    passed: bool
    warnings: tuple[str, ...]
    factor_glob: str
    factor_sha256: str
