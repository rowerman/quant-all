"""逐日 RankIC 的 Newey-West 检验和 BH 多重检验诊断。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Any, Iterable, Mapping

from mining.contract_types import sha256_json
from mining.search_evaluation_evidence import SearchEvaluationEvidence

MULTIPLE_TESTING_STAGE = "multiple_testing"
MULTIPLE_TESTING_SCHEMA_VERSION = "mining_multiple_testing_v1"
MULTIPLE_TESTING_MIN_DAYS = 20


@dataclass(frozen=True)
class NeweyWestResult:
    """单个候选逐日 RankIC 均值的 HAC 检验。"""

    candidate_id: str
    status: str
    valid_days: int
    mean_rank_ic: float | None
    standard_error: float | None
    t_stat: float | None
    p_value: float | None
    bh_q_value: float | None
    lag: int | None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "valid_days": self.valid_days,
            "mean_rank_ic": self.mean_rank_ic,
            "standard_error": self.standard_error,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "bh_q_value": self.bh_q_value,
            "lag": self.lag,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MultipleTestingReport:
    """运行内搜索规模和多重检验诊断。"""

    run_id: str
    attempt_count: int
    evaluated_count: int
    tested_count: int
    denominator: int
    candidates: tuple[NeweyWestResult, ...]
    warnings: tuple[str, ...]
    selection_rule: str = (
        "Newey-West HAC 双侧 p 值进行 BH 调整；q-value 不参与候选排序、"
        "审批或发布"
    )
    schema_version: str = MULTIPLE_TESTING_SCHEMA_VERSION

    def material(self) -> dict[str, Any]:
        """返回稳定哈希输入。"""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "method": "newey_west_bartlett_bh",
            "bandwidth_rule": "floor(4 * (n / 100) ** (2 / 9))",
            "min_days": MULTIPLE_TESTING_MIN_DAYS,
            "attempt_count": self.attempt_count,
            "evaluated_count": self.evaluated_count,
            "tested_count": self.tested_count,
            "denominator": self.denominator,
            "candidates": [item.to_dict() for item in self.candidates],
            "warnings": list(self.warnings),
            "selection_rule": self.selection_rule,
        }

    @property
    def evidence_hash(self) -> str:
        """返回统计报告哈希。"""

        return sha256_json(self.material())

    def to_dict(self) -> dict[str, Any]:
        """返回持久化字典。"""

        return {**self.material(), "evidence_hash": self.evidence_hash}

    def to_evaluation_evidence(
        self,
        candidate_id: str,
    ) -> SearchEvaluationEvidence:
        """把运行报告登记到指定候选。"""

        return SearchEvaluationEvidence(
            run_id=self.run_id,
            candidate_id=candidate_id,
            stage=MULTIPLE_TESTING_STAGE,
            input_hash=self.evidence_hash,
            status="available" if self.tested_count else "not_computable",
            metrics=self.to_dict(),
            warnings=self.warnings,
        )


def _hac(
    values: Iterable[float],
) -> tuple[float, float, float, int] | None:
    """返回均值、HAC 标准误、t 值和带宽。"""

    sample = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    count = len(sample)
    if count < MULTIPLE_TESTING_MIN_DAYS:
        return None
    mean = sum(sample) / count
    centered = [value - mean for value in sample]
    lag = min(
        count - 1,
        int(math.floor(4 * (count / 100) ** (2 / 9))),
    )
    variance = sum(value * value for value in centered) / count
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1)
        covariance = sum(
            centered[index] * centered[index - offset]
            for index in range(offset, count)
        ) / count
        variance += 2.0 * weight * covariance
    variance = max(0.0, variance)
    standard_error = math.sqrt(variance / count)
    if standard_error == 0:
        return None
    t_stat = mean / standard_error
    return mean, standard_error, t_stat, lag


def _bh_q_values(
    p_values: Mapping[str, float],
    *,
    denominator: int,
) -> dict[str, float]:
    """按给定保守分母计算稳定 BH q-value。"""

    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 1.0
    count = len(ordered)
    for reverse_index in range(count - 1, -1, -1):
        candidate_id, p_value = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, p_value * max(denominator, count) / rank)
        adjusted[candidate_id] = min(1.0, running)
    return adjusted


def build_multiple_testing_report(
    *,
    run_id: str,
    series: Mapping[str, tuple[tuple[str, ...], tuple[float, ...]]],
    attempt_count: int,
    evaluated_count: int,
) -> MultipleTestingReport:
    """生成运行级 Newey-West 和 BH 诊断。"""

    raw: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {}
    normal = NormalDist()
    for candidate_id in sorted(series):
        dates, values = series[candidate_id]
        result = _hac(values)
        if result is None:
            raw[candidate_id] = {
                "status": "not_computable",
                "valid_days": len(values),
                "reason": (
                    "有效日不足"
                    if len(values) < MULTIPLE_TESTING_MIN_DAYS
                    else "RankIC 标准差为零"
                ),
            }
            continue
        mean, standard_error, t_stat, lag = result
        p_value = min(1.0, 2.0 * (1.0 - normal.cdf(abs(t_stat))))
        p_values[candidate_id] = p_value
        raw[candidate_id] = {
            "status": "available",
            "valid_days": len(values),
            "mean_rank_ic": mean,
            "standard_error": standard_error,
            "t_stat": t_stat,
            "p_value": p_value,
            "lag": lag,
        }
    denominator = max(int(attempt_count), len(p_values), 1)
    q_values = _bh_q_values(p_values, denominator=denominator)
    candidates = tuple(
        NeweyWestResult(
            candidate_id=candidate_id,
            status=str(item["status"]),
            valid_days=int(item["valid_days"]),
            mean_rank_ic=item.get("mean_rank_ic"),
            standard_error=item.get("standard_error"),
            t_stat=item.get("t_stat"),
            p_value=item.get("p_value"),
            bh_q_value=q_values.get(candidate_id),
            lag=item.get("lag"),
            reason=item.get("reason"),
        )
        for candidate_id, item in sorted(raw.items())
    )
    warnings = (
        "多重检验提示只反映当前搜索规模，不能替代人工研究判断",
        f"本次报告按 {denominator} 个尝试的保守分母计算 BH q-value",
    )
    return MultipleTestingReport(
        run_id=run_id,
        attempt_count=max(0, int(attempt_count)),
        evaluated_count=max(0, int(evaluated_count)),
        tested_count=len(p_values),
        denominator=denominator,
        candidates=candidates,
        warnings=warnings,
    )


__all__ = [
    "MULTIPLE_TESTING_MIN_DAYS",
    "MULTIPLE_TESTING_SCHEMA_VERSION",
    "MULTIPLE_TESTING_STAGE",
    "MultipleTestingReport",
    "NeweyWestResult",
    "build_multiple_testing_report",
]
