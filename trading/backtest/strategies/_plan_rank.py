"""读取通用评估候选分数的显式调仓目标权重提供者。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.artifacts import file_sha256
from alpha.config import EVALUATION_RUNS_DIR
from alpha.evaluation_plan import EvaluationPlan
from alpha.portfolio import HoldingPolicy, select_holdings
from backtest.data import MarketData
from backtest.errors import DataAlignmentError, SpecError
from backtest.providers import register_provider


def _rebalance_dates(value) -> tuple[str, ...]:
    """校验显式调仓日并返回排序后的 ISO 日期。"""

    if not isinstance(value, (list, tuple)) or not value:
        raise SpecError("plan_rank.params.rebalance_dates 必须是非空日期序列")
    normalized: list[str] = []
    for item in value:
        try:
            if (
                not isinstance(item, str)
                or date.fromisoformat(item).isoformat() != item
            ):
                raise ValueError
        except (TypeError, ValueError) as error:
            raise SpecError(
                f"plan_rank.params.rebalance_dates 日期必须是 YYYY-MM-DD：{item!r}"
            ) from error
        normalized.append(item)
    return tuple(sorted(dict.fromkeys(normalized)))


class PlanRankProvider:
    """按评估 ID 和候选名读取不可变分数并生成目标权重。"""

    name = "plan_rank"
    version = "1"
    description = "通用评估计划候选的显式日期 Top-N 或 rank-buffer 目标"
    required_factors: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._signal_history: dict[tuple[str, str], dict] = {}

    @staticmethod
    def validate_params(params: dict) -> dict:
        """校验评估、候选和调仓日参数。"""

        unknown = set(params) - {
            "evaluation_id",
            "candidate",
            "rebalance_dates",
        }
        if unknown:
            raise SpecError(
                f"plan_rank.params 包含未知字段：{sorted(unknown)}"
            )
        evaluation_id = params.get("evaluation_id")
        candidate = params.get("candidate")
        for name, value in (
            ("evaluation_id", evaluation_id),
            ("candidate", candidate),
        ):
            if not isinstance(value, str) or not value.strip():
                raise SpecError(f"plan_rank.params.{name} 必须是非空字符串")
            if Path(value).name != value:
                raise SpecError(f"plan_rank.params.{name} 不能包含路径")
        dates = _rebalance_dates(params.get("rebalance_dates"))
        return {
            "evaluation_id": evaluation_id.strip(),
            "candidate": candidate.strip(),
            "rebalance_dates": list(dates),
        }

    @staticmethod
    def _load(
        evaluation_id: str,
        candidate_name: str,
        *,
        root: Path | None = None,
    ) -> tuple[Path, dict, EvaluationPlan, pd.DataFrame]:
        """读取评估 manifest、计划、候选规则和标准分数。"""

        folder = Path(root or EVALUATION_RUNS_DIR) / evaluation_id
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            raise DataAlignmentError(f"plan_rank 缺少评估 manifest：{folder}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("analysis_id") != evaluation_id
            or manifest.get("status") != "pass_evaluation"
        ):
            raise DataAlignmentError("plan_rank 评估状态或 ID 不一致")
        plan = EvaluationPlan.from_dict(manifest.get("plan") or {})
        plan.by_name(candidate_name)
        relative = f"scores/{candidate_name}.parquet"
        score_path = folder / relative
        declared = (manifest.get("files") or {}).get(relative)
        if not score_path.is_file() or not declared:
            raise DataAlignmentError(f"plan_rank 缺少候选分数：{score_path}")
        if file_sha256(score_path) != str(declared):
            raise DataAlignmentError(f"plan_rank 候选分数哈希不一致：{score_path}")
        scores = pd.read_parquet(score_path)
        required = {"trade_date", "stock_code", "score"}
        missing = sorted(required - set(scores.columns))
        if missing:
            raise DataAlignmentError(
                f"plan_rank 候选分数缺少列：{missing}"
            )
        return folder, manifest, plan, scores

    def build(
        self,
        data: MarketData,
        eligible: pd.DataFrame,
        params: dict,
    ) -> pd.DataFrame:
        """返回显式信号日的计划候选目标权重。"""

        settings = self.validate_params(params)
        _, _, plan, predictions = self._load(
            settings["evaluation_id"],
            settings["candidate"],
        )
        candidate = plan.by_name(settings["candidate"])
        signal_dates = pd.DatetimeIndex(
            settings["rebalance_dates"]
        ).intersection(data.index)
        if signal_dates.empty:
            return pd.DataFrame(
                np.nan,
                index=data.index,
                columns=data.columns,
                dtype="float64",
            )
        work = predictions.copy()
        work["trade_date"] = pd.to_datetime(work["trade_date"])
        work["stock_code"] = work["stock_code"].astype(str)
        score = work.pivot(
            index="trade_date",
            columns="stock_code",
            values="score",
        ).reindex(index=signal_dates)
        if score.isna().all(axis=1).any():
            raise DataAlignmentError("plan_rank 存在没有候选分数的调仓日")

        holding = candidate.holding_policy
        policy = HoldingPolicy(
            kind=str(holding["kind"]),
            buffer_rank=holding.get("buffer_rank"),
        )
        top_n = int(holding["top_n"])
        weights = pd.DataFrame(
            np.nan,
            index=data.index,
            columns=data.columns,
            dtype="float64",
        )
        history_key = (
            settings["evaluation_id"],
            settings["candidate"],
        )
        history = self._signal_history.get(history_key)
        first_date = signal_dates[0]
        previous = (
            pd.Index([])
            if history is None
            or first_date <= pd.Timestamp(history["last_date"])
            else history["previous"]
        )
        for day in signal_dates:
            weights.loc[day] = 0.0
            row = score.loc[day].reindex(data.columns)
            pool = (
                eligible.loc[day]
                .reindex(data.columns, fill_value=False)
                .astype(bool)
            )
            chosen = select_holdings(
                row,
                pool,
                previous,
                top_n=top_n,
                policy=policy,
            )
            previous = chosen
            if len(chosen):
                weights.loc[day, chosen] = 1.0 / len(chosen)
        self._signal_history[history_key] = {
            "last_date": signal_dates[-1],
            "previous": previous,
        }
        return weights

    def artifact_snapshot(self, params: dict) -> dict:
        """返回评估、计划和候选分数的可复现摘要。"""

        settings = self.validate_params(params)
        folder, manifest, _, _ = self._load(
            settings["evaluation_id"],
            settings["candidate"],
        )
        return {
            "schema_version": "plan_rank_artifacts_v1",
            "evaluation_id": settings["evaluation_id"],
            "candidate": settings["candidate"],
            "evaluation_manifest_sha256": file_sha256(
                folder / "manifest.json"
            ),
            "score_sha256": (manifest.get("files") or {}).get(
                f"scores/{settings['candidate']}.parquet"
            ),
            "sources": manifest.get("sources") or {},
            "plan": manifest.get("plan") or {},
        }


register_provider(PlanRankProvider())


def ensure_registered() -> None:
    """显式导入时确保 provider 已注册。"""

    return None


__all__ = ["PlanRankProvider", "ensure_registered"]
