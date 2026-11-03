"""按通用评估计划执行候选评分和决策。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from alpha.artifacts import file_sha256, prediction_sha256, write_json
from alpha.config import MODEL_DATA_DIR
from alpha.dataset import load_dataset
from alpha.ensemble import combine_rank_predictions
from alpha.errors import AlphaEvaluationError
from alpha.evaluate import (
    evaluate_score_frame,
    p3_decision_gate,
)
from alpha.evaluation_plan import EvaluationCandidate, EvaluationPlan
from alpha.exposure import compute_portfolio_exposure
from alpha.portfolio import HoldingPolicy, smooth_rank_scores
from alpha.predict import load_predictions
from alpha.risk import RiskGatePolicy, evaluate_risk_gate
from common.paths import PROJECT_ROOT
from scripts.build_alpha.baselines import scores_for_baseline
from scripts.build_alpha.weekly_sim import (
    _auxiliary_panels,
    simulate_score_frame,
    yearly_summary,
)

EVALUATION_SCHEMA_VERSION = "alpha_model_evaluation_v1"


def evaluation_code_sha256() -> str:
    """汇总评估计划执行和模拟代码哈希。"""

    root = PROJECT_ROOT
    digest = hashlib.sha256()
    for relative in (
        "research/modeling/alpha/evaluate.py",
        "research/modeling/alpha/evaluation_plan.py",
        "research/modeling/alpha/ensemble.py",
        "research/modeling/alpha/exposure.py",
        "research/modeling/alpha/portfolio.py",
        "research/modeling/alpha/predict.py",
        "research/modeling/alpha/risk.py",
        "core/data/scripts/build_alpha/baselines.py",
        "core/data/scripts/build_alpha/evaluation_executor.py",
        "core/data/scripts/build_alpha/weekly_sim.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluation_analysis_id(
    plan: EvaluationPlan,
    sources: Mapping[str, dict],
    dataset_manifest: dict,
    code_sha256: str,
) -> str:
    """由计划、来源、数据集和代码生成确定性分析 ID。"""

    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "plan": plan.to_dict(),
        "sources": sources,
        "dataset_manifest": dataset_manifest,
        "code_sha256": code_sha256,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "evaluation_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    """原子写入 CSV 或 Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _wide_frames(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把数据集标签与股票池转成宽表。"""

    labels = data.pivot(
        index="trade_date",
        columns="stock_code",
        values="label_raw",
    ).astype("float64")
    eligible = data.pivot(
        index="trade_date",
        columns="stock_code",
        values="eligible",
    ).astype("boolean").fillna(False).astype(bool)
    return labels, eligible


def _wide_scores(
    predictions: pd.DataFrame,
    dates: pd.DatetimeIndex,
    columns: pd.Index,
) -> pd.DataFrame:
    """把标准预测表转换为对齐的宽表。"""

    return predictions.pivot(
        index="trade_date",
        columns="stock_code",
        values="score",
    ).reindex(index=dates, columns=columns).astype("float64")


def _factor_names(data: pd.DataFrame) -> tuple[str, ...]:
    """从数据集中恢复参与 B4 的因子列。"""

    excluded = {
        "trade_date",
        "stock_code",
        "eligible",
        "sample_weight",
        "label_raw",
        "label_rank",
        "industry_l1",
        "log_mv",
        "turnover_rate",
        "listed_days",
    }
    return tuple(name for name in data.columns if name not in excluded)


def _model_inputs(
    plan: EvaluationPlan,
    root: Path,
) -> tuple[dict[str, dict], dict[str, pd.DataFrame]]:
    """读取计划引用的全部模型并校验共同口径。"""

    manifests: dict[str, dict] = {}
    predictions: dict[str, pd.DataFrame] = {}
    for model_id in plan.model_ids():
        _, manifest, frame = load_predictions(model_id, root=root)
        manifests[model_id] = manifest
        predictions[model_id] = frame
    first_id = plan.model_ids()[0]
    first_manifest = manifests[first_id]
    first_frame = predictions[first_id]
    first_keys = first_frame.loc[:, ["trade_date", "stock_code"]]
    for model_id in plan.model_ids()[1:]:
        manifest = manifests[model_id]
        for field in ("dataset_id", "target"):
            if manifest.get(field) != first_manifest.get(field):
                raise AlphaEvaluationError(
                    f"模型 {model_id} 的 {field} 与 {first_id} 不一致"
                )
        left_folds = [
            (str(item["fold_id"]), str(item["test_start"]), str(item["test_end"]))
            for item in first_manifest.get("folds") or ()
        ]
        right_folds = [
            (str(item["fold_id"]), str(item["test_start"]), str(item["test_end"]))
            for item in manifest.get("folds") or ()
        ]
        if left_folds != right_folds:
            raise AlphaEvaluationError(
                f"模型 {model_id} 的折或测试区间与 {first_id} 不一致"
            )
        if not first_keys.equals(
            predictions[model_id].loc[:, ["trade_date", "stock_code"]]
        ):
            raise AlphaEvaluationError(
                f"模型 {model_id} 的预测日期或股票与 {first_id} 不一致"
            )
    return manifests, predictions


def _source_hashes(
    plan: EvaluationPlan,
    manifests: Mapping[str, dict],
    predictions: Mapping[str, pd.DataFrame],
    root: Path,
) -> dict[str, dict]:
    """记录计划引用的模型与预测哈希。"""

    payload: dict[str, dict] = {}
    for model_id in plan.model_ids():
        manifest = manifests[model_id]
        model_folder = root / "models" / model_id
        if not model_folder.is_dir():
            model_folder = root / "archive" / "models" / model_id
        payload[model_id] = {
            "model_id": model_id,
            "manifest_sha256": file_sha256(model_folder / "manifest.json"),
            "model_sha256": manifest.get("model_sha256"),
            "prediction_sha256": prediction_sha256(predictions[model_id]),
            "dataset_id": manifest.get("dataset_id"),
        }
    return payload


def _base_sources(
    plan: EvaluationPlan,
    predictions: Mapping[str, pd.DataFrame],
    data: pd.DataFrame,
    dates: pd.DatetimeIndex,
    columns: pd.Index,
) -> dict[tuple[str, str], pd.DataFrame]:
    """构造计划中每个来源的原始宽分数。"""

    factors = _factor_names(data)
    output: dict[tuple[str, str], pd.DataFrame] = {}
    for candidate in (*plan.candidates, *plan.controls):
        source = candidate.source
        key = (source.kind, source.id)
        if key in output:
            continue
        if source.kind == "model":
            output[key] = _wide_scores(
                predictions[source.id],
                dates,
                columns,
            )
        elif source.kind == "baseline":
            output[key] = scores_for_baseline(
                data,
                source.id,
                factors,
            ).reindex(index=dates, columns=columns).astype("float64")
        else:
            frames = [predictions[item] for item in source.members]
            blended = combine_rank_predictions(
                frames,
                weights=source.weights,
                names=source.members,
            )
            output[key] = _wide_scores(blended, dates, columns)
    return output


def _combined_metrics(model_metrics: dict, portfolio: dict) -> dict:
    """合并模型层与组合层核心指标。"""

    return {
        "rank_ic_mean": model_metrics.get("rank_ic_mean"),
        "icir": model_metrics.get("icir"),
        "rank_ic_t": model_metrics.get("rank_ic_t"),
        "rank_ic_win_rate": model_metrics.get("rank_ic_win_rate"),
        "monotonicity": model_metrics.get("monotonicity"),
        "score_autocorrelation_1w": model_metrics.get(
            "score_autocorrelation_1w"
        ),
        "annual_return": portfolio.get("annual_return"),
        "excess_annual_return": portfolio.get("excess_annual_return"),
        "annual_volatility": portfolio.get("annual_volatility"),
        "sharpe": portfolio.get("sharpe"),
        "max_drawdown": portfolio.get("max_drawdown"),
        "turnover_two_way_annual": portfolio.get(
            "turnover_two_way_annual"
        ),
        "total_cost": portfolio.get("total_cost"),
    }


def _yearly_rows(
    model_yearly: list[dict],
    daily: pd.DataFrame,
) -> list[dict]:
    """合并模型层 RankIC 与组合层分年结果。"""

    model_by_year = {int(row["year"]): row for row in model_yearly}
    rows: list[dict] = []
    for item in yearly_summary(daily):
        year = int(item["year"])
        group = daily.loc[pd.DatetimeIndex(daily.index).year == year]
        benchmark = float(
            (1.0 + group["benchmark_return"].astype(float)).prod() - 1.0
        )
        periods = int(item["periods"])
        rows.append(
            {
                **item,
                "rank_ic_mean": model_by_year.get(year, {}).get(
                    "rank_ic_mean"
                ),
                "benchmark_total_return": benchmark,
                "excess_total_return": (
                    float(item["total_return"]) - benchmark
                ),
                "turnover_two_way_annual": (
                    float(item["turnover"]) * 52.0 / periods
                    if periods
                    else None
                ),
            }
        )
    return rows


def evaluate_candidate(
    candidate: EvaluationCandidate,
    source_scores: pd.DataFrame,
    labels: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    start: str,
    end: str,
    quantiles: int,
    auxiliary,
    score_frame: pd.DataFrame | None = None,
) -> tuple[dict, list[dict], pd.DataFrame, pd.DataFrame]:
    """计算一个计划候选的模型层、组合层、年度和周结果。"""

    score = source_scores if score_frame is None else score_frame
    model_metrics = evaluate_score_frame(
        score,
        labels,
        eligible,
        quantiles=quantiles,
    )
    holding = candidate.holding_policy
    policy = HoldingPolicy(
        kind=str(holding["kind"]),
        buffer_rank=holding.get("buffer_rank"),
    )
    portfolio, daily, orders = simulate_score_frame(
        score,
        labels,
        eligible,
        start=start,
        end=end,
        top_n=int(holding["top_n"]),
        auxiliary=auxiliary,
        holding_policy=policy,
    )
    return (
        _combined_metrics(model_metrics, portfolio),
        _yearly_rows(model_metrics.get("yearly", []), daily),
        daily,
        orders,
    )


def _policy_score(
    candidate: EvaluationCandidate,
    source_scores: pd.DataFrame,
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    """按候选规则生成评估和完整账本共用的分数。"""

    if candidate.score_policy.kind != "smooth_rank":
        return source_scores
    return smooth_rank_scores(
        source_scores,
        eligible,
        window=candidate.score_policy.smoothing_weeks,
        min_periods=1,
    )


def _long_scores(score: pd.DataFrame) -> pd.DataFrame:
    """把候选宽分数转换为标准预测长表。"""

    payload = (
        score.rename_axis(index="trade_date", columns="stock_code")
        .stack(future_stack=True)
        .rename("score")
        .reset_index()
    )
    return payload.loc[payload["score"].notna()].reset_index(drop=True)


def _risk_styles(start: str, end: str) -> pd.DataFrame:
    """读取风险门槛区间内的原始风格因子。"""

    from calculate.params import EvalParams
    from calculate.styles import build_styles

    return build_styles(
        EvalParams(
            factor="MODEL_SCORE",
            start=start,
            end=end,
            horizon=5,
            price="open",
        )
    )


def _candidate_risk(
    candidate: EvaluationCandidate,
    score: pd.DataFrame,
    data: pd.DataFrame,
    styles: pd.DataFrame,
    policy: RiskGatePolicy,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """计算候选目标组合的风险门槛和可落盘时序。"""

    holding = candidate.holding_policy
    exposure = compute_portfolio_exposure(
        _long_scores(score),
        data,
        top_n=int(holding["top_n"]),
        holding_policy=HoldingPolicy(
            kind=str(holding["kind"]),
            buffer_rank=holding.get("buffer_rank"),
        ),
        styles=styles,
    )
    risk = evaluate_risk_gate(
        exposure.style_series,
        exposure.industry_series,
        policy,
    )
    return risk.summary, risk.style_timeseries, risk.industry_timeseries


def _combine_gate(economic: dict | None, risk: dict | None) -> dict:
    """把经济门槛和组合风险门槛合并为候选结论。"""

    if risk is None:
        return dict(economic or {})
    economic_passed = bool((economic or {}).get("passed", False))
    risk_passed = bool(risk.get("passed"))
    passed = economic_passed and risk_passed
    reasons = list((economic or {}).get("reasons") or ())
    if not risk_passed:
        failed = [
            name
            for name, value in (risk.get("checks") or {}).items()
            if not value
        ]
        reasons.append("组合风险门槛未通过：" + ", ".join(failed))
    return {
        **(economic or {}),
        "passed": passed,
        "status": "pass" if passed else "fail",
        "economic_gate": economic,
        "risk_gate": risk,
        "reasons": reasons,
    }


def _selection_key(
    name: str,
    candidate: EvaluationCandidate,
    metrics: Mapping[str, object],
    metrics_order: Sequence[str],
) -> tuple:
    """按计划指标顺序生成候选排序键。"""

    values: list[float] = []
    for metric in metrics_order:
        value = float(metrics.get(metric) or 0.0)
        values.append(value if metric == "turnover_two_way_annual" else -value)
    return (*values, candidate.priority, name)


def select_candidate(
    plan: EvaluationPlan,
    metrics: Mapping[str, dict],
    gates: Mapping[str, dict],
) -> dict:
    """从通过门槛且可选择的候选中按计划排序。"""

    eligible = [
        candidate
        for candidate in plan.candidates
        if candidate.selectable
        and candidate.name != plan.reference
        and bool(gates.get(candidate.name, {}).get("passed"))
    ]
    if not eligible:
        return {
            "selected": None,
            "ranked": [],
            "reason": (
                f"没有可选择候选通过评估门，保留 {plan.reference}"
            ),
        }
    ranked = sorted(
        eligible,
        key=lambda candidate: _selection_key(
            candidate.name,
            candidate,
            metrics[candidate.name],
            plan.selection_metrics,
        ),
    )
    return {
        "selected": ranked[0].name,
        "ranked": [candidate.name for candidate in ranked],
        "reason": f"按计划排序选择 {ranked[0].name}",
    }


def execute_plan(
    plan: EvaluationPlan,
    *,
    root: Path | None = None,
) -> dict:
    """在已登记模型和数据集上执行完整评估计划。"""

    output_root = Path(root or MODEL_DATA_DIR)
    manifests, predictions = _model_inputs(plan, output_root)
    first_id = plan.model_ids()[0]
    dataset_id = str(manifests[first_id]["dataset_id"])
    data, dataset_manifest = load_dataset(dataset_id)
    labels, eligible = _wide_frames(data)
    first_predictions = predictions[first_id]
    dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(first_predictions["trade_date"]).unique())
    )
    columns = pd.Index(sorted(first_predictions["stock_code"].astype(str).unique()))
    start = str(dates.min().date())
    end = str(dates.max().date())
    source_scores = _base_sources(
        plan,
        predictions,
        data,
        dates,
        columns,
    )
    auxiliary = _auxiliary_panels(start, end)
    risk_styles = (
        _risk_styles(start, end) if plan.risk_gate is not None else None
    )
    quantiles = int(manifests[first_id].get("quantiles", 10))
    metrics: dict[str, dict] = {}
    risks: dict[str, dict] = {}
    candidate_scores: dict[str, pd.DataFrame] = {}
    yearly: dict[str, list[dict]] = {}
    weekly: list[pd.DataFrame] = []
    orders: list[pd.DataFrame] = []
    for candidate in (*plan.candidates, *plan.controls):
        source = source_scores[(candidate.source.kind, candidate.source.id)]
        score = _policy_score(candidate, source, eligible)
        candidate_scores[candidate.name] = _long_scores(score)
        if plan.risk_gate is not None:
            risk_summary, style_risk, industry_risk = _candidate_risk(
                candidate,
                score,
                data,
                risk_styles,
                plan.risk_gate,
            )
            risks[candidate.name] = {
                "summary": risk_summary,
                "style_timeseries": style_risk,
                "industry_timeseries": industry_risk,
            }
        candidate_metrics, candidate_yearly, daily, candidate_orders = (
            evaluate_candidate(
                candidate,
                source,
                labels,
                eligible,
                start=start,
                end=end,
                quantiles=quantiles,
                auxiliary=auxiliary,
                score_frame=score,
            )
        )
        metrics[candidate.name] = candidate_metrics
        yearly[candidate.name] = candidate_yearly
        weekly.append(daily.assign(candidate=candidate.name).reset_index())
        orders.append(candidate_orders.assign(candidate=candidate.name))

    reference = metrics[plan.reference]
    gates: dict[str, dict] = {}
    for candidate in (*plan.candidates, *plan.controls):
        economic = None
        if (
            candidate.name != plan.reference
            and candidate in plan.candidates
        ):
            economic = p3_decision_gate(
                metrics[candidate.name],
                reference,
                yearly[candidate.name],
                expected_years=plan.gate.expected_years or None,
                required_improvements=plan.gate.required_improvements,
                turnover_multiple_limit=plan.gate.turnover_multiple_limit,
            )
        gates[candidate.name] = _combine_gate(
            economic,
            risks.get(candidate.name, {}).get("summary"),
        )
    selection = select_candidate(plan, metrics, gates)
    selected_name = selection["selected"] or plan.reference
    selected_candidate = plan.by_name(selected_name)
    selected_source = selected_candidate.source
    decision = {
        "status": (
            "pass_evaluation"
            if selection["selected"] is not None
            else "fail_keep_reference"
        ),
        "selected_candidate": selected_name,
        "selected_source": selected_source.to_dict(),
        "selected_model_id": (
            selected_source.id if selected_source.kind == "model" else None
        ),
        "reference_candidate": plan.reference,
        "reference_source": plan.by_name(plan.reference).source.to_dict(),
        "score_policy": {
            "kind": selected_candidate.score_policy.kind,
            "smoothing_weeks": selected_candidate.score_policy.smoothing_weeks,
        },
        "holding_policy": dict(selected_candidate.holding_policy),
        "selection": selection,
        "gate": gates.get(selected_name, {}),
        "risk_gate": risks.get(selected_name, {}).get("summary"),
    }
    return {
        "start": start,
        "end": end,
        "metrics": metrics,
        "yearly": yearly,
        "gates": gates,
        "risks": risks,
        "candidate_scores": candidate_scores,
        "selection": selection,
        "selected_candidate": selected_name,
        "status": decision["status"],
        "decision": decision,
        "weekly": pd.concat(weekly, ignore_index=True),
        "orders": pd.concat(orders, ignore_index=True),
    }


def candidate_table(result: Mapping) -> pd.DataFrame:
    """展开候选整体指标、差值和决策门。"""

    plan = result["plan"]
    reference = result["metrics"][plan.reference]
    rows = []
    for candidate in (*plan.candidates, *plan.controls):
        metrics = result["metrics"][candidate.name]
        gate = result["gates"].get(candidate.name, {})
        risk = (result.get("risks") or {}).get(candidate.name, {}).get(
            "summary",
            {},
        )
        rows.append(
            {
                "name": candidate.name,
                "source_kind": candidate.source.kind,
                "source_id": candidate.source.id,
                "score_kind": candidate.score_policy.kind,
                "smoothing_weeks": candidate.score_policy.smoothing_weeks,
                "holding_kind": candidate.holding_policy["kind"],
                "top_n": candidate.holding_policy["top_n"],
                "buffer_rank": candidate.holding_policy.get("buffer_rank"),
                "priority": candidate.priority,
                "selectable": candidate.selectable,
                **metrics,
                "delta_rank_ic": (
                    float(metrics["rank_ic_mean"])
                    - float(reference["rank_ic_mean"])
                ),
                "delta_annual_return": (
                    float(metrics["annual_return"])
                    - float(reference["annual_return"])
                ),
                "delta_max_drawdown": (
                    float(metrics["max_drawdown"])
                    - float(reference["max_drawdown"])
                ),
                "delta_turnover": (
                    float(metrics["turnover_two_way_annual"])
                    - float(reference["turnover_two_way_annual"])
                ),
                "gate_status": gate.get("status"),
                "gate_improvements": gate.get("improvement_count"),
                "gate_turnover_multiple": gate.get("turnover_multiple"),
                "gate_yearly_stable": gate.get("yearly_stable"),
                "risk_gate_status": risk.get("status"),
                "risk_max_size_abs_exposure": risk.get(
                    "max_size_abs_exposure"
                ),
                "risk_max_liquidity_abs_exposure": risk.get(
                    "max_liquidity_abs_exposure"
                ),
                "risk_max_industry_active_weight": risk.get(
                    "max_industry_active_weight"
                ),
                "risk_mature_windows": risk.get("mature_windows"),
                "selected": candidate.name == result["selected_candidate"],
            }
        )
    return pd.DataFrame(rows)


def yearly_table(result: Mapping) -> pd.DataFrame:
    """展开全部候选的分年结果。"""

    rows = []
    for name, items in result["yearly"].items():
        rows.extend({"candidate": name, **item} for item in items)
    return pd.DataFrame(rows)


def build_report(result: Mapping, plan: EvaluationPlan) -> str:
    """生成通用评估报告。"""

    table = candidate_table(result)
    lines = [
        "# 模型评估报告",
        "",
        f"- 区间：{result['start']}～{result['end']}",
        f"- 参照：{plan.reference}",
        f"- 选中：{result['selected_candidate']}",
        f"- 状态：{result['status']}",
        f"- 原因：{result['selection']['reason']}",
        "",
        "| 候选 | 来源 | 年化净收益 | 净超额 | 夏普 | 最大回撤 | 年换手 | 门槛 | 风险 | 选中 |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            "| {name} | {source} | {annual:.4f} | {excess:.4f} | "
            "{sharpe:.4f} | {drawdown:.4f} | {turnover:.2f} | "
            "{gate} | {risk} | {selected} |".format(
                name=row.name,
                source=f"{row.source_kind}:{row.source_id}",
                annual=float(row.annual_return),
                excess=float(row.excess_annual_return),
                sharpe=float(row.sharpe),
                drawdown=float(row.max_drawdown),
                turnover=float(row.turnover_two_way_annual),
                gate=row.gate_status or "参照/控制",
                risk=row.risk_gate_status or "未启用",
                selected="是" if row.selected else "否",
            )
        )
    return "\n".join(lines) + "\n"


def _load_reusable_evaluation(
    output: Path,
    *,
    analysis_id: str,
    plan: EvaluationPlan,
    sources: Mapping[str, dict],
    dataset_manifest: dict,
    code_sha256: str,
) -> dict:
    """完整校验已存在的评估产物，避免损坏目录被静默复用。"""

    manifest_path = output / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AlphaEvaluationError(
            f"评估目录存在但 manifest 不可读取：{output}；如需重建请使用 --force"
        ) from error
    expected = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "plan": plan.to_dict(),
        "sources": dict(sources),
        "dataset_manifest": dataset_manifest,
        "code_sha256": code_sha256,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise AlphaEvaluationError(
                f"评估产物 {name} 与当前输入不一致；如需重建请使用 --force"
            )
    if manifest.get("status") not in {
        "pass_evaluation",
        "fail_keep_reference",
    }:
        raise AlphaEvaluationError(
            f"评估产物状态非法：{manifest.get('status')!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise AlphaEvaluationError("评估产物没有文件哈希清单")
    base = output.resolve()
    for relative, expected_hash in files.items():
        path = (output / str(relative)).resolve()
        if base not in path.parents or not path.is_file():
            raise AlphaEvaluationError(f"评估产物文件缺失或越界：{relative}")
        if file_sha256(path) != str(expected_hash):
            raise AlphaEvaluationError(
                f"评估产物文件哈希不一致：{relative}；如需重建请使用 --force"
            )
    return manifest


def run_evaluation_plan(
    plan_path: Path,
    *,
    root: Path | None = None,
    force: bool = False,
) -> dict:
    """执行计划并原子落盘评估产物。"""

    plan_path = Path(plan_path)
    plan = EvaluationPlan.from_file(plan_path)
    output_root = Path(root or MODEL_DATA_DIR)
    evaluation_root = output_root / "runs" / "evaluation"
    manifests, predictions = _model_inputs(plan, output_root)
    first_id = plan.model_ids()[0]
    dataset_id = str(manifests[first_id]["dataset_id"])
    _, dataset_manifest = load_dataset(dataset_id)
    sources = _source_hashes(plan, manifests, predictions, output_root)
    code_hash = evaluation_code_sha256()
    analysis_id = evaluation_analysis_id(
        plan,
        sources,
        dataset_manifest,
        code_hash,
    )
    output = evaluation_root / analysis_id
    if output.exists():
        if not force:
            return _load_reusable_evaluation(
                output,
                analysis_id=analysis_id,
                plan=plan,
                sources=sources,
                dataset_manifest=dataset_manifest,
                code_sha256=code_hash,
            )
        if output.parent.resolve() != evaluation_root.resolve():
            raise AlphaEvaluationError("拒绝删除评估目录之外的路径")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    result = execute_plan(plan, root=output_root)
    result["plan"] = plan
    candidates = candidate_table(result)
    yearly = yearly_table(result)
    _write_frame(candidates, output / "candidates.csv")
    _write_frame(yearly, output / "yearly.csv")
    _write_frame(result["weekly"], output / "weekly_results.parquet")
    _write_frame(result["orders"], output / "orders.parquet")
    for name, frame in result["candidate_scores"].items():
        _write_frame(frame, output / "scores" / f"{name}.parquet")
    for name, payload in result["risks"].items():
        risk_folder = output / "risk" / name
        _write_frame(
            payload["style_timeseries"],
            risk_folder / "style_series.csv",
        )
        _write_frame(
            payload["industry_timeseries"],
            risk_folder / "industry_series.csv",
        )
        write_json(risk_folder / "summary.json", payload["summary"])
    write_json(output / "plan.json", plan.to_dict())
    write_json(output / "decision.json", result["decision"])
    (output / "report.md").write_text(build_report(result, plan), encoding="utf-8")
    files = {
        path.relative_to(output).as_posix(): file_sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "status": result["status"],
        "selected_candidate": result["selected_candidate"],
        "selected_source": result["decision"]["selected_source"],
        "start": result["start"],
        "end": result["end"],
        "plan": plan.to_dict(),
        "sources": sources,
        "dataset_manifest": dataset_manifest,
        "risk_gate": (
            None if plan.risk_gate is None else plan.risk_gate.to_dict()
        ),
        "code_sha256": code_hash,
        "candidate_count": int(len(candidates)),
        "files": files,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "build_report",
    "candidate_table",
    "evaluate_candidate",
    "evaluation_analysis_id",
    "evaluation_code_sha256",
    "execute_plan",
    "run_evaluation_plan",
    "select_candidate",
    "yearly_table",
]
