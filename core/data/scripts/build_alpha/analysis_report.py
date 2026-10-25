"""P3-4 分析结果的 Markdown 与 JSON 报告。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha.artifacts import write_json


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """把 DataFrame 渲染成紧凑 Markdown 表。"""

    if frame.empty:
        return "无数据。\n"
    selected = frame.loc[:, columns].copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
            )
    header = "| " + " | ".join(selected.columns) + " |"
    divider = "| " + " | ".join("---" for _ in selected.columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *body]) + "\n"


def write_report(
    folder: Path,
    *,
    model_id: str,
    analysis_id: str,
    baseline: dict,
    gain_features: pd.DataFrame,
    gain_family: pd.DataFrame,
    shap_features: pd.DataFrame,
    shap_family: pd.DataFrame,
    family_table: pd.DataFrame,
    permutation_table: pd.DataFrame,
    correlation_pairs: pd.DataFrame,
    exposure,
    config: dict,
) -> dict:
    """写入 JSON 摘要和可读 Markdown 报告。"""

    top_gain = gain_features.head(10).loc[
        :, ["rank", "feature", "root_feature", "family", "share_mean"]
    ]
    top_shap = shap_features.head(10).loc[
        :, ["rank", "feature", "root_feature", "family", "mean_abs_shap_mean"]
    ]
    top_family = family_table.sort_values(
        "delta_rank_ic", ascending=False, kind="mergesort"
    ).head(12)
    top_single = permutation_table.sort_values(
        "delta_rank_ic", ascending=False, kind="mergesort"
    ).head(10)
    top_industry = exposure.industry_summary.head(10)
    summary = {
        "model_id": model_id,
        "analysis_id": analysis_id,
        "baseline": baseline,
        "config": config,
        "top_gain": top_gain.to_dict(orient="records"),
        "top_shap": top_shap.to_dict(orient="records"),
        "gain_family": gain_family.to_dict(orient="records"),
        "shap_family": shap_family.to_dict(orient="records"),
        "top_family_ablation": top_family.to_dict(orient="records"),
        "top_single_permutation": top_single.to_dict(orient="records"),
        "style_summary": exposure.style_summary.to_dict(orient="records"),
        "top_industries": top_industry.to_dict(orient="records"),
        "correlation_pair_count": int(len(correlation_pairs)),
        "exposure_summary": exposure.summary,
    }
    write_json(folder / "report.json", summary)
    report = [
        f"# P3-4 重要性与消融：{model_id}",
        "",
        f"- 分析 ID：`{analysis_id}`",
        f"- 样本外区间：{baseline['start']} ~ {baseline['end']}",
        "- 正式模型未改写，2024-2026 最终测试集未运行。",
        "",
        "## seed=0 消融基线",
        "",
        "本节及消融比较使用 `seed=0` 成员；五种子集成的正式指标见模型 `metrics.json`。",
        "",
        markdown_table(
            pd.DataFrame([baseline]),
            [
                "rank_ic_mean",
                "icir",
                "annual_return",
                "max_drawdown",
                "turnover_two_way_annual",
            ],
        ),
        "## Gain 重要性（前 10）",
        "",
        markdown_table(top_gain, list(top_gain.columns)),
        "## TreeSHAP 重要性（前 10）",
        "",
        markdown_table(top_shap, list(top_shap.columns)),
        "## 分族 Gain 重要性",
        "",
        markdown_table(
            gain_family,
            ["rank", "family", "label", "share_mean"],
        ),
        "## 分族 TreeSHAP 重要性",
        "",
        markdown_table(
            shap_family,
            ["rank", "family", "label", "mean_abs_shap_mean"],
        ),
        "## 分族重训消融",
        "",
        "正值表示删除该组后指标下降，即该组提供信息。",
        "",
        markdown_table(
            top_family,
            ["group", "rank_ic_mean", "delta_rank_ic", "delta_icir", "delta_annual_return"],
        ),
        "## 单因子置换消融（前 10）",
        "",
        markdown_table(
            top_single,
            ["group", "rank_ic_mean", "delta_rank_ic", "delta_icir"],
        ),
        "## 相关性",
        "",
        f"共报告 {len(correlation_pairs)} 对 `|rho|>=0.8` 特征。",
        "",
        markdown_table(
            correlation_pairs.head(10),
            ["feature_a", "feature_b", "mean_spearman", "abs_spearman", "days"],
        ),
        "## 组合风格暴露",
        "",
        markdown_table(
            exposure.style_summary,
            ["style", "label", "mean", "t", "positive_share", "mean_coverage"],
        ),
        "## 行业主动暴露（前 10）",
        "",
        markdown_table(
            top_industry,
            ["industry", "mean_weight", "max_weight", "mean_active_weight"],
        ),
        "## 解释边界",
        "",
        "- gain 是树分裂收益，重复分裂会让相关特征互相分摊重要性。",
        "- TreeSHAP 在确定性样本上计算，反映当前模型的关联贡献，不是因果作用。",
        "- 分族重训只使用 seed=0；单因子置换固定 seed=0，避免种子差异混入贡献。",
        "- 相关特征的贡献不可直接相加，置换某因子时模型可能从同族特征补足信息。",
        "- 组合暴露对应周频模拟器的 Top-30 等权目标持仓，不是完整账本成交持仓。",
        "- P3-5 决策门尚未执行，正式模型状态保持 `pending_p3`。",
        "",
    ]
    (folder / "report.md").write_text("\n".join(report), encoding="utf-8")
    return summary


__all__ = ["markdown_table", "write_report"]
