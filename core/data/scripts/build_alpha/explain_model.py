"""生成 P3-4 模型重要性、消融、相关性和组合暴露报告。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from alpha.ablation import fit_family_ablation, predict_permuted_group
from alpha.artifacts import file_sha256, model_dir, write_json
from alpha.dataset import load_dataset
from alpha.errors import AlphaArtifactError
from alpha.explain import (
    aggregate_group_importance,
    aggregate_importance,
    compute_feature_correlation,
    compute_gain_importance,
)
from alpha.exposure import compute_portfolio_exposure
from alpha.feature_groups import build_feature_groups
from alpha.logging_conf import get_logger
from alpha.preprocess import PreprocessConfig, transform
from scripts.build_alpha.analysis_common import (
    ANALYSIS_SCHEMA_VERSION,
    analysis_id,
    code_sha256,
    delta,
    full_metrics,
    load_model_manifest,
    load_models,
    load_prediction_parts,
    rank_metrics,
    shap_records,
    wide_frames,
    write_csv,
    yearly_deltas,
)
from scripts.build_alpha.analysis_report import write_report
from scripts.build_alpha.weekly_sim import _auxiliary_panels

logger = get_logger("scripts.build_alpha.explain_model")


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="生成 P3-4 模型解释与消融报告")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--sample-rows", type=int, default=5000)
    parser.add_argument("--permutation-seed", type=int, default=20261020)
    parser.add_argument("--ablation-seed", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser


def run_analysis(
    model_id: str,
    *,
    top_n: int = 30,
    sample_rows: int = 5000,
    permutation_seed: int = 20261020,
    ablation_seed: int = 0,
    num_threads: int = 4,
    force: bool = False,
) -> dict:
    """运行完整的 P3-4 分析。"""

    folder, manifest = load_model_manifest(model_id)
    config = {
        "top_n": int(top_n),
        "sample_rows": int(sample_rows),
        "permutation_seed": int(permutation_seed),
        "ablation_seed": int(ablation_seed),
        "num_threads": int(num_threads),
        "correlation_dates": 100,
    }
    code_hash = code_sha256()
    current_analysis_id = analysis_id(manifest, config, code_hash)
    output = folder / "analysis" / current_analysis_id
    manifest_path = output / "manifest.json"
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            logger.info("复用解释分析：%s", output)
            return existing
    output.mkdir(parents=True, exist_ok=True)
    logger.info("开始解释分析：%s -> %s", model_id, output)

    data, dataset_manifest = load_dataset(manifest["dataset_id"])
    groups = build_feature_groups(manifest)
    preprocessed = transform(
        data,
        PreprocessConfig(**manifest["preprocess"]),
        feature_columns=tuple(manifest["raw_features"])
        + tuple(manifest["context_features"]),
    )
    if preprocessed.feature_names != tuple(manifest["features"]):
        raise AlphaArtifactError("预处理特征与模型 manifest 不一致")
    model_refs, models, seed_zero_models = load_models(folder, manifest)
    baseline_predictions = load_prediction_parts(
        folder / "member_predictions" / "seed=0",
        manifest["member_predictions"]["0"]["files"],
        str(manifest["member_predictions"]["0"]["sha256"]),
        label="seed=0 模型预测",
    )
    labels, eligible = wide_frames(data)
    dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(baseline_predictions["trade_date"]).unique())
    )
    start = str(dates.min().date())
    end = str(dates.max().date())
    auxiliary = _auxiliary_panels(start, end)
    baseline, baseline_yearly = full_metrics(
        baseline_predictions,
        labels,
        eligible,
        manifest,
        auxiliary,
        top_n=top_n,
    )
    baseline.update({"start": start, "end": end})

    logger.info("计算 gain 重要性")
    gain_records = compute_gain_importance(
        model_refs, tuple(manifest["features"])
    )
    gain_features = aggregate_importance(
        gain_records,
        groups,
        value_columns=("gain", "share"),
    )
    gain_groups, gain_family = aggregate_group_importance(
        gain_features, groups, metric="share"
    )
    gain_groups["family"] = gain_groups["root_feature"].map(
        groups.family_by_root
    )
    gain_groups["label"] = gain_groups["family"].map(
        {group.key: group.label for group in groups.families}
    )
    write_json(output / "feature_groups.json", groups.to_dict())
    write_csv(gain_features, output / "gain_importance.csv")
    write_csv(gain_groups, output / "gain_group_importance.csv")
    write_csv(gain_family, output / "gain_family_importance.csv")

    logger.info("计算 TreeSHAP 重要性")
    shap_model_records = shap_records(
        preprocessed,
        manifest,
        models,
        sample_rows,
    )
    shap_features = aggregate_importance(
        shap_model_records,
        groups,
        value_columns=("mean_abs_shap", "mean_signed_shap", "share"),
    )
    shap_groups, shap_family = aggregate_group_importance(
        shap_features, groups, metric="mean_abs_shap"
    )
    shap_groups["family"] = shap_groups["root_feature"].map(
        groups.family_by_root
    )
    shap_groups["label"] = shap_groups["family"].map(
        {group.key: group.label for group in groups.families}
    )
    write_csv(shap_model_records, output / "shap_model_records.csv")
    write_csv(shap_features, output / "shap_importance.csv")
    write_csv(shap_groups, output / "shap_group_importance.csv")
    write_csv(shap_family, output / "shap_family_importance.csv")

    logger.info("计算特征相关性")
    roots = tuple(group.root_features[0] for group in groups.singles)
    correlation_pairs, correlation_clusters = compute_feature_correlation(
        data,
        roots,
        max_dates=int(config["correlation_dates"]),
    )
    if correlation_pairs.empty:
        correlation_pairs = pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "mean_spearman",
                "abs_spearman",
                "days",
            ]
        )
    write_csv(correlation_pairs, output / "correlation_pairs.csv")
    write_csv(correlation_clusters, output / "correlation_clusters.csv")

    logger.info("开始 12 组分族重训消融")
    family_rows: list[dict] = []
    family_yearly: list[dict] = []
    family_models: list[dict] = []
    for index, group in enumerate(groups.families, start=1):
        logger.info(
            "分族重训 %d/%d：%s（%d 个转换特征）",
            index,
            len(groups.families),
            group.key,
            len(group.transformed_features),
        )
        result = fit_family_ablation(
            preprocessed,
            manifest,
            group,
            output_dir=output / "ablation_models" / group.key,
            seed=ablation_seed,
            num_threads=num_threads,
        )
        metrics, yearly = full_metrics(
            result.predictions,
            labels,
            eligible,
            manifest,
            auxiliary,
            top_n=top_n,
        )
        family_rows.append(
            {
                "group": group.key,
                "label": group.label,
                "kind": group.kind,
                "root_features": ";".join(group.root_features),
                "removed_feature_count": len(group.transformed_features),
                **metrics,
                "delta_rank_ic": delta(baseline, metrics, "rank_ic_mean"),
                "delta_icir": delta(baseline, metrics, "icir"),
                "delta_annual_return": delta(
                    baseline, metrics, "annual_return"
                ),
                "delta_max_drawdown": delta(
                    baseline, metrics, "max_drawdown"
                ),
                "delta_turnover": delta(
                    baseline, metrics, "turnover_two_way_annual"
                ),
            }
        )
        family_yearly.extend(
            yearly_deltas(
                baseline_yearly,
                yearly,
                group.key,
                model_level_only=False,
            )
        )
        family_models.extend(result.model_records)
    family_table = pd.DataFrame(family_rows).sort_values(
        ["delta_rank_ic", "group"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    family_table["rank"] = np.arange(1, len(family_table) + 1)
    write_csv(family_table, output / "family_ablation.csv")
    write_csv(pd.DataFrame(family_yearly), output / "family_ablation_yearly.csv")
    write_csv(pd.DataFrame(family_models), output / "family_ablation_models.csv")

    logger.info("开始 50 个单因子置换消融")
    baseline_rank, baseline_rank_yearly = rank_metrics(baseline_predictions, data)
    permutation_rows: list[dict] = []
    permutation_yearly: list[dict] = []
    for index, group in enumerate(groups.singles, start=1):
        logger.info(
            "单因子置换 %d/%d：%s",
            index,
            len(groups.singles),
            group.key,
        )
        predictions = predict_permuted_group(
            preprocessed,
            manifest,
            group,
            seed_zero_models,
            permutation_seed=permutation_seed + index * 100003,
        )
        metrics, yearly = rank_metrics(predictions, data)
        permutation_rows.append(
            {
                "group": group.key,
                "label": group.label,
                "root_features": ";".join(group.root_features),
                "permuted_feature_count": len(group.transformed_features),
                **metrics,
                "delta_rank_ic": delta(
                    baseline_rank, metrics, "rank_ic_mean"
                ),
                "delta_icir": delta(baseline_rank, metrics, "icir"),
            }
        )
        permutation_yearly.extend(
            yearly_deltas(
                baseline_rank_yearly,
                yearly,
                group.key,
                model_level_only=True,
            )
        )
    permutation_table = pd.DataFrame(permutation_rows).sort_values(
        ["delta_rank_ic", "group"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    permutation_table["rank"] = np.arange(1, len(permutation_table) + 1)
    write_csv(permutation_table, output / "permutation_ablation.csv")
    write_csv(
        pd.DataFrame(permutation_yearly),
        output / "permutation_ablation_yearly.csv",
    )

    logger.info("计算 Top-%d 组合暴露", top_n)
    exposure = compute_portfolio_exposure(
        baseline_predictions,
        data,
        top_n=top_n,
    )
    write_csv(exposure.style_summary, output / "portfolio_style_summary.csv")
    write_csv(exposure.style_series, output / "portfolio_style_series.csv")
    write_csv(exposure.industry_summary, output / "portfolio_industry_summary.csv")
    write_csv(exposure.industry_series, output / "portfolio_industry_series.csv")

    report = write_report(
        output,
        model_id=model_id,
        analysis_id=current_analysis_id,
        baseline=baseline,
        gain_features=gain_features,
        gain_family=gain_family,
        shap_features=shap_features,
        shap_family=shap_family,
        family_table=family_table,
        permutation_table=permutation_table,
        correlation_pairs=correlation_pairs,
        exposure=exposure,
        config=config,
    )
    files: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        files[str(path.relative_to(output))] = file_sha256(path)
    payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_id": current_analysis_id,
        "model_id": model_id,
        "dataset_id": manifest["dataset_id"],
        "model_sha256": manifest["model_sha256"],
        "prediction_sha256": manifest["prediction_sha256"],
        "dataset_manifest": dataset_manifest,
        "code_sha256": code_hash,
        "config": config,
        "start": start,
        "end": end,
        "seeds": list(manifest["seeds"]),
        "feature_count": int(len(manifest["features"])),
        "root_feature_count": len(groups.singles),
        "family_group_count": len(groups.families),
        "files": files,
        "summary": report,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, payload)
    logger.info("解释分析完成：%s", output)
    return payload


def main() -> None:
    """命令行入口。"""

    args = _parser().parse_args()
    result = run_analysis(
        args.model_id,
        top_n=args.top_n,
        sample_rows=args.sample_rows,
        permutation_seed=args.permutation_seed,
        ablation_seed=args.ablation_seed,
        num_threads=args.num_threads,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "analysis_id": result["analysis_id"],
                "model_id": result["model_id"],
                "status": result["status"],
                "path": str(
                    model_dir(result["model_id"])
                    / "analysis"
                    / result["analysis_id"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
