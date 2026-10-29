"""V2 选择前的开发期邻域稳定性评估。"""

from __future__ import annotations

from typing import Any, Mapping

from mining.contract_candidates import CandidateDefinition
from mining.search_evaluation_adapter import development_context
from mining.search_stability import generate_primary_window_variants
from mining.search_stability_contracts import (
    STABILITY_STAGE,
    WindowVariantObservation,
    build_stability_report,
    optional_float,
)


class SearchStabilityMixin:
    """执行和复用候选主窗口邻域稳定性证据。"""

    def _record_stability_reports(
        self,
        *,
        run: Mapping[str, Any],
        context: Any,
        selected_results: Mapping[
            str, tuple[Mapping[str, Any], CandidateDefinition, Any]
        ],
    ) -> None:
        """为选择期候选登记只在开发期计算的稳定性报告。"""

        existing = {
            str(row["evidence"]["candidate_id"])
            for row in self.store.list_evaluation_evidence(
                run["run_id"],
                stage=STABILITY_STAGE,
            )
        }
        for candidate_id, (_, candidate, _) in selected_results.items():
            if candidate_id in existing:
                continue
            development_rows = self.store.list_evaluation_evidence(
                run["run_id"],
                candidate_id,
                stage="development",
            )
            base_rank_ic = (
                optional_float(
                    development_rows[-1]["evidence"]["metrics"].get(
                        "rank_ic_mean"
                    )
                )
                if development_rows
                else None
            )
            plan = generate_primary_window_variants(
                candidate,
                limit=context.control_policy.stability_variant_limit,
            )
            observations: list[WindowVariantObservation] = []
            for variant in plan.variants:
                stage_root = (
                    self.store.candidate_dir(run["run_id"], candidate_id)
                    / STABILITY_STAGE
                    / variant.candidate.candidate_id
                )
                try:
                    result = self._development_evaluator(
                        run_id=run["run_id"],
                        candidate=variant.candidate,
                        context=development_context(context),
                        stage_root=stage_root,
                    )
                    quick_metrics = (
                        result.quick.metrics if result.quick is not None else {}
                    )
                    observations.append(
                        WindowVariantObservation(
                            candidate_id=variant.candidate.candidate_id,
                            candidate_hash=variant.candidate.content_hash,
                            window=variant.variant_window,
                            delta=variant.delta,
                            status=(
                                result.quick.status
                                if result.quick is not None
                                else result.technical.status
                            ),
                            gate_passed=(
                                result.quick is not None
                                and result.technical.status == "passed"
                            ),
                            rank_ic=optional_float(
                                quick_metrics.get("rank_ic_mean")
                            ),
                            rank_ic_ir=optional_float(
                                quick_metrics.get("rank_ic_ir")
                            ),
                            error_code=getattr(result.technical, "error_code", None),
                        )
                    )
                except Exception as exc:
                    observations.append(
                        WindowVariantObservation(
                            candidate_id=variant.candidate.candidate_id,
                            candidate_hash=variant.candidate.content_hash,
                            window=variant.variant_window,
                            delta=variant.delta,
                            status="error",
                            gate_passed=False,
                            rank_ic=None,
                            rank_ic_ir=None,
                            error_code=getattr(exc, "code", "MINING-106"),
                        )
                    )
            report = build_stability_report(
                run_id=run["run_id"],
                candidate=candidate,
                plan=plan,
                base_rank_ic=base_rank_ic,
                variants=tuple(observations),
                rank_ic_ratio_floor=(
                    context.control_policy.stability_rank_ic_ratio_floor
                ),
            )
            self.store.record_evaluation_evidence(
                report.to_evaluation_evidence()
            )
            self.store.append_event(
                run["run_id"],
                event_key=f"stability_{candidate_id}",
                stage="selecting",
                event_type="stability_report_recorded",
                payload={
                    "candidate_id": candidate_id,
                    "status": report.status,
                    "variant_count": len(report.variants),
                    "evidence_hash": report.evidence_hash,
                },
            )


__all__ = ["SearchStabilityMixin"]
