"""V2 选择期多重检验证据。"""

from __future__ import annotations

from typing import Any, Mapping

from mining.search_statistical_diagnostics import (
    MULTIPLE_TESTING_STAGE,
    build_multiple_testing_report,
)


class SearchStatisticsMixin:
    """执行和复用运行级 Newey-West 与 BH 诊断。"""

    def _record_multiple_testing(
        self,
        *,
        run: Mapping[str, Any],
        candidate_ids: tuple[str, ...],
    ) -> None:
        """为选择期候选登记同一运行级统计报告。"""

        if not candidate_ids:
            return
        existing = {
            str(row["evidence"]["candidate_id"])
            for row in self.store.list_evaluation_evidence(
                run["run_id"],
                stage=MULTIPLE_TESTING_STAGE,
            )
        }
        if set(candidate_ids) <= existing:
            return
        diagnostics = {
            str(row["evidence"]["candidate_id"]): row["evidence"]["metrics"]
            for row in self.store.list_evaluation_evidence(
                run["run_id"],
                stage="selection_diagnostics",
            )
        }
        series: dict[str, tuple[tuple[str, ...], tuple[float, ...]]] = {}
        for candidate_id in candidate_ids:
            payload = diagnostics.get(candidate_id)
            incremental = (
                payload.get("incremental")
                if isinstance(payload, Mapping)
                else None
            )
            if not isinstance(incremental, Mapping):
                continue
            dates = tuple(str(item) for item in incremental.get("rank_ic_dates") or ())
            values = tuple(
                float(item)
                for item in incremental.get("rank_ic_values") or ()
            )
            if len(dates) == len(values):
                series[candidate_id] = (dates, values)
        counts = self.store.trial_counts(run["run_id"])
        stability_rows = self.store.list_evaluation_evidence(
            run["run_id"],
            stage="stability",
        )
        variant_count = sum(
            len(row["evidence"]["metrics"].get("variants") or ())
            for row in stability_rows
        )
        attempt_count = (
            counts.proposed
            + counts.invalid
            + counts.duplicate
            + variant_count
        )
        report = build_multiple_testing_report(
            run_id=run["run_id"],
            series=series,
            attempt_count=attempt_count,
            evaluated_count=counts.evaluated,
        )
        for candidate_id in candidate_ids:
            self.store.record_evaluation_evidence(
                report.to_evaluation_evidence(candidate_id)
            )
        self.store.append_event(
            run["run_id"],
            event_key="multiple_testing_report",
            stage="selecting",
            event_type="multiple_testing_recorded",
            payload={
                "attempt_count": report.attempt_count,
                "evaluated_count": report.evaluated_count,
                "tested_count": report.tested_count,
                "denominator": report.denominator,
                "evidence_hash": report.evidence_hash,
            },
        )


__all__ = ["SearchStatisticsMixin"]
