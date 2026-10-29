"""挖掘 V2 尝试台账查询。"""

from __future__ import annotations

import json
from typing import Any

from mining.errors import MiningNotFoundError
from mining.search_budget import TrialCounts


class V2TrialQueryMixin:
    """尝试记录和分类计数只读入口。"""

    def get_trial(self, run_id: str, trial_id: str) -> dict[str, Any]:
        """读取一条尝试记录。"""

        with self._connection() as conn:
            row = self._row(
                conn,
                "SELECT * FROM mining_v2.trial_record "
                "WHERE run_id=? AND trial_id=?",
                [run_id, trial_id],
            )
        if row is None:
            raise MiningNotFoundError(f"V2 尝试记录不存在：{trial_id}")
        result = self._public_row(row)
        result["record"] = json.loads(result.pop("payload_json"))
        return result

    def list_trials(self, run_id: str) -> list[dict[str, Any]]:
        """按发生时间列出全部尝试，不区分成功或失败。"""

        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM mining_v2.trial_record "
                "WHERE run_id=? ORDER BY occurred_at, trial_id",
                [run_id],
            ).fetchall()
            columns = [item[0] for item in conn.description]
            invalidated = {
                row[0]
                for row in conn.execute(
                    "SELECT trial_id FROM mining_v2.trial_invalidation "
                    "WHERE run_id=?",
                    [run_id],
                ).fetchall()
            }
        result = []
        for values in rows:
            item = self._public_row(dict(zip(columns, values, strict=True)))
            item["record"] = json.loads(item.pop("payload_json"))
            item["invalidated"] = item["trial_id"] in invalidated
            result.append(item)
        return result

    def trial_counts(self, run_id: str) -> TrialCounts:
        """读取运行的全部分类尝试计数。"""

        with self._connection() as conn:
            row = self._counter_conn(conn, run_id)
        return TrialCounts(
            proposed=int(row["proposed"]),
            rejected=int(row["rejected"]),
            duplicate=int(row["duplicate"]),
            invalid=int(row["invalid"]),
            gate_failed=int(row["gate_failed"]),
            evaluated=int(row["evaluated"]),
            full_evaluated=int(row["full_evaluated"]),
            holdout_claimed=int(row["holdout_claimed"]),
            invalidated=int(row["invalidated"]),
        )

    def trial_summary(self, run_id: str) -> dict[str, Any]:
        """返回审核页面所需的完整尝试和预算摘要。"""

        counts = self.trial_counts(run_id).to_dict()
        with self._connection() as conn:
            self._require_run_conn(conn, run_id)
            budget = self._budget_for_run_conn(conn, run_id)
        return {
            "counts": counts,
            "total_events": sum(counts.values()),
            "attempted_candidates": counts["proposed"],
            "failed_attempts": (
                counts["rejected"]
                + counts["invalid"]
                + counts["gate_failed"]
            ),
            "evaluated_candidates": counts["evaluated"],
            "full_evaluated_candidates": counts["full_evaluated"],
            "holdout_claims": counts["holdout_claimed"],
            "budget": budget.to_dict(),
        }
