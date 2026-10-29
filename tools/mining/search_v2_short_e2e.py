"""用短窗口和真实评估器验收 V2 搜索闭环。"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from calculate.correlation import available_factors
from common.paths import DB_FILE, FACTOR_DIR
from mining.catalog import load_catalog
from mining.search_budget import SearchBudget
from mining.search_context import FrozenSearchContext, SearchWindows
from mining.search_orchestrator import MiningSearchOrchestrator
from mining.search_run import SearchRunRequest
from mining.search_scorecard import build_run_scorecards
from mining.snapshot import DEFAULT_DATA_MANIFEST, data_snapshot_fingerprint
from mining.storage_v2 import MiningV2Store
from mining_worker.protocol import WorkerCandidateDraft, WorkerJobView
from mining_worker.protocol_v2 import (
    WorkerDecisionProposal,
    WorkerSearchHypothesis,
    WorkerSearchResponse,
)
from mining_worker.worker_config import load_worker_config

WINDOWS = SearchWindows(
    development_start="2019-01-02",
    development_end="2019-03-29",
    selection_start="2019-04-01",
    selection_end="2019-06-28",
    final_start="2019-07-01",
    final_end="2019-09-30",
    holdout_start="2019-10-08",
    holdout_end="2019-12-31",
)


class ShortRunWorker:
    """第一轮提出候选，第二轮基于真实开发期反馈停止。"""

    def __init__(self) -> None:
        self.requests = []

    def health(self):
        return {"status": "ready", "mode": "deterministic-short-run"}

    def submit(self, request):
        self.requests.append(request)
        if request.turn_index == 1:
            response = WorkerSearchResponse(
                decision=WorkerDecisionProposal(
                    action="refine",
                    reason="验证短窗口均值反转信号",
                ),
                next_hypothesis=WorkerSearchHypothesis(
                    statement="第二轮确认短窗口反转是否具有开发期证据",
                    mechanism="短期价格修复",
                    expected_direction="positive",
                    falsification_criteria=("开发期没有有效 RankIC",),
                ),
                candidates=(
                    WorkerCandidateDraft(
                        expression="rank(mean(close,5))",
                        mechanism="短期价格修复",
                        description="五日收盘价均值的截面排名",
                        direction=1,
                        steps=("计算五日收盘价均值", "按交易日做截面排名"),
                    ),
                ),
                usage={"calls": 1},
            )
        else:
            response = WorkerSearchResponse(
                decision=WorkerDecisionProposal(
                    action="stop",
                    reason="短窗口开发期证据已收集",
                ),
                next_hypothesis=None,
                candidates=(),
                usage={"calls": 1},
            )
        return WorkerJobView(
            job_id=request.job_id,
            request_id=request.request_id,
            status="succeeded",
            result=response.to_dict(),
        )

    def get_job(self, job_id):
        request = next(item for item in self.requests if item.job_id == job_id)
        return WorkerJobView(
            job_id=request.job_id,
            request_id=request.request_id,
            status="succeeded",
            result=self.submit_without_record(request).result,
        )

    def submit_without_record(self, request):
        stored = request
        original = self.requests
        self.requests = []
        try:
            result = self.submit(stored)
        finally:
            self.requests = original
        return result

    def cancel(self, job_id):
        return WorkerJobView(
            job_id=job_id,
            request_id="short-run-cancel",
            status="cancelled",
        )


def main() -> int:
    """执行一次不接触正式候选库的短窗口闭环。"""

    catalog = load_catalog()
    budget = SearchBudget(
        max_search_turns=2,
        max_hypothesis_branches=1,
        max_refines_per_branch=1,
        max_candidates_per_turn=1,
        max_candidates_per_run=2,
        max_full_evaluations=1,
        stop_after_no_improvement_turns=2,
        max_tool_calls_per_turn=0,
    )
    existing = tuple(
        sorted(
            name
            for names in available_factors(FACTOR_DIR).values()
            for name in names
        )
    )
    context = FrozenSearchContext(
        data_fingerprint=data_snapshot_fingerprint(
            DB_FILE,
            DEFAULT_DATA_MANIFEST,
        ),
        windows=WINDOWS,
        llm_model=load_worker_config().default_model,
        field_catalog=catalog.frozen_field_mappings(),
        operator_catalog=catalog.frozen_operator_mappings(),
        existing_factors=existing,
        budget=budget,
    )
    with TemporaryDirectory(prefix="mining-v2-short-") as directory:
        root = Path(directory)
        store = MiningV2Store(root / "mining.duckdb", root / "data")
        orchestrator = MiningSearchOrchestrator(
            store=store,
            worker=ShortRunWorker(),
            snapshot_check=lambda *args, **kwargs: None,
        )
        try:
            run = orchestrator.create_run(
                SearchRunRequest(
                    "short-real-window-20261029",
                    "短期价格反转可能带来开发期超额收益",
                    budget=budget,
                ),
                context,
                background=False,
            )
            result = orchestrator.execute(run["run_id"])
            trace = store.get_search_trace(run["run_id"])
            scorecards = build_run_scorecards(store, run["run_id"])
            output = {
                "run_id": run["run_id"],
                "status": result["status"],
                "context_hash": context.context_hash,
                "trace_hash": trace["trace_hash"],
                "decisions": [
                    {
                        "turn_index": item["turn_index"],
                        "action": item["action"],
                    }
                    for item in trace["decisions"]
                ],
                "candidate_statuses": [
                    {
                        "candidate_id": item["candidate_id"],
                        "status": item["status"],
                    }
                    for item in trace["candidates"]
                ],
                "evaluation_evidence_count": len(
                    store.list_evaluation_evidence(run["run_id"])
                ),
                "trial_summary": store.trial_summary(run["run_id"]),
                "scorecard_hashes": [
                    item.scorecard_hash for item in scorecards
                ],
            }
        finally:
            orchestrator.close()
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if output["status"] == "review_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
