"""S7 浏览器验收专用应用：临时候选库、确定性候选和快速评估结果。"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import shutil
import tempfile
from threading import Thread
import time
from types import SimpleNamespace
from typing import Any

from backend.app import create_app
from backend.mining_service import MiningWebService
from mining.contract_evidence import EvaluationReport, TechnicalGate
from mining.contract_requests import FrozenContext, MiningConstraints, MiningRunRequest
from mining.contract_types import sha256_json, utc_now_iso
from mining.expression_validation import validate_expression
from mining.orchestration_support import context_from_run, progress_for_run
from mining.storage import MiningStore


class FixtureOrchestrator:
    """把新建运行立即物化为可审核候选，不访问 Worker 或行情库。"""

    def __init__(self, store: MiningStore) -> None:
        self.store = store

    def create_run(self, request: MiningRunRequest, *, background: bool = True):
        context = _context(request)
        run = self.store.create_run(request, context, worker_mode="rdagent")
        run_id = run["run_id"]
        run = self.store.get_run(run_id)
        context = context_from_run(run)
        first = _candidate(
            "mean(close,5)",
            "收盘价短期均值",
            1,
            description="收盘价五日均值可能反映短期趋势延续。",
        )
        second = _candidate(
            "mean(volume_share,5)",
            "成交量占比短期均值",
            2,
            description="成交量占比抬升可能反映资金关注度变化。",
        )
        rejected = _candidate(
            "mean(turnover_rate,5)",
            "换手率短期均值",
            3,
            description="换手率变化可能反映交易拥挤程度。",
        )
        failed = _candidate(
            "mean(volume_share,10)",
            "成交量占比中期均值",
            4,
            description="较慢成交量均值用于验证测试段失败候选。",
        )
        for definition in (first, second, rejected, failed):
            self.store.add_candidate(run_id, definition)
        self.store.transition_run(
            run_id,
            "generating",
            event_key="fixture_generating",
            stage="generating",
        )
        thread = Thread(
            target=self._finish_run,
            args=(run_id, context, first, second, rejected, failed),
            name=f"fixture-{run_id}",
            daemon=True,
        )
        thread.start()
        return self.store.get_run(run_id)

    def _finish_run(
        self,
        run_id: str,
        context: FrozenContext,
        first,
        second,
        rejected,
        failed,
    ) -> None:
        """稍后生成候选，给浏览器留出连接并验证 SSE 续传的时间。"""

        time.sleep(2)
        for definition in (first, second, rejected):
            _prepare_candidate(self.store, run_id, context, definition)
        self.store.claim_holdout(
            run_id=run_id,
            candidate_ids=[first.candidate_id, second.candidate_id, rejected.candidate_id],
            evidence_hash="e" * 64,
        )
        for definition in (first, second, rejected):
            self.store.transition_candidate(
                run_id,
                definition.candidate_id,
                "review_ready",
                event_key=f"fixture_review_{definition.candidate_id}",
                stage="holdout",
            )
        _prepare_failed_candidate(self.store, run_id, context, failed)
        for target in ("screening", "evaluating", "holding_out", "review_ready"):
            self.store.transition_run(
                run_id,
                target,
                event_key=f"fixture_run_{target}",
                stage=target,
            )

    def progress(self, run_id: str):
        return progress_for_run(self.store, run_id)

    def cancel(self, run_id: str):
        return self.store.terminate_run(
            run_id,
            "cancelled",
            event_key="fixture_cancelled",
        )

    def resume(self, run_id: str, *, background: bool = True):
        return self.store.resume_run(run_id)

    def close(self) -> None:
        return None


def _context(request: MiningRunRequest) -> FrozenContext:
    return FrozenContext(
        data_fingerprint="a" * 64,
        field_catalog=({"name": "close"},),
        operator_catalog=({"name": "mean"},),
        budget=request.constraints,
        llm_profile=request.llm_profile,
    )


def _candidate(expression: str, mechanism: str, index: int, *, description: str):
    return validate_expression(expression, direction=1).to_candidate(
        mechanism=mechanism,
        description=description,
        steps=("读取登记字段", "按五日窗口计算均值", "按候选方向解释得分"),
        discovery_round=1,
        discovery_index=index,
    )


def _prepare_candidate(
    store: MiningStore,
    run_id: str,
    context: FrozenContext,
    definition,
) -> None:
    store.transition_candidate(
        run_id,
        definition.candidate_id,
        "validating",
        event_key=f"fixture_validating_{definition.candidate_id}",
        stage="validating",
    )
    store.record_technical_gate(
        TechnicalGate(
            run_id=run_id,
            candidate_id=definition.candidate_id,
            candidate_content_hash=definition.content_hash,
            context_hash=context.context_hash,
            rule_version="fixture-quality-v1",
            passed=True,
            checks={
                "values": {"coverage": 0.82, "non_finite": 0},
                "lookahead": {"passed": True, "max_relative_error": 0.0},
                "correlation": {"passed": True, "max_abs_rank_corr": 0.31},
            },
        )
    )
    store.record_evaluation(
        _report(run_id, context, definition, "technical", {"values": {"coverage": 0.82}})
    )
    store.transition_candidate(
        run_id,
        definition.candidate_id,
        "screening",
        event_key=f"fixture_screening_{definition.candidate_id}",
        stage="screening",
    )
    store.record_evaluation(
        _report(
            run_id,
            context,
            definition,
            "quick",
            {
                "rank_ic_mean": 0.037,
                "rank_ic_ir": 0.52,
                "positive_ratio": 0.58,
                "year_sign_consistency": 0.75,
                "valid_days": 2830,
                "min_valid_days": 20,
            },
        )
    )
    store.transition_candidate(
        run_id,
        definition.candidate_id,
        "evaluating",
        event_key=f"fixture_evaluating_{definition.candidate_id}",
        stage="evaluating",
    )
    store.record_evaluation(
        _report(
            run_id,
            context,
            definition,
            "full",
            {
                "events": {
                    "overview": {
                        "metrics": {
                            "ic_stats": {
                                "ic": {"mean": 0.021, "ir": 0.34, "win_rate": 0.55},
                                "rank_ic": {"mean": 0.037, "ir": 0.52, "win_rate": 0.58},
                                "valid_days": 2830,
                            }
                        }
                    },
                    "ic_series": {
                        "metrics": {
                            "ic_series": {
                                "dates": ["2024-01-02", "2024-01-03", "2024-01-04"],
                                "cumulative": {
                                    "ic": [0.01, 0.025, 0.04],
                                    "rank_ic": [0.02, 0.045, 0.07],
                                },
                            }
                        }
                    },
                },
                "elapsed_ms": 1250,
            },
        )
    )
    store.transition_candidate(
        run_id,
        definition.candidate_id,
        "evaluated",
        event_key=f"fixture_evaluated_{definition.candidate_id}",
        stage="evaluating",
    )
    store.transition_candidate(
        run_id,
        definition.candidate_id,
        "holdout_running",
        event_key=f"fixture_holdout_{definition.candidate_id}",
        stage="holdout",
    )
    store.record_evaluation(
        _report(
            run_id,
            context,
            definition,
            "holdout",
            {
                "valid_days": 243,
                "min_valid_days": 20,
                "complete": True,
                "stages": {
                    "overview": {
                        "metrics": {
                            "ic_stats": {
                                "ic": {"mean": 0.018, "ir": 0.27},
                                "rank_ic": {"mean": 0.031, "ir": 0.43},
                                "valid_days": 243,
                            }
                        }
                    }
                },
            },
        )
    )


def _prepare_failed_candidate(
    store: MiningStore,
    run_id: str,
    context: FrozenContext,
    definition,
) -> None:
    for target in ("validating", "screening", "evaluating", "evaluated", "holdout_running"):
        store.transition_candidate(
            run_id,
            definition.candidate_id,
            target,
            event_key=f"fixture_failed_{target}_{definition.candidate_id}",
            stage=target,
        )
    store.record_evaluation(
        _report(
            run_id,
            context,
            definition,
            "holdout",
            {"valid_days": 3, "min_valid_days": 20, "complete": False},
            status="failed",
            warnings=("模块测试段有效交易日不足或评估阶段不完整",),
        )
    )
    store.transition_candidate(
        run_id,
        definition.candidate_id,
        "holdout_failed",
        event_key=f"fixture_holdout_failed_{definition.candidate_id}",
        stage="holdout",
        failure_stage="holdout",
    )


def _report(
    run_id: str,
    context: FrozenContext,
    definition,
    stage: str,
    metrics: dict[str, Any],
    *,
    status: str = "passed",
    warnings: tuple[str, ...] = (),
) -> EvaluationReport:
    now = utc_now_iso()
    return EvaluationReport(
        run_id=run_id,
        candidate_id=definition.candidate_id,
        stage=stage,
        candidate_content_hash=definition.content_hash,
        context_hash=context.context_hash,
        input_hash=sha256_json(
            {"run_id": run_id, "candidate_id": definition.candidate_id, "stage": stage}
        ),
        status=status,
        metrics=metrics,
        warnings=warnings,
        artifact_path=None,
        artifact_sha256=None,
        error_code=None,
        error_message=None,
        started_at=now,
        finished_at=now,
    )


_configured_root = os.environ.get("MINING_FIXTURE_ROOT", "").strip()
_TEMP_ROOT = (
    Path(_configured_root)
    if _configured_root
    else Path(tempfile.mkdtemp(prefix="quant-mining-web-"))
)
_STORE = MiningStore(_TEMP_ROOT / "factor_mining.duckdb", _TEMP_ROOT / "data")
_SERVICE = MiningWebService(
    store=_STORE,
    orchestrator=FixtureOrchestrator(_STORE),
    defaults=MiningConstraints(
        max_rounds=1,
        max_candidates=4,
        full_eval_top_k=2,
        max_llm_calls=4,
        min_quality_checks_for_failure_stop=4,
    ),
    reviewer="browser-reviewer",
    snapshot_check=lambda _context, **_kwargs: None,
)


def _fixture_prepare_publication(**kwargs) -> SimpleNamespace:
    """浏览器夹具只验证页面状态，不重复执行真实全量构建。"""

    names = tuple(item["name"] for item in kwargs["candidates"])
    return SimpleNamespace(
        manifest_hash="a" * 64,
        artifact_sha256="b" * 64,
        factor_names=names,
    )


_SERVICE.publication_service._prepare = _fixture_prepare_publication
_SERVICE.publication_service._commit = lambda **_kwargs: None
_SERVICE.publication_service._rollback = lambda **_kwargs: None
app = create_app(mining_service=_SERVICE)


def cleanup() -> None:
    """删除浏览器夹具的临时候选库。"""

    shutil.rmtree(_TEMP_ROOT, ignore_errors=True)


atexit.register(cleanup)
