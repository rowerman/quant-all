"""计划驱动完整账本的候选、策略和产物契约测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from alpha.evaluation_plan import EvaluationPlan
from alpha.ledger_acceptance import (
    LEDGER_CANDIDATE_SCHEMA_VERSION,
    candidate_names,
    candidate_strategy,
    ledger_analysis_id,
    load_evaluation,
)
from common.paths import CONFIG_DIR
from common.paths import MODEL_DATA_DIR
from scripts.build_alpha.backtest_accept_artifacts import (
    candidate_is_complete,
    write_candidate_artifacts,
)
@pytest.fixture(scope="module", autouse=True)
def _remove_private_provider():
    """测试结束后移除只在通用账本命令中注册的私有 provider。"""

    yield
    from backtest import providers

    providers._REGISTRY.pop("plan_rank", None)
    providers._ORIGINS.pop("plan_rank", None)
    providers._VERSIONS.pop(("plan_rank", "1"), None)
    providers._VERSION_ORIGINS.pop(("plan_rank", "1"), None)


def _plan() -> EvaluationPlan:
    return EvaluationPlan.from_file(
        CONFIG_DIR / "alpha" / "development_evaluation_v85.json"
    )


def test_candidate_selection_and_strategy_come_from_evaluation():
    """候选名来自评估 manifest，策略由计划规则统一生成。"""

    plan = _plan()
    _folder, manifest, loaded = load_evaluation("evaluation_d57dd9e6e750")
    assert loaded.to_dict() == plan.to_dict()
    assert candidate_names(plan, manifest, "selected") == (
        "LIGHTGBM_SMOOTH_4W",
    )
    assert candidate_names(
        plan,
        manifest,
        "SKLEARN_HIST_RAW",
    ) == ("SKLEARN_HIST_RAW",)
    strategy = candidate_strategy(
        "evaluation_d57dd9e6e750",
        plan.by_name("SKLEARN_HIST_RAW"),
        ["2023-01-06"],
    )
    assert strategy.signal.provider.name == "plan_rank"
    assert strategy.signal.provider.params["candidate"] == "SKLEARN_HIST_RAW"
    assert strategy.portfolio.top_n == 30


def test_ledger_id_is_deterministic_and_scope_sensitive():
    """账本 ID 绑定评估来源、候选、范围和代码。"""

    scope = {
        "start": "2018-01-12",
        "end": "2023-12-29",
        "init_cash": 1_000_000.0,
    }
    left = ledger_analysis_id(
        "evaluation_demo",
        "manifest",
        ("SKLEARN_HIST_RAW",),
        scope,
        "code",
    )
    right = ledger_analysis_id(
        "evaluation_demo",
        "manifest",
        ("SKLEARN_HIST_RAW",),
        dict(scope),
        "code",
    )
    changed = ledger_analysis_id(
        "evaluation_demo",
        "manifest",
        ("SKLEARN_HIST_RAW",),
        {**scope, "init_cash": 2_000_000.0},
        "code",
    )
    assert left == right
    assert left != changed
    assert left.startswith("acceptance_")


def test_plan_rank_provider_reads_hash_checked_candidate_scores():
    """provider 从评估候选分数生成计划持仓权重。"""

    from backtest.strategies._plan_rank import PlanRankProvider

    evaluation_id = "evaluation_d57dd9e6e750"
    scores = pd.read_parquet(
        MODEL_DATA_DIR
        / "runs"
        / "evaluation"
        / evaluation_id
        / "scores"
        / "SKLEARN_HIST_RAW.parquet"
    )
    day = pd.Timestamp(scores["trade_date"].min())
    columns = pd.Index(
        sorted(
            scores.loc[
                scores["trade_date"].eq(day), "stock_code"
            ].astype(str).unique()
        )[:40]
    )
    data = SimpleNamespace(
        index=pd.DatetimeIndex([day]),
        columns=columns,
    )
    eligible = pd.DataFrame(True, index=data.index, columns=columns)
    weights = PlanRankProvider().build(
        data,
        eligible,
        {
            "evaluation_id": evaluation_id,
            "candidate": "SKLEARN_HIST_RAW",
            "rebalance_dates": [str(day.date())],
        },
    )
    positive = weights.loc[day].loc[weights.loc[day].gt(0)]
    assert len(positive) == 30
    assert positive.sum() == pytest.approx(1.0)


def test_plan_rank_preserves_buffer_state_across_chunks():
    """分块调用会延续同一候选的上一块目标持仓。"""

    from backtest.strategies._plan_rank import PlanRankProvider

    evaluation_id = "evaluation_d57dd9e6e750"
    scores = pd.read_parquet(
        MODEL_DATA_DIR
        / "runs"
        / "evaluation"
        / evaluation_id
        / "scores"
        / "RIDGE_BUFFER_45.parquet"
    )
    dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(scores["trade_date"]).unique())
    )[:2]
    columns = pd.Index(
        sorted(scores["stock_code"].astype(str).unique())[:50]
    )
    provider = PlanRankProvider()
    for day in dates:
        data = SimpleNamespace(
            index=pd.DatetimeIndex([day]),
            columns=columns,
        )
        provider.build(
            data,
            pd.DataFrame(True, index=data.index, columns=columns),
            {
                "evaluation_id": evaluation_id,
                "candidate": "RIDGE_BUFFER_45",
                "rebalance_dates": [str(value.date()) for value in dates],
            },
        )
    history = provider._signal_history[
        (evaluation_id, "RIDGE_BUFFER_45")
    ]
    assert history["last_date"] == dates[-1]


def test_generic_candidate_completion_schema(tmp_path):
    """通用账本完成标记使用独立 schema 并能校验文件哈希。"""

    folder = tmp_path / "candidate"
    write_candidate_artifacts(
        folder,
        analysis_id="acceptance_demo",
        candidate="DEMO",
        metrics={"candidate": "DEMO"},
        strategy={"name": "DEMO"},
        yearly=pd.DataFrame({"year": [2023]}),
        daily=pd.DataFrame(
            {"return": [0.01]},
            index=pd.DatetimeIndex(["2023-01-03"]),
        ),
        orders=pd.DataFrame({"stock_code": ["SH600000"]}),
        holdings=pd.DataFrame({"stock_code": ["SH600000"]}),
        run_manifest={"snapshot_id": "a" * 64},
        schema_version=LEDGER_CANDIDATE_SCHEMA_VERSION,
    )
    assert candidate_is_complete(
        folder,
        "acceptance_demo",
        "DEMO",
        schema_version=LEDGER_CANDIDATE_SCHEMA_VERSION,
    )
