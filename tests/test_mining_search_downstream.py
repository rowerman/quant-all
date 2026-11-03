"""V2 正式发布到因子、评估、模型和回测的联合验收。"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

import factors
import quant
from alpha.config import DEFAULT_CONTRACT
from alpha.dataset import DAILY_FIELDS, DatasetSpec, assemble_dataset
from backtest.data import load_market_data
from backtest.snapshot import build_input_snapshot
from backtest.specs import SignalSpec, StrategySpec
from calculate.panel import build_panel
from calculate.params import EvalParams
from common.paths import DB_FILE, FACTOR_DB_FILE
from mining.publication_service import MiningPublicationService
from tests.mining_v2_fixtures import approved_search_run


pytestmark = pytest.mark.skipif(
    not DB_FILE.exists() or not FACTOR_DB_FILE.exists(),
    reason="需要真实行情库和因子目录库",
)

START = "2024-05-06"
END = "2024-05-10"
STOCKS = ("SH600000", "SZ000001")


def test_v2_published_factors_reach_all_downstream_consumers(tmp_path: Path):
    """真实临时发布后，V2 因子可被现有下游完整解析。"""

    store, run_id, candidate_ids = approved_search_run(tmp_path)
    candidate_names = tuple(
        store.get_candidate(run_id, item)["name"]
        for item in candidate_ids
    )
    factor_dir = tmp_path / "factors"
    factor_dir.mkdir()
    catalog_file = tmp_path / "factors.duckdb"
    shutil.copy2(FACTOR_DB_FILE, catalog_file)
    service = MiningPublicationService(
        store=store,
        db_file=DB_FILE,
        factor_dir=factor_dir,
        catalog_file=catalog_file,
        data_root=tmp_path / "publication-data",
        snapshot_check=lambda *_args, **_kwargs: None,
    )

    published = service.start_batch(
        run_id,
        candidate_ids,
        expected_context_hash=store.get_run(run_id)["context_hash"],
        background=False,
    )

    assert published["status"] == "published"
    assert published["source_kind"] == "v2_search"
    assert {
        store.get_candidate(run_id, item)["status"]
        for item in candidate_ids
    } == {"published"}
    manifest = json.loads(
        (factor_dir / "mined" / "definitions.json").read_text(encoding="utf-8")
    )
    manifest_by_name = {
        item["name"]: item for item in manifest["factors"]
    }
    for name in candidate_names:
        assert manifest_by_name[name]["source_kind"] == "v2_search"
        assert manifest_by_name[name]["source_context_hash"]
        assert manifest_by_name[name]["source_trace_hash"]
        assert manifest_by_name[name]["scorecard_hash"]
        assert manifest_by_name[name]["approval_evidence_hash"]
        assert factors.resolve(name, factor_dir).family == "mined"

    listed = quant.load_factor_list(
        family="mined",
        db_path=catalog_file,
    )
    assert set(candidate_names) <= set(listed["name"])
    panel = quant.load_factor_panel(
        start=START,
        end=END,
        names=list(candidate_names),
        stocks=list(STOCKS),
        factor_dir=factor_dir,
    )
    assert not panel.empty
    assert set(candidate_names) <= set(panel.columns)
    wide = quant.load_factor_wide(
        candidate_names[0],
        start=START,
        end=END,
        factor_dir=factor_dir,
    )
    assert not wide.empty

    evaluated = build_panel(
        EvalParams(
            factor=candidate_names[0],
            start=START,
            end=END,
        ),
        db_file=DB_FILE,
        factor_dir=factor_dir,
    )
    assert evaluated.spec.name == candidate_names[0]
    assert evaluated.summary["rows"] > 0

    calendar = pd.DatetimeIndex(
        quant.load_calendar(start=START, end=END, db_path=DB_FILE)["trade_date"]
    )
    daily = quant.load_panel(
        start=START,
        end=END,
        fields=list(DAILY_FIELDS),
        db_path=DB_FILE,
    )
    labels = quant.load_label(
        start=START,
        end=END,
        horizons=[DEFAULT_CONTRACT.horizon],
        price=DEFAULT_CONTRACT.price,
        db_path=DB_FILE,
    )
    model_data = assemble_dataset(
        panel,
        daily,
        labels,
        pd.DataFrame(columns=["stock_code", "trade_date", "industry_l1"]),
        calendar,
        DatasetSpec(
            start=START,
            end=END,
            features=candidate_names,
            allow_holdout=True,
        ),
        candidate_names,
        signal_dates=calendar[-1:],
    )
    assert not model_data.empty
    assert set(candidate_names) <= set(model_data.columns)

    market_data = load_market_data(
        START,
        END,
        list(candidate_names),
        db_path=DB_FILE,
        factor_dir=factor_dir,
        benchmark_code=None,
        stock_codes=STOCKS,
    )
    assert set(market_data.factors) == set(candidate_names)
    assert all(not frame.empty for frame in market_data.factors.values())
    snapshot = build_input_snapshot(
        StrategySpec(
            universe="all_a",
            signal=SignalSpec(
                factors=candidate_names,
                weights=(1.0, 1.0),
            ),
        ),
        market_data.data_snapshot,
    )
    assert set(snapshot.scope["factors"]) == set(candidate_names)
    assert set(snapshot.components["factors"]) == set(
        candidate_names
    )

    service.initialize()
    assert service.get_batch(run_id, published["batch_id"])["status"] == (
        "published"
    )
    assert store.validate_search_trace(run_id) is True
