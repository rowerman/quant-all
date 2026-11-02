"""V3 正式发布到因子、评估、模型和回测的联合验收。"""

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
from tests.mining_v3_fixtures import approved_v3_run


pytestmark = pytest.mark.skipif(
    not DB_FILE.exists() or not FACTOR_DB_FILE.exists(),
    reason="需要真实行情库和因子目录库",
)

START = "2024-05-06"
END = "2024-05-10"
STOCKS = ("SH600000", "SZ000001")


def test_v3_published_factor_reaches_all_downstream_consumers(tmp_path: Path):
    """真实临时发布后，V3 因子可被现有下游完整解析。"""

    store, input_envelope, envelope = approved_v3_run(
        tmp_path,
        name="V3_DOWNSTREAM_TREND",
    )
    candidate_id = envelope.candidate.candidate_id
    factor_name = envelope.candidate.name
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
        input_envelope.run_id,
        (candidate_id,),
        expected_context_hash=input_envelope.context_hash,
        background=False,
    )

    assert published["status"] == "published"
    assert published["source_kind"] == "v3_research"
    assert store.get_candidate(
        input_envelope.run_id,
        candidate_id,
    )["status"] == "published"
    manifest = json.loads(
        (factor_dir / "mined" / "definitions.json").read_text(encoding="utf-8")
    )
    entry = next(
        item for item in manifest["factors"] if item["name"] == factor_name
    )
    assert entry["source_kind"] == "v3_research"
    assert entry["candidate_envelope_hash"] == envelope.envelope_hash
    assert entry["source_reference_hash"] == envelope.source.reference_hash
    assert entry["approval_evidence_hash"]
    assert factors.resolve(factor_name, factor_dir).family == "mined"

    listed = quant.load_factor_list(
        family="mined",
        db_path=catalog_file,
    )
    assert factor_name in set(listed["name"])
    panel = quant.load_factor_panel(
        start=START,
        end=END,
        names=[factor_name],
        stocks=list(STOCKS),
        factor_dir=factor_dir,
    )
    assert not panel.empty
    assert factor_name in panel.columns
    wide = quant.load_factor_wide(
        factor_name,
        start=START,
        end=END,
        factor_dir=factor_dir,
    )
    assert not wide.empty

    evaluated = build_panel(
        EvalParams(
            factor=factor_name,
            start=START,
            end=END,
        ),
        db_file=DB_FILE,
        factor_dir=factor_dir,
    )
    assert evaluated.spec.name == factor_name
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
            features=[factor_name],
            allow_holdout=True,
        ),
        (factor_name,),
        signal_dates=calendar[-1:],
    )
    assert not model_data.empty
    assert factor_name in model_data.columns

    market_data = load_market_data(
        START,
        END,
        [factor_name],
        db_path=DB_FILE,
        factor_dir=factor_dir,
        benchmark_code=None,
        stock_codes=STOCKS,
    )
    assert set(market_data.factors) == {factor_name}
    assert not market_data.factors[factor_name].empty
    snapshot = build_input_snapshot(
        StrategySpec(
            universe="all_a",
            signal=SignalSpec(
                factors=(factor_name,),
                weights=(1.0,),
            ),
        ),
        market_data.data_snapshot,
    )
    assert factor_name in snapshot.scope["factors"]
    assert factor_name in snapshot.components["factors"]

    service.initialize()
    assert service.get_batch(
        input_envelope.run_id,
        published["batch_id"],
    )["status"] == "published"
