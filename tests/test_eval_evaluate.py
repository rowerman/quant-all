"""evaluate 生成器的单元测试：stage 顺序、可序列化与默认口径跑通。"""

import json

import pytest

from calculate import EvalParams, evaluate, planned_stages
from calculate.config import DB_FILE

pytestmark = pytest.mark.skipif(not DB_FILE.exists(), reason="需要真实主库 db/quant.duckdb")

START, END = "2024-05-06", "2024-05-31"


def test_stage_sequence_and_payloads(momentum_dir):
    """stage 顺序固定为契约，所有 payload 都必须是 JSON 可序列化的。"""
    params = EvalParams(factor="MOM_20", start=START, end=END)
    events = list(evaluate("MOM_20", params, factor_dir=momentum_dir))
    assert [stage for stage, _ in events] == [
        "meta", "overview", "ic_series", "ic_dist",
        "bins", "bin_curves", "spread", "bin_dist", "turnover", "decay", "done",
    ]
    for _, payload in events:
        assert isinstance(json.loads(json.dumps(payload)), dict)

    meta = events[0][1]
    assert meta["factor"] == "MOM_20"
    assert meta["params"]["label"] == "fwd_ret_close_5"
    assert meta["universe"]["key"] == "all_a"
    assert [metric["key"] for metric in meta["metrics"]] == [
        "coverage", "ic_stats", "ic_series", "ic_dist",
        "bin_returns", "bin_curves", "spread", "bin_dist", "turnover", "decay",
    ]
    assert meta["panel"]["days"] == 20
    assert planned_stages() == (
        "overview", "ic_series", "ic_dist", "bins", "bin_curves", "spread", "bin_dist",
        "turnover", "decay",
    )

    payloads = dict(events)
    assert payloads["overview"]["metrics"]["ic_stats"]["valid_days"] == 20
    assert payloads["ic_series"]["metrics"]["ic_series"]["rolling"]["window"] == 20
    assert len(payloads["bins"]["metrics"]["bin_returns"]["equal"]["mean"]) == 10
    assert len(payloads["bin_curves"]["metrics"]["bin_curves"]["dates"]) == 20
    assert payloads["spread"]["metrics"]["spread"]["long_bin"] == 10
    assert len(payloads["bin_dist"]["metrics"]["bin_dist"]["daily"]["market_cap"]) == 10
    assert set(payloads["turnover"]["metrics"]["turnover"]["autocorrelation"]) == {
        "lag1", "lag5", "lag20"
    }
    assert [item["horizon"] for item in payloads["decay"]["metrics"]["decay"]["curve"]] == [
        1, 3, 5, 10, 20
    ]
    assert [item["stage"] for item in payloads["done"]["stages"]] == [
        "meta", "overview", "ic_series", "ic_dist",
        "bins", "bin_curves", "spread", "bin_dist", "turnover", "decay",
    ]


def test_stage_filter(momentum_dir):
    """只跑指定 stage 时其余 stage 直接跳过。"""
    params = EvalParams(factor="MOM_20", start=START, end=END)
    events = list(evaluate("MOM_20", params, factor_dir=momentum_dir, stages=["ic_series"]))
    assert [stage for stage, _ in events] == ["meta", "ic_series", "done"]


def test_factor_name_mismatch_raises(momentum_dir):
    """参数里的因子名与调用参数不一致时直接报错。"""
    params = EvalParams(factor="MOM_60", start=START, end=END)
    with pytest.raises(Exception) as error:
        list(evaluate("MOM_20", params, factor_dir=momentum_dir))
    assert "不一致" in str(error.value)


def test_default_window_on_real_factor_library():
    """默认口径（最近 3 年 + 全A池 + close_5）在真实因子库上跑通且数值可用。"""
    events = list(evaluate("MOM_20"))
    payloads = dict(events)
    stats = payloads["overview"]["metrics"]["ic_stats"]
    assert stats["insufficient"] is False
    assert stats["valid_days"] > 500
    assert stats["rank_ic"]["mean"] is not None
    assert payloads["done"]["total_elapsed_ms"] < 30_000
