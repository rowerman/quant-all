"""回测内存剖析工具的静态口径测试。"""

import json

import pandas as pd

from tools.monitor.backtest_profile import (
    case_payloads,
    read_vm_mb,
    summarize_hierarchy,
    summarize_stages,
)
from tools.monitor.backtest_profile_fingerprint import (
    call_signature,
    stable_frame_digest,
)


def test_profile_cases_cover_large_pool_long_window_and_dual():
    """三个固定案例分别覆盖大股票池、长区间和双任务并发。"""
    all_a = case_payloads("all_a_3y")
    index = case_payloads("index_10y")
    dual = case_payloads("dual")
    assert len(all_a) == 1 and all_a[0]["universe"] == "all_a"
    assert len(index) == 1 and index[0]["universe"] == "index:000905.SH"
    assert dual == [all_a[0], index[0]]


def test_profile_reader_returns_current_process_memory():
    """读取当前进程 RSS 与 VmHWM 应返回正数。"""
    assert read_vm_mb("VmRSS") and read_vm_mb("VmRSS") > 0
    assert read_vm_mb("VmHWM") and read_vm_mb("VmHWM") > 0


def test_stage_summary_aggregates_repeated_events():
    """大量分段事件要聚合，避免剖析输出被单个 chunk 行淹没。"""
    summary = summarize_stages(
        [
            {"stage": "chunk", "elapsed_s": 1.5},
            {"stage": "chunk", "elapsed_s": 2.5},
            {"stage": "execute", "elapsed_s": 0.5},
        ]
    )
    assert summary["chunk"] == {
        "count": 2, "elapsed_s": 4.0, "max_elapsed_s": 2.5
    }
    assert summary["execute"]["count"] == 1


def test_hierarchy_accounts_for_each_event_once():
    """段外层和段内事件分组后必须与原始记录总耗时一致。"""
    stages = [
        {"stage": "meta", "elapsed_s": 0.5},
        {"stage": "load", "elapsed_s": 2.0},
        {"stage": "chunk", "elapsed_s": 0.1, "phase": "prepare"},
        {"stage": "signals", "elapsed_s": 0.3},
        {"stage": "chunk", "elapsed_s": 0.2, "phase": "segment"},
        {"stage": "probe", "elapsed_s": 0.4},
        {"stage": "allocation", "elapsed_s": 0.5},
        {"stage": "done", "elapsed_s": 1.0},
    ]
    result = summarize_hierarchy(stages)
    assert result["accounted_elapsed_s"] == result["grouped_elapsed_s"] == 5.0
    assert result["groups"]["chunk.prepare"]["elapsed_s"] == 0.4
    assert result["groups"]["chunk.segment"]["elapsed_s"] == 1.1
    assert result["children"]["allocation"] == {"count": 1, "elapsed_s": 0.5}


def test_call_signature_compresses_large_values():
    """调用签名只保留大表和序列的形状，不把整段内容写入证据。"""
    signature, example = call_signature(
        "engine.run_backtest",
        (pd.DataFrame({"value": [1.0, 2.0]}),),
        {"mask": pd.Series([True, False]), "state": object()},
    )
    encoded = json.dumps(example)
    assert signature.startswith("engine.run_backtest:")
    assert len(encoded) < 1000
    assert example["args"] == [{"frame_shape": [2, 1]}]
    assert example["kwargs"]["mask"]["series_length"] == 2


def test_result_digest_ignores_random_run_id():
    """同一业务结果更换 run_id 后指纹保持不变。"""
    first = pd.DataFrame({"run_id": ["first"], "value": [1.5]})
    second = pd.DataFrame({"run_id": ["second"], "value": [1.5]})
    changed = pd.DataFrame({"run_id": ["first"], "value": [2.5]})
    assert stable_frame_digest(first) == stable_frame_digest(second)
    assert stable_frame_digest(first)["sha256"] != stable_frame_digest(changed)["sha256"]
