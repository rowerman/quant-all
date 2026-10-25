"""回测内存剖析工具的静态口径测试。"""

from tools.monitor.backtest_profile import case_payloads, read_vm_mb, summarize_stages


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
