"""模型预处理基准门禁函数的单元测试。"""

from tools.monitor.modeling_preprocess_baseline import (
    _correction_pass,
    _performance_pass,
)


def _result(hash_value: str, elapsed_s: float = 1.0) -> dict:
    """构造最小结果记录。"""

    return {
        "combined_sha256": hash_value,
        "column_sha256": {"X": hash_value},
        "elapsed_s": elapsed_s,
    }


def test_correction_gate_requires_all_backends_identical():
    """任一后端列哈希不同都不能通过正确性门禁。"""

    passing = {"rank": {"python": _result("a")}}
    passing["rank"]["native"] = _result("a")
    passing["rank"]["auto"] = _result("a")
    assert _correction_pass(passing)

    passing["rank"]["native"] = _result("b")
    assert not _correction_pass(passing)


def test_performance_gate_requires_native_and_auto_not_slower():
    """native 或 auto 慢于 Python 都不能通过不退化门禁。"""

    passing = {
        "rank": {
            "python": _result("a", 3.0),
            "native": _result("a", 1.0),
            "auto": _result("a", 1.1),
        }
    }
    assert _performance_pass(passing)

    passing["rank"]["auto"] = _result("a", 3.1)
    assert not _performance_pass(passing)
