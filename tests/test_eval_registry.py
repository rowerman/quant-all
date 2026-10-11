"""指标注册表的单元测试：注册、重名、顺序与扩展方式。"""

import calculate
from calculate.metrics import (
    Metric,
    MetricParam,
    get_metric,
    list_metrics,
    metrics_by_stage,
    register_metric,
    unregister_metric,
)
from calculate.errors import MetricError

BUILTIN_KEYS = [
    "coverage", "ic_stats", "ic_series", "ic_dist",
    "bin_returns", "bin_curves", "spread", "bin_dist", "turnover",
]


def test_builtin_metrics_registered_in_order():
    """内置指标按 stage 顺序排列，前端卡片顺序直接用它。"""
    assert [metric.key for metric in list_metrics()] == BUILTIN_KEYS


def test_metric_describe_is_flat():
    """指标的描述信息要能直接塞进 JSON。"""
    described = get_metric("coverage").describe()
    assert described["stage"] == "overview"
    assert described["params"] == []
    assert described["requires_quantiles"] is False


def test_duplicate_registration_raises_with_code():
    """同名指标重复注册要报 EVAL-201，不能悄悄覆盖。"""
    duplicated = Metric(
        key="coverage", label="重复的覆盖率", stage="overview",
        compute=lambda context, options: {},
    )
    try:
        register_metric(duplicated)
    except MetricError as error:
        assert error.code == "EVAL-201"
    else:  # pragma: no cover - 走到这里说明校验失效
        raise AssertionError("重复注册没有报错")


def test_custom_metric_extends_stage_plan():
    """加一个指标只需要注册一次：stage 计划与参数声明都会自动跟着走。"""
    fake = Metric(
        key="fake_decay",
        label="假的衰减",
        stage="decay",
        params=(MetricParam(key="quantiles", label="组数", kind="enum",
                            default=10, choices=(5, 10, 20)),),
        compute=lambda context, options: {"ok": True},
    )
    assert "decay" not in calculate.planned_stages()
    register_metric(fake)
    try:
        assert "decay" in calculate.planned_stages()
        assert metrics_by_stage("decay") == (fake,)
        described = get_metric("fake_decay").describe()
        assert described["params"][0]["choices"] == [5, 10, 20]
        assert described["params"][0]["default"] == 10
    finally:
        unregister_metric("fake_decay")
    assert "decay" not in calculate.planned_stages()


def test_unknown_metric_raises():
    """取不存在的指标要报错并列出已注册指标。"""
    try:
        get_metric("not_exist")
    except MetricError as error:
        assert error.code == "EVAL-201"
        assert "coverage" in error.message
    else:  # pragma: no cover - 走到这里说明校验失效
        raise AssertionError("未知指标没有报错")
