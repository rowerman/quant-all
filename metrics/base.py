"""指标注册表：把「算一个指标」变成一条可插拔的声明。

每个指标声明自己的 key、中文名、stage（同时是前端卡片与 SSE 事件名）、
参数与计算函数。新增指标只要加一个模块并注册，接口层与前端都不用改。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping as MappingABC
from typing import TYPE_CHECKING, Any, Callable, Mapping

from calculate.errors import MetricError
from calculate.stages import STAGE_ORDER

if TYPE_CHECKING:  # 只用于类型标注，避免与 panel 相互导入
    from calculate.panel import EvalContext


@dataclass(frozen=True)
class MetricParam:
    """指标自己的参数。

    参数:
        key: 参数名。
        label: 中文名，展示到前端。
        kind: 类型，``int`` / ``bool`` / ``enum``。
        default: 默认值。
        choices: ``enum`` 类型的可选值。
    """

    key: str
    label: str
    kind: str
    default: Any = None
    choices: tuple = ()


@dataclass(frozen=True)
class Metric:
    """一个可注册的指标。

    参数:
        key: 唯一标识，前端卡片 id。
        label: 中文标题。
        stage: 所属阶段，同时作为 SSE 事件名。
        compute: 计算函数，签名 ``compute(ctx, options) -> dict``。
        params: 指标参数声明。
        requires_quantiles: 是否依赖分层结果（留给后续 binplot 阶段使用）。
    """

    key: str
    label: str
    stage: str
    compute: Callable[["EvalContext", Mapping[str, Any]], dict]
    params: tuple[MetricParam, ...] = ()
    requires_quantiles: bool = False

    def describe(self) -> dict:
        """摊平成可序列化字典，供接口层与 meta 事件使用。"""
        return {
            "key": self.key,
            "label": self.label,
            "stage": self.stage,
            "requires_quantiles": self.requires_quantiles,
            "params": [
                {
                    "key": param.key,
                    "label": param.label,
                    "kind": param.kind,
                    "default": param.default,
                    "choices": list(param.choices),
                }
                for param in self.params
            ],
        }


_METRICS: dict[str, Metric] = {}
_STAGE_POSITION = {stage: index for index, stage in enumerate(STAGE_ORDER)}


def register_metric(metric: Metric, overwrite: bool = False) -> Metric:
    """登记一个指标。

    参数:
        metric: 指标定义。
        overwrite: 同名时是否覆盖，默认报错。
    """
    if not metric.key or not metric.stage:
        raise MetricError("指标必须有 key 与 stage")
    if metric.stage not in STAGE_ORDER[1:-1]:
        raise MetricError(f"非法指标 stage {metric.stage!r}；可选：{list(STAGE_ORDER[1:-1])}")
    if not callable(metric.compute):
        raise MetricError(f"指标 {metric.key} 的 compute 必须可调用")
    if metric.key in _METRICS and not overwrite:
        raise MetricError(f"指标 {metric.key} 重复注册；要覆盖请显式传 overwrite=True")
    _METRICS[metric.key] = metric
    return metric


def get_metric(key: str) -> Metric:
    """按 key 取指标。

    参数:
        key: 指标标识。
    """
    metric = _METRICS.get(key)
    if metric is None:
        raise MetricError(f"未知指标 {key!r}；已注册：{sorted(_METRICS)}")
    return metric


def unregister_metric(key: str) -> None:
    """注销一个指标，找不到就静默返回。

    参数:
        key: 指标标识。
    """
    _METRICS.pop(key, None)


def list_metrics() -> tuple[Metric, ...]:
    """返回全部指标，按 stage 顺序排列（即前端卡片顺序）。"""
    return tuple(
        sorted(_METRICS.values(), key=lambda item: _STAGE_POSITION[item.stage])
    )


def metrics_by_stage(stage: str) -> tuple[Metric, ...]:
    """返回某个阶段下的指标，保持注册顺序。

    参数:
        stage: 阶段名。
    """
    return tuple(metric for metric in list_metrics() if metric.stage == stage)


def default_options(metric: Metric) -> dict:
    """取出某个指标的默认参数取值。

    参数:
        metric: 指标定义。
    """
    return {param.key: param.default for param in metric.params}


def resolve_options(metric: Metric, options: Mapping[str, Any] | None = None) -> dict:
    """合并并校验指标参数，错误在读库前返回。

    参数:
        metric: 指标定义。
        options: 参数名到取值的映射，未提供的参数取默认值。
    """
    if options is not None and not isinstance(options, MappingABC):
        raise MetricError(f"指标 {metric.key} 的参数必须是映射")
    supplied = {} if options is None else dict(options)
    declared = {param.key: param for param in metric.params}
    unknown = supplied.keys() - declared.keys()
    if unknown:
        raise MetricError(f"指标 {metric.key} 未声明参数：{sorted(unknown)}")
    values = default_options(metric) | supplied
    for key, param in declared.items():
        value = values[key]
        valid = (param.kind == "bool" and isinstance(value, bool)
                 or param.kind == "int" and isinstance(value, int) and not isinstance(value, bool)
                 or param.kind == "enum" and value in param.choices)
        if not valid:
            raise MetricError(f"指标 {metric.key} 参数 {key} 非法：{value!r}；"
                              f"类型={param.kind}，可选={list(param.choices)}")
    return values


def clear_metrics() -> None:
    """清空注册表，仅供测试使用。"""
    _METRICS.clear()
