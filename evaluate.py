"""评估入口：按固定 stage 顺序产出结果，供接口层流式推送。

stage 顺序是前后端的公共契约。后续阶段（binplot、衰减、拆分、相关性）
只是往 ``STAGE_ORDER`` 里已经留好的位置填内容，接口层与前端都不用改。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator
from collections.abc import Mapping

from calculate.jsonutil import json_safe
from calculate.metrics import list_metrics, metrics_by_stage, resolve_options
from calculate.panel import EvalContext, FactorView, build_panel, default_params
from calculate.params import EvalParams
from calculate.stages import STAGE_ORDER
from calculate.config import DB_FILE, FACTOR_DIR
from calculate.errors import EvalParamError
from calculate.logging_conf import get_logger

logger = get_logger("calculate.evaluate")


def planned_stages() -> tuple[str, ...]:
    """列出当前已注册指标会真正产出的 stage（不含 meta 与 done）。"""
    return tuple(stage for stage in STAGE_ORDER[1:-1] if metrics_by_stage(stage))


def _resolve_metric_options(metric_options: Mapping | None) -> dict[str, dict]:
    """校验调用方传入的全部指标参数；在读库前失败。"""

    if metric_options is not None and not isinstance(metric_options, Mapping):
        raise EvalParamError("metric_options 必须是指标 key 到参数映射")
    supplied = {} if metric_options is None else dict(metric_options)
    unknown_metrics = supplied.keys() - {metric.key for metric in list_metrics()}
    if unknown_metrics:
        raise EvalParamError(f"未知指标参数：{sorted(unknown_metrics)}")
    return {
        metric.key: resolve_options(metric, supplied.get(metric.key))
        for metric in list_metrics()
    }


def meta_payload(context, stages=None, metric_options=None) -> dict:
    """组装 meta 事件：参数回显、因子定义、面板概览与指标清单。

    参数:
        context: 评估上下文。
        stages: 本次实际执行的阶段；None 表示全部已实现阶段。
        metric_options: 本次已校验的指标参数，用于回显。
    """
    spec = context.spec
    return {
        "factor": spec.name,
        "spec": {
            "name": spec.name,
            "family": spec.family,
            "formula": spec.formula,
            "description": spec.description,
            "steps": list(spec.steps),
            "direction": spec.direction,
            "lookback": spec.lookback,
            "fields": list(spec.fields),
            "extra_sources": list(spec.extra_sources),
        },
        "params": context.params.describe(),
        "universe": context.summary["universe"],
        "panel": {key: value for key, value in context.summary.items() if key != "universe"},
        "metrics": [metric.describe() for metric in list_metrics()],
        "stages": list(planned_stages() if stages is None else stages),
        "metric_options": metric_options or {},
    }


def evaluate(factor: str, params: EvalParams | None = None, *,
             db_file: Path | str | None = None, factor_dir: Path | str | None = None,
             stages=None, metric_options: Mapping | None = None,
             factor_view: FactorView | None = None) -> Iterator[tuple[str, dict]]:
    """跑一次因子评估，按 stage 逐段产出结果。

    参数:
        factor: 因子名。
        params: 评估参数；留空按「最近 3 年 + 默认口径」构造。
        db_file: 日线主库路径，默认取项目配置。
        factor_dir: 因子数据根目录，默认取项目配置。
        stages: 只跑这些已实现 stage；留空表示全部。
        metric_options: 指标 key 到参数映射，例如
            {"exposure": {"industry": "stock_basic"}}。不改变其他指标的参数。
        factor_view: 显式因子来源；供候选临时目录复用同一评估入口。

    产出:
        ``(stage, payload)`` 序列，payload 保证是 JSON 可序列化的 dict。
    """
    available = planned_stages()
    if isinstance(stages, str):
        raise EvalParamError("stages 必须是阶段名序列，不能是字符串")
    selected = None if stages is None else set(stages)
    unknown = set() if selected is None else selected - set(available)
    if unknown:
        raise EvalParamError(f"未实现或未知 stage：{sorted(unknown)}；可选：{list(available)}")
    _resolve_metric_options(metric_options)
    actual_stages = tuple(stage for stage in available if selected is None or stage in selected)
    target_db = db_file or DB_FILE
    target_dir = factor_dir or FACTOR_DIR
    if params is None:
        params = default_params(factor, db_file=target_db)
    elif params.factor != factor:
        raise EvalParamError(
            f"参数里的因子名 {params.factor!r} 与调用参数 {factor!r} 不一致"
        )

    context = build_panel(
        params,
        db_file=target_db,
        factor_dir=target_dir,
        factor=factor_view,
    )
    yield from evaluate_context(
        context,
        stages=actual_stages,
        metric_options=metric_options,
    )


def evaluate_context(
    context: EvalContext,
    *,
    stages=None,
    metric_options: Mapping | None = None,
) -> Iterator[tuple[str, dict]]:
    """在已构建的正式或候选面板上按固定 stage 顺序计算指标。"""

    started = time.perf_counter()
    available = planned_stages()
    selected = None if stages is None else set(stages)
    unknown = set() if selected is None else selected - set(available)
    if unknown:
        raise EvalParamError(f"未实现或未知 stage：{sorted(unknown)}；可选：{list(available)}")
    options = _resolve_metric_options(metric_options)
    actual_stages = tuple(
        stage for stage in available if selected is None or stage in selected
    )
    params = context.params
    timings = [{"stage": "meta", "elapsed_ms": round(context.panel_elapsed_ms, 1)}]
    yield ("meta", json_safe(meta_payload(context, actual_stages, options)))

    for stage in STAGE_ORDER[1:-1]:
        metrics = metrics_by_stage(stage)
        if not metrics or (selected is not None and stage not in selected):
            continue
        stage_started = time.perf_counter()
        results = {
            metric.key: json_safe(metric.compute(context, options[metric.key]))
            for metric in metrics
        }
        elapsed_ms = (time.perf_counter() - stage_started) * 1000
        timings.append({"stage": stage, "elapsed_ms": round(elapsed_ms, 1)})
        logger.info("stage %s 完成：%s 耗时 %.0fms", stage, list(results), elapsed_ms)
        yield (stage, {"stage": stage, "metrics": results, "elapsed_ms": round(elapsed_ms, 1)})

    total_ms = (time.perf_counter() - started) * 1000
    logger.info("评估完成：因子=%s 池=%s 总耗时 %.0fms", params.factor, params.universe, total_ms)
    yield ("done", json_safe({
        "stages": timings,
        "total_elapsed_ms": round(total_ms, 1),
        "panel": {key: value for key, value in context.summary.items() if key != "universe"},
    }))
