"""回测编排：读取策略版本 → 取数 → 执行 → 落库，并按阶段上报进度。

``iter_run`` 是一个生成器，每完成一个阶段就产出一条 ``(stage, payload)``；
后端把它交给调度器流式消费，命令行直接遍历打印。
VectorBT 的执行放在一个工作线程里，生成器本身负责分阶段汇报与取消检查。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from queue import Empty, Queue
from threading import Thread
from typing import Callable, Iterator

from backtest.chunked import plan_chunks, run_chunked, run_loaded
from backtest.config import MAX_PANEL_CELLS
from backtest.coverage import validate_window
from backtest.data import estimate_panel, load_market_data
from backtest.engine import BacktestResult
from backtest.errors import BacktestError, RunCancelled, SpecError
from backtest.logging_conf import get_logger
from backtest.specs import BacktestParams, BenchmarkSpec, StrategySpec
from backtest.store import BacktestStore
from backtest.universe import resolve_universe

logger = get_logger("backtest.pipeline")

Cancelled = Callable[[], bool]


@dataclass(frozen=True)
class RunRequest:
    """一次回测请求：策略版本加区间，带可选的运行级覆盖。"""

    strategy_id: str
    start: str
    end: str
    version: int | None = None
    init_cash: float = 1_000_000.0
    universe: str | None = None
    benchmark: str | None = None
    clear_benchmark: bool = False

    def __post_init__(self) -> None:
        if self.clear_benchmark and self.benchmark is not None:
            raise SpecError("benchmark 与 clear_benchmark 不能同时指定")

    @property
    def params(self) -> BacktestParams:
        """转换成回测参数。"""
        return BacktestParams(self.start, self.end, self.init_cash)


def effective_spec(spec: StrategySpec, request: RunRequest) -> StrategySpec:
    """应用运行级覆盖，得到真正执行的策略配置。

    策略版本本身不可变；覆盖后的配置会随运行元数据一起保存，保证可复现。

    参数:
        spec: 策略版本里的配置。
        request: 回测请求。
    """
    changes = {}
    if request.universe is not None:
        resolve_universe(request.universe)
        changes["universe"] = request.universe
    if request.clear_benchmark:
        changes["benchmark"] = BenchmarkSpec(index_code=None)
    elif request.benchmark is not None:
        changes["benchmark"] = BenchmarkSpec(index_code=request.benchmark)
    return replace(spec, **changes) if changes else spec


def _run_in_thread(runner: Callable[[Callable[[str, dict], None]], BacktestResult], events: Queue) -> None:
    """在工作线程里执行回测，把阶段事件推进队列。

    参数:
        runner: 接收阶段回调、返回回测结果的执行函数。
        events: 阶段事件队列。
    """

    try:
        result = runner(lambda stage, payload: events.put((stage, payload)))
        events.put(("__result__", result))
    except BaseException as error:  # 线程里的异常要带回主流程
        events.put(("__error__", error))
    finally:
        events.put(None)


def iter_run(
    store: BacktestStore,
    request: RunRequest,
    should_cancel: Cancelled | None = None,
) -> Iterator[tuple[str, dict]]:
    """执行一次回测并按阶段产出事件。

    参数:
        store: 回测存储。
        request: 回测请求。
        should_cancel: 可选取消检查，返回 True 时在当前阶段边界退出。
    """
    spec = effective_spec(store.get_version(request.strategy_id, request.version), request)
    params = request.params
    # 覆盖检查放在运行记录和面板读取之前，避免可预期的缺日留下失败运行。
    validate_window(spec.benchmark.index_code, params.start, params.end)
    run_id = store.start_run(
        request.strategy_id, spec.version, params,
        universe=spec.universe, benchmark=spec.benchmark.index_code,
    )
    logger.info("运行 %s：策略=%s 区间=%s~%s", run_id, spec.name, params.start, params.end)
    yield (
        "meta",
        {
            "run_id": run_id,
            "strategy_id": request.strategy_id,
            "version": spec.version,
            "name": spec.name,
            "strategy_hash": spec.config_hash,
            "universe": spec.universe,
            "benchmark": spec.benchmark.index_code,
            "factors": list(spec.signal.factors),
            "params": {"start": params.start, "end": params.end, "init_cash": params.init_cash},
        },
    )
    events: Queue = Queue()
    result_holder: dict = {}
    try:
        _check_cancelled(should_cancel)
        days, stocks = estimate_panel(params.start, params.end, spec.universe)
        # 面板超过单块上限就按时间分块；分块结果与不分块逐位一致，只是逐块占用内存
        windows = (
            plan_chunks(params.start, params.end, spec.universe)
            if days * stocks > MAX_PANEL_CELLS
            else None
        )
        if windows is None:
            data = load_market_data(
                params.start,
                params.end,
                list(spec.signal.factors),
                universe=spec.universe,
                benchmark_code=spec.benchmark.index_code,
                industry_level="L1" if spec.risk.max_industry_weight is not None else None,
            )
            yield (
                "load",
                {
                    "trading_days": int(len(data.index)),
                    "stocks": int(len(data.columns)),
                    "factors": list(data.factors),
                    "membership": data.membership is not None,
                    "benchmark": data.benchmark is not None,
                },
            )
            runner = lambda report: run_loaded(  # noqa: E731
                data, spec, params, progress=report, should_cancel=should_cancel
            )
        else:
            logger.info("面板 %d×%d 超过单块上限，改为 %d 块分块回测", days, stocks, len(windows))
            yield (
                "load",
                {
                    "trading_days": int(days),
                    "stocks": int(stocks),
                    "factors": list(spec.signal.factors),
                    "membership": spec.universe != "all_a",
                    "benchmark": spec.benchmark.index_code is not None,
                    "chunks": len(windows),
                },
            )
            runner = lambda report: run_chunked(  # noqa: E731
                spec,
                params,
                progress=report,
                should_cancel=should_cancel,
                windows=windows,
            )
        worker = Thread(target=_run_in_thread, args=(runner, events), daemon=True)
        worker.start()
        while True:
            try:
                item = events.get(timeout=0.5)
            except Empty:
                _check_cancelled(should_cancel)
                continue
            if item is None:
                break
            stage, payload = item
            if stage == "__result__":
                result_holder["result"] = payload
                continue
            if stage == "__error__":
                raise payload
            yield stage, payload
        worker.join()
        result = result_holder.get("result")
        if result is None:
            raise BacktestError("回测没有产出结果")
        store.finish_run(run_id, result)
    except RunCancelled as error:
        store.cancel_run(run_id)
        raise
    except BacktestError as error:
        store.fail_run(run_id, error)
        raise
    except Exception as error:
        store.fail_run(run_id, error)
        raise
    except GeneratorExit:
        store.cancel_run(run_id)
        raise
    yield (
        "done",
        {
            "run_id": run_id,
            "strategy_hash": result.metadata["strategy_hash"],
            "orders": int(len(result.orders)),
            "rejections": int(len(result.rejections)),
            "calibration": result.metadata["calibration"],
            "metrics": result.metrics,
            "elapsed_seconds": result.metadata["elapsed_seconds"],
        },
    )


def _check_cancelled(should_cancel: Cancelled | None) -> None:
    """取消检查，命中就抛 RunCancelled。"""
    if should_cancel is not None and should_cancel():
        raise RunCancelled("回测已取消")


def run(
    store: BacktestStore,
    request: RunRequest,
    should_cancel: Cancelled | None = None,
) -> dict:
    """执行一次回测并返回最后一个阶段的结果摘要。

    参数:
        store: 回测存储。
        request: 回测请求。
        should_cancel: 可选取消检查。
    """
    summary: dict = {}
    for stage, payload in iter_run(store, request, should_cancel):
        if stage == "done":
            summary = payload
    return summary

