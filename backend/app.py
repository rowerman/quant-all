"""FastAPI路由与SSE传输，静态前端同源托管。"""

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from queue import Empty
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from backend import service
from backend import backtest_service
from backend.errors import error_payload, MonitorError
from backtest.errors import BacktestError
from backend.logging_conf import get_logger
from backend.scheduler import Scheduler
from calculate.jsonutil import json_safe

logger = get_logger('backend.app')


def create_app():
    """构造独立应用实例，调度器在lifespan内启动和关闭。"""
    @asynccontextmanager
    async def lifespan(app):
        app.state.scheduler = Scheduler()
        # 回测单次占 2~3GB，默认串行；需要并发时用环境变量放宽
        app.state.backtest_scheduler = Scheduler(
            workers=int(os.environ.get('BACKTEST_WORKERS', 1)),
            waiting=int(os.environ.get('BACKTEST_WAITING', 2)),
        )
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.scheduler.close)
            await asyncio.to_thread(app.state.backtest_scheduler.close)

    app = FastAPI(title='因子质量监测', lifespan=lifespan)

    @app.exception_handler(Exception)
    async def handle_error(request, error):
        """将请求异常映射为JSON；内部失败记录堆栈。"""
        request_id = getattr(request.state, 'request_id', uuid.uuid4().hex)
        status, payload = error_payload(error, request_id)
        logger.error('request=%s status=%s %s', request_id, status, error, exc_info=status >= 500)
        return JSONResponse(payload, status_code=status)

    # 预期异常注册独立处理器，避免Starlette在测试/中间件中重抛。
    from calculate.errors import CalculateError
    for kind in (MonitorError, CalculateError, FileNotFoundError):
        app.add_exception_handler(kind, handle_error)
    app.add_exception_handler(BacktestError, handle_error)

    @app.get('/api/options')
    async def options():
        """返回可选参数、指标声明和限制。"""
        return json_safe(await asyncio.to_thread(service.options))

    @app.get('/api/factors')
    async def factors():
        """列出实际已落盘因子。"""
        return json_safe(await asyncio.to_thread(service.factor_list))

    @app.get('/api/factors/{name}')
    async def factor(name: str):
        """返回指定因子的定义与覆盖率，参数name为因子名。"""
        return json_safe(await asyncio.to_thread(service.factor_detail, name))

    @app.post('/api/strategies')
    async def create_strategy(payload: dict):
        """创建策略第一版配置。"""
        return json_safe(await asyncio.to_thread(backtest_service.create_strategy, payload))

    @app.get('/api/strategies')
    async def list_strategies():
        """列出策略版本。"""
        return json_safe(await asyncio.to_thread(backtest_service.list_strategies))

    @app.get('/api/strategies/{strategy_id}')
    async def get_strategy(strategy_id: str, version: int | None = None):
        """读取策略指定版本。"""
        return json_safe(await asyncio.to_thread(backtest_service.get_strategy, strategy_id, version))

    @app.get('/api/backtest/options')
    async def backtest_options():
        """回测页可选策略、股票池、因子与资源上限。"""
        return json_safe(await asyncio.to_thread(backtest_service.options))

    @app.get('/api/backtests')
    async def list_backtests(limit: int = 50):
        """列出历史回测运行。"""
        return json_safe(await asyncio.to_thread(backtest_service.list_runs, limit))

    @app.get('/api/backtests/compare')
    async def compare_backtests(run_ids: str):
        """按共同交易日对比 2～4 次成功运行。"""
        chosen = [item.strip() for item in run_ids.split(',') if item.strip()]
        return json_safe(await asyncio.to_thread(backtest_service.compare_runs, chosen))

    @app.post('/api/backtests')
    async def run_backtest(request: Request, payload: dict):
        """提交一次回测并同步等待调度器返回结果。"""
        scheduler = app.state.backtest_scheduler
        job = scheduler.reserve()
        request.state.request_id = job.request_id
        try:
            job.mark_status('preflight')
            spec = await asyncio.to_thread(backtest_service.prepare, payload)
            if await request.is_disconnected():
                raise MonitorError('请求已断开，未提交回测任务', 400, 'MONITOR-105')
            scheduler.start(
                job,
                lambda: backtest_service.stream_strategy(spec, job.cancelled.is_set),
            )
            result = await asyncio.to_thread(scheduler.wait, job)
            result = dict(result)
            result.pop('request_id', None)
            return JSONResponse(
                json_safe(result),
                headers={'X-Request-ID': job.request_id},
            )
        except asyncio.CancelledError:
            job.cancel()
            raise
        except BaseException:
            if job.future is None or not job.finished.is_set():
                scheduler.discard(job)
            raise

    @app.get('/api/backtests/stream')
    async def stream_backtest(request: Request):
        """流式运行回测，逐阶段推送进度。"""
        if len(set(request.query_params.keys())) != len(request.query_params.multi_items()):
            raise MonitorError('同一查询参数不能重复提供')
        payload = dict(request.query_params)
        scheduler = app.state.backtest_scheduler
        job = scheduler.reserve()
        queued_status = job.snapshot()
        request.state.request_id = job.request_id
        try:
            job.mark_status('preflight')
            spec = await asyncio.to_thread(backtest_service.prepare, payload)
            preflight_status = job.snapshot()
            if await request.is_disconnected():
                raise MonitorError('请求已断开，未提交回测任务', 400, 'MONITOR-105')
            scheduler.start(
                job,
                lambda: backtest_service.stream_strategy(spec, job.cancelled.is_set),
            )
        except BaseException:
            scheduler.discard(job)
            raise

        async def stream():
            """消费回测阶段的线程事件，关闭连接时在阶段边界取消。"""
            ticks = 0
            last_status_version = preflight_status['version']
            terminal = False
            try:
                yield 'event: queued\ndata: ' + json.dumps(queued_status, ensure_ascii=False) + '\n\n'
                yield 'event: status\ndata: ' + json.dumps(
                    preflight_status, ensure_ascii=False, allow_nan=False,
                ) + '\n\n'
                while True:
                    if await request.is_disconnected():
                        break
                    status = job.snapshot()
                    if status['version'] != last_status_version:
                        last_status_version = status['version']
                        yield 'event: status\ndata: ' + json.dumps(
                            status, ensure_ascii=False, allow_nan=False,
                        ) + '\n\n'
                    try:
                        stage, payload = job.events.get_nowait()
                    except Empty:
                        if job.finished.is_set():
                            break
                        await asyncio.sleep(.1)
                        ticks += 1
                        if ticks % 100 == 0:
                            yield ': heartbeat\n\n'
                        continue
                    yield f'event: {stage}\ndata: ' + json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n\n'
                    if stage in ('done', 'error'):
                        terminal = True
                        break
            finally:
                if not terminal:
                    job.cancel()

        return StreamingResponse(stream(), media_type='text/event-stream', background=BackgroundTask(job.cancel),
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no',
                                          'X-Request-ID': job.request_id})

    @app.get('/api/backtests/{run_id}')
    async def get_backtest(run_id: str):
        """读取回测运行状态。"""
        return json_safe(await asyncio.to_thread(backtest_service.get_run, run_id))

    @app.delete('/api/backtests/{run_id}')
    async def delete_backtest(run_id: str):
        """删除一次已结束回测及其全部结果。"""
        return json_safe(await asyncio.to_thread(backtest_service.delete_run, run_id))

    @app.get('/api/backtests/{run_id}/metrics')
    async def get_backtest_metrics(run_id: str):
        """读取回测绩效指标。"""
        return json_safe(await asyncio.to_thread(backtest_service.get_run_metrics, run_id))

    @app.get('/api/backtests/{run_id}/holding')
    async def get_backtest_holding(run_id: str):
        """读取逐日持仓。"""
        return json_safe(await asyncio.to_thread(backtest_service.get_run_table, run_id, 'holding'))

    @app.get('/api/backtests/{run_id}/{table}')
    async def get_backtest_table(run_id: str, table: str):
        """读取回测日结果、成交、拒单或目标权重。"""
        return json_safe(await asyncio.to_thread(backtest_service.get_run_table, run_id, table))

    async def start_stream(request, compare):
        """开启有界SSE；参数request为HTTP请求，compare决定单因子或对比。"""
        request.state.request_id = uuid.uuid4().hex
        if len(set(request.query_params.keys())) != len(request.query_params.multi_items()):
            raise MonitorError('同一查询参数不能重复提供')
        scheduler = app.state.scheduler
        job = scheduler.reserve()
        queued_status = job.snapshot()
        try:
            job.mark_status('preflight')
            spec = await asyncio.to_thread(service.prepare, dict(request.query_params), compare)
            preflight_status = job.snapshot()
            if await request.is_disconnected():
                raise MonitorError('请求已断开，未提交计算任务', 400, 'MONITOR-105')
            scheduler.start(job, lambda: service.run(spec))
        except BaseException:
            scheduler.discard(job)
            raise
        request.state.request_id = job.request_id

        async def stream():
            """消费线程事件，长阶段发送心跳，关闭时标记取消。"""
            ticks = 0
            last_status_version = preflight_status['version']
            terminal = False
            try:
                yield 'event: queued\ndata: ' + json.dumps(queued_status, ensure_ascii=False) + '\n\n'
                yield 'event: status\ndata: ' + json.dumps(
                    preflight_status, ensure_ascii=False, allow_nan=False,
                ) + '\n\n'
                while True:
                    if await request.is_disconnected():
                        break
                    status = job.snapshot()
                    if status['version'] != last_status_version:
                        last_status_version = status['version']
                        yield 'event: status\ndata: ' + json.dumps(
                            status, ensure_ascii=False, allow_nan=False,
                        ) + '\n\n'
                    try:
                        stage, payload = job.events.get_nowait()
                    except Empty:
                        if job.finished.is_set():
                            break
                        await asyncio.sleep(.1)
                        ticks += 1
                        if ticks % 100 == 0:
                            yield ': heartbeat\n\n'
                        continue
                    yield f'event: {stage}\ndata: ' + json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n\n'
                    if stage in ('done', 'error'):
                        terminal = True
                        break
            finally:
                if not terminal:
                    job.cancel()

        return StreamingResponse(stream(), media_type='text/event-stream', background=BackgroundTask(job.cancel),
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no',
                                          'X-Request-ID': job.request_id})

    @app.get('/api/evaluate/stream')
    async def evaluate(request: Request):
        """流式单因子评估。"""
        return await start_stream(request, False)

    @app.get('/api/compare/stream')
    async def compare(request: Request):
        """流式顺序对比2～4个因子。"""
        return await start_stream(request, True)

    app.mount('/', StaticFiles(directory=Path(__file__).resolve().parents[1] / 'frontend', html=True), name='frontend')
    return app


app = create_app()
