"""阶段隔离下的慢消费者、取消、名额预留及HTTP错误契约。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
import time

import pytest
from fastapi.testclient import TestClient

from backend import service
from backend.app import create_app
from backend.errors import MonitorError
from backend.scheduler import Scheduler
from backend.service import RequestSpec
from calculate import EvalParams
from calculate.native import CALCULATION_LOCK


def test_slow_consumer_does_not_hold_native_lock(monkeypatch):
    """一个请求的事件队列已满时，另一请求和元数据仍可继续。"""
    third = Event()
    closed = Event()

    def fake(params, stages):
        """按阶段产出事件，用第三次推进标记事件队列即将背压。"""
        try:
            for number in range(6):
                if number == 2:
                    third.set()
                yield 'overview', {'number': number}
        finally:
            closed.set()

    monkeypatch.setattr(service, 'ic_events', fake)
    spec = RequestSpec(('MOM_20',), EvalParams('MOM_20', '2024-01-01', '2024-02-01'),
                       'ic', (), {})
    scheduler = Scheduler()
    first = scheduler.submit(lambda: service.run(spec))
    try:
        assert third.wait(2)
        assert first.events.qsize() == 2 and not first.finished.is_set()
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(service.factor_detail, 'MOM_20').result(timeout=5)['name'] == 'MOM_20'
        second = scheduler.submit(lambda: iter([('done', {})]))
        assert second.events.get(timeout=2)[0] == 'done'
        assert second.finished.wait(2)
        first.cancel()
        assert first.finished.wait(2) and closed.is_set()
    finally:
        scheduler.close()


def test_cancel_while_waiting_for_native_lock_never_creates_generator():
    """尚未获得原生锁的取消任务不应进入工厂或等待其他任务完成。"""
    created = Event()

    def factory():
        """标记工厂被调用，返回一个无需计算的迭代器。"""
        created.set()
        return iter([('done', {})])

    scheduler = Scheduler()
    try:
        with CALCULATION_LOCK:
            job = scheduler.submit(factory)
            assert job.started.wait(2)
            job.cancel()
            assert job.finished.wait(2)
            assert not created.is_set() and job.events.empty()
    finally:
        scheduler.close()


def test_native_lock_wait_is_visible_in_job_status():
    """任务等待共享原生锁时应记录等待状态和耗时。"""
    scheduler = Scheduler(workers=1, waiting=0)

    def factory():
        return iter([('done', {})])

    try:
        with CALCULATION_LOCK:
            job = scheduler.submit(factory)
            assert job.started.wait(2)
            deadline = time.monotonic() + 2
            while job.snapshot()['status'] != 'waiting_native' and time.monotonic() < deadline:
                time.sleep(.01)
            assert job.snapshot()['status'] == 'waiting_native'
        assert job.finished.wait(2)
        status = job.snapshot()
        assert status['status'] == 'success'
        assert status['native_wait_seconds'] > 0
    finally:
        scheduler.close()


def test_factor_and_backtest_schedulers_share_native_lock():
    """因子与回测独立调度器不能同时进入原生计算阶段。"""
    factor_scheduler = Scheduler(workers=1, waiting=0)
    backtest_scheduler = Scheduler(workers=1, waiting=0)
    state_lock = Lock()
    active = 0
    maximum = 0

    def factory(name):
        def events():
            nonlocal active, maximum
            with state_lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(.15)
            with state_lock:
                active -= 1
            yield 'done', {'name': name}
        return events

    try:
        factor = factor_scheduler.submit(factory('factor'))
        backtest = backtest_scheduler.submit(factory('backtest'))
        assert factor.finished.wait(3)
        assert backtest.finished.wait(3)
        assert maximum == 1
        assert factor.events.get(timeout=1)[0] == 'done'
        assert backtest.events.get(timeout=1)[0] == 'done'
    finally:
        factor_scheduler.close()
        backtest_scheduler.close()


def test_reservations_count_toward_capacity_and_close_releases_them():
    """预检中的请求占用容量，失败释放后可再次预留。"""
    scheduler = Scheduler()
    jobs = [scheduler.reserve() for _ in range(6)]
    try:
        with pytest.raises(MonitorError) as error:
            scheduler.reserve()
        assert error.value.status == 429
        scheduler.discard(jobs.pop())
        jobs.append(scheduler.reserve())
        scheduler.close()
        assert not scheduler.jobs
        assert all(job.finished.is_set() for job in jobs)
    finally:
        scheduler.close()


def test_http_busy_rejects_before_waiting_for_preflight(monkeypatch):
    """名额满时即使原生锁被占用也应立刻返回429，不调用预检。"""
    called = Event()

    def prepare(*args):
        """记录错误地进入预检的请求，参数不参与计算。"""
        called.set()
        raise AssertionError('busy request reached preflight')

    monkeypatch.setattr(service, 'prepare', prepare)
    with TestClient(create_app()) as client:
        scheduler = client.app.state.scheduler
        jobs = [scheduler.reserve() for _ in range(6)]
        try:
            with CALCULATION_LOCK:
                response = client.get('/api/evaluate/stream', params={'factor': 'MOM_20'})
            assert response.status_code == 429 and not called.is_set()
        finally:
            for job in jobs:
                scheduler.discard(job)


def test_http_preflight_failure_releases_reservation():
    """参数错误保持开流前JSON响应，失败的预留名额全部释放。"""
    with TestClient(create_app()) as client:
        for _ in range(8):
            response = client.get('/api/evaluate/stream', params={'factor': 'unknown'})
            assert response.status_code == 404
            assert not client.app.state.scheduler.jobs


def test_close_failure_reports_error_and_releases_capacity():
    """关闭迭代器失败须发出公开错误，不遗留任务名额或原生锁。"""
    class Events:
        """在结束阶段测试清理错误的事件源。"""

        def __iter__(self):
            """返回自身迭代器。"""
            return self

        def __next__(self):
            """直接结束，使调度器执行关闭操作。"""
            raise StopIteration

        def close(self):
            """模拟关闭期间的错误。"""
            raise RuntimeError('private close failure')

    scheduler = Scheduler(workers=1, waiting=0)
    try:
        job = scheduler.submit(Events)
        stage, payload = job.events.get(timeout=2)
        assert stage == 'error' and payload['code'] == 'MONITOR-104'
        assert 'private' not in payload['message']
        assert job.finished.wait(2) and not scheduler.jobs
        next_job = scheduler.submit(lambda: iter([('done', {})]))
        assert next_job.events.get(timeout=2)[0] == 'done'
        assert next_job.finished.wait(2)
    finally:
        scheduler.close()


def test_reserve_and_start_after_close_are_rejected():
    """关闭后的预留与启动都返回503，不产生新的运行任务。"""
    scheduler = Scheduler()
    job = scheduler.reserve()
    scheduler.close()
    with pytest.raises(MonitorError) as reserved:
        scheduler.reserve()
    assert reserved.value.status == 503
    with pytest.raises(MonitorError) as started:
        scheduler.start(job, lambda: iter([('done', {})]))
    assert started.value.status == 503
    assert job.finished.is_set() and not scheduler.jobs


def test_discard_is_idempotent_and_releases_capacity_once():
    """重复丢弃已取消的预留不会破坏容量计数。"""
    scheduler = Scheduler(workers=1, waiting=0)
    try:
        job = scheduler.reserve()
        scheduler.discard(job)
        scheduler.discard(job)
        assert not scheduler.jobs and job.finished.is_set()
        again = scheduler.submit(lambda: iter([('done', {})]))
        assert again.finished.wait(2)
    finally:
        scheduler.close()


def test_close_returns_while_job_waits_for_native_lock():
    """有任务在等原生锁时关闭服务不会挂住，等待中的任务按取消退出。"""
    scheduler = Scheduler(workers=1, waiting=0)

    def factory():
        """正常情况下会立即结束的工厂，用于验证它根本没有被调用。"""
        return iter([('done', {})])

    with CALCULATION_LOCK:
        job = scheduler.submit(factory)
        assert job.started.wait(2)
        scheduler.close()
    assert job.finished.is_set() and not scheduler.jobs
    assert job.events.empty()
