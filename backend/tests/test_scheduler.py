"""调度容量、背压、错误和取消，独立于真实计算耗时。"""

from threading import Event
import time

import pytest

from backend.scheduler import Scheduler
from backend.errors import MonitorError


def test_capacity_and_waiting_cancel():
    gate = Event()
    scheduler = Scheduler()
    def blocked():
        gate.wait(5)
        yield 'done', {}
    jobs = [scheduler.submit(blocked) for _ in range(6)]
    try:
        assert jobs[0].started.wait(2) and jobs[1].started.wait(2)
        with pytest.raises(MonitorError) as error:
            scheduler.submit(blocked)
        assert error.value.status == 429
        jobs[5].cancel()
        assert jobs[5].finished.wait(2)
        assert len(scheduler.jobs) == 5
        assert not jobs[5].started.is_set()
    finally:
        gate.set()
        scheduler.close()
    assert not scheduler.jobs


def test_running_slot_retained_and_backpressure():
    gate = Event()
    scheduler = Scheduler(workers=1, waiting=0)
    def blocked():
        gate.wait(5)
        yield 'overview', {}
    job = scheduler.submit(blocked)
    try:
        assert job.started.wait(2)
        job.cancel()
        assert not job.finished.is_set()
        with pytest.raises(MonitorError): scheduler.submit(blocked)
        gate.set()
        assert job.finished.wait(2)
        job2 = scheduler.submit(lambda: iter([('overview', {})] * 10))
        deadline = time.monotonic() + 2
        while job2.events.qsize() < 2 and time.monotonic() < deadline: time.sleep(.01)
        assert job2.events.qsize() == 2 and not job2.finished.is_set()
        job2.cancel()
        assert job2.finished.wait(2)
    finally:
        gate.set()
        scheduler.close()


def test_error_is_public_and_generator_closed():
    scheduler = Scheduler()
    closed = Event()
    def failed():
        try:
            yield 'overview', {}
            raise ValueError('private detail')
        finally:
            closed.set()
    try:
        job = scheduler.submit(failed)
        assert job.events.get(timeout=2)[0] == 'overview'
        stage, payload = job.events.get(timeout=2)
        assert stage == 'error' and payload['code'] == 'MONITOR-104'
        assert 'private' not in payload['message']
        assert job.finished.wait(2) and closed.is_set()
    finally:
        scheduler.close()
