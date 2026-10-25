"""两计算线程、四等待任务；断连后安全退出并释放任务槽。"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from queue import Empty, Queue, Full
from threading import Event, Lock, RLock
import time
import traceback
import uuid

from backend.errors import MonitorError, error_payload
from backend.logging_conf import get_logger
from calculate.jsonutil import json_safe
from calculate.native import CalculationCancelled, native_section

logger = get_logger('backend.scheduler')


@dataclass
class Job:
    """一次流请求的状态，事件缓冲最多2条，取消标记跨线程可见。"""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: Queue = field(default_factory=lambda: Queue(maxsize=2))
    cancelled: Event = field(default_factory=Event)
    started: Event = field(default_factory=Event)
    finished: Event = field(default_factory=Event)
    terminal_sent: Event = field(default_factory=Event)
    future: object = None
    status: str = 'queued'
    status_version: int = 0
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    native_wait_started: float | None = None
    native_wait_seconds: float = 0.0
    cancelled_at: float | None = None
    current_stage: str | None = None
    run_id: str | None = None
    failure: BaseException | None = None
    status_lock: RLock = field(default_factory=RLock, repr=False)

    def send(self, stage, payload):
        """带背压地发送事件；参数为stage和结果，取消后返回False。"""
        event = (stage, json_safe(payload))
        while not self.cancelled.is_set():
            try:
                self.events.put(event, timeout=.2)
                return True
            except Full:
                continue
        return False

    def cancel(self):
        """取消排队任务或标记运行任务在下一阶段边界退出。"""
        with self.status_lock:
            if self.terminal_sent.is_set() or self.finished.is_set():
                return
            if self.cancelled_at is None:
                self.cancelled_at = time.monotonic()
            self.mark_status('cancelling')
        self.cancelled.set()
        if self.future is not None:
            self.future.cancel()

    def mark_status(self, status: str) -> None:
        """更新任务状态与计时，参数status为新的生命周期状态。"""
        now = time.monotonic()
        with self.status_lock:
            if self.status == status and status != 'waiting_native':
                return
            if self.status == 'waiting_native' and self.native_wait_started is not None:
                self.native_wait_seconds += now - self.native_wait_started
                self.native_wait_started = None
            if status == 'waiting_native':
                self.native_wait_started = now
            if status == 'running' and self.started_at is None:
                self.started_at = now
            if status in {'success', 'failed', 'cancelled'}:
                self.finished_at = now
            self.status = status
            self.status_version += 1

    def snapshot(self) -> dict:
        """返回任务状态快照，供SSE与日志使用。"""
        now = time.monotonic()
        with self.status_lock:
            native_wait = self.native_wait_seconds
            if self.status == 'waiting_native' and self.native_wait_started is not None:
                native_wait += now - self.native_wait_started
            return {
                'request_id': self.request_id,
                'status': self.status,
                'stage': self.current_stage,
                'run_id': self.run_id,
                'version': self.status_version,
                'queue_wait_seconds': max(0.0, (self.started_at or now) - self.created_at),
                'native_wait_seconds': native_wait,
                'run_seconds': (self.finished_at or now) - self.started_at
                if self.started_at is not None else 0.0,
                'cancel_wait_seconds': (self.finished_at or now) - self.cancelled_at
                if self.cancelled_at is not None else 0.0,
            }

    def mark_stage(self, stage: str, payload: dict) -> None:
        """记录当前业务阶段及已知运行编号。"""
        with self.status_lock:
            self.current_stage = stage
            if payload.get('run_id'):
                self.run_id = payload['run_id']
            self.status_version += 1


class Scheduler:
    """任务容量固定为workers+waiting，参数默认对应两运行、四等待。"""

    def __init__(self, workers=2, waiting=4):
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='monitor')
        self.workers = workers
        self.waiting = waiting
        self.capacity = workers + waiting
        self.jobs = {}
        self.lock = Lock()
        self.closed = False

    def reserve(self):
        """预检前预留任务名额，避免等待计算锁的请求越过容量上限。"""
        with self.lock:
            if self.closed:
                raise MonitorError('服务正在关闭', 503, 'MONITOR-103')
            if len(self.jobs) >= self.capacity:
                raise MonitorError(
                    f'计算繁忙：{self.workers}个运行任务及{self.waiting}个等待名额已占满',
                    429,
                    'MONITOR-102',
                )
            job = Job()
            self.jobs[job.request_id] = job
        return job

    def start(self, job, factory):
        """启动已预留任务；参数为Job和事件迭代器工厂。"""
        with self.lock:
            if self.closed:
                raise MonitorError('服务正在关闭', 503, 'MONITOR-103')
            if job.cancelled.is_set() or job.request_id not in self.jobs:
                raise MonitorError('请求已取消', 400, 'MONITOR-105')
            job.future = self.executor.submit(self._run, job, factory)
        job.future.add_done_callback(lambda future: self._release(job))
        return job

    def discard(self, job):
        """预检失败或断连时取消预留任务，参数为Job。"""
        job.cancel()
        if job.future is None:
            job.mark_status('cancelled')
            self._release(job)

    def submit(self, factory):
        """预留并启动事件迭代器工厂，容量已满则抛429。"""
        job = self.reserve()
        try:
            return self.start(job, factory)
        except Exception:
            self.discard(job)
            raise

    def _release(self, job):
        """仅在Future实际结束或排队取消后释放容量，参数为Job。"""
        if job.cancelled.is_set() and not job.started.is_set():
            job.mark_status('cancelled')
        with self.lock:
            self.jobs.pop(job.request_id, None)
        job.finished.set()
        state = job.snapshot()
        logger.info(
            'request=%s released status=%s stage=%s run=%s queue_wait=%.3fs native_wait=%.3fs '
            'run=%.3fs cancel_wait=%.3fs',
            job.request_id, state['status'], state['stage'], state['run_id'],
            state['queue_wait_seconds'], state['native_wait_seconds'], state['run_seconds'],
            state['cancel_wait_seconds'],
        )

    def _run(self, job, factory):
        """线程内创建并消费生成器，阶段边界响应取消；参数为Job与工厂。"""
        started = time.perf_counter()
        job.started.set()
        iterator = None
        job.mark_status('running')
        logger.info('request=%s started', job.request_id)
        try:
            try:
                with self._native(job):
                    iterator = iter(factory())
                while not job.cancelled.is_set():
                    with self._native(job):
                        try:
                            stage, payload = next(iterator)
                        except StopIteration:
                            break
                        payload = json_safe({**payload, 'request_id': job.request_id})
                        job.mark_stage(stage, payload)
                        if stage == 'done':
                            job.mark_status('success')
                        elif stage == 'error':
                            job.mark_status('failed')
                    if not job.send(stage, payload):
                        break
                    if stage in {'done', 'error'}:
                        job.terminal_sent.set()
                    logger.info('request=%s factor=%s event=%s elapsed=%.3fs',
                                job.request_id, payload.get('factor'), stage, time.perf_counter() - started)
                    if stage in {'done', 'error'}:
                        break
            finally:
                if iterator is not None:
                    with native_section():
                        try:
                            if hasattr(iterator, 'close'):
                                iterator.close()
                        finally:
                            iterator = None
            job.mark_status('cancelled' if job.cancelled.is_set() else 'success')
        except CalculationCancelled:
            job.mark_status('cancelled')
            logger.info('request=%s cancelled while waiting for calculation', job.request_id)
        except Exception as error:
            job.failure = error
            job.mark_status('failed')
            with native_section():
                logger.exception('request=%s failed', job.request_id)
                _, payload = error_payload(error, job.request_id)
                traceback.clear_frames(error.__traceback__)
            if job.send('error', payload):
                job.terminal_sent.set()

    @contextmanager
    def _native(self, job):
        """进入共享原生锁，并记录锁等待状态，参数为任务。"""
        def waiting():
            job.mark_status('waiting_native')
            logger.info('request=%s waiting for native calculation lock', job.request_id)

        with native_section(
            job.cancelled,
            on_wait=waiting,
        ):
            job.mark_status('running')
            yield

    def wait(self, job):
        """排空任务事件并返回done结果，参数为已启动或已取消的任务。"""
        result = None
        while True:
            try:
                stage, payload = job.events.get(timeout=.2)
            except Empty:
                if job.finished.is_set() and job.events.empty():
                    break
                continue
            if stage == 'done':
                result = payload
            if job.finished.is_set() and job.events.empty():
                break
        if job.failure is not None:
            raise job.failure
        if result is None:
            raise MonitorError('回测任务未产生完成结果', 500, 'MONITOR-104')
        return result

    def close(self):
        """取消所有任务并等待线程在阶段边界退出，供服务关闭调用。"""
        with self.lock:
            self.closed = True
            jobs = list(self.jobs.values())
        for job in jobs:
            self.discard(job)
        self.executor.shutdown(wait=True, cancel_futures=True)
