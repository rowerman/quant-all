"""两计算线程、四等待任务；断连后安全退出并释放任务槽。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Queue, Full
from threading import Event, Lock
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
    future: object = None

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
        self.cancelled.set()
        if self.future is not None:
            self.future.cancel()


class Scheduler:
    """任务容量固定为workers+waiting，参数默认对应两运行、四等待。"""

    def __init__(self, workers=2, waiting=4):
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='monitor')
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
                raise MonitorError('计算繁忙：2个运行任务及4个等待名额已占满', 429, 'MONITOR-102')
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
        with self.lock:
            self.jobs.pop(job.request_id, None)
        job.finished.set()
        logger.info('request=%s released cancelled=%s', job.request_id, job.cancelled.is_set())

    def _run(self, job, factory):
        """线程内创建并消费生成器，阶段边界响应取消；参数为Job与工厂。"""
        started = time.perf_counter()
        job.started.set()
        iterator = None
        logger.info('request=%s started', job.request_id)
        try:
            try:
                with native_section(job.cancelled):
                    iterator = iter(factory())
                while not job.cancelled.is_set():
                    with native_section(job.cancelled):
                        try:
                            stage, payload = next(iterator)
                        except StopIteration:
                            break
                        payload = json_safe({**payload, 'request_id': job.request_id})
                    if not job.send(stage, payload):
                        break
                    logger.info('request=%s factor=%s event=%s elapsed=%.3fs',
                                job.request_id, payload.get('factor'), stage, time.perf_counter() - started)
            finally:
                if iterator is not None:
                    with native_section():
                        try:
                            if hasattr(iterator, 'close'):
                                iterator.close()
                        finally:
                            iterator = None
        except CalculationCancelled:
            logger.info('request=%s cancelled while waiting for calculation', job.request_id)
        except Exception as error:
            with native_section():
                logger.exception('request=%s failed', job.request_id)
                _, payload = error_payload(error, job.request_id)
                traceback.clear_frames(error.__traceback__)
            job.send('error', payload)

    def close(self):
        """取消所有任务并等待线程在阶段边界退出，供服务关闭调用。"""
        with self.lock:
            self.closed = True
            jobs = list(self.jobs.values())
        for job in jobs:
            self.discard(job)
        self.executor.shutdown(wait=True, cancel_futures=True)
