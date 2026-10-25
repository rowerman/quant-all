"""原生阶段隔离、事件转换及对象释放，使用小型数据验证生命周期边界。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
import weakref

import numpy as np
import pytest

from calculate.native import CALCULATION_LOCK, serialized, serialized_events


def test_events_release_lock_before_delivery_and_protect_conversion_and_close():
    """事件只包含Python值，转换和清理时锁定，消费者暂停时允许元数据读取。"""
    observations = []
    references = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        def observe(label):
            """从其他线程检测当前锁状态，参数label表示观察位置。"""
            def probe():
                """尝试获得锁，立即归还成功获得的锁。"""
                acquired = CALCULATION_LOCK.acquire(blocking=False)
                if acquired:
                    CALCULATION_LOCK.release()
                return acquired
            observations.append((label, executor.submit(probe).result(timeout=2)))

        class Payload:
            """以转换及析构回调记录原生对象处理边界。"""

            def __str__(self):
                """转换时观察锁，并返回JSON可用的字符串。"""
                observe('convert')
                return 'value'

        @serialized_events
        def events():
            """产出含数组和对象的事件，结束时记录清理。"""
            payload = Payload()
            references.append(weakref.ref(payload))
            weakref.finalize(payload, observe, 'release')
            try:
                yield 'meta', {'array': np.array([1.0, np.nan]), 'payload': payload}
            finally:
                observe('close')

        iterator = events()
        assert next(iterator) == ('meta', {'array': [1.0, None], 'payload': 'value'})
        observe('delivery')
        iterator.close()
        assert references[0]() is None
    assert observations == [('convert', False), ('delivery', True),
                            ('close', False), ('release', False)]


def test_native_generators_do_not_overlap_and_metadata_waits_for_stage():
    """第一个阶段被阻塞时，另一原生阶段和元数据读取不能进入。"""
    entered = Event()
    release = Event()
    second_entered = Event()
    metadata_entered = Event()

    @serialized_events
    def first():
        """阻塞一个原生阶段，直到测试允许继续。"""
        entered.set()
        assert release.wait(5)
        yield 'done', {}

    @serialized_events
    def second():
        """记录第二个阶段真正进入的时刻。"""
        second_entered.set()
        yield 'done', {}

    @serialized
    def metadata():
        """记录元数据函数进入的时刻。"""
        metadata_entered.set()

    with ThreadPoolExecutor(max_workers=3) as executor:
        first_result = executor.submit(lambda: list(first()))
        assert entered.wait(2)
        second_result = executor.submit(lambda: list(second()))
        metadata_result = executor.submit(metadata)
        try:
            assert not second_entered.wait(.1)
            assert not metadata_entered.wait(.1)
        finally:
            release.set()
        assert first_result.result(timeout=3) == [('done', {})]
        assert second_result.result(timeout=3) == [('done', {})]
        metadata_result.result(timeout=3)


@pytest.mark.parametrize('failure', ['advance', 'close'])
def test_exception_releases_native_lock(failure):
    """推进和关闭失败都应向上反馈并允许下一次计算。"""
    @serialized_events
    def events():
        """在指定位置抛出异常，failure由测试参数控制。"""
        try:
            if failure == 'advance':
                raise ValueError('advance')
            yield 'meta', {}
        finally:
            if failure == 'close':
                raise ValueError('close')

    iterator = events()
    with pytest.raises(ValueError, match=failure):
        if failure == 'close':
            next(iterator)
            iterator.close()
        else:
            next(iterator)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(lambda: list(serialized_events(lambda: iter([('done', {})]))()))
        assert result.result(timeout=2) == [('done', {})]
        result = executor.submit(serialized(lambda: 'released'))
        assert result.result(timeout=2) == 'released'
