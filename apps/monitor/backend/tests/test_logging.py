"""后端日志保留模块信息与首尾时间，在128MB边界轮转。"""

import logging

from backend.logging_conf import SegmentHandler


def test_log_segment_end_and_rotation(tmp_path):
    handler = SegmentHandler(tmp_path)
    first = logging.LogRecord('backend.test', logging.INFO, '', 0, 'request=first', (), None)
    handler.handle(first)
    old = handler.path
    second = logging.LogRecord('backend.test', logging.INFO, '', 0, 'request=second', (), None)
    second.created = first.created + .1
    handler.handle(second)
    assert not old.exists()
    assert handler.path.read_text().splitlines() == ['request=first', 'request=second']
    previous = handler.path
    with previous.open('a') as stream: stream.truncate(128 * 1024**2)
    third = logging.LogRecord('backend.test', logging.INFO, '', 0, 'request=third', (), None)
    third.created = second.created + .1
    handler.handle(third)
    assert previous.exists() and handler.path != previous
    assert handler.path.read_text().strip() == 'request=third'
    handler.close()
