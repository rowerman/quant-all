"""后端按模块记录日志，128MB轮转并维护段内第一/最后消息时间。"""

import logging
from pathlib import Path
from datetime import datetime


class SegmentHandler(logging.Handler):
    """时间段日志处理器；参数 directory 为模块目录。"""

    def __init__(self, directory):
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = None
        self.started = None

    def emit(self, record):
        """写日志、更新时间段文件名；写入失败交给 logging 报告。"""
        try:
            stamp = datetime.fromtimestamp(record.created).strftime('%Y%m%d%H%M%S%f')
            message = self.format(record) + '\n'
            if self.path is None or self.path.stat().st_size + len(message.encode()) > 128 * 1024**2:
                self.started = stamp
                self.path = self.directory / f'{stamp}-{stamp}.log'
            target = self.directory / f'{self.started}-{stamp}.log'
            if target != self.path:
                self.path.rename(target)
                self.path = target
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(message)
        except Exception:
            self.handleError(record)


def get_logger(module):
    """返回后端模块logger，参数module同时确定logs下目录。"""
    logger = logging.getLogger(module)
    if not logger.handlers:
        handler = SegmentHandler(Path(__file__).parent / 'logs' / module.replace('.', '_'))
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
