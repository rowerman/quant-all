"""在独立进程反复并发预检，原生层崩溃不应被Python异常处理掩盖。"""

import os
from pathlib import Path
import subprocess
import sys


def test_repeated_concurrent_preflight():
    """交替预检不同因子，覆盖独立连接的创建、挂载、列元数据读取与关闭。"""
    code = '''
from concurrent.futures import ThreadPoolExecutor
from backend.service import prepare
names = ['MOM_20', 'RSI_14'] * 20
with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(lambda name: prepare({'factor': name}), names))
assert [r.params.factor for r in results] == names
'''
    result = subprocess.run([sys.executable, '-c', code],
                            cwd=Path(__file__).resolve().parents[2],
                            env={**os.environ, 'PYTHONFAULTHANDLER': '1'},
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
