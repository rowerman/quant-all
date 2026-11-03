"""在独立进程反复并发预检，原生层崩溃不应被Python异常处理掩盖。"""

import os
from pathlib import Path
import subprocess
import sys


def _subprocess_env() -> tuple[Path, dict]:
    """子进程要同时导入仓库根的 calculate 与 apps/monitor 下的 backend。"""
    backend_root = Path(__file__).resolve().parents[2]
    project_root = Path(__file__).resolve().parents[4]
    roots = [str(project_root), str(backend_root)]
    if os.environ.get('PYTHONPATH'):
        roots.append(os.environ['PYTHONPATH'])
    return project_root, {
        **os.environ,
        'PYTHONFAULTHANDLER': '1',
        'PYTHONPATH': os.pathsep.join(roots),
    }


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
    cwd, env = _subprocess_env()
    result = subprocess.run([sys.executable, '-c', code], cwd=cwd, env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
