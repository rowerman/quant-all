"""在独立进程内测真实计算的时间、峰值RSS与事件量；结果仅打印不写数据产物。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import resource
import sys
import time

# 前后端在 apps/monitor 下，脚本从仓库根运行时需要补上导入路径
_BACKEND_ROOT = Path(__file__).resolve().parents[2] / 'apps' / 'monitor'
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from backend import service
from backend.errors import error_payload
from calculate.jsonutil import json_safe


def measure(query, compare=False):
    """计算并统计stage耗时和字节，参数query为真实接口参数，compare控制对比。"""
    started = time.perf_counter()
    spec = service.prepare(query, compare)
    stages = []
    size = 0
    for stage, payload in service.run(spec):
        size += len(json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False).encode())
        if stage not in ('meta', 'done', 'factor_done'):
            stages.append({'factor': payload['factor'], 'stage': stage,
                           'elapsed_ms': payload.get('elapsed_ms')})
    return {'elapsed_s': time.perf_counter()-started, 'payload_bytes': size, 'stages': stages}


def benchmark(case):
    """运行一个独立压测案例，参数case为命令行枚举。"""
    base = {'factor': 'MOM_20'}
    if case == 'default':
        result = measure(base)
    elif case == 'near_limit':
        result = measure({**base, 'start': '2021-01-01', 'end': '2026-09-28'})
    elif case == 'full_ic':
        result = measure({**base, 'mode': 'ic', 'start': '2000-01-04', 'end': '2026-09-28'})
    elif case == 'compare':
        result = measure({'factors': 'MOM_20,MOM_60,REV_5,RSI_14'}, True)
    elif case == 'concurrent':
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(measure, [base, {**base, 'factor': 'RSI_14'}]))
        result = {'requests': results}
    elif case == 'reject':
        results = []
        for query in ({**base, 'start': '2000-01-04', 'end': '2026-09-28'},
                      {**base, 'start': '2020-01-01', 'end': '2026-09-28'}):
            try:
                service.prepare(query)
            except Exception as error:
                status, payload = error_payload(error, 'benchmark')
                results.append({'status': status, **payload})
            else:
                raise AssertionError('超限请求应被拒绝')
        result = {'rejections': results}
    else:
        raise ValueError(case)
    result.update(case=case, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['default', 'near_limit', 'full_ic', 'compare', 'concurrent', 'reject'], required=True)
    benchmark(parser.parse_args().case)
