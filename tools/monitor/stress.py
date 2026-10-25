"""并发验收：真实调度器与HTTP下的双任务、取消与慢消费者。

本地模式进程内启动应用，走真实Scheduler与SSE，只做双任务验收：TestClient会把整段
响应缓冲后才交给调用方，读不到中间事件，因此取消与背压必须在HTTP模式下对真实
uvicorn验证。每轮结果追加写入AiRecord/benchmarks/V11/日期-stress-模式.jsonl，
进程被杀也保留已完成轮次。

用法:
    .venv/bin/python -m tools.monitor.stress --mode local --case concurrent --rounds 5
    .venv/bin/python -m tools.monitor.stress --mode http --server-pid <pid> --case concurrent
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import httpx
from fastapi.testclient import TestClient

# 前后端在 apps/monitor 下，脚本从仓库根运行时需要补上导入路径
_BACKEND_ROOT = Path(__file__).resolve().parents[2] / 'apps' / 'monitor'
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from backend.app import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = PROJECT_ROOT / 'AiRecord/benchmarks/V11'
SCHEDULER_LOG_DIR = PROJECT_ROOT / 'logs/apps/monitor/backend_scheduler'
LIGHT_QUERY = {'factor': 'MOM_20', 'mode': 'ic', 'start': '2024-05-06', 'end': '2024-05-31'}


def decode_block(block):
    """解析一个SSE块；参数block为两空行之间的文本，心跳与空块返回None。"""
    stage, data = None, None
    for line in block.splitlines():
        if line.startswith('event: '):
            stage = line[7:].strip()
        elif line.startswith('data: '):
            data = line[6:]
    if stage is None or data is None:
        return None
    return stage, json.loads(data)


def consume(open_stream, sink, delay=0.0, stop_after=None):
    """消费SSE流直到结束或stop_after命中；参数open_stream返回流上下文，sink收集事件。"""
    buffer = ''
    with open_stream() as response:
        response.raise_for_status()
        for chunk in response.iter_text():
            buffer += chunk
            while '\n\n' in buffer:
                block, buffer = buffer.split('\n\n', 1)
                event = decode_block(block)
                if event is None:
                    continue
                sink.append(event)
                if event[0] in ('done', 'error'):
                    return
                if stop_after is not None and stop_after(sink):
                    return
            if delay:
                time.sleep(delay)


class LocalTransport:
    """进程内TestClient传输，复用应用生命周期内的真实调度器。"""

    def __init__(self, app, client):
        self.app = app
        self.client = client

    def stream(self, query):
        """返回指定查询的SSE流上下文，参数query为接口参数。"""
        return self.client.stream('GET', '/api/evaluate/stream', params=query)

    def factors(self):
        """请求元数据接口，返回耗时秒数与状态码。"""
        started = time.perf_counter()
        response = self.client.get('/api/factors')
        return round(time.perf_counter() - started, 3), response.status_code

    def slots(self):
        """返回当前占用的任务名额数。"""
        return len(self.app.state.scheduler.jobs)

    def close(self):
        """进程内传输随TestClient上下文关闭，无需额外动作。"""


class HttpTransport:
    """对真实uvicorn的httpx传输。"""

    def __init__(self, url):
        self.client = httpx.Client(base_url=url, timeout=httpx.Timeout(30.0, read=300.0))

    def stream(self, query):
        """返回指定查询的SSE流上下文，参数query为接口参数。"""
        return self.client.stream('GET', '/api/evaluate/stream', params=query)

    def factors(self):
        """请求元数据接口，返回耗时秒数与状态码。"""
        started = time.perf_counter()
        response = self.client.get('/api/factors')
        return round(time.perf_counter() - started, 3), response.status_code

    def slots(self):
        """真实服务不暴露名额，返回None由日志判断。"""
        return None

    def close(self):
        """关闭底层连接池。"""
        self.client.close()


def read_vmhwm(status_path):
    """读取进程峰值常驻内存；参数status_path为/proc下的status文件。"""
    match = re.search(r'VmHWM:\s+(\d+) kB', Path(status_path).read_text())
    return round(int(match.group(1)) / 1024, 1) if match else None


def find_server_pid():
    """查找uvicorn服务进程号，找不到返回None。"""
    result = subprocess.run(['pgrep', '-f', 'uvicorn backend.app:app'], capture_output=True, text=True)
    return int(result.stdout.split()[0]) if result.stdout.split() else None


def peak_rss_mb(args):
    """返回被测进程峰值内存：本地取自身，HTTP取服务进程。"""
    if args.mode == 'http':
        pid = args.server_pid or find_server_pid()
        status = Path(f'/proc/{pid}/status') if pid else None
        return read_vmhwm(status) if status and status.exists() else None
    return read_vmhwm('/proc/self/status')


def one_request(transport, query, sink, delay=0.0, stop_after=None):
    """发起一路SSE请求并汇总结果；参数transport为传输、query为接口参数、sink收集事件。"""
    started = time.perf_counter()
    error = None
    try:
        consume(lambda: transport.stream(query), sink, delay=delay, stop_after=stop_after)
    except Exception as failure:  # 断连与传输错误都要记录，不静默
        error = f'{type(failure).__name__}: {failure}'
    stages = [stage for stage, _ in sink]
    request_id = sink[0][1].get('request_id') if sink else None
    return {'query': query, 'request_id': request_id, 'elapsed_s': round(time.perf_counter() - started, 3),
            'events': len(sink), 'stages': stages, 'error': error,
            'ok': not error and bool(stages) and stages[-1] == 'done'}


def wait_slots_zero(transport, timeout=30.0):
    """等待任务名额清零，返回等待秒数；不适用时返回None。"""
    if transport.slots() is None:
        return None
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if transport.slots() == 0:
            return round(time.perf_counter() - started, 3)
        time.sleep(0.2)
    return None


def tail_text(path, limit=256 * 1024):
    """读取日志文件尾部文本；参数path为文件、limit为读取字节上限。"""
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - limit))
        return stream.read().decode('utf-8', errors='ignore')


def wait_release_line(request_id, timeout=600.0):
    """等待调度器日志出现该请求的释放记录，返回该行或None。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for path in sorted(SCHEDULER_LOG_DIR.glob('*.log')):
            for line in tail_text(path).splitlines():
                if f'request={request_id} ' in line and 'released' in line:
                    return line.strip()
        time.sleep(1.0)
    return None


def build_queries(args):
    """构造两个完整分析查询，参数为命令行参数。"""
    first = {'factor': args.factor}
    second = {'factor': args.second_factor}
    for query in (first, second):
        if args.start:
            query['start'] = args.start
        if args.end:
            query['end'] = args.end
    return first, second


class Evidence:
    """逐轮追加JSONL证据；每条立即落盘，进程崩溃也能保留已完成轮次。"""

    def __init__(self, mode, enabled):
        self.enabled = enabled
        self.path = EVIDENCE_DIR / f'{date.today():%Y%m%d}-stress-{mode}.jsonl'
        if enabled:
            EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    def write(self, record):
        """写入并打印一条记录；参数record为可JSON化的字典。"""
        text = json.dumps(record, ensure_ascii=False, allow_nan=False)
        if self.enabled:
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(text + '\n')
        print(text, flush=True)


def run_concurrent(transport, queries, args, evidence, run_id):
    """重复双任务完整分析；参数包含两路查询、轮数与证据。"""
    all_ok = True
    for index in range(1, args.rounds + 1):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(one_request, transport, query, []) for query in queries]
            results = [future.result() for future in futures]
        settle = wait_slots_zero(transport)
        settled = transport.slots() is None or settle is not None
        round_ok = all(item['ok'] for item in results) and settled
        record = {'run': run_id, 'case': 'concurrent', 'mode': args.mode, 'round': index,
                  'round_elapsed_s': round(time.perf_counter() - started, 3), 'requests': results,
                  'slots_settle_s': settle, 'peak_rss_mb': peak_rss_mb(args), 'ok': bool(round_ok)}
        evidence.write(record)
        all_ok = all_ok and record['ok']
    return all_ok


def run_cancel(transport, queries, args, evidence, run_id):
    """并发两路，其中一路读到第二阶段后断连，验证取消释放名额且服务仍可用。"""
    victim_sink, keeper_sink = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        keeper = pool.submit(one_request, transport, queries[1], keeper_sink)
        victim = pool.submit(one_request, transport, queries[0], victim_sink, 0.0,
                             lambda events: len(events) >= 2)
        victim_result = victim.result()
        abort_at = time.perf_counter()
        release_line = wait_release_line(victim_result['request_id'], args.release_timeout)
        release_s = round(time.perf_counter() - abort_at, 3) if release_line else None
        keeper_result = keeper.result()
    probe = one_request(transport, LIGHT_QUERY, [])
    victim_stopped = victim_result['stages'][-1] != 'done'
    record = {'run': run_id, 'case': 'cancel', 'mode': args.mode, 'victim': victim_result,
              'keeper': keeper_result, 'probe': probe, 'release_s': release_s,
              'release_line': release_line,
              'peak_rss_mb': peak_rss_mb(args),
              'ok': bool(keeper_result['ok'] and probe['ok'] and victim_stopped
                         and release_s is not None)}
    evidence.write(record)
    return record['ok']


def run_slow(transport, queries, args, evidence, run_id):
    """一路慢读制造背压，验证另一路与元数据接口仍能推进。"""
    slow_sink = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        slow = pool.submit(one_request, transport, queries[0], slow_sink, args.read_delay)
        time.sleep(args.backpressure_wait)
        fast_sink = []
        fast = one_request(transport, LIGHT_QUERY, fast_sink)
        slow_progress = len(slow_sink)
        meta_elapsed, meta_status = transport.factors()
        slow_result = slow.result()
    record = {'run': run_id, 'case': 'slow', 'mode': args.mode, 'slow': slow_result,
              'slow_events_when_fast_done': slow_progress, 'slow_events_total': slow_result['events'],
              'fast': fast, 'meta_elapsed_s': meta_elapsed, 'meta_status': meta_status,
              'peak_rss_mb': peak_rss_mb(args),
              'ok': bool(slow_result['ok'] and fast['ok'] and meta_status == 200
                         and slow_progress < slow_result['events'])}
    evidence.write(record)
    return record['ok']


def main():
    """解析参数并执行一种并发验收场景。"""
    parser = argparse.ArgumentParser(description='并发验收：双任务、取消与慢消费者')
    parser.add_argument('--mode', choices=('local', 'http'), default='local')
    parser.add_argument('--case', choices=('concurrent', 'cancel', 'slow'), default='concurrent')
    parser.add_argument('--url', default='http://127.0.0.1:8800')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--factor', default='MOM_20')
    parser.add_argument('--second-factor', default='RSI_14')
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--read-delay', type=float, default=0.5)
    parser.add_argument('--backpressure-wait', type=float, default=2.0)
    parser.add_argument('--release-timeout', type=float, default=600.0)
    parser.add_argument('--server-pid', type=int)
    parser.add_argument('--no-evidence', action='store_true')
    args = parser.parse_args()

    queries = build_queries(args)
    evidence = Evidence(args.mode, not args.no_evidence)
    run_id = f'{date.today():%Y%m%d}T{time.strftime("%H%M%S")}'
    cases = {'concurrent': run_concurrent, 'cancel': run_cancel, 'slow': run_slow}

    if args.mode == 'local' and args.case != 'concurrent':
        raise SystemExit('本地模式读不到中间事件，取消与背压请用 --mode http 对真实服务验证')

    if args.mode == 'local':
        with TestClient(create_app()) as client:
            transport = LocalTransport(client.app, client)
            ok = cases[args.case](transport, queries, args, evidence, run_id)
    else:
        transport = HttpTransport(args.url)
        try:
            ok = cases[args.case](transport, queries, args, evidence, run_id)
        finally:
            transport.close()
    print('验收结果：' + ('通过' if ok else '未通过'), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
