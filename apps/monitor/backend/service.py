"""参数预检、目录读取与计算包适配；指标公式仍由calculate负责。"""

from dataclasses import dataclass, replace
import json
from types import SimpleNamespace
import time

import calculate
from calculate.config import FACTOR_DIR
from calculate.correlation import available_factors
from calculate.errors import EvalParamError, EvalPanelError
from calculate.metrics import resolve_options
from calculate.metrics.ic import compute_stats, compute_series, compute_dist
from calculate.native import serialized, serialized_events
from calculate.panel import count_trade_days, estimate_panel_rows, check_label_columns
from backend.errors import MonitorError

IC_STAGES = ('overview', 'ic_series', 'ic_dist')


@dataclass(frozen=True)
class RequestSpec:
    """已校验的请求；params统一口径，factors按用户顺序计算。"""

    factors: tuple[str, ...]
    params: calculate.EvalParams
    mode: str
    stages: tuple[str, ...]
    metric_options: dict


@serialized
def options():
    """读取交易区间并返回前端所需枚举、指标清单和保护上限。"""
    with calculate.open_source() as conn:
        first, last = conn.execute('SELECT min(trade_date),max(trade_date) FROM src.md.calendar').fetchone()
    default = calculate.default_params('MOM_20').describe()
    return {'start': str(first), 'end': str(last), 'defaults': default,
            'universes': [{'key': item.key, 'label': item.label, 'detail': item.detail}
                          for item in calculate.list_universes()],
            'horizons': list(calculate.HORIZONS), 'prices': list(calculate.PRICE_CHANNELS),
            'quantiles': list(calculate.QUANTILE_CHOICES),
            'metrics': [item.describe() for item in calculate.list_metrics()],
            'stages': list(calculate.planned_stages()), 'ic_stages': list(IC_STAGES),
            'limits': {'trade_days': calculate.MAX_TRADE_DAYS, 'panel_rows': calculate.MAX_PANEL_ROWS,
                       'compare': 4, 'workers': 2, 'waiting': 4},
            'note': '全历史只支持轻量IC；完整分析保留2400交易日/800万行上限'}


@serialized
def factor_list():
    """目录库元数据与实际落盘清单交叉确认，返回可评估因子。"""
    import quant
    present = {name for names in available_factors().values() for name in names}
    frame = quant.load_factor_list(status='implemented')
    return frame.loc[frame.name.isin(present)].to_dict(orient='records')


@serialized
def factor_detail(name):
    """获取单因子元数据，参数name必须为已落盘因子。"""
    calculate.resolve_factor(name)
    rows = [row for row in factor_list() if row['name'] == name]
    if not rows:
        raise MonitorError('因子目录库缺少该因子元数据，请检查目录库版本', 503, 'MONITOR-103')
    return rows[0]


@serialized
def prepare(query, compare=False):
    """校验GET查询并预检资源上限；参数query为查询映射，compare为对比开关。"""
    allowed = {'factor', 'factors', 'start', 'end', 'universe', 'horizon', 'price',
               'quantiles', 'winsorize', 'mode', 'stages', 'metric_options'}
    if set(query) - allowed:
        raise MonitorError(f'未知请求参数：{sorted(set(query)-allowed)}')
    names = tuple(query.get('factors', '').split(',')) if compare else (query.get('factor', ''),)
    if not all(names) or (compare and (not 2 <= len(names) <= 4 or len(set(names)) != len(names))):
        raise MonitorError('对比需要2～4个不同因子' if compare else '需要提供factor')
    mode = query.get('mode', 'full')
    if mode not in ('full', 'ic'):
        raise MonitorError('mode可选full或ic')
    for name in names:
        calculate.resolve_factor(name)
    defaults = calculate.default_params(names[0])
    flag = query.get('winsorize', 'true')
    if flag not in ('true', 'false'):
        raise MonitorError('winsorize必须为true或false')
    try:
        horizon = int(query.get('horizon', defaults.horizon))
        quantiles = int(query.get('quantiles', defaults.quantiles))
        supplied = json.loads(query.get('metric_options', '{}'))
    except (ValueError, TypeError) as error:
        raise MonitorError('周期/分组需要整数，metric_options需要JSON对象') from error
    params = calculate.EvalParams(names[0], query.get('start', defaults.start),
                                 query.get('end', defaults.end), query.get('universe', 'all_a'),
                                 horizon, query.get('price', 'close'), quantiles, flag == 'true')
    calculate.resolve_universe(params.universe)
    if not isinstance(supplied, dict):
        raise MonitorError('metric_options必须是JSON对象')
    metrics = {metric.key: metric for metric in calculate.list_metrics()}
    if supplied.keys() - metrics.keys():
        raise MonitorError('metric_options包含未知指标')
    resolved = {key: resolve_options(metrics[key], value) for key, value in supplied.items()}
    if 'correlation' in resolved and not 1 <= resolved['correlation']['top_n'] <= 100:
        raise MonitorError('top_n必须在1～100之间')
    available = IC_STAGES if mode == 'ic' else calculate.planned_stages()
    chosen = tuple(query['stages'].split(',')) if 'stages' in query else available
    if not chosen or set(chosen) - set(available):
        raise MonitorError(f'非法stage，可选：{list(available)}')
    stages = tuple(stage for stage in available if stage in chosen)
    with calculate.open_source() as conn:
        days = count_trade_days(conn, params.start, params.end)
        if days == 0:
            raise EvalParamError('所选区间没有交易日')
        check_label_columns(conn, [params.label_column])
        if mode == 'full':
            if days > calculate.MAX_TRADE_DAYS:
                raise EvalParamError(f'{days}交易日超过2400日上限；请缩小区间或选择轻量IC')
            for name in names:
                spec = calculate.resolve_factor(name)
                estimate = estimate_panel_rows(conn, spec.family, FACTOR_DIR, params.start, params.end)
                if estimate > calculate.MAX_PANEL_ROWS:
                    raise EvalPanelError(f'面板上界{estimate:,}行超过800万；请缩小区间或选择轻量IC')
    return RequestSpec(names, params, mode, stages, resolved)


def ic_events(params, stages):
    """适配轻量IC为既定stage；参数为评估参数与选择的IC阶段。"""
    metrics = [m.describe() for m in calculate.list_metrics()
               if m.key in ('ic_stats', 'ic_series', 'ic_dist') and m.stage in stages]
    spec = calculate.resolve_factor(params.factor)
    yield 'meta', {'factor': params.factor, 'params': params.describe(), 'mode': 'ic',
                   'stages': list(stages), 'metrics': metrics, 'spec': {'name': spec.name}}
    series = calculate.load_daily_ic(params)
    context = SimpleNamespace(params=params, spec=spec, daily_ic=lambda: series)
    computes = {'overview': ('ic_stats', compute_stats), 'ic_series': ('ic_series', compute_series),
                'ic_dist': ('ic_dist', compute_dist)}
    for stage in stages:
        start = time.perf_counter()
        key, compute = computes[stage]
        yield stage, {'stage': stage, 'metrics': {key: compute(context, {})},
                      'elapsed_ms': (time.perf_counter()-start)*1000}
    yield 'done', {'ic_load_elapsed_ms': series.attrs['elapsed_ms']}


@serialized_events
def run(spec):
    """依次计算请求里的因子，单因子done改为factor_done，最后发送总done。"""
    started = time.perf_counter()
    for name in spec.factors:
        params = replace(spec.params, factor=name)
        events = ic_events(params, spec.stages) if spec.mode == 'ic' else calculate.evaluate(
            name, params, stages=spec.stages, metric_options=spec.metric_options)
        try:
            for stage, payload in events:
                yield ('factor_done' if stage == 'done' else stage), {**payload, 'factor': name}
        finally:
            events.close()
            events = None
    yield 'done', {'factors': list(spec.factors), 'total_elapsed_ms': (time.perf_counter()-started)*1000}
