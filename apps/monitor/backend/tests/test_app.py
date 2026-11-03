"""HTTP/SSE协议测试与真实库轻量集成。"""

import json
from threading import Event

import pytest
import quant
from fastapi.testclient import TestClient

from backend.app import create_app
from backend import service
from backend.service import RequestSpec
from calculate import EvalParams


def decode(body):
    return [(block.splitlines()[0][7:], json.loads(block.splitlines()[1][6:]))
            for block in body.strip().split('\n\n') if block.startswith('event:')]


def test_routes_and_real_ic():
    with TestClient(create_app()) as client:
        options = client.get('/api/options').json()
        assert options['limits']['compare'] == 4
        factors = client.get('/api/factors').json()
        assert len(factors) == len(quant.load_factor_list(status='implemented'))
        assert client.get('/api/factors/MOM_20').json()['formula']
        result = client.get('/api/evaluate/stream', params={
            'factor': 'MOM_20', 'start': '2024-05-06', 'end': '2024-05-31', 'mode': 'ic'})
        assert result.status_code == 200
        events = decode(result.text)
        business = [(stage, payload) for stage, payload in events if stage != 'status']
        assert [s for s, _ in business] == ['queued', 'meta', 'overview', 'ic_series',
                                           'ic_dist', 'factor_done', 'done']
        assert any(stage == 'status' for stage, _ in events)
        assert any(payload['status'] == 'preflight' for stage, payload in events if stage == 'status')
        assert business[2][1]['metrics']['ic_stats']['valid_days'] == 20
        assert all('request_id' in payload for _, payload in events)


@pytest.mark.parametrize('query,status', [
    ({'factor': 'unknown'}, 404),
    ({'factor': 'MOM_20', 'start': 'bad'}, 400),
    ({'factor': 'MOM_20', 'metric_options': '[]'}, 400),
    ({'factor': 'MOM_20', 'stages': 'unknown'}, 400),
    ({'factor': 'MOM_20', 'start': '2000-01-04', 'end': '2026-09-28'}, 400),
    ({'factor': 'MOM_20', 'horizon': 'x'}, 400),
    ({'factor': 'MOM_20', 'winsorize': '1'}, 400),
    ({'factor': 'MOM_20', 'metric_options': '{"correlation":{"top_n":0}}'}, 400),
])
def test_preflight_errors(query, status):
    with TestClient(create_app()) as client:
        result = client.get('/api/evaluate/stream', params=query)
        assert result.status_code == status
        assert result.headers['content-type'].startswith('application/json')
        assert {'code', 'message', 'request_id'} <= result.json().keys()


def test_compare_order_and_error(monkeypatch):
    def fake(params, stages):
        yield 'meta', {}
        yield 'overview', {'metrics': {}}
        yield 'done', {}
    monkeypatch.setattr(service, 'ic_events', fake)
    with TestClient(create_app()) as client:
        result = client.get('/api/compare/stream', params={'factors': 'MOM_20,MOM_60', 'mode': 'ic'})
        events = decode(result.text)
        assert [p['factor'] for s, p in events if s == 'meta'] == ['MOM_20', 'MOM_60']
        assert sum(s == 'done' for s, _ in events) == 1
        assert client.get('/api/compare/stream', params={'factors': 'MOM_20,MOM_20'}).status_code == 400


def test_http_busy_and_runtime_error(monkeypatch):
    gate = Event()
    def blocked():
        gate.wait(5)
        yield 'done', {}
    with TestClient(create_app()) as client:
        scheduler = client.app.state.scheduler
        jobs = [scheduler.submit(blocked) for _ in range(6)]
        try:
            assert client.get('/api/evaluate/stream', params={'factor': 'MOM_20'}).status_code == 429
        finally:
            gate.set()
            for job in jobs: job.cancel()
            for job in jobs: assert job.finished.wait(3)
        def failed(spec):
            yield 'meta', {}
            raise RuntimeError('private details')
        monkeypatch.setattr(service, 'run', failed)
        events = decode(client.get('/api/evaluate/stream', params={'factor': 'MOM_20'}).text)
        assert events[-1][0] == 'error'
        assert events[-1][1]['code'] == 'MONITOR-104'


def test_run_closes_generator_when_cancelled(monkeypatch):
    closed = Event()
    def fake(params, stages):
        try:
            yield 'meta', {}
            yield 'done', {}
        finally: closed.set()
    monkeypatch.setattr(service, 'ic_events', fake)
    spec = RequestSpec(('MOM_20',), EvalParams('MOM_20', '2024-01-01', '2024-02-01'), 'ic', (), {})
    iterator = service.run(spec)
    next(iterator)
    iterator.close()
    assert closed.is_set()


@pytest.mark.parametrize('error,status,code', [
    (FileNotFoundError('private file'), 503, 'MONITOR-103'),
    (RuntimeError('private stack'), 500, 'MONITOR-104'),
])
def test_metadata_failure_is_json(monkeypatch, error, status, code):
    def failed():
        raise error
    monkeypatch.setattr(service, 'options', failed)
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.get('/api/options')
        assert response.status_code == status and response.json()['code'] == code
        assert 'private' not in response.json()['message']
