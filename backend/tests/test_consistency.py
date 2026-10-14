"""经后端适配的IC与完整面板保持相同样本和排名口径。"""

import numpy as np
import pytest

from backend import service
from calculate import EvalParams, evaluate
from backend.errors import MonitorError, error_payload


@pytest.mark.parametrize('price,horizon,universe', [
    ('close', 5, 'all_a'), ('open', 10, 'all_a'), ('close', 1, 'index:000300.SH'),
])
def test_light_ic_matches_full(price, horizon, universe):
    params = EvalParams('MOM_20', '2024-05-06', '2024-05-31', universe, horizon, price)
    full = dict(evaluate('MOM_20', params, stages=['overview', 'ic_series', 'ic_dist']))
    light = dict(service.ic_events(params, service.IC_STAGES))
    for key in ('ic', 'rank_ic', 'n'):
        np.testing.assert_allclose(full['ic_series']['metrics']['ic_series'][key],
                                   light['ic_series']['metrics']['ic_series'][key], atol=1e-10)
    for key in ('ic', 'rank_ic'):
        np.testing.assert_allclose(
            full['ic_series']['metrics']['ic_series']['cumulative'][key],
            light['ic_series']['metrics']['ic_series']['cumulative'][key],
            atol=1e-10,
            equal_nan=True,
        )


def test_data_missing_and_unknown_option(monkeypatch):
    def missing(*args):
        raise FileNotFoundError('private path')
    monkeypatch.setattr(service.calculate, 'resolve_factor', missing)
    with pytest.raises(FileNotFoundError) as error:
        service.prepare({'factor': 'MOM_20'})
    status, payload = error_payload(error.value, 'test')
    assert status == 503 and 'private path' not in payload['message']
    with pytest.raises(MonitorError):
        service.prepare({'factor': 'MOM_20', 'unrecognized': 'x'})
