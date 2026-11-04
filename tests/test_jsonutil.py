import json

import pandas as pd

from calculate.jsonutil import json_safe


def test_json_safe_converts_missing_timestamp_to_none():
    payload = json_safe({"finished_at": pd.NaT, "nested": [pd.Timestamp("2026-11-04")]})

    assert payload == {"finished_at": None, "nested": ["2026-11-04"]}
    assert json.loads(json.dumps(payload)) == payload
