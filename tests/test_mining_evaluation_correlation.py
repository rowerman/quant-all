"""挖掘候选正式因子相关性的三后端和边界测试。"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from mining.evaluation_contract import QualityPolicy
from mining.evaluation_correlation import (
    CandidateCorrelationCache,
    _correlate_frame,
)

requires_native = pytest.mark.skipif(
    importlib.util.find_spec("quant_native") is None,
    reason="quant_native 扩展未构建",
)


def _policy(**overrides) -> QualityPolicy:
    """构造只关注相关性门槛的质量规则。"""

    values = {
        "schema_version": "mining_quality_test",
        "forward_look_samples": 3,
        "forward_look_relative_tolerance": 1e-5,
        "max_flat_day_share": 0.01,
        "correlation_sample_days": 24,
        "correlation_min_stocks": 5,
        "correlation_min_days": 20,
        "correlation_max_absolute": 0.9,
        "quick_min_valid_days": 20,
        "warning_rank_ic": 0.02,
        "warning_icir": 0.2,
        "warning_year_sign_ratio": 0.6,
    }
    values.update(overrides)
    return QualityPolicy(**values)


def _frame() -> tuple[pd.DataFrame, tuple[str, ...]]:
    """构造打乱顺序且包含并列、缺失、无穷和常数 peer 的宽表。"""

    rng = np.random.default_rng(1211)
    dates = pd.bdate_range("2024-01-02", periods=24)
    stocks = [f"SH600{index:03d}" for index in range(12)]
    frame = pd.MultiIndex.from_product(
        [dates, stocks],
        names=["trade_date", "stock_code"],
    ).to_frame(index=False)
    target = rng.integers(0, 6, size=len(frame)).astype(np.float64)
    frame["candidate"] = target
    frame["peer_a"] = target + rng.normal(scale=0.2, size=len(frame))
    frame["peer_b"] = -target
    frame["peer_c"] = 1.0
    frame.loc[frame.index[::31], "candidate"] = np.nan
    frame.loc[frame.index[::37], "peer_a"] = np.inf
    frame.loc[frame.index[::41], "peer_b"] = np.nan
    frame = frame.sample(frac=1.0, random_state=1223).reset_index(drop=True)
    return frame, ("peer_a", "peer_b", "peer_c")


def _reference(
    frame: pd.DataFrame,
    peers: tuple[str, ...],
    policy: QualityPolicy,
) -> dict[str, object]:
    """用 pandas 逐日排名生成独立金标准。"""

    sums = np.zeros(len(peers), dtype=np.float64)
    days = np.zeros(len(peers), dtype=np.int64)
    for _, daily in frame.groupby("trade_date", sort=False):
        values = daily[["candidate", *peers]].to_numpy(dtype=np.float64)
        for index, _peer in enumerate(peers, start=1):
            pair = values[:, [0, index]]
            pair = pair[np.isfinite(pair).all(axis=1)]
            if len(pair) < policy.correlation_min_stocks:
                continue
            left = pd.Series(pair[:, 0]).rank(method="average").to_numpy()
            right = pd.Series(pair[:, 1]).rank(method="average").to_numpy()
            if np.std(left) == 0 or np.std(right) == 0:
                continue
            correlation = float(np.corrcoef(left, right)[0, 1])
            if np.isfinite(correlation):
                sums[index - 1] += correlation
                days[index - 1] += 1
    means = np.divide(
        sums,
        days,
        out=np.full(len(peers), np.nan),
        where=days >= policy.correlation_min_days,
    )
    finite = np.isfinite(means)
    if not finite.any():
        return {"reason": "insufficient_days", "passed": False}
    positions = np.flatnonzero(finite)
    best = int(positions[np.argmax(np.abs(means[positions]))])
    maximum = float(abs(means[best]))
    return {
        "passed": maximum < policy.correlation_max_absolute,
        "reason": None,
        "observed_days": int(days[best]),
        "peer": peers[best],
        "correlation": float(means[best]),
        "max_absolute": maximum,
    }


def test_python_correlation_matches_pandas_reference(monkeypatch):
    """Python 回退必须与原有逐日 pandas 排名口径一致。"""

    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "python")
    frame, peers = _frame()
    policy = _policy()

    actual = _correlate_frame(
        frame,
        target_name="candidate",
        peers=peers,
        policy=policy,
    )
    expected = _reference(frame, peers, policy)

    assert actual["passed"] is expected["passed"]
    assert actual["reason"] is expected["reason"]
    assert actual["peer"] == expected["peer"]
    assert actual["observed_days"] == expected["observed_days"]
    assert actual["correlation"] == pytest.approx(
        expected["correlation"],
        abs=1e-12,
    )
    assert actual["max_absolute"] == pytest.approx(
        expected["max_absolute"],
        abs=1e-12,
    )


@requires_native
def test_native_and_auto_match_python(monkeypatch):
    """native/auto 的目标、有效天数和门禁结论保持一致。"""

    frame, peers = _frame()
    policy = _policy()
    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "python")
    expected = _correlate_frame(
        frame,
        target_name="candidate",
        peers=peers,
        policy=policy,
    )
    for backend in ("native", "auto"):
        monkeypatch.setenv("QUANT_KERNEL_BACKEND", backend)
        actual = _correlate_frame(
            frame,
            target_name="candidate",
            peers=peers,
            policy=policy,
        )
        assert actual == expected


def test_insufficient_days_fails_closed(monkeypatch):
    """没有任何 peer 达到最小有效日时不得放行。"""

    monkeypatch.setenv("QUANT_KERNEL_BACKEND", "python")
    frame, peers = _frame()
    result = _correlate_frame(
        frame,
        target_name="candidate",
        peers=peers,
        policy=_policy(correlation_min_days=99),
    )

    assert result == {
        "passed": False,
        "reason": "insufficient_days",
        "max_absolute": None,
        "peer": None,
        "min_days": 99,
    }


def test_cache_reuses_peer_frame_only_for_the_same_window(monkeypatch):
    """同一筛选窗口复用缓存，窗口变化时重新读取正式 peer。"""

    from mining import evaluation_correlation

    frame, peers = _frame()
    peer_frame = frame[["trade_date", "stock_code", *peers]]
    calls = []

    def load_peer_frame(conn, *, peer_factor_dir, dates):
        calls.append((conn, tuple(dates)))
        return peer_frame, peers

    monkeypatch.setattr(
        evaluation_correlation,
        "_load_peer_frame",
        load_peer_frame,
    )
    cache = CandidateCorrelationCache("data/factors")
    dates = tuple(str(day.date()) for day in sorted(frame["trade_date"].unique()))

    first = cache.load("first", dates)
    second = cache.load("second", dates)
    assert first is second
    assert calls == [("first", dates)]

    cache.load("third", dates[:-1])
    assert calls == [("first", dates), ("third", dates[:-1])]
