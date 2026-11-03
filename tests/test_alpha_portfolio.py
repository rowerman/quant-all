"""P3-5 分数平滑与持仓规则测试。"""

import pandas as pd
import pytest

from alpha.errors import AlphaEvaluationError
from alpha.portfolio import HoldingPolicy, select_holdings, smooth_rank_scores


def test_smooth_rank_scores_is_causal_and_ignores_ineligible():
    """未来分数和池外极端值不能改变历史平滑结果。"""

    dates = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    stocks = ["A", "B", "C"]
    scores = pd.DataFrame(
        [
            [3.0, 2.0, 1.0],
            [1.0, 3.0, 2.0],
            [2.0, 1.0, 3.0],
            [3.0, 1.0, 2.0],
        ],
        index=dates,
        columns=stocks,
    )
    eligible = pd.DataFrame(True, index=dates, columns=stocks)
    eligible.loc[dates[0], "C"] = False
    left = smooth_rank_scores(scores, eligible, window=2)

    changed = scores.copy()
    changed.loc[dates[-1]] = [100.0, -100.0, 0.0]
    changed.loc[dates[0], "C"] = -1000.0
    right = smooth_rank_scores(changed, eligible, window=2)

    pd.testing.assert_frame_equal(
        left.loc[dates[:-1], ["A", "B"]],
        right.loc[dates[:-1], ["A", "B"]],
    )


def test_smooth_rank_scores_rejects_bad_window():
    """平滑参数必须是合法正整数。"""

    scores = pd.DataFrame([[1.0]], index=["2020-01-03"], columns=["A"])
    eligible = pd.DataFrame([[True]], index=scores.index, columns=scores.columns)
    with pytest.raises(AlphaEvaluationError, match="窗口"):
        smooth_rank_scores(scores, eligible, window=0)


def test_buffer_keeps_rank_inside_cutoff_and_sells_outside():
    """buffer 保留前 45 名内旧持仓，并卖出排名更低的旧持仓。"""

    score = pd.Series(
        [5.0, 4.0, 3.0, 2.0, 1.0],
        index=["A", "B", "C", "D", "E"],
    )
    eligible = pd.Series(True, index=score.index)
    policy = HoldingPolicy(kind="rank_buffer", buffer_rank=3)

    kept = select_holdings(
        score,
        eligible,
        ["C"],
        top_n=2,
        policy=policy,
    )
    sold = select_holdings(
        score,
        eligible,
        ["D"],
        top_n=2,
        policy=policy,
    )

    assert set(kept) == {"A", "C"}
    assert set(sold) == {"A", "B"}


def test_buffer_sells_ineligible_and_reduces_turnover():
    """不可买旧持仓必须退出，buffer 可避免边际换仓。"""

    day1 = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0})
    day2 = pd.Series({"A": 4.0, "B": 3.0, "C": 5.0})
    policy = HoldingPolicy(kind="rank_buffer", buffer_rank=3)
    previous = select_holdings(
        day1,
        pd.Series(True, index=day1.index),
        None,
        top_n=2,
        policy=policy,
    )
    held = select_holdings(
        day2,
        pd.Series(True, index=day2.index),
        previous,
        top_n=2,
        policy=policy,
    )
    raw = select_holdings(
        day2,
        pd.Series(True, index=day2.index),
        None,
        top_n=2,
        policy=HoldingPolicy(),
    )
    ineligible = pd.Series({"A": True, "B": False, "C": True})
    forced = select_holdings(
        day2,
        ineligible,
        previous,
        top_n=2,
        policy=policy,
    )

    assert set(previous) == {"A", "B"}
    assert set(held) == {"A", "B"}
    assert set(raw) == {"A", "C"}
    assert "B" not in set(forced)
    assert len(held) == 2


def test_select_holdings_is_deterministic_for_ties():
    """同分时按股票代码稳定排序。"""

    score = pd.Series([1.0, 1.0, 1.0], index=["C", "A", "B"])
    eligible = pd.Series(True, index=score.index)
    selected = select_holdings(score, eligible, None, top_n=2)
    assert selected.tolist() == ["A", "B"]


def test_buffer_rank_cannot_be_less_than_top_n():
    """buffer 排名范围必须覆盖目标持仓数。"""

    score = pd.Series([2.0, 1.0], index=["A", "B"])
    eligible = pd.Series(True, index=score.index)
    with pytest.raises(AlphaEvaluationError, match="不能小于"):
        select_holdings(
            score,
            eligible,
            pd.Index(["A"]),
            top_n=2,
            policy=HoldingPolicy(kind="rank_buffer", buffer_rank=1),
        )
