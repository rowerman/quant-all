"""Alpha 研究契约的单元测试。"""

import pytest

from alpha.config import DEFAULT_CONTRACT, ResearchContract
from alpha.errors import AlphaConfigError


def test_default_contract_matches_first_version():
    """默认契约固定标签、日期和 purge/embargo。"""

    assert DEFAULT_CONTRACT.label == "fwd_ret_open_5"
    assert DEFAULT_CONTRACT.dev_start == "2012-01-01"
    assert DEFAULT_CONTRACT.dev_end == "2023-12-31"
    assert DEFAULT_CONTRACT.holdout_start == "2024-01-01"
    assert DEFAULT_CONTRACT.purge_days == 6
    assert DEFAULT_CONTRACT.embargo_days == 6


def test_contract_rejects_label_mismatch():
    """标签必须与价格通道和周期一致。"""

    with pytest.raises(AlphaConfigError, match="标签"):
        ResearchContract(label="fwd_ret_close_5")


def test_contract_rejects_holdout_overlap_and_short_purge():
    """最终测试集不能与开发期重叠，purge 不能小于 h+1。"""

    with pytest.raises(AlphaConfigError, match="最终测试集"):
        ResearchContract(holdout_start="2023-12-01")
    with pytest.raises(AlphaConfigError, match="purge_days"):
        ResearchContract(purge_days=5)
    with pytest.raises(AlphaConfigError, match="embargo_days"):
        ResearchContract(embargo_days=5)
