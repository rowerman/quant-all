"""代码归一与板块判定的单元测试。"""

import pytest

from quant.codes import (
    board_of,
    extract_code6,
    infer_exchange,
    normalize_code,
    normalize_codes,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sh600000", "SH600000"),
        ("SH600000", "SH600000"),
        ("600000.SH", "SH600000"),
        ("sh.600000", "SH600000"),
        ("600000", "SH600000"),
        ("T600018.SH", "SH600018"),
        ("000001", "SZ000001"),
        ("000001.SZ", "SZ000001"),
        ("300750", "SZ300750"),
        ("688981", "SH688981"),
        ("430017", "BJ430017"),
        ("920000", "BJ920000"),
        ("830799", "BJ830799"),
    ],
)
def test_normalize_code(raw, expected):
    """各种来源写法都应归一到 SH/SZ/BJ + 6 位数字。"""
    assert normalize_code(raw) == expected


def test_normalize_code_rejects_garbage():
    """无法解析的代码要抛错，不能静默返回空值。"""
    with pytest.raises(ValueError):
        normalize_code("浦发银行")


@pytest.mark.parametrize(
    ("code6", "board"),
    [
        ("600000", "main_sh"),
        ("601318", "main_sh"),
        ("603444", "main_sh"),
        ("000001", "main_sz"),
        ("002415", "main_sz"),
        ("300750", "chinext"),
        ("301029", "chinext"),
        ("688981", "star"),
        ("689009", "star"),
        ("430017", "bse"),
        ("920000", "bse"),
        ("900901", "b_sh"),
        ("200012", "b_sz"),
    ],
)
def test_board_of(code6, board):
    """板块判定决定涨跌停限幅，必须与交易所分类一致。"""
    assert board_of(code6) == board


def test_infer_exchange_fallback():
    """缺少交易所信息时按代码前缀兜底。"""
    assert infer_exchange("600000") == "SH"
    assert infer_exchange("000001") == "SZ"
    assert infer_exchange("920000") == "BJ"


def test_extract_code6():
    """抽取 6 位数字部分。"""
    assert extract_code6("T600018.SH") == "600018"
    assert extract_code6("sh.600000") == "600000"


def test_normalize_codes_keeps_unknown_as_none():
    """批量归一供两个阶段共用，认不出的历史代码置空而不是中断。"""
    assert normalize_codes(["600050.SH", "000001.SZ"], "测试") == ["SH600050", "SZ000001"]
    assert normalize_codes(["600050.SH", "T00018.SH"], "测试") == ["SH600050", None]
