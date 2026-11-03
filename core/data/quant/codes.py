"""股票代码归一与板块判定。

项目内统一使用 ``SH600000``（大写交易所前缀 + 6 位数字）作为唯一代码口径。
各数据源的原始格式需要在导入时经过 :func:`normalize_code` 转换：

===============  ==================
来源             原始形态
===============  ==================
qlib features     ``sh600000``
instruments      ``SH600000``
Tushare          ``600000.SH`` / ``T600018.SH``
BaoStock         ``sh.600000``
AKShare/东财      ``600000``、``000001``
===============  ==================
"""

from __future__ import annotations

import re

_CODE_RE = re.compile(r"^[A-Za-z]{0,2}\.?(\d{6})$")

# 交易所前缀 -> 板块，用于涨跌停限幅判定
_BOARD_PREFIXES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("600", "601", "603", "605"), "main_sh"),
    (("000", "001", "002", "003"), "main_sz"),
    (("300", "301"), "chinext"),
    (("688", "689"), "star"),
    (("900",), "b_sh"),
    (("200",), "b_sz"),
)

_BSE_PREFIXES_3 = ("430", "920")
_BSE_PREFIXES_2 = ("83", "87", "88")


def board_of(code6: str) -> str:
    """按 6 位数字代码判定板块。

    参数:
        code6: 6 位数字代码，例如 "600000"。

    返回:
        main_sh / main_sz / chinext / star / bse / b_sh / b_sz / other。
    """
    for prefixes, board in _BOARD_PREFIXES:
        if code6.startswith(prefixes):
            return board
    if code6.startswith(_BSE_PREFIXES_3) or code6[:2] in _BSE_PREFIXES_2:
        return "bse"
    return "other"


def infer_exchange(code6: str) -> str:
    """在缺少交易所信息时，按代码前缀推断交易所。

    参数:
        code6: 6 位数字代码。
    """
    board = board_of(code6)
    if board in ("main_sh", "star", "b_sh"):
        return "SH"
    if board in ("main_sz", "chinext", "b_sz"):
        return "SZ"
    if board == "bse":
        return "BJ"
    return "UNKNOWN"


def extract_code6(raw: str) -> str:
    """从各种写法的代码中抽出 6 位数字部分。

    参数:
        raw: 原始代码字符串。
    """
    text = str(raw).strip().upper()
    match = _CODE_RE.match(text)
    if match:
        return match.group(1)
    digits = re.findall(r"\d{6}", text)
    if digits:
        return digits[0]
    raise ValueError(f"无法解析股票代码: {raw!r}")


def normalize_code(raw: str) -> str:
    """把任意来源的股票代码转成项目统一口径 ``SH600000``。

    参数:
        raw: 原始代码，支持 ``sh600000``、``600000.SH``、``sh.600000``、
            ``T600018.SH``、``600000`` 等写法。

    返回:
        归一后的代码。
    """
    text = str(raw).strip().upper()
    code6 = extract_code6(text)

    # 形如 600000.SH / T600018.SH：交易所写在点号后面
    suffix = ""
    if "." in text:
        tail = text.split(".")[-1]
        if tail in ("SH", "SZ", "BJ"):
            suffix = tail
    # 形如 SH600000 / sh.600000 / SZ000001：交易所写在前面
    prefix = ""
    head = text.split(".")[0]
    head_letters = re.match(r"^[A-Z]{2}", head)
    if head_letters and head_letters.group(0) in ("SH", "SZ", "BJ"):
        prefix = head_letters.group(0)

    exchange = suffix or prefix or infer_exchange(code6)
    return f"{exchange}{code6}"


def normalize_codes(values, where: str, logger=None) -> list[str | None]:
    """批量归一代码，认不出的历史遗留代码记 None 并留痕。

    Tushare 对「代码被历史主体占用过」的情况加 T 前缀（如上港集箱 T00018.SH），
    这类行都落在现上市主体的上市期之外，跳过但不能静默丢弃。

    参数:
        values: 原始代码列。
        where: 调用来源，用于告警定位。
        logger: 可选的日志对象，传入时对无法归一的代码记 WARNING。

    返回:
        与输入等长的代码列表，无法识别的元素为 None。
    """
    codes: list[str | None] = []
    unknown: dict[str, int] = {}
    for value in values:
        try:
            codes.append(normalize_code(value))
        except ValueError:
            text = str(value)
            unknown[text] = unknown.get(text, 0) + 1
            codes.append(None)
    if unknown and logger is not None:
        logger.warning(
            "%s 跳过无法归一化的历史代码 %s（合计 %d 行）",
            where,
            unknown,
            sum(unknown.values()),
        )
    return codes
