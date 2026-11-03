"""Tushare 落地数据解析的单元测试。"""


from scripts.clean import sources
from scripts.sources import tushare_src


def test_to_iso_converts_compact_dates():
    """8 位数字日期要转成 YYYY-MM-DD，其余保持前 10 位。"""
    assert sources._to_iso("20240102") == "2024-01-02"
    assert sources._to_iso("2024-01-02") == "2024-01-02"


def test_is_date_like_filters_missing_values():
    """空值、nan、None 都不算日期。"""
    assert sources._is_date_like("20240102")
    assert not sources._is_date_like("nan")
    assert not sources._is_date_like("")
    assert not sources._is_date_like("None")


def test_normalize_codes_skips_unknown_legacy_code():
    """Tushare 历史遗留代码跳过而不是中断清洗，正常代码不受影响。"""
    assert sources.normalize_codes(["600000.SH", "T00018.SH"], "测试") == ["SH600000", None]


def test_classify_error_kinds():
    """把服务端报错正确归类，决定要不要退避重试。"""
    assert tushare_src._classify("您的并发请求过多（上限 2个）") == "concurrency"
    assert tushare_src._classify("频率超限(1次/分钟)") == "rate"
    assert tushare_src._classify("抱歉，您没有接口(stk_limit)访问权限") == "permission"
    assert tushare_src._classify("您的token不对，请确认。") == "token"
    assert tushare_src._classify("接口用法错误，请检查代码后重试！") == "usage"
    assert tushare_src._classify("网络错误") == "other"
