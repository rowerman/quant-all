"""20260928-2 修复的定点替换脚本（已执行，保留用于复现与追溯）。

本环境里 apply_patch 的 Update 不可用（读文件即触发 bwrap 报错），
因此改用「锚点字符串 + 命中次数断言」的方式落盘：每个锚点必须恰好命中一次，
否则整体不写盘。改动内容见 `AiRecord/plans/20260928-2.md`。

用法：
    .venv/bin/python tools/v5/apply_fix_20260928_2.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

NEW_FUNC = '''def normalize_codes(values, where: str) -> list[str | None]:
    """批量把原始代码归一成项目口径，认不出的历史遗留代码记 None。

    参数:
        values: 原始代码列。
        where: 调用来源，用于告警定位。

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
    if unknown:
        # Tushare 给「代码被历史主体占用过」的情况加 T 前缀（如上港集箱 T00018.SH），
        # 这类行都落在现上市主体的上市期之外，跳过但必须留痕，不能静默丢弃
        logger.warning(
            "%s 跳过无法归一化的历史代码 %s（合计 %d 行）",
            where,
            unknown,
            sum(unknown.values()),
        )
    return codes


'''

EDITS: dict[str, list[tuple[str, str]]] = {
    "scripts/clean/sources.py": [
        (
            "def _shards(api_name: str, year: int | None = None) -> list[Path]:",
            NEW_FUNC + "def _shards(api_name: str, year: int | None = None) -> list[Path]:",
        ),
        (
            '    frame["stock_code"] = [normalize_code(value) for value in frame["ts_code"]]\n'
            '    frame["trade_date"] = frame["trade_date"].astype(str)\n'
            '    return frame[["stock_code", "trade_date"]].drop_duplicates()',
            '    frame["stock_code"] = normalize_codes(frame["ts_code"], "停牌记录 suspend_d")\n'
            '    frame = frame[frame["stock_code"].notna()]\n'
            '    frame["trade_date"] = frame["trade_date"].astype(str)\n'
            '    return frame[["stock_code", "trade_date"]].drop_duplicates()',
        ),
    ],
    "scripts/clean/io.py": [
        ("from quant.codes import normalize_code\n", ""),
        (
            "def _prepare(frame: pd.DataFrame) -> pd.DataFrame:\n"
            '    """统一代码与日期格式。\n'
            "\n"
            "    参数:\n"
            "        frame: 接口原始长表。\n"
            '    """\n'
            "    result = frame.copy()\n"
            '    result["stock_code"] = [normalize_code(value) for value in result["ts_code"]]',
            "def _prepare(frame: pd.DataFrame, label: str) -> pd.DataFrame:\n"
            '    """统一代码与日期格式，认不出的历史遗留代码整行丢弃。\n'
            "\n"
            "    参数:\n"
            "        frame: 接口原始长表。\n"
            "        label: 调用来源描述，用于告警定位。\n"
            '    """\n'
            "    result = frame.copy()\n"
            '    result["stock_code"] = sources.normalize_codes(result["ts_code"], label)',
        ),
        (
            '    daily = _prepare(sources.load_api("daily", year))',
            '    daily = _prepare(sources.load_api("daily", year), f"daily {year}")',
        ),
        (
            '    adjust = _prepare(sources.load_api("adj_factor", year))',
            '    adjust = _prepare(sources.load_api("adj_factor", year), f"adj_factor {year}")',
        ),
        (
            '    limit = _prepare(sources.load_api("stk_limit", year))',
            '    limit = _prepare(sources.load_api("stk_limit", year), f"stk_limit {year}")',
        ),
        (
            '    basic = _prepare(sources.load_api("daily_basic", year))',
            '    basic = _prepare(sources.load_api("daily_basic", year), f"daily_basic {year}")',
        ),
    ],
    "tests/test_sources.py": [
        (
            "def test_classify_error_kinds():",
            "def test_normalize_codes_skips_unknown_legacy_code():\n"
            '    """Tushare 历史遗留代码跳过而不是中断清洗，正常代码不受影响。"""\n'
            '    assert sources.normalize_codes(["600000.SH", "T00018.SH"], "测试") == ["SH600000", None]\n'
            "\n"
            "\n"
            "def test_classify_error_kinds():",
        ),
    ],
}


def main() -> int:
    """逐文件做定点替换，全部命中后一次性写盘。"""
    pending: list[tuple[Path, str]] = []
    for relative, edits in EDITS.items():
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        for old, new in edits:
            hits = text.count(old)
            if hits != 1:
                print(f"FAIL {relative}: 锚点命中 {hits} 次（期望 1 次）—— {old[:60]!r}")
                return 1
            text = text.replace(old, new)
        pending.append((path, text))
    for path, text in pending:
        path.write_text(text, encoding="utf-8")
        print(f"OK 写入 {path}")
    return 0


raise SystemExit(main())
