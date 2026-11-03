"""清洗逐层剖析的 Markdown 报告。"""

from __future__ import annotations

from typing import Any

from tools.monitor.clean_profile_contract import CASE_ORDER, PRIMARY_CASE


def _case_table(cases: dict[str, dict[str, Any]]) -> list[str]:
    """生成案例汇总 Markdown 表。"""

    lines = [
        "| 案例 | 角色 | 中位总耗时(s) | 峰值RSS(MB) | 最大纯数组规则 | 占比 | DuckDB 日线(s) |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for name in CASE_ORDER:
        if name not in cases:
            continue
        case = cases[name]
        median = case["median"]
        profile = median["profile"]
        rule = profile["largest_pure_array_stage"] or "—"
        lines.append(
            f"| {name} | {case['definition']['role']} | "
            f"{median['elapsed_s']:.3f} | {median['peak_rss_mb']:.1f} | "
            f"`{rule}` | {profile['largest_pure_array_share']:.2%} | "
            f"{median['duckdb_daily_s']:.3f} |"
        )
    return lines


def _stage_table(case: dict[str, Any]) -> list[str]:
    """生成主案例阶段耗时 Markdown 表。"""

    stages = case["median"]["profile"]["stages"]
    ordered = sorted(
        stages.items(),
        key=lambda item: float(item[1]["elapsed_s"]),
        reverse=True,
    )
    lines = [
        "| 阶段 | 类别 | 纯数组 | 次数 | 中位耗时(s) | 占比 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, stage in ordered:
        lines.append(
            f"| `{name}` | {stage['category']} | "
            f"{'是' if stage['pure_array'] else '否'} | {stage['count']} | "
            f"{stage['elapsed_s']:.6f} | {stage['share']:.2%} |"
        )
    return lines


def render_report(result: dict[str, Any]) -> str:
    """生成人类可读的剖析与决策报告。"""

    decision = result["decision"]
    lines = [
        "# V124 / P3-02 清洗层条件化剖析",
        "",
        "## 结论",
        "",
        f"结论为 `{decision['status']}`。{decision['reason']}",
        "",
        f"- 主案例：`{decision['primary_case']}`",
        f"- C++ 门槛：最大纯数组规则占清洗总耗时至少 `{decision['threshold']:.0%}`",
        f"- 实际新增 C++ 替换模块：`{decision['session_cpp_modules_added']}`",
        f"- 当前累计替换模块数：`{decision['existing_cpp_replacement_total']}`",
        "",
        "## 案例",
        "",
        *_case_table(result["cases"]),
        "",
        "## 主案例阶段",
        "",
        *_stage_table(result["cases"][PRIMARY_CASE]),
        "",
        "## 口径",
        "",
        "- 每案独立子进程运行，目标年度重复三次取中位数。",
        "- 先运行上一年度生成 `carry_close`，但不计入目标年度耗时。",
        "- 阶段均为单层叶调用，避免父函数和子函数重复累计。",
        "- DuckDB 只隔离构建日线三张同构表，单独报告，不进入清洗 C++ 决策分母。",
        "- 所有清洗和数据库输出均位于临时目录，不读取或覆盖生产清洗产物。",
        "",
        "## 正确性",
        "",
    ]
    for name, case in result["cases"].items():
        stability = case["stability"]
        lines.append(
            f"- `{name}` {case['repeat_count']} 次内容摘要一致："
            f"{stability['content_exact']}；"
            f"稳定统计一致：{stability['stats_exact']}；"
            f"摘要 `{stability['content_sha256']}`。"
        )
    lines.extend(
        [
            "",
            "## 复现",
            "",
            "```bash",
            ".venv/bin/python -m tools.monitor.clean_profile --case all --repeats 3",
            "```",
            "",
            "## 边界",
            "",
            "- 历史案例只作规则与耗时交叉检查，生产决策以 `2025` 为准。",
            "- 本轮只形成 C++ 路线决策，不优化剖析发现的 Python 瓶颈。",
            "- 未修改生产清洗数据、主库、正式因子、模型、候选或回测结果。",
        ]
    )
    return "\n".join(lines) + "\n"
