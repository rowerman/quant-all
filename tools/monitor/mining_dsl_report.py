"""汇总受控 DSL 原生执行器测量并生成报告。"""

from __future__ import annotations

from pathlib import Path
import statistics
from typing import Any

from tools.monitor.baseline_contract import atomic_write_text
from tools.monitor.mining_dsl_analysis import (
    ROUTE_COSTS,
    SCHEMA_VERSION,
    decision_from_measurements,
)
from tools.monitor.mining_native_common import QUICK_END, QUICK_START


def _median(values: list[float]) -> float:
    """返回浮点中位数。"""

    return float(statistics.median(values))


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总多轮阶段耗时、收益上限和稳定结果。"""

    if not runs:
        raise ValueError("至少需要一轮测量")
    stage_names = tuple(runs[0]["stages_s"])
    stage_s = {
        name: _median([float(run["stages_s"][name]) for run in runs])
        for name in stage_names
    }
    profile_categories = tuple(runs[0]["profile"]["category_cpu_s"])
    category_cpu_s = {
        name: _median(
            [float(run["profile"]["category_cpu_s"][name]) for run in runs]
        )
        for name in profile_categories
    }
    category_wall_s = {
        name: _median(
            [
                float(run["profile"]["category_wall_equivalent_s"][name])
                for run in runs
            ]
        )
        for name in profile_categories
    }
    query_wall_s = _median(
        [float(run["profile"]["query_wall_s"]) for run in runs]
    )
    query_cpu_s = _median(
        [float(run["profile"]["query_cpu_s"]) for run in runs]
    )
    expression_s = category_wall_s["expression"]
    materialization_s = stage_s["materialization"]
    total_s = stage_s["total_candidate"]
    expression_pipeline_share = expression_s / total_s
    expression_materialization_share = expression_s / materialization_s
    decision = decision_from_measurements(
        expression_pipeline_share,
        expression_pipeline_share,
    )
    stable_hashes = {str(run["stable_payload_sha256"]) for run in runs}
    if len(stable_hashes) != 1:
        raise RuntimeError("多轮候选稳定 payload 哈希不一致")
    decision["stable_payload_sha256"] = stable_hashes.pop()
    return {
        "schema_version": SCHEMA_VERSION,
        "repeat_count": len(runs),
        "stage_s": {
            name: round(value, 6)
            for name, value in stage_s.items()
        },
        "profile": {
            "query_wall_s": round(query_wall_s, 6),
            "query_cpu_s": round(query_cpu_s, 6),
            "category_cpu_s": {
                name: round(value, 6)
                for name, value in category_cpu_s.items()
            },
            "category_wall_equivalent_s": {
                name: round(value, 6)
                for name, value in category_wall_s.items()
            },
            "expression_share_of_query_cpu": (
                category_cpu_s["expression"] / query_cpu_s
                if query_cpu_s
                else 0.0
            ),
            "expression_share_of_query_wall_equivalent": (
                expression_s / query_wall_s if query_wall_s else 0.0
            ),
            "expression_share_of_materialization": (
                expression_materialization_share
            ),
            "expression_share_of_pipeline": expression_pipeline_share,
            "zero_cost_improvement_upper_bound": expression_pipeline_share,
        },
        "peak_rss_mb": round(
            max(float(run["peak_rss_mb"]) for run in runs),
            3,
        ),
        "threads": runs[0]["threads"],
        "decision": decision,
        "routes": ROUTE_COSTS,
    }


def _route_table(summary: dict[str, Any]) -> list[str]:
    """返回路线成本 Markdown 表格行。"""

    lines = [
        "| 路线 | 版本锁定 | 构建 | 调试 | 崩溃隔离 | 测试 | 相对成本 | 结论 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, route in summary["routes"].items():
        lines.append(
            f"| {name} | {route['version_lock']} | {route['build']} | "
            f"{route['debug']} | {route['crash_isolation']} | {route['test']} | "
            f"{route['relative_cost']} | {route['summary']} |"
        )
    return lines


def write_report(
    output: Path,
    corpus: dict[str, Any],
    summary: dict[str, Any],
    input_hash: str,
) -> None:
    """写出人工可读的决策报告。"""

    decision = summary["decision"]
    stage = summary["stage_s"]
    profile = summary["profile"]
    operator_counts = "、".join(
        f"{name}={count}"
        for name, count in corpus["operator_counts"].items()
    )
    lines = [
        "# P2-03 受控 DSL 原生执行器决策",
        "",
        "## 口径",
        "",
        f"- 区间：`{QUICK_START}`～`{QUICK_END}`；固定四候选、单进程、"
        f"{summary['repeat_count']} 轮中位数。",
        f"- 输入树 SHA-256：`{input_hash}`。",
        f"- DuckDB 线程：{summary['threads']['duckdb']}；native 记录线程："
        f"{summary['threads']['native']}。",
        f"- 覆盖语料：{corpus['candidate_count']} 个去重表达式，覆盖 "
        f"{corpus['coverage']['covered_operator_count']}/"
        f"{corpus['coverage']['catalog_operator_count']} 个冻结算子。",
        "- 当前无正式候选库，语料频率只代表仓库覆盖样本，不代表线上分布。",
        "",
        "## 阶段",
        "",
        "| 阶段 | 中位耗时(s) |",
        "|---|---:|",
        f"| DAG 计划 | {stage['plan']:.6f} |",
        f"| 批量 SQL 编译 | {stage['compile']:.6f} |",
        f"| DuckDB COPY | {stage['query_copy']:.6f} |",
        f"| 候选拆分发布 | {stage['split_publish']:.6f} |",
        f"| 候选物化合计 | {stage['materialization']:.6f} |",
        f"| 技术门禁与快速筛选 | {stage['evaluation']:.6f} |",
        f"| 完整候选流程 | {stage['total_candidate']:.6f} |",
        "",
        "## Profile",
        "",
        "算子时间为 DuckDB 多线程累计 CPU 时间；墙钟等效值按查询 CPU/墙钟比例折算。",
        "",
        "| 分类 | 中位 CPU(s) | 墙钟等效(s) |",
        "|---|---:|---:|",
    ]
    for name, value in profile["category_cpu_s"].items():
        lines.append(
            f"| {name} | {value:.6f} | "
            f"{profile['category_wall_equivalent_s'][name]:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- 查询墙钟：{profile['query_wall_s']:.6f}s；累计 CPU："
            f"{profile['query_cpu_s']:.6f}s。",
            f"- 表达式占批量 SQL CPU："
            f"{profile['expression_share_of_query_cpu']:.2%}；墙钟等效占比："
            f"{profile['expression_share_of_query_wall_equivalent']:.2%}。",
            f"- 表达式占候选物化："
            f"{profile['expression_share_of_materialization']:.2%}。",
            f"- 表达式占完整候选流程："
            f"{profile['expression_share_of_pipeline']:.2%}。",
            "- 若表达式耗时降为零，完整流程改善上限："
            f"{profile['zero_cost_improvement_upper_bound']:.2%}。",
            "",
            "## 算子复杂度",
            "",
            f"- 深度：均值 {corpus['depth']['mean']:.3f}，中位 "
            f"{corpus['depth']['median']:.3f}，P95 "
            f"{corpus['depth']['p95']:.3f}，最大 {corpus['depth']['max']}。",
            f"- 节点数：均值 {corpus['node_count']['mean']:.3f}，中位 "
            f"{corpus['node_count']['median']:.3f}，P95 "
            f"{corpus['node_count']['p95']:.3f}，最大 "
            f"{corpus['node_count']['max']}。",
            f"- 算子出现次数：{operator_counts}。",
            "",
            "## 路线成本",
            "",
            *_route_table(summary),
            "",
            "成本分值：0=无新增，1=中，2=高。",
            "",
            "## 结论",
            "",
            f"- 决策：`{decision['decision']}`。",
            f"- 原因：{', '.join(decision['reasons']) or '达到后续设计门槛'}。",
            "- 本轮未编写原生表达式 VM，未修改生产路由。",
            "- 实际 C++ 替换模块新增：0；C++ 替换台账不变。",
            "",
            "## 复现",
            "",
            "```bash",
            ".venv/bin/python -m tools.monitor.mining_dsl_decision "
            "--output AiRecord/benchmarks/V122/p2-03-20261028",
            "```",
        ]
    )
    atomic_write_text(output / "report.md", "\n".join(lines) + "\n")
