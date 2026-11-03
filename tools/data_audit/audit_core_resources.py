"""审计 V22 下载的财务、事件与历史行业原始资源。"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sources.core_resources import ALWAYS_NONEMPTY_APIS, MANIFEST_PATH, TUSHARE_DIR


def _full_path(value: str | None) -> Path | None:
    """把清单里的相对路径还原成文件路径。"""
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else TUSHARE_DIR / path


def audit() -> dict:
    """读取资源清单和所有非空分片，返回按接口汇总的审计结果。"""
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"资源清单不存在：{MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    grouped: dict[str, dict] = defaultdict(
        lambda: {
            "partitions": 0, "complete": 0, "failed": 0, "empty": 0, "rows": 0,
            "files": 0, "missing_files": 0, "exact_duplicates": 0,
            "min_date": None, "max_date": None,
        }
    )
    failures = []
    empties = []
    for key, item in manifest.get("items", {}).items():
        api = str(item.get("api") or key.split(":", 1)[0])
        stats = grouped[api]
        stats["partitions"] += 1
        status = item.get("status")
        stats["complete"] += status == "complete"
        stats["failed"] += status == "failed"
        stats["empty"] += status == "empty"
        stats["rows"] += int(item.get("rows", 0))
        if status == "failed":
            failures.append({"key": key, "error": item.get("error")})
        if status == "empty":
            empties.append({"key": key, "reason": item.get("error")})
            continue
        if (
            api in ALWAYS_NONEMPTY_APIS
            and str(item.get("partition", "")) >= "20050101"
            and status == "complete"
            and int(item.get("rows", 0)) == 0
        ):
            failures.append({"key": key, "error": "2005 年后财务主表异常为空"})
        for field, fn in (("min_date", min), ("max_date", max)):
            value = item.get(field)
            if value:
                stats[field] = value if stats[field] is None else fn(stats[field], value)
        path = _full_path(item.get("path"))
        if int(item.get("rows", 0)) == 0:
            continue
        if path is None or not path.exists():
            stats["missing_files"] += 1
            continue
        frame = pd.read_parquet(path)
        stats["files"] += 1
        stats["exact_duplicates"] += int(frame.duplicated().sum())
        if len(frame) != int(item.get("rows", 0)):
            failures.append(
                {"key": key, "error": f"清单行数 {item.get('rows')} != 文件行数 {len(frame)}"}
            )
        if api == "index_member" and not frame.empty:
            required = {"index_code", "con_code", "in_date", "out_date", "is_new"}
            missing = required - set(frame.columns)
            if missing:
                failures.append({"key": key, "error": f"缺少字段 {sorted(missing)}"})
                continue
            expected = str(item.get("partition"))
            if key.startswith("index_member_ts_code:"):
                actual = set(frame["con_code"].astype(str))
            else:
                actual = set(frame["index_code"].astype(str))
            if actual != {expected}:
                failures.append({"key": key, "error": "分片混入其它代码"})
            if frame["in_date"].isna().any():
                failures.append({"key": key, "error": "存在空 in_date"})
            dated = frame.loc[frame["out_date"].notna(), ["in_date", "out_date"]].astype(str)
            if (dated["out_date"] < dated["in_date"]).any():
                failures.append({"key": key, "error": "存在 out_date 早于 in_date"})
    return {
        "resources": dict(sorted(grouped.items())),
        "failures": failures,
        "empty": empties,
    }


def main() -> None:
    """打印 JSON 审计报告，失败时返回非零退出码。"""
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
