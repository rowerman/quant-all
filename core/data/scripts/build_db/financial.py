"""财务三表与财务指标入库。

原始 parquet 保留 Tushare 全字段；正式表只保留当前因子和追溯所需的核心字段。
时点口径统一为 ``trade_date > available_date``：公告日次一交易日起才可用于信号，
避免收盘后公告被当天收盘信号提前使用。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import time

import duckdb

from scripts.config import EXTRA_DIR
from scripts.errors import DbWriteError
from scripts.logging_conf import get_logger

logger = get_logger("build_db.financial")

TUSHARE_DIR = EXTRA_DIR / "tushare"
FINANCIAL_ROOT = TUSHARE_DIR / "raw_financial"
MIN_SAFETY_BYTES = 4 * 1024**3


@dataclass(frozen=True)
class FinancialTable:
    """一张财务表的源目录、目标表和核心指标。"""

    source_dir: str
    target_table: str
    metrics: tuple[str, ...]
    has_f_ann_date: bool = True


FINANCIAL_TABLES: tuple[FinancialTable, ...] = (
    FinancialTable(
        source_dir="income",
        target_table="income_statement",
        metrics=(
            "total_revenue", "revenue", "oper_cost", "operate_profit",
            "n_income", "n_income_attr_p", "rd_exp",
        ),
    ),
    FinancialTable(
        source_dir="balancesheet",
        target_table="balance_sheet",
        metrics=(
            "total_assets", "total_liab", "total_hldr_eqy_exc_min_int",
            "total_cur_assets", "total_cur_liab", "money_cap",
            "accounts_receiv", "inventories", "goodwill", "total_share",
        ),
    ),
    FinancialTable(
        source_dir="cashflow",
        target_table="cash_flow",
        metrics=(
            "n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act",
            "c_pay_acq_const_fiolta", "free_cashflow", "depr_fa_coga_dpba",
            "n_incr_cash_cash_equ",
        ),
    ),
    FinancialTable(
        source_dir="fina_indicator",
        target_table="fina_indicator",
        metrics=(
            "profit_dedt", "roe", "roe_waa", "roe_dt", "roe_yearly",
            "roa", "roa_yearly", "roa2_yearly", "roic",
            "grossprofit_margin", "netprofit_margin", "debt_to_assets",
            "current_ratio", "quick_ratio", "cash_ratio", "assets_turn",
            "or_yoy", "tr_yoy", "netprofit_yoy", "op_yoy", "ocf_yoy",
            "assets_yoy", "eqt_yoy", "q_sales_yoy", "q_op_qoq",
        ),
        has_f_ann_date=False,
    ),
)

KEY_COLUMNS = (
    "stock_code", "ts_code", "report_date", "ann_date", "f_ann_date",
    "available_date", "report_type", "update_flag", "comp_type",
)


def _source_glob(root: Path, spec: FinancialTable) -> str:
    """返回一张财务表的 parquet 通配路径。"""
    return str(root / "raw_financial" / spec.source_dir / "period=*.parquet")


def _source_files(root: Path, spec: FinancialTable) -> list[Path]:
    """列出源分片文件。"""
    return sorted((root / "raw_financial" / spec.source_dir).glob("period=*.parquet"))


def _source_digest(root: Path) -> tuple[str, str | None]:
    """生成源数据摘要和清单更新时间。"""
    digest = hashlib.sha256()
    updated_at = None
    manifest = root / "manifests" / "core_resources.json"
    if manifest.exists():
        payload = manifest.read_bytes()
        digest.update(payload)
        try:
            updated_at = json.loads(payload).get("updated_at")
        except json.JSONDecodeError:
            updated_at = None
    for spec in FINANCIAL_TABLES:
        for path in _source_files(root, spec):
            stat = path.stat()
            digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest(), updated_at


def _code_sql() -> str:
    """把 Tushare 代码转成项目口径，非法代码返回 NULL。"""
    return """
    CASE
      WHEN regexp_matches(ts_code, '^[0-9]{6}\\.SH$') THEN 'SH' || substr(ts_code, 1, 6)
      WHEN regexp_matches(ts_code, '^[0-9]{6}\\.SZ$') THEN 'SZ' || substr(ts_code, 1, 6)
      WHEN regexp_matches(ts_code, '^[0-9]{6}\\.BJ$') THEN 'BJ' || substr(ts_code, 1, 6)
    END
    """


def _date_sql(column: str) -> str:
    """把 YYYYMMDD 文本转成 DATE，空值保持 NULL。"""
    return f"CAST(strptime({column}, '%Y%m%d') AS DATE)"


def _available_sql(spec: FinancialTable) -> str:
    """生成公告可用日表达式。"""
    ann = _date_sql("ann_date")
    if not spec.has_f_ann_date:
        return ann
    return f"COALESCE({_date_sql('f_ann_date')}, {ann})"


def _f_ann_sql(spec: FinancialTable) -> str:
    """生成实际公告日表达式。"""
    if not spec.has_f_ann_date:
        return "CAST(NULL AS DATE)"
    return _date_sql("f_ann_date")


def _report_type_sql(spec: FinancialTable) -> str:
    """财务指标接口没有 report_type，按主表口径记 1。"""
    if spec.has_f_ann_date:
        return "CAST(report_type AS VARCHAR)"
    return "CAST('1' AS VARCHAR)"


def _comp_type_sql(spec: FinancialTable) -> str:
    """财务指标接口没有 comp_type。"""
    if spec.has_f_ann_date:
        return "CAST(comp_type AS VARCHAR)"
    return "CAST(NULL AS VARCHAR)"


def _metric_sql(spec: FinancialTable) -> str:
    """把核心指标统一转成 DOUBLE。"""
    return ",\n".join(f"CAST({name} AS DOUBLE) AS {name}" for name in spec.metrics)


def _raw_view_sql(root: Path, spec: FinancialTable) -> str:
    """构造带代码、日期和可用日的原始视图。"""
    glob = _source_glob(root, spec)
    return f"""
    SELECT *,
           {_code_sql()} AS stock_code,
           {_date_sql("end_date")} AS report_date,
           {_date_sql("ann_date")} AS ann_date_value,
           {_f_ann_sql(spec)} AS f_ann_date_value,
           {_available_sql(spec)} AS available_date
    FROM read_parquet('{glob}', union_by_name = true)
    """


def _ranked_view_sql(root: Path, spec: FinancialTable) -> str:
    """过滤非法键，并按报告修订优先级去重。"""
    raw = _raw_view_sql(root, spec)
    return f"""
    WITH raw AS ({raw}),
    valid AS (
      SELECT * FROM raw
      WHERE stock_code IS NOT NULL
        AND report_date IS NOT NULL
        AND ann_date_value IS NOT NULL
        AND available_date IS NOT NULL
        AND available_date >= report_date
    ),
    ranked AS (
      SELECT *,
             row_number() OVER (
               PARTITION BY stock_code, report_date, available_date, {_report_type_sql(spec)}
               ORDER BY TRY_CAST(update_flag AS INTEGER) DESC NULLS LAST,
                        f_ann_date_value DESC NULLS LAST,
                        ann_date_value DESC NULLS LAST
             ) AS rn
      FROM valid
    )
    SELECT * FROM ranked
    """


def _metrics_from_ranked(spec: FinancialTable) -> str:
    """生成最终表的列清单。"""
    metric_columns = ",\n           ".join(spec.metrics)
    return f"""
           stock_code,
           ts_code,
           report_date,
           ann_date_value AS ann_date,
           f_ann_date_value AS f_ann_date,
           available_date,
           {_report_type_sql(spec)} AS report_type,
           CAST(update_flag AS VARCHAR) AS update_flag,
           {_comp_type_sql(spec)} AS comp_type,
           {metric_columns}
    """


def _table_sql(root: Path, spec: FinancialTable) -> str:
    """生成一张正式财务表的建表 SQL。"""
    ranked = _ranked_view_sql(root, spec)
    return f"""
    CREATE OR REPLACE TABLE fin.{spec.target_table} AS
    SELECT {_metrics_from_ranked(spec)}
    FROM ({ranked})
    WHERE rn = 1
    """


def _count_sql(root: Path, spec: FinancialTable) -> str:
    """统计源行、非法键、日期倒置和重复行。"""
    raw = _raw_view_sql(root, spec)
    return f"""
    WITH raw AS ({raw}),
    bad_keys AS (
      SELECT count(*) AS n FROM raw
      WHERE stock_code IS NULL OR report_date IS NULL OR ann_date_value IS NULL
    ),
    bad_dates AS (
      SELECT count(*) AS n FROM raw
      WHERE stock_code IS NOT NULL AND report_date IS NOT NULL
        AND ann_date_value IS NOT NULL AND available_date IS NOT NULL
        AND available_date < report_date
    ),
    valid AS (
      SELECT * FROM raw
      WHERE stock_code IS NOT NULL AND report_date IS NOT NULL
        AND ann_date_value IS NOT NULL AND available_date IS NOT NULL
        AND available_date >= report_date
    ),
    ranked AS (
      SELECT *,
             row_number() OVER (
               PARTITION BY stock_code, report_date, available_date, {_report_type_sql(spec)}
               ORDER BY TRY_CAST(update_flag AS INTEGER) DESC NULLS LAST,
                        f_ann_date_value DESC NULLS LAST,
                        ann_date_value DESC NULLS LAST
             ) AS rn
      FROM valid
    )
    SELECT
      (SELECT count(*) FROM raw) AS source_rows,
      (SELECT n FROM bad_keys) AS invalid_codes,
      (SELECT n FROM bad_dates) AS invalid_dates,
      (SELECT count(*) FROM ranked WHERE rn > 1) AS duplicates
    """


def validate_inputs(root: Path = TUSHARE_DIR) -> dict:
    """在建库前检查财务源文件和必需字段。

    参数:
        root: ``data/source/tushare`` 目录。
    """
    root = Path(root)
    conn = duckdb.connect()
    try:
        result = {}
        for spec in FINANCIAL_TABLES:
            files = _source_files(root, spec)
            if not files:
                raise DbWriteError(f"缺少财务分片：{root / 'raw_financial' / spec.source_dir}")
            columns = {
                row[0] for row in conn.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{_source_glob(root, spec)}', union_by_name = true)"
                ).fetchall()
            }
            required = {"ts_code", "ann_date", "end_date", "update_flag", *spec.metrics}
            if spec.has_f_ann_date:
                required.update({"f_ann_date", "report_type", "comp_type"})
            missing = required - columns
            if missing:
                raise DbWriteError(
                    f"{spec.source_dir} 缺少财务字段：{sorted(missing)}"
                )
            result[spec.target_table] = {"files": len(files)}
        return result
    finally:
        conn.close()


def preflight(db_file: Path, db_dir: Path, root: Path = TUSHARE_DIR) -> dict:
    """检查磁盘空间是否足够删除旧库后重建。

    参数:
        db_file: 主库路径。
        db_dir: 主库目录。
        root: ``data/source/tushare`` 目录。
    """
    root = Path(root)
    validate_inputs(root)
    free = shutil.disk_usage(db_dir).free
    db_size = db_file.stat().st_size if db_file.exists() else 0
    raw_size = sum(
        path.stat().st_size
        for spec in FINANCIAL_TABLES
        for path in _source_files(root, spec)
    )
    estimate = db_size + max(1024**3, int(raw_size * 1.5))
    after_delete = free + db_size
    if after_delete < estimate + MIN_SAFETY_BYTES:
        raise DbWriteError(
            "磁盘空间不足，未删除旧库："
            f"当前可用 {free / 1024**3:.1f}GB，旧库 {db_size / 1024**3:.1f}GB，"
            f"预计新库 {estimate / 1024**3:.1f}GB，需要保留 4GB 安全余量"
        )
    return {
        "free_gb": round(free / 1024**3, 2),
        "old_db_gb": round(db_size / 1024**3, 2),
        "estimated_new_db_gb": round(estimate / 1024**3, 2),
        "after_delete_gb": round(after_delete / 1024**3, 2),
    }


def build_financial(
    conn: duckdb.DuckDBPyConnection,
    root: Path = TUSHARE_DIR,
) -> dict:
    """把四张财务表导入 DuckDB。

    参数:
        conn: 已创建 ``fin`` 与 ``meta`` schema 的连接。
        root: ``data/source/tushare`` 目录。
    """
    root = Path(root)
    started = time.time()
    digest, source_updated_at = _source_digest(root)
    rows: list[dict] = []
    for spec in FINANCIAL_TABLES:
        stats = conn.execute(_count_sql(root, spec)).fetchone()
        conn.execute(_table_sql(root, spec))
        summary = conn.execute(
            f"""
            SELECT count(*), count(DISTINCT stock_code),
                   min(report_date), max(report_date),
                   min(available_date), max(available_date)
            FROM fin.{spec.target_table}
            """
        ).fetchone()
        row = {
            "table_name": spec.target_table,
            "rows": int(summary[0]),
            "stocks": int(summary[1]),
            "source_rows": int(stats[0]),
            "source_files": len(_source_files(root, spec)),
            "min_report_date": summary[2],
            "max_report_date": summary[3],
            "min_available_date": summary[4],
            "max_available_date": summary[5],
            "invalid_codes_dropped": int(stats[1]),
            "invalid_dates_dropped": int(stats[2]),
            "duplicates_dropped": int(stats[3]),
        }
        rows.append(row)
        logger.info(
            "fin.%s 导入 %d 行、%d 只股票；源 %d 行，去重 %d，异常日期 %d",
            spec.target_table, row["rows"], row["stocks"], row["source_rows"],
            row["duplicates_dropped"], row["invalid_dates_dropped"],
        )
    conn.execute(
        """
        CREATE OR REPLACE TABLE meta.financial_manifest (
          table_name VARCHAR,
          rows BIGINT,
          stocks BIGINT,
          source_rows BIGINT,
          source_files INTEGER,
          min_report_date DATE,
          max_report_date DATE,
          min_available_date DATE,
          max_available_date DATE,
          invalid_codes_dropped BIGINT,
          invalid_dates_dropped BIGINT,
          duplicates_dropped BIGINT,
          source_digest VARCHAR,
          source_updated_at VARCHAR,
          pit_rule VARCHAR,
          built_at TIMESTAMP
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO meta.financial_manifest VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
        """,
        [
            (
                row["table_name"], row["rows"], row["stocks"], row["source_rows"],
                row["source_files"], row["min_report_date"], row["max_report_date"],
                row["min_available_date"], row["max_available_date"],
                row["invalid_codes_dropped"], row["invalid_dates_dropped"],
                row["duplicates_dropped"], digest, source_updated_at,
                "trade_date > available_date",
            )
            for row in rows
        ],
    )
    elapsed = round(time.time() - started, 1)
    logger.info("财务表导入完成：%d 张，耗时 %.1f 秒", len(rows), elapsed)
    return {"tables": rows, "source_digest": digest, "elapsed_seconds": elapsed}
